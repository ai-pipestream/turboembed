//! MASC named entities as a separate four-type BIO head. Uses the same
//! release pin, held-out documents and Twitter exclusion as `masc_pos`.
//! Incomplete, overlapping and discontinuous entities are refused before fit.

use std::path::Path;

mod document;
pub use document::{UnsupportedAnnotation, document};

use turbo_formats::masc::{MascCorpus, MascDocument, MascError};
use turbo_train_recipes::masc::{self, RELEASE, Release, SPLIT};
use turbo_train_recipes::native_ner;

use crate::Error;
use crate::corpus::{Corpus, CorpusRecord, Counts, Sentence, Splits, Task};
type Result<T> = std::result::Result<T, ReadError>;

pub const ID: &str = "masc-ne";
pub const TYPES: [&str; 4] = native_ner::TYPES;
/// MASC's licence (`masc::TERMS`), as the zoo names it.
pub const SPDX: &str = "CC-BY-3.0-US";
/// Training-side documents at `DEV_EVERY - 1` modulo `DEV_EVERY`, in
/// corpus order, are dev.
pub const DEV_EVERY: usize = 10;

/// Typed annotation failures are preserved for callers before the CLI formats them.
#[derive(Debug)]
pub enum ReadError {
    Annotation(MascError),
    Invalid(Error),
    Unsupported(UnsupportedAnnotation),
}
impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Annotation(e) => e.fmt(f),
            Self::Invalid(e) => e.fmt(f),
            Self::Unsupported(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for ReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Annotation(e) => Some(e),
            Self::Invalid(e) => Some(e),
            Self::Unsupported(e) => Some(e),
        }
    }
}
impl From<Error> for ReadError {
    fn from(value: Error) -> Self {
        Self::Invalid(value)
    }
}
impl From<MascError> for ReadError {
    fn from(value: MascError) -> Self {
        Self::Annotation(value)
    }
}

/// Construct checked sentence labels from original token memberships.
///
/// # Errors
///
/// Cut entities produce `ReadError::Annotation(MascError::CutEntity)` with
/// document identity and source offsets; other unsupported annotations fail too.
pub fn sentences(doc: &MascDocument) -> std::result::Result<Vec<Sentence>, ReadError> {
    doc.require_complete_entities().map_err(ReadError::Annotation)?;
    let labels = native_ner::head().labels;
    native_ner::sentences(doc)
        .map_err(|e| ReadError::Invalid(Error(e.to_string())))?
        .iter()
        .map(|s| {
            let words = s
                .tokens()
                .spans
                .iter()
                .map(|(_, span)| s.view().text()[span.start as usize..span.end as usize].to_owned())
                .collect();
            Ok(Sentence { words, labels: s.labels().iter().map(|&i| labels[i as usize].clone()).collect() })
        })
        .collect()
}

/// A source document's membership and exact byte offsets in its original text.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentSamples {
    pub path: String,
    pub source_sha256: String,
    pub first: usize,
    pub end: usize,
    pub token_spans: Vec<[u32; 2]>,
}

/// Training data plus exhaustive document groups for held-out resampling.
pub struct Prepared {
    pub corpus: Corpus,
    pub train_documents: Vec<DocumentSamples>,
    pub dev_documents: Vec<DocumentSamples>,
    pub test_documents: Vec<DocumentSamples>,
}

fn samples(corpus: &MascCorpus, root: &Path) -> Result<(Vec<Sentence>, Vec<DocumentSamples>)> {
    let mut out = Vec::new();
    let mut documents = Vec::new();
    for doc in corpus.documents() {
        if let Some(sample) = document(doc)? {
            let first = out.len();
            out.push(sample);
            let path = doc.path().strip_prefix(root).map_err(|e| Error(e.to_string()))?;
            let path = path.to_str().ok_or_else(|| Error("non-UTF-8 document path".into()))?.replace('\\', "/");
            let tokens = doc.document_tokens().expect("document() checked Penn tokens");
            documents.push(DocumentSamples {
                path,
                source_sha256: turbo::bundle::sha256_hex(doc.text().as_bytes()),
                first,
                end: out.len(),
                token_spans: doc.token_spans(tokens)?.iter().map(|s| [s.start, s.end]).collect(),
            });
        }
    }
    Ok((out, documents))
}

/// Reads the extracted release at `dir` (the `MASC-3.0.0` directory).
/// A directory whose digest is not the recorded release's is refused
/// unless `unrecorded`, and the record then says `recorded: false`.
///
/// # Errors
///
/// A directory that cannot be read, a document MASC's reader refuses, or
/// a digest mismatch.
pub fn read(dir: &Path, unrecorded: bool) -> Result<Corpus> {
    Ok(read_with_documents(dir, unrecorded)?.corpus)
}

/// Read the same pinned recipe while retaining source document groups.
///
/// # Errors
///
/// The same corpus and annotation refusals as [`read`].
pub fn read_with_documents(dir: &Path, unrecorded: bool) -> Result<Prepared> {
    let digest = masc::corpus_digest(dir).map_err(|e| Error(format!("MASC: {e}")))?;
    let recorded = RELEASE.is(&digest);
    if !recorded && !unrecorded {
        let want = match RELEASE {
            Release::Recorded(r) => r.corpus_sha256,
            Release::NotRecorded => "nothing",
        };
        return Err(Error(format!(
            "MASC: {} has corpus digest {} over {} files; the release records {want} \
             (pass --unrecorded-corpus to train on it anyway)",
            dir.display(),
            digest.sha256,
            digest.files
        ))
        .into());
    }
    let corpus = MascCorpus::read(dir)?;
    let split = SPLIT.apply(corpus, dir).map_err(|e| Error(format!("MASC: {e}")))?;
    let (mut train_docs, mut dev_docs) = (Vec::new(), Vec::new());
    for (i, d) in split.train.into_documents().into_iter().enumerate() {
        if i % DEV_EVERY == DEV_EVERY - 1 { dev_docs.push(d) } else { train_docs.push(d) }
    }
    let (train, train_documents) = samples(&MascCorpus::from_documents(train_docs), dir)?;
    let (dev, dev_documents) = samples(&MascCorpus::from_documents(dev_docs), dir)?;
    let (test, test_documents) = samples(&split.held_out, dir)?;
    if digest != masc::corpus_digest(dir).map_err(|e| Error(format!("MASC: {e}")))? {
        return Err(Error("MASC source files changed during preparation".into()).into());
    }
    let record = CorpusRecord {
        id: ID.into(),
        name: masc::NAME.into(),
        task: Task::Bio,
        licence: "CC BY 3.0 US".into(),
        spdx: SPDX.into(),
        terms: masc::TERMS.into(),
        attribution: masc::ATTRIBUTION.into(),
        recorded,
        corpus_sha256: digest.sha256,
        files: Vec::new(),
        split: format!(
            "test: documents at {} modulo {} with {} left out (the recipes' split); dev: training-side \
             documents at {} modulo {DEV_EVERY}",
            SPLIT.held_out_at(),
            SPLIT.every(),
            masc::EXCLUSIONS.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
            DEV_EVERY - 1
        ),
        train: Counts::of(&train),
        dev: Counts::of(&dev),
        test: Counts::of(&test),
        skipped: 0,
        recipe_only: false,
    };
    Ok(Prepared {
        corpus: Corpus { record, splits: Splits { train, dev, test } },
        train_documents,
        dev_documents,
        test_documents,
    })
}
