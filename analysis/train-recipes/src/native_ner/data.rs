use std::collections::HashSet;
use std::path::{Component, Path};

use turbo::status::Result;
use turbo_analysis::{Document, Key, Spans, View};
use turbo_analysis_layers::Tokens;
use turbo_formats::masc::{MascCorpus, MascDocument};

use super::{TYPES, digest, field, invalid};
use crate::masc::{self, ExcludedDocument};

/// Membership is fixed before NE filtering. Test is the existing MASC
/// held-out set; dev is position 8 modulo 10 of the remaining train side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Partition {
    Train,
    Dev,
    Test,
}

/// Relative identity, content digest and partition, including documents
/// with no NE annotation. Such documents produce no training examples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentId {
    pub path: String,
    pub text_sha256: String,
    pub partition: Partition,
    pub annotated: bool,
}

/// A checked sentence, retaining full source text and original byte spans.
pub struct Sentence {
    view: View,
    tokens: Tokens,
    labels: Vec<u32>,
}

impl Sentence {
    pub fn view(&self) -> &View {
        &self.view
    }

    pub fn tokens(&self) -> &Tokens {
        &self.tokens
    }

    pub fn labels(&self) -> &[u32] {
        &self.labels
    }
}

/// Inputs and partition evidence; construction checks the source files.
pub struct Prepared {
    pub(super) train: Vec<Sentence>,
    pub(super) dev: Vec<Sentence>,
    test: Vec<Sentence>,
    documents: Vec<DocumentId>,
    excluded: Vec<ExcludedDocument>,
    corpus_sha256: String,
    membership_sha256: String,
}

impl Prepared {
    pub fn samples(&self, partition: Partition) -> &[Sentence] {
        match partition {
            Partition::Train => &self.train,
            Partition::Dev => &self.dev,
            Partition::Test => &self.test,
        }
    }

    pub fn documents(&self) -> &[DocumentId] {
        &self.documents
    }

    pub fn excluded(&self) -> &[ExcludedDocument] {
        &self.excluded
    }

    pub fn corpus_sha256(&self) -> &str {
        &self.corpus_sha256
    }

    pub fn membership_sha256(&self) -> &str {
        &self.membership_sha256
    }
}

/// Read the recorded MASC split, preserving its test documents and licence
/// exclusions. This does not certify the input's publisher or licence.
/// The corpus digest must be matched to approved data before a release run.
pub fn prepare(root: &Path) -> Result<Prepared> {
    let root = std::path::absolute(root).map_err(invalid)?;
    let before = masc::corpus_digest(&root).map_err(invalid)?;
    let corpus = MascCorpus::read(&root).map_err(invalid)?;
    let split = masc::SPLIT.apply(corpus, &root).map_err(invalid)?;
    let mut out = Prepared {
        train: Vec::new(),
        dev: Vec::new(),
        test: Vec::new(),
        documents: Vec::new(),
        excluded: split.excluded,
        corpus_sha256: before.sha256.clone(),
        membership_sha256: String::new(),
    };
    let mut seen = HashSet::new();
    let mut membership = Vec::new();
    for (docs, held_out) in [(split.train.documents(), false), (split.held_out.documents(), true)] {
        for (i, doc) in docs.iter().enumerate() {
            let rel = doc.path().strip_prefix(&root).map_err(invalid)?;
            if !rel.components().all(|c| matches!(c, Component::Normal(_))) {
                return Err(invalid("document identity is not relative"));
            }
            let path = rel.to_str().ok_or_else(|| invalid("non-UTF-8 document identity"))?.replace('\\', "/");
            if !seen.insert(path.clone()) {
                return Err(invalid("duplicate document identity"));
            }
            let partition = if held_out {
                Partition::Test
            } else if i % 10 == 8 {
                Partition::Dev
            } else {
                Partition::Train
            };
            let text_sha256 = digest(doc.text().as_bytes());
            field(&mut membership, path.as_bytes());
            field(&mut membership, text_sha256.as_bytes());
            membership.push(partition as u8);
            membership.push(u8::from(doc.has_named_entities()));
            out.documents.push(DocumentId { path, text_sha256, partition, annotated: doc.has_named_entities() });
            let samples = sentences(doc)?;
            match partition {
                Partition::Train => out.train.extend(samples),
                Partition::Dev => out.dev.extend(samples),
                Partition::Test => out.test.extend(samples),
            }
        }
    }
    if out.train.is_empty() || out.dev.is_empty() || out.test.is_empty() {
        return Err(invalid("train, development and test each need annotated sentences"));
    }
    if before != masc::corpus_digest(&root).map_err(invalid)? {
        return Err(invalid("source files changed during preparation"));
    }
    out.membership_sha256 = digest(&membership);
    Ok(out)
}

/// Read NE-annotated sentences only. Unsupported annotations are errors,
/// not discarded positives or widened spans. This uses `entities`, never
/// the legacy reader's lossy `names` projection.
pub fn sentences(doc: &MascDocument) -> Result<Vec<Sentence>> {
    if !doc.has_named_entities() {
        return Ok(Vec::new());
    }
    if !doc.unplaced_entities().is_empty() {
        return Err(invalid("document has unplaced entities"));
    }
    let document = Document::new(doc.text());
    let view = document.require(&Key::<View>::DEFAULT).map_err(invalid)?;
    let mut out = Vec::new();
    for sentence in doc.sentences() {
        let mut spans = Spans::new(view.id());
        for span in doc.token_spans(sentence).map_err(invalid)? {
            if span.start == span.end {
                return Err(invalid("empty token"));
            }
            spans.push(span);
        }
        let tokens = Tokens { spans, sentence: None };
        tokens.check_contiguous_in(view).map_err(invalid)?;
        if tokens.spans.is_empty() {
            return Err(invalid("annotated sentence has no tokens"));
        }
        let mut labels = vec![0; tokens.spans.len()];
        for entity in sentence.entities() {
            let kind = match entity.kind.as_deref() {
                Some("org") => "organization",
                Some(kind) => kind,
                None => return Err(invalid("entity has no label")),
            };
            let kind = TYPES.iter().position(|&k| k == kind).ok_or_else(|| invalid("unsupported entity label"))?;
            if !entity.is_contiguous() || entity.start >= entity.end || entity.end as usize > labels.len() {
                return Err(invalid("discontinuous or invalid entity token bounds"));
            }
            for (j, label) in labels[entity.start as usize..entity.end as usize].iter_mut().enumerate() {
                if *label != 0 {
                    return Err(invalid("overlapping entity annotations"));
                }
                *label = (1 + kind * 2 + usize::from(j != 0)) as u32;
            }
        }
        out.push(Sentence { view: view.clone(), tokens, labels });
    }
    if !doc.warnings().is_empty() {
        return Err(invalid("document has annotation warnings"));
    }
    Ok(out)
}
