//! MASC 3.0.0's Penn part-of-speech tags (CC BY 3.0 US), read from the
//! release `turbo-train-recipes` records, with its fixed split: every
//! tenth document held out (the test split here), the Twitter genre left
//! out of both sides. The test documents are the ones the MASC tagger's
//! gate scores, so the numbers compare.
//!
//! MASC has no dev split. The trainer takes every tenth training-side
//! document, in corpus order, as dev (`DEV_EVERY`), so early stopping
//! never sees a test document.

use std::path::Path;

use turbo_formats::masc::MascCorpus;
use turbo_train_recipes::masc::{self, RELEASE, Release, SPLIT};
use turbo_train_recipes::postag::read_samples;

use crate::corpus::{Corpus, CorpusRecord, Counts, Sentence, Splits, Task};
use crate::{Error, Result};

pub const ID: &str = "masc-pos";
/// MASC's licence (`masc::TERMS`), as the zoo names it.
pub const SPDX: &str = "CC-BY-3.0-US";
/// Training-side documents at `DEV_EVERY - 1` modulo `DEV_EVERY`, in
/// corpus order, are dev.
pub const DEV_EVERY: usize = 10;

fn sentences(corpus: &MascCorpus) -> Result<(Vec<Sentence>, usize)> {
    let mut out = Vec::new();
    let skipped = read_samples(corpus, |s| {
        out.push(Sentence {
            words: s.tokens().iter().map(|t| t.text.to_string()).collect(),
            labels: s.tags().iter().map(|t| t.text.to_string()).collect(),
        });
    })
    .map_err(|e| Error(format!("MASC: {e}")))?;
    Ok((out, skipped))
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
        )));
    }
    let corpus = MascCorpus::read(dir).map_err(|e| Error(format!("MASC: {e}")))?;
    let split = SPLIT.apply(corpus, dir).map_err(|e| Error(format!("MASC: {e}")))?;
    let (mut train_docs, mut dev_docs) = (Vec::new(), Vec::new());
    for (i, d) in split.train.into_documents().into_iter().enumerate() {
        if i % DEV_EVERY == DEV_EVERY - 1 { dev_docs.push(d) } else { train_docs.push(d) }
    }
    let (train, a) = sentences(&MascCorpus::from_documents(train_docs))?;
    let (dev, b) = sentences(&MascCorpus::from_documents(dev_docs))?;
    let (test, c) = sentences(&split.held_out)?;
    let record = CorpusRecord {
        id: ID.into(),
        name: masc::NAME.into(),
        task: Task::Tags,
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
        skipped: a + b + c,
    };
    Ok(Corpus { record, splits: Splits { train, dev, test } })
}
