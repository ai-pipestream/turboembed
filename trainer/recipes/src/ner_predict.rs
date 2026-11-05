//! Native BIO predictions from a verified assembled bundle. Assembly and
//! storage resolution belong to the shared bundle and zoo interfaces.
use std::fmt::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};
use turbo::bundle::{Bundle, sha256_hex};
use turbo::manifest::{HeadSpec, LicenseClass, Scheme};
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::HeadRunner;

use crate::corpus::{Corpus, CorpusRecord, FileRecord, Task};
use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionRecord {
    pub schema: String,
    pub commit: String,
    pub corpus: CorpusRecord,
    pub bundle_manifest_sha256: String,
    pub head: HeadSpec,
    pub backend: String,
    pub device: String,
    pub precision: String,
    pub predictions: FileRecord,
}

pub struct Prediction {
    pub bio: String,
    pub record: PredictionRecord,
}

/// Predict only the corpus's held-out words, without re-tokenizing or
/// supplying its labels to the encoder. An explicit head is mandatory
/// because German outer and inner heads write the same layer.
pub fn predict(
    corpus: &Corpus,
    bundle: &Path,
    head_name: &str,
    options: &EncoderOptions,
    commit: &str,
) -> Result<Prediction> {
    if corpus.record.task != Task::Bio || corpus.splits.test.is_empty() {
        return Err(Error("predict-ner needs a nonempty BIO test corpus".into()));
    }
    if commit.trim().is_empty() || head_name.trim().is_empty() {
        return Err(Error("predict-ner needs a commit and explicit head name".into()));
    }
    // Check the exchange domain before doing encoder work.
    crate::ner_peer::export(&corpus.splits.test)?;
    let sealed = Bundle::open(bundle).map_err(|e| Error(e.message))?;
    let spec = sealed
        .manifest
        .heads
        .iter()
        .find(|h| h.name == head_name)
        .ok_or_else(|| Error(format!("no head {head_name:?}")))?;
    if spec.scheme != Scheme::Bio || !turbo_tagger::annotate::same_layer(&spec.layer, "turbo:entities") {
        return Err(Error("predict-ner needs a BIO entity head".into()));
    }
    if spec.labels != corpus.labels() {
        return Err(Error("head label schema differs from the corpus training schema".into()));
    }
    if corpus.record.id == crate::multiconer::ID
        && (spec.license_class != Some(LicenseClass::ShareAlike)
            || sealed.manifest.heads.iter().any(|h| h.name != head_name))
    {
        return Err(Error("MultiCoNER requires its own share-alike bundle, separate from other heads".into()));
    }
    let runner = HeadRunner::open(bundle, Some(head_name), "turbo:entities", options).map_err(|e| Error(e.message))?;
    if runner.model().sha256 != sealed.manifest_sha256 {
        return Err(Error("bundle manifest changed while opening prediction".into()));
    }
    let mut predicted = Vec::with_capacity(corpus.splits.test.len());
    for chunk in corpus.splits.test.chunks(crate::run::PREDICT_CHUNK) {
        let words: Vec<Vec<&str>> = chunk.iter().map(|s| s.words.iter().map(String::as_str).collect()).collect();
        let decoded = runner.decode(&words).map_err(|e| Error(e.message))?;
        if decoded.len() != words.len() {
            return Err(Error("decoder sentence count differs".into()));
        }
        for (row, words) in decoded.iter().zip(&words) {
            if row.tags.len() != words.len() {
                return Err(Error("decoder token count differs".into()));
            }
            let labels: Vec<String> = row
                .tags
                .iter()
                .map(|&t| {
                    spec.labels.get(t as usize).cloned().ok_or_else(|| Error("decoder label outside schema".into()))
                })
                .collect::<Result<_>>()?;
            predicted.push(labels);
        }
    }
    crate::ner_peer::validate(&corpus.record.id, &predicted)?;
    let mut bio = String::new();
    for (sentence, labels) in corpus.splits.test.iter().zip(&predicted) {
        for (word, label) in sentence.words.iter().zip(labels) {
            writeln!(bio, "{word}\t{label}").expect("String write");
        }
        bio.push('\n');
    }
    let execution = runner.execution();
    let record = PredictionRecord {
        schema: "turboembed.ner.prediction/1".into(),
        commit: commit.into(),
        corpus: corpus.record.clone(),
        bundle_manifest_sha256: sealed.manifest_sha256,
        head: spec.clone(),
        backend: execution.backend.clone(),
        device: execution.device.clone(),
        precision: execution.precision.clone(),
        predictions: FileRecord {
            path: "predictions.bio".into(),
            size: bio.len() as u64,
            sha256: sha256_hex(bio.as_bytes()),
        },
    };
    Ok(Prediction { bio, record })
}
