//! The trained SPLATE study receipt (#913): the checkpoints a run names
//! and the record it writes.
//!
//! A receipt names each checkpoint by the upstream commit its bundle was
//! sealed from (`manifest.model.source.commit`), never by the recipe's
//! revision counter, and carries the manifest SHA-256 of every bundle
//! that ran together with the encoder, cut and fit settings, so a
//! regenerated record holds everything the archived one names.

use std::path::Path;

use serde_json::{Value, json};
use turbo::bundle::Bundle;
use turbo::manifest::Model;
use turbo::sparse::SparseOptions;

use super::colbert::{DOC_MAXLEN, QUERY_MAXLEN};
use super::trained::{TrainedConfig, TrainedProvenance};
use crate::Result;

/// What the record says it is.
pub const KIND: &str = "trained four-arm SPLATE study receipt";
/// Generic test invocation; the build block records the actual Cargo settings.
/// This command alone does not reproduce custom profile or environment overrides.
pub const RUN: &str = "cargo test --locked -p turbo-eval --test splate_trained -- --ignored --nocapture";

/// A bundle as the receipt names it: model id, the upstream commit the
/// bundle was sealed from, and the SHA-256 of the manifest that loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedBundle {
    pub id: String,
    pub commit: String,
    pub manifest_sha256: String,
}

impl PinnedBundle {
    pub fn open(dir: &Path) -> Result<Self> {
        let b = Bundle::open(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        Self::from_model(&b.manifest.model, &b.manifest_sha256)
    }

    /// Refuses a model whose source commit is not a 40-digit lowercase hex
    /// commit: a receipt that cannot name its checkpoint is not written,
    /// and git's own spelling keeps receipts byte-comparable.
    pub fn from_model(model: &Model, manifest_sha256: &str) -> Result<Self> {
        let commit = &model.source.commit;
        if commit.len() != 40 || !commit.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
            return Err(format!("{}: source commit {commit:?} is not a 40-digit lowercase hex commit", model.id));
        }
        Ok(Self { id: model.id.clone(), commit: commit.clone(), manifest_sha256: manifest_sha256.to_owned() })
    }

    /// `id@commit`, as the record names the checkpoint.
    pub fn pinned(&self) -> String {
        format!("{}@{}", self.id, self.commit)
    }
}

/// The four bundles a trained run loads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptBundles {
    pub sparse: PinnedBundle,
    pub colbert: PinnedBundle,
    pub static_model: PinnedBundle,
    pub cross_encoder: PinnedBundle,
}

/// How a run was made: the device, the timing rounds and every encoder
/// and fit knob.
#[derive(Debug, Clone, Copy)]
pub struct ReceiptRun<'a> {
    pub device: &'a str,
    pub rounds: usize,
    pub cross_encoder_threads: u32,
    pub encoder: &'a TrainedConfig,
    pub doc_cut: &'a SparseOptions,
    pub query_cut: &'a SparseOptions,
}

/// The fit's mean squared error at the six places the record keeps.
fn six_places(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

/// A cut as the record names it; the floor is written as its shortest
/// `f32` form (0.05, not 0.05000000074505806).
fn cut(c: &SparseOptions) -> Value {
    let floor: f64 = c.floor.to_string().parse().unwrap_or(f64::from(c.floor));
    json!({ "top_k": c.top_k, "floor": floor })
}

/// One set's entry: its size, its pinned checkpoints, the adapter fit and
/// the per-depth tables as `(C, markdown)`.
pub fn set_record(
    name: &str,
    docs: usize,
    queries: usize,
    bundles: &ReceiptBundles,
    prov: &TrainedProvenance,
    tables: &[(usize, String)],
) -> Value {
    json!({
        "set": name,
        "docs": docs,
        "queries": queries,
        "sparse": bundles.sparse.pinned(),
        "colbert": bundles.colbert.pinned(),
        "static": bundles.static_model.pinned(),
        "cross_encoder": bundles.cross_encoder.pinned(),
        "bundles": {
            "sparse": bundles.sparse.manifest_sha256,
            "colbert": bundles.colbert.manifest_sha256,
            "static": bundles.static_model.manifest_sha256,
            "cross_encoder": bundles.cross_encoder.manifest_sha256,
        },
        "token_dim": prov.token_dim,
        "static_dim": prov.static_dim,
        "vocab": prov.vocab_size,
        "fit_docs": prov.fit_docs_used,
        "fit_tokens": prov.fit_tokens,
        "fit_terms": prov.fit_terms,
        "fit_misaligned_docs": prov.fit_misaligned_docs,
        "fit_mse": six_places(prov.fit_mse),
        "tables": tables.iter().map(|(c, md)| json!({ "c": c, "markdown": md })).collect::<Vec<_>>(),
    })
}

/// The whole record over the sets [`set_record`] wrote.
pub fn record(run: &ReceiptRun<'_>, sets: Vec<Value>) -> Value {
    let e = run.encoder;
    json!({
        "issue": 913,
        "kind": KIND,
        "run": RUN,
        "device": run.device,
        "rounds": run.rounds,
        "build": {
            "profile": env!("TURBO_EVAL_PROFILE"),
            "opt_level": env!("TURBO_EVAL_OPT_LEVEL"),
            "debug_info": env!("TURBO_EVAL_DEBUG"),
            "debug_assertions": cfg!(debug_assertions),
        },
        "config": {
            "cross_encoder_threads": run.cross_encoder_threads,
            "max_doc_tokens": e.max_doc_tokens,
            "max_query_tokens": e.max_query_tokens,
            "fit_docs": e.fit_docs,
            "adapter_terms": e.adapter_terms,
            "token_targets": e.token_targets,
            "ridge_lambda": e.ridge_lambda,
            "doc_cut": cut(run.doc_cut),
            "query_cut": cut(run.query_cut),
            "colbert_protocol": {
                "query_maxlen": QUERY_MAXLEN,
                "doc_maxlen": DOC_MAXLEN,
                "query_marker": "[unused0]",
                "doc_marker": "[unused1]",
                "mask_augmentation": true,
                "attend_to_mask_tokens": false,
                "doc_punctuation_filtered": true,
            },
        },
        "sets": sets,
    })
}
