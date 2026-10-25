//! `turbo-bench tag`: one tier of the BiLSTM-CRF part-of-speech tagger
//! (device and precision) on a CoNLL-U corpus, the mode every backend
//! records through (docs/benchmarks.md, "Tagger records").
//!
//! A tier is a device and a precision. On the CPU, `exact` is the f64
//! JVM parity numerics, `model` the packed numerics on the F32 weights,
//! and `fastest` the packed numerics on a bundle's F16 copy. The record
//! holds the accuracy, the words per second and call latency of the
//! first pass of a freshly loaded tagger (cold) and of the passes after
//! it (warm), the tier's `config_hash`, and the provenance block every
//! record carries: made by the tool from a clean, pushed commit.

#[cfg(feature = "postag")]
mod run;

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use turbo::record::{self, Device, Library, Machine};

use crate::Result;
use crate::postag::Treebank;

/// The version of `TagRecord`.
pub const TAG_RECORD_VERSION: u32 = 1;

/// Where tagger records go, under the repository: a directory the embed
/// and postag record readers do not descend into.
pub const TAG_DIR: &str = "benchmarks/records/tag/";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TagRecord {
    pub record_version: u32,
    pub recorded_at: String,
    pub machine: Machine,
    pub device: Device,
    pub library: Library,
    pub procedure: String,
    /// The command that made the record, its paths as placeholders.
    pub command: String,
    pub model: TagModel,
    pub tier: Tier,
    pub corpus: Treebank,
    pub sentences_per_call: u32,
    /// Warm passes, after the one cold pass.
    pub rounds: u32,
    /// Words whose tag matches the corpus's UPOS.
    pub correct: u32,
    pub cold: Speed,
    pub warm: Speed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TagModel {
    /// The model's ModelRef as loaded.
    pub id: String,
    pub revision: String,
    pub sha256: String,
    /// SHA-256 of the `ONLP-BLPT` file, read or named by the bundle.
    pub source_sha256: String,
    /// SHA-256 of the bundle's `postag.json`; empty for a model file.
    pub bundle_sha256: String,
    /// `f64` (the model file), `f32` or `f16` (a bundle's weights).
    pub weights: String,
    pub tags: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tier {
    pub device: String,
    pub precision: String,
    /// `Packed` or `JvmParity`.
    pub numerics: String,
    /// Annotator::config_hash of the tagger, as hex.
    pub config_hash: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Speed {
    /// Words over seconds: of the one pass (cold), the median pass (warm).
    pub words_per_second: f64,
    /// Over every call of those passes, in milliseconds.
    pub call_p50_ms: f64,
    pub call_p99_ms: f64,
}

/// The CPU tiers: a precision, the numerics it runs and the weights it
/// may run on (a model file's f64, a bundle's F32 or its F16 copy).
pub const CPU_TIERS: [(&str, &str, &[&str]); 3] =
    [("exact", "JvmParity", &["f64", "f32"]), ("model", "Packed", &["f64", "f32"]), ("fastest", "Packed", &["f16"])];

fn slug(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_end_matches('-').to_owned()
}

/// The record's file name: machine, backend, tier, model and commit.
pub fn file_name(r: &TagRecord) -> String {
    format!(
        "{}.{}.tag.bilstm-crf.{}.{}.{}.json",
        slug(&r.machine.arch),
        slug(&r.device.backend),
        slug(&r.tier.precision),
        r.model.source_sha256.get(..8).unwrap_or(""),
        r.library.commit.get(..12).unwrap_or("")
    )
}

/// What a committed tagger record must be: from a pushed commit, a
/// known tier, counts within the corpus, every rate and latency finite
/// and positive with p99 at or over p50, and no host path in its text.
pub fn check(r: &TagRecord) -> Result<()> {
    if r.record_version != TAG_RECORD_VERSION {
        return Err(format!("record_version {}, expected {TAG_RECORD_VERSION}", r.record_version));
    }
    let c = &r.library.commit;
    if c.len() != 40 || !c.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("library.commit {c} is not 40 hex"));
    }
    if r.library.pushed_to.is_empty() {
        return Err("library.pushed_to is empty: made from a commit on no branch of origin".into());
    }
    let hex = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_hexdigit());
    if !hex(&r.model.source_sha256, 64) || !hex(&r.corpus.sha256, 64) || !hex(&r.tier.config_hash, 16) {
        return Err("model.source_sha256, corpus.sha256 and tier.config_hash must be hex digests".into());
    }
    if !(r.model.bundle_sha256.is_empty() || hex(&r.model.bundle_sha256, 64)) {
        return Err("model.bundle_sha256 is neither empty nor a SHA-256".into());
    }
    let t = &r.tier;
    if r.device.backend == "cpu" || t.device == "cpu" {
        let known = CPU_TIERS
            .iter()
            .any(|(p, n, w)| *p == t.precision && *n == t.numerics && w.contains(&r.model.weights.as_str()));
        if t.device != r.device.backend || !known {
            return Err(format!("tier {t:?} with {} weights is not a CPU tier", r.model.weights));
        }
    }
    let words = r.corpus.words;
    if words == 0 || r.corpus.sentences == 0 || r.correct > words {
        return Err(format!("correct {} past the corpus's {words} words", r.correct));
    }
    if r.rounds == 0 || r.sentences_per_call == 0 {
        return Err("no warm pass, or no sentence per call".into());
    }
    let positive = |x: f64| x.is_finite() && x > 0.0;
    for (what, s) in [("cold", &r.cold), ("warm", &r.warm)] {
        if !positive(s.words_per_second) || !positive(s.call_p50_ms) || !positive(s.call_p99_ms) {
            return Err(format!("{what}: a rate or latency that is not finite and positive"));
        }
        if s.call_p99_ms < s.call_p50_ms {
            return Err(format!("{what}: p99 under p50"));
        }
    }
    let text = serde_json::to_string(r).map_err(|e| e.to_string())?;
    let text = text.replace("\\\\", "\\");
    if let Some(home) = record::HOST_PATHS.iter().find(|h| text.contains(*h)) {
        return Err(format!("the record holds a host path ({home}...)"));
    }
    Ok(())
}

/// Write the record under `dir`, checked; a record is never replaced.
pub fn write(r: &TagRecord, dir: &Path) -> Result<PathBuf> {
    check(r)?;
    let mut bytes = serde_json::to_vec_pretty(r).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(file_name(r));
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| format!("{}: {e}; a record is never replaced", path.display()))?;
    std::io::Write::write_all(&mut f, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

/// `turbo-bench tag`, with its arguments after the subcommand.
#[cfg(feature = "postag")]
pub fn cmd(args: &[String]) -> Result<()> {
    run::cmd(args)
}

/// Without the feature the tool has no tagger to measure.
#[cfg(not(feature = "postag"))]
pub fn cmd(_: &[String]) -> Result<()> {
    Err("this turbo-bench was built without the postag feature: cargo run --release -p turbo-bench --features \
         postag -- tag ..."
        .into())
}

#[cfg(test)]
mod tests;
