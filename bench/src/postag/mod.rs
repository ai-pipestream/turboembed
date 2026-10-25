//! `turbo-bench postag`: the BiLSTM-CRF part-of-speech tagger's two
//! numerics, `Numerics::Packed` (f32 weights on the CPU backend's packed
//! matrix product) and `Numerics::JvmParity` (the f64 reference), on one
//! treebank on the CPU (docs/benchmarks.md, "Part-of-speech records").
//!
//! The record holds what each numerics gets right, how often the two
//! agree, how far apart their emissions and marginals are, and per
//! configuration (sentences per call, sentence lengths) the words per
//! second and the latency of a call. The two are timed in turn, packed,
//! parity, parity, packed in every round, so their ratio holds however
//! the clock moves. Like every record it is made by the tool from a
//! clean, pushed commit; nothing in it is typed by hand.

#[cfg(feature = "postag")]
mod measure;

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use turbo::record::{self, Device, Library, Machine};

use crate::Result;

/// The version of `PosRecord`.
pub const POS_RECORD_VERSION: u32 = 1;

/// Where part-of-speech records go, under the repository. A directory of
/// benchmarks/records/ the embed records' readers (core/build.rs, the
/// tree test) do not descend into.
pub const POSTAG_DIR: &str = "benchmarks/records/postag/";

/// The configurations each record times: every sentence of the treebank at
/// these many sentences per call, then at 32 per call the sentences of
/// each length range (words, inclusive; `None`: no upper end).
pub const SENTENCES_PER_CALL: [u32; 4] = [1, 8, 32, 128];
pub const LENGTHS: [(u32, Option<u32>); 3] = [(1, Some(9)), (10, Some(24)), (25, None)];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PosRecord {
    pub record_version: u32,
    pub recorded_at: String,
    pub machine: Machine,
    pub device: Device,
    pub library: Library,
    /// The processors the tool ran on.
    pub procedure: String,
    /// The command that made the record, with the model and treebank paths
    /// as placeholders.
    pub command: String,
    pub model: PosModel,
    pub treebank: Treebank,
    /// Timed passes per numerics and configuration: two per round.
    pub rounds: u32,
    pub accuracy: Accuracy,
    pub drift: Drift,
    pub runs: Vec<PosRun>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PosModel {
    /// The model's own ModelRef: id, revision and the SHA-256 it names.
    pub id: String,
    pub revision: String,
    pub sha256: String,
    /// The model file as read: its SHA-256 and size.
    pub file_sha256: String,
    pub file_bytes: u64,
    pub tags: u32,
    /// Annotator::config_hash of each numerics on this model, as hex.
    pub packed_config_hash: String,
    pub parity_config_hash: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Treebank {
    /// The file's name, without its directory.
    pub file: String,
    pub sha256: String,
    pub sentences: u32,
    /// Syntactic words: a multiword token's range line is not one.
    pub words: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Accuracy {
    /// Words whose tag matches the treebank's UPOS, per numerics.
    pub packed_correct: u32,
    pub parity_correct: u32,
    /// Words the two numerics tag the same.
    pub tags_agree: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Drift {
    /// The largest difference between the two numerics' emission scores,
    /// over every word and tag of the treebank.
    pub max_emission_abs_diff: f64,
    /// The largest difference between the two numerics' marginal
    /// probability of a word's tag, over the words they tag the same.
    pub max_marginal_abs_diff: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PosRun {
    pub sentences_per_call: u32,
    /// The sentence lengths in words, as `all`, `1-9` or `25-`.
    pub lengths: String,
    pub sentences: u32,
    pub words: u32,
    pub packed: Side,
    pub parity: Side,
    /// packed.words_per_second over parity.words_per_second.
    pub packed_over_parity: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Side {
    /// The median over the passes of the words of a pass over its time.
    pub words_per_second: f64,
    /// Over every call of every pass, in milliseconds.
    pub call_p50_ms: f64,
    pub call_p95_ms: f64,
}

/// A length range as the record names it.
pub fn lengths_name(min: u32, max: Option<u32>) -> String {
    match max {
        Some(max) => format!("{min}-{max}"),
        None => format!("{min}-"),
    }
}

/// The record's file name: machine, backend, model and commit.
pub fn file_name(r: &PosRecord) -> String {
    format!(
        "{}.{}.postag.bilstm-crf.{}.{}.json",
        crate::throughput::slug(&r.machine.arch),
        crate::throughput::slug(&r.device.backend),
        r.model.file_sha256.get(..8).unwrap_or(""),
        r.library.commit.get(..12).unwrap_or("")
    )
}

/// What a committed part-of-speech record must be: made from a pushed
/// commit, each count within the treebank, every rate positive and each
/// ratio the quotient of its rates, the two numerics under two hashes,
/// and no path on the machine that made it in its text.
pub fn check(r: &PosRecord) -> Result<()> {
    if r.record_version != POS_RECORD_VERSION {
        return Err(format!("record_version {}, expected {POS_RECORD_VERSION}", r.record_version));
    }
    let c = &r.library.commit;
    if c.len() != 40 || !c.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("library.commit {c} is not 40 hex"));
    }
    if r.library.pushed_to.is_empty() {
        return Err("library.pushed_to is empty: made from a commit on no branch of origin".into());
    }
    let hex = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_hexdigit());
    if !hex(&r.model.file_sha256, 64) || !hex(&r.treebank.sha256, 64) {
        return Err("model.file_sha256 and treebank.sha256 must be SHA-256 hex".into());
    }
    if r.model.packed_config_hash == r.model.parity_config_hash {
        return Err("the two numerics have one config hash".into());
    }
    let words = r.treebank.words;
    let a = &r.accuracy;
    if words == 0 || a.packed_correct > words || a.parity_correct > words || a.tags_agree > words {
        return Err(format!("accuracy counts past the treebank's {words} words"));
    }
    let d = &r.drift;
    if !(d.max_emission_abs_diff.is_finite() && d.max_emission_abs_diff >= 0.0)
        || !(d.max_marginal_abs_diff.is_finite() && d.max_marginal_abs_diff >= 0.0)
    {
        return Err("drift is negative or not finite".into());
    }
    if r.rounds == 0 || r.runs.is_empty() {
        return Err("no timed run".into());
    }
    let positive = |x: f64| x.is_finite() && x > 0.0;
    for run in &r.runs {
        let what = format!("{} sentences per call, lengths {}", run.sentences_per_call, run.lengths);
        if run.words > words || run.sentences > r.treebank.sentences || run.sentences == 0 {
            return Err(format!("{what}: counts past the treebank"));
        }
        for s in [&run.packed, &run.parity] {
            let latency = positive(s.call_p50_ms) && positive(s.call_p95_ms) && s.call_p95_ms >= s.call_p50_ms;
            if !positive(s.words_per_second) || !latency {
                return Err(format!("{what}: a rate or latency that is not finite and positive, or p95 under p50"));
            }
        }
        let q = run.packed.words_per_second / run.parity.words_per_second;
        if !positive(q) || !positive(run.packed_over_parity) {
            return Err(format!("{what}: packed_over_parity or the quotient of the rates is not finite and positive"));
        }
        if (run.packed_over_parity - q).abs() > 1e-12 * q {
            return Err(format!("{what}: packed_over_parity is not the quotient of the rates"));
        }
    }
    let text = serde_json::to_string(r).map_err(|e| e.to_string())?;
    // JSON writes a backslash as two: fold them back before looking.
    let text = text.replace("\\\\", "\\");
    if let Some(home) = record::HOST_PATHS.iter().find(|h| text.contains(*h)) {
        return Err(format!("the record holds a host path ({home}...)"));
    }
    Ok(())
}

/// Write the record under `dir`, checked; a record is never replaced.
pub fn write(r: &PosRecord, dir: &Path) -> Result<PathBuf> {
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

/// `turbo-bench postag`, with its arguments after the subcommand.
#[cfg(feature = "postag")]
pub fn cmd(args: &[String]) -> Result<()> {
    measure::cmd(args)
}

/// Without the feature the tool has no tagger to measure.
#[cfg(not(feature = "postag"))]
pub fn cmd(_: &[String]) -> Result<()> {
    Err("this turbo-bench was built without the postag feature: cargo run --release -p turbo-bench --features \
         postag -- postag ..."
        .into())
}

#[cfg(test)]
mod tests;
