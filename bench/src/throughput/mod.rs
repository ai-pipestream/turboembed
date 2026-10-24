//! `turbo-bench lattice` and `turbo-bench grammar`: characters per
//! second for the lattice tokenizer (`analysis/tokenize`) and the
//! number, unit and time grammars (`analysis/grammar`), recorded
//! with the provenance of every record this tool writes
//! (docs/benchmarks.md, "Throughput records").
//!
//! Each mode times named phases over one committed text (the
//! goldens inputs of a set, in name order, repeated to a floor
//! size) and records the median pass's seconds beside the counts
//! it must divide into, so a checked record holds nothing a
//! measurement cannot give. Like every record it is made by the
//! tool from a clean, pushed commit; nothing in it is typed by
//! hand.

#[cfg(feature = "grammar")]
mod grammar;
#[cfg(feature = "lattice")]
mod lattice;
#[cfg(any(feature = "grammar", feature = "lattice"))]
mod measure;

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use turbo::record::{self, Device, Library, Machine};

use crate::Result;

/// The version of `ThroughputRecord`.
pub const THROUGHPUT_RECORD_VERSION: u32 = 1;

/// Where throughput records go, under the repository. A directory
/// of benchmarks/records/ the embed records' readers do not
/// descend into, like records/postag.
pub const THROUGHPUT_DIR: &str = "benchmarks/records/throughput/";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThroughputRecord {
    pub record_version: u32,
    pub recorded_at: String,
    pub machine: Machine,
    pub device: Device,
    pub library: Library,
    /// How the tool ran (pinning, threads).
    pub procedure: String,
    /// The command that made the record, with paths as
    /// placeholders.
    pub command: String,
    /// `lattice` or `grammar`.
    pub mode: String,
    pub subject: Subject,
    pub text: RecordText,
    /// Timed passes per phase, after one untimed warm pass.
    pub rounds: u32,
    pub runs: Vec<PhaseRun>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Subject {
    /// The dictionary (`ipadic`, `ko-dic`, `unidic`) or grammar
    /// set (`english`) measured.
    pub name: String,
    /// The dictionary's manifest SHA-256; empty for the grammar,
    /// whose rules are compiled into the library the record
    /// names by commit.
    pub sha256: String,
    /// Seconds to load the dictionary; `None` for the grammar.
    pub load_seconds: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordText {
    /// Where the text comes from, as a repository-relative
    /// description (the goldens input set and its floor size).
    pub source: String,
    pub bytes: u64,
    /// Unicode scalar values in the text.
    pub chars: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseRun {
    /// The phase timed: for the lattice, `analyze` and, when
    /// asked, `analyzer-readings`; for the grammar, `lex`,
    /// `parse`, `parse-units`, `find`, and `times`.
    pub phase: String,
    /// What the phase produced over the text (morphemes,
    /// tokens, lexemes, chart nodes, or mentions).
    pub units: u64,
    /// The median pass's seconds over the whole text.
    pub seconds: f64,
    /// `text.chars / seconds`.
    pub chars_per_second: f64,
    /// `text.bytes / seconds`.
    pub bytes_per_second: f64,
}

/// The record's file name: machine, backend, mode, subject and commit.
pub fn file_name(r: &ThroughputRecord) -> String {
    format!(
        "{}.{}.throughput.{}.{}.{}.json",
        slug(&r.machine.arch),
        slug(&r.device.backend),
        slug(&r.mode),
        slug(&r.subject.name),
        r.library.commit.get(..12).unwrap_or("")
    )
}

pub(crate) fn slug(s: &str) -> String {
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

/// What a committed throughput record must be: made from a
/// pushed commit, rates the quotients of their counts and
/// seconds, every figure finite and positive, and no path on
/// the machine that made it in its text.
pub fn check(r: &ThroughputRecord) -> Result<()> {
    if r.record_version != THROUGHPUT_RECORD_VERSION {
        return Err(format!("record_version {}, expected {THROUGHPUT_RECORD_VERSION}", r.record_version));
    }
    let c = &r.library.commit;
    if c.len() != 40 || !c.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("library.commit {c} is not 40 hex"));
    }
    if r.library.pushed_to.is_empty() {
        return Err("library.pushed_to is empty: made from a commit on no branch of origin".into());
    }
    if !matches!(r.mode.as_str(), "lattice" | "grammar") {
        return Err(format!("mode {:?} is neither lattice nor grammar", r.mode));
    }
    if r.device.backend != "cpu" || r.device.kind != "DEVICE_CPU" {
        return Err(format!(
            "device {}/{} is not the CPU device these modes measure on",
            r.device.backend, r.device.kind
        ));
    }
    // The phases a mode times, in order. A lattice record without
    // --readings holds the first alone; partial phase sets are not
    // a record this tool makes.
    let phases: Vec<&str> = r.runs.iter().map(|run| run.phase.as_str()).collect();
    let allowed: &[&[&str]] = match r.mode.as_str() {
        "grammar" => &[&["find", "lex", "parse", "parse-units", "times"]],
        _ => &[&["analyze"], &["analyze", "analyzer-readings"]],
    };
    if !allowed.contains(&phases.as_slice()) {
        return Err(format!("{} record with phases {phases:?}, which the tool never times", r.mode));
    }
    let hex = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_hexdigit());
    if !hex(&r.text.sha256, 64) || !(r.subject.sha256.is_empty() || hex(&r.subject.sha256, 64)) {
        return Err("text.sha256 and subject.sha256 must be SHA-256 hex (or empty)".into());
    }
    if r.mode == "lattice" && r.subject.sha256.is_empty() {
        return Err("a lattice record names no dictionary hash".into());
    }
    if let Some(load) = r.subject.load_seconds
        && (!load.is_finite() || load < 0.0)
    {
        return Err("load_seconds is negative or not finite".into());
    }
    if r.text.bytes == 0 || r.text.chars == 0 || r.text.chars > r.text.bytes {
        return Err(format!("text of {} chars in {} bytes", r.text.chars, r.text.bytes));
    }
    if r.rounds == 0 || r.runs.is_empty() {
        return Err("no timed run".into());
    }
    let positive = |x: f64| x.is_finite() && x > 0.0;
    for run in &r.runs {
        if run.phase.is_empty() || run.units == 0 {
            return Err(format!("phase {:?}: no units", run.phase));
        }
        if !positive(run.seconds) || !positive(run.chars_per_second) || !positive(run.bytes_per_second) {
            return Err(format!("phase {}: a rate or time that is not finite and positive", run.phase));
        }
        for (rate, count, what) in [
            (run.chars_per_second, r.text.chars, "chars_per_second"),
            (run.bytes_per_second, r.text.bytes, "bytes_per_second"),
        ] {
            let q = count as f64 / run.seconds;
            if !positive(q) || (rate - q).abs() > 1e-12 * q {
                return Err(format!("phase {}: {what} is not the quotient of its count and seconds", run.phase));
            }
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
pub fn write(r: &ThroughputRecord, dir: &Path) -> Result<PathBuf> {
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

/// Throughput records claim release rates, so an unoptimized
/// build refuses to measure at all.
pub(super) fn require_optimized(mode: &str) -> Result<()> {
    if cfg!(debug_assertions) {
        return Err(format!(
            "this turbo-bench was built without optimizations: cargo run --release -p turbo-bench              --features {mode} -- {mode} ..."
        ));
    }
    Ok(())
}

/// `turbo-bench lattice`, with its arguments after the subcommand.
#[cfg(feature = "lattice")]
pub fn lattice_cmd(args: &[String]) -> Result<()> {
    require_optimized("lattice")?;
    lattice::cmd(args)
}

/// Without the feature the tool has no lattice tokenizer to measure.
#[cfg(not(feature = "lattice"))]
pub fn lattice_cmd(_: &[String]) -> Result<()> {
    Err("this turbo-bench was built without the lattice feature: cargo run --release -p turbo-bench \
         --features lattice -- lattice ..."
        .into())
}

/// `turbo-bench grammar`, with its arguments after the subcommand.
#[cfg(feature = "grammar")]
pub fn grammar_cmd(args: &[String]) -> Result<()> {
    require_optimized("grammar")?;
    grammar::cmd(args)
}

/// Without the feature the tool has no grammar to measure.
#[cfg(not(feature = "grammar"))]
pub fn grammar_cmd(_: &[String]) -> Result<()> {
    Err("this turbo-bench was built without the grammar feature: cargo run --release -p turbo-bench \
         --features grammar -- grammar ..."
        .into())
}

#[cfg(test)]
mod tests;
