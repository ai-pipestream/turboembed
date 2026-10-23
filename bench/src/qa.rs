//! `turbo-bench qa`: extractive QA throughput on a device, in questions
//! and windows per second, by the context's length in tokens
//! (docs/extractive-qa.md, "Throughput").
//!
//! The questions come from a SQuAD-format JSON file. Each question goes
//! into the bin of its context's token count (the bundle's tokenizer, no
//! special tokens). A pass runs every question of a bin through
//! `ExtractiveQa::logits_many`, `batch` questions per call, so a session
//! runs as many windows at once as it holds; the record times passes.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use turbo::bundle::Bundle;
use turbo::manifest::{PromptRole, Truncation};
use turbo::qa::{ExtractiveQa, QaOptions};
use turbo::record::{self, Device, Library, Machine};
use turbo::tokenizer::{Encode, Tokenizer};

use crate::Result;
use crate::api::field;
use crate::git::Provenance;
use crate::measure::percentile;

/// The version of `QaRecord`.
pub const QA_RECORD_VERSION: u32 = 1;

/// Where QA records go, under the repository.
pub const QA_DIR: &str = "benchmarks/qa/";

/// The bins' lower bounds in context tokens; a bin runs to the next bound.
pub const BINS: [u32; 4] = [0, 128, 256, 384];

/// What to measure, as the command line gave it.
#[derive(Debug, Clone)]
pub struct QaPlan {
    pub bundle: PathBuf,
    pub device: String,
    pub precision: u32,
    /// Questions per `logits_many` call.
    pub batch: u32,
    /// Windows per run; 0 is the model's max_batch.
    pub max_batch: u32,
    /// SQuAD-format questions.
    pub questions: PathBuf,
    /// Questions per bin, at most.
    pub per_bin: usize,
    pub warmup: u32,
    pub iterations: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QaRecord {
    pub record_version: u32,
    pub recorded_at: String,
    pub machine: Machine,
    pub device: Device,
    pub library: Library,
    pub precision: String,
    pub model_id: String,
    pub revision: String,
    pub manifest_sha256: String,
    /// The questions file's SHA-256.
    pub questions_sha256: String,
    pub batch: u32,
    pub max_batch: u32,
    pub bins: Vec<Bin>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bin {
    /// Context tokens, `min_tokens..max_tokens`; 0 for no upper bound.
    pub min_tokens: u32,
    pub max_tokens: u32,
    pub questions: u32,
    /// The windows one pass runs.
    pub windows: u32,
    pub warmup: u32,
    pub iterations: u32,
    pub p50_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub questions_per_second: f64,
    pub windows_per_second: f64,
}

/// One question and its context, from a SQuAD-format file: `data`, each
/// with `paragraphs`, each with `context` and `qas`, each with `question`.
pub fn questions(text: &str) -> Result<Vec<(String, String)>> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("questions: {e}"))?;
    let mut out = Vec::new();
    for article in v["data"].as_array().ok_or("questions: no data array")? {
        for p in article["paragraphs"].as_array().ok_or("questions: an article with no paragraphs")? {
            let context = p["context"].as_str().ok_or("questions: a paragraph with no context")?;
            for qa in p["qas"].as_array().ok_or("questions: a paragraph with no qas")? {
                let q = qa["question"].as_str().ok_or("questions: a qa with no question")?;
                out.push((q.to_owned(), context.to_owned()));
            }
        }
    }
    Ok(out)
}

/// The bin of a context of `tokens` tokens.
pub fn bin_of(tokens: u32) -> usize {
    BINS.iter().rposition(|&b| tokens >= b).unwrap_or(0)
}

pub fn measure(plan: &QaPlan, p: &Provenance, recorded_at: String) -> Result<QaRecord> {
    if plan.batch == 0 || plan.iterations == 0 || plan.per_bin == 0 {
        return Err("--batch, --iterations and --per-bin must be above 0".into());
    }
    let bytes = std::fs::read(&plan.questions).map_err(|e| format!("{}: {e}", plan.questions.display()))?;
    let text = std::str::from_utf8(&bytes).map_err(|e| format!("{}: {e}", plan.questions.display()))?;
    let all = questions(text)?;
    let bundle = Bundle::open(&plan.bundle).map_err(|e| e.message)?;
    let tokenizer = Tokenizer::load(&bundle).map_err(|e| e.message)?;
    let opts = Encode {
        add_special_tokens: false,
        truncation: Truncation::None,
        max_tokens: u32::MAX,
        prompt: PromptRole::None,
    };
    let mut bins: Vec<Vec<(&str, &str)>> = vec![Vec::new(); BINS.len()];
    let mut ids = Vec::new();
    for (q, c) in &all {
        ids.clear();
        tokenizer.encode_into(c, opts, &mut ids).map_err(|e| e.message)?;
        let b = &mut bins[bin_of(ids.len() as u32)];
        if b.len() < plan.per_bin {
            b.push((q, c));
        }
    }
    let o = QaOptions {
        backend: Some(plan.device.clone()),
        precision: plan.precision,
        cpu_threads: 0,
        max_batch: plan.max_batch,
    };
    let mut qa = ExtractiveQa::open_with(&plan.bundle, &o).map_err(|e| e.message)?;
    let mut out = Vec::new();
    for (i, set) in bins.iter().enumerate().filter(|(_, s)| !s.is_empty()) {
        let mut pass = || -> Result<u32> {
            let mut windows = 0;
            for chunk in set.chunks(plan.batch as usize) {
                windows += qa.logits_many(chunk).map_err(|e| e.message)?.iter().map(Vec::len).sum::<usize>() as u32;
            }
            Ok(windows)
        };
        let mut windows = 0;
        for _ in 0..plan.warmup {
            windows = pass()?;
        }
        let mut ms = Vec::with_capacity(plan.iterations as usize);
        for _ in 0..plan.iterations {
            let t = Instant::now();
            windows = pass()?;
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(f64::total_cmp);
        let p50 = percentile(&ms, 50.0);
        out.push(Bin {
            min_tokens: BINS[i],
            max_tokens: BINS.get(i + 1).copied().unwrap_or(0),
            questions: set.len() as u32,
            windows,
            warmup: plan.warmup,
            iterations: plan.iterations,
            p50_ms: p50,
            min_ms: ms[0],
            max_ms: ms[ms.len() - 1],
            questions_per_second: set.len() as f64 / (p50 / 1e3),
            windows_per_second: f64::from(windows) / (p50 / 1e3),
        });
    }
    let d = qa.device_info().map_err(|e| e.message)?;
    let backend = field(&d.backend);
    Ok(QaRecord {
        record_version: QA_RECORD_VERSION,
        recorded_at,
        machine: Machine { arch: field(&d.arch), host_cpu: String::new(), os: std::env::consts::OS.into() },
        device: Device {
            backend: backend.clone(),
            kind: crate::kind_name(d.kind).into(),
            name: field(&d.name),
            vendor: field(&d.vendor),
            driver_version: field(&d.driver_version),
            runtime_version: field(&d.runtime_version),
            memory_total: d.memory_total,
        },
        library: Library {
            version: crate::api::version().split(' ').next().unwrap_or("").into(),
            build: crate::api::version(),
            commit: p.commit.clone(),
            pushed_to: p.pushed_to.clone(),
            settings: Vec::new(),
        },
        precision: record::precision_name(plan.precision).ok_or("precision is not a TURBO_PRECISION_* value")?.into(),
        model_id: qa.model_id.clone(),
        revision: qa.revision.clone(),
        manifest_sha256: qa.manifest_sha256.clone(),
        questions_sha256: turbo::bundle::sha256_hex(&bytes),
        batch: plan.batch,
        max_batch: qa.max_batch(),
        bins: out,
    })
}

/// The record's file name: machine, backend, precision, model and commit.
pub fn file_name(r: &QaRecord) -> String {
    let slug = |s: &str| {
        let mut out = String::new();
        for c in s.chars() {
            if c.is_ascii_alphanumeric() {
                out.push(c.to_ascii_lowercase());
            } else if !out.is_empty() && !out.ends_with('-') {
                out.push('-');
            }
        }
        out.trim_end_matches('-').to_owned()
    };
    let mut model = slug(r.model_id.rsplit('/').next().unwrap_or(""));
    model.truncate(32);
    format!(
        "{}.{}.qa.{}.{}-{}.{}.json",
        slug(&r.machine.arch),
        slug(&r.device.backend),
        slug(r.precision.strip_prefix("PRECISION_").unwrap_or(&r.precision)),
        model.trim_end_matches('-'),
        r.manifest_sha256.get(..8).unwrap_or(""),
        r.library.commit.get(..12).unwrap_or("")
    )
}

/// What a committed QA record must be: made from a pushed commit, each
/// rate the quotient of its count and p50, and no host path in its text.
pub fn check(r: &QaRecord) -> Result<()> {
    if r.record_version != QA_RECORD_VERSION {
        return Err(format!("record_version {}, expected {QA_RECORD_VERSION}", r.record_version));
    }
    let c = &r.library.commit;
    if c.len() != 40 || !c.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("library.commit {c} is not 40 hex"));
    }
    if r.library.pushed_to.is_empty() {
        return Err("library.pushed_to is empty: made from a commit on no branch of origin".into());
    }
    let same = |a: f64, b: f64| a.is_finite() && a > 0.0 && (a - b).abs() <= 1e-9 * a.abs();
    for b in &r.bins {
        let s = b.p50_ms / 1e3;
        if !same(b.questions_per_second, f64::from(b.questions) / s)
            || !same(b.windows_per_second, f64::from(b.windows) / s)
        {
            return Err(format!("bin {}: the rates are not the counts over the p50", b.min_tokens));
        }
    }
    let text = serde_json::to_string(r).map_err(|e| e.to_string())?.replace("\\\\", "\\");
    if let Some(home) = record::HOST_PATHS.iter().find(|h| text.contains(*h)) {
        return Err(format!("the record holds a host path ({home}...)"));
    }
    Ok(())
}

/// Write the record under `dir`, checked; a record is never replaced.
pub fn write(r: &QaRecord, dir: &Path) -> Result<PathBuf> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn questions_read_squad_layout_in_order() {
        let text = r#"{"version":"v2.0","data":[{"title":"A","paragraphs":[
            {"context":"Paris is in France.","qas":[{"id":"1","question":"Where is Paris?","answers":[],"is_impossible":false},
                                                {"id":"2","question":"Who?","answers":[],"is_impossible":true}]},
            {"context":"Second.","qas":[{"id":"3","question":"What?","answers":[]}]}]}]}"#;
        let q = questions(text).unwrap();
        assert_eq!(q.len(), 3);
        assert_eq!(q[0], ("Where is Paris?".into(), "Paris is in France.".into()));
        assert_eq!(q[2], ("What?".into(), "Second.".into()));
        assert!(questions(r#"{"data":[{"paragraphs":[{"qas":[]}]}]}"#).is_err());
        assert!(questions("[]").is_err());
    }

    #[test]
    fn bins_start_at_their_bounds() {
        assert_eq!([0, 127, 128, 255, 256, 383, 384, 5000].map(bin_of), [0, 0, 1, 1, 2, 2, 3, 3]);
    }
}
