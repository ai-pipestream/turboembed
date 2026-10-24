//! `turbo-bench rerank`: cross-encoder rerank throughput on a device, in
//! pairs per second, with text-embeddings-inference's /rerank as the
//! reference on the same pairs (docs/reranking.md, "Throughput").
//!
//! The pairs come from a SQuAD-format file: request `i` is the first
//! question of paragraph `i` and the contexts of the `passages`
//! paragraphs from `i` on that fit one window with it, so neither side
//! truncates or windows. A pass sends every request in turn, one query
//! with its passages; the library scores them through `Reranker::score`,
//! TEI through POST /rerank with raw scores. The record keeps the p50 of
//! the passes, the pairs per second it gives, and the largest difference
//! between the two sides' scores.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::json;
use turbo::bundle::Bundle;
use turbo::manifest::{PromptRole, Truncation};
use turbo::record::{self, Device, Library, Machine};
use turbo::rerank::{RerankOptions, RerankSessionOptions, Reranker};
use turbo::tokenizer::{Encode, Tokenizer};

use crate::Result;
use crate::api::field;
use crate::docker::{self, Log, Running, argv};
use crate::git::Provenance;
use crate::measure::percentile;

/// The version of `RerankRecord`.
pub const RERANK_RECORD_VERSION: u32 = 1;

/// Where rerank records go, under the repository.
pub const RERANK_DIR: &str = "benchmarks/rerank/";

/// How long TEI may take to load the model and answer /health.
const START_TIMEOUT: Duration = Duration::from_secs(600);

/// What to measure, as the command line gave it.
#[derive(Debug, Clone)]
pub struct RerankPlan {
    pub bundle: PathBuf,
    pub device: String,
    pub precision: u32,
    /// SQuAD-format questions and contexts.
    pub questions: PathBuf,
    pub requests: usize,
    /// Passages a request.
    pub passages: usize,
    pub warmup: u32,
    pub iterations: u32,
    /// TEI's image, pinned by digest, and the model in the upstream
    /// layout; None records that TEI was not run.
    pub tei: Option<(String, PathBuf)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RerankRecord {
    pub record_version: u32,
    pub recorded_at: String,
    pub machine: Machine,
    pub device: Device,
    pub library: Library,
    pub precision: String,
    pub model_id: String,
    pub revision: String,
    pub manifest_sha256: String,
    pub questions_sha256: String,
    pub requests: u32,
    pub passages: u32,
    pub turbo: Side,
    pub tei: Option<Tei>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Side {
    pub warmup: u32,
    pub iterations: u32,
    /// One pass: every request once.
    pub p50_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub pairs_per_second: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tei {
    /// `name@sha256:<64 hex>`.
    pub image: String,
    pub commands: Vec<Vec<String>>,
    pub side: Side,
    /// The largest difference of a TEI score from the library's.
    pub max_abs_score_diff: f64,
    /// TEI's pairs per second over the library's.
    pub turbo_speedup: f64,
}

/// One query and its passages.
pub type Request = (String, Vec<String>);

/// The requests of a SQuAD-format file that fit one window each for
/// `tokenizer`: up to `requests`, each with `passages` passages.
pub fn requests(
    text: &str,
    tokenizer: &Tokenizer,
    fits: &dyn Fn(&Tokenizer, &str, &str) -> bool,
    requests: usize,
    passages: usize,
) -> Result<Vec<Request>> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("questions: {e}"))?;
    let mut paragraphs = Vec::new();
    for article in v["data"].as_array().ok_or("questions: no data array")? {
        for p in article["paragraphs"].as_array().ok_or("questions: an article with no paragraphs")? {
            let context = p["context"].as_str().ok_or("questions: a paragraph with no context")?;
            let question = p["qas"][0]["question"].as_str().ok_or("questions: a paragraph with no question")?;
            paragraphs.push((question.to_owned(), context.to_owned()));
        }
    }
    let mut out = Vec::new();
    for (i, (q, _)) in paragraphs.iter().enumerate() {
        if out.len() == requests {
            break;
        }
        let ps: Vec<String> =
            paragraphs[i..].iter().map(|(_, c)| c).filter(|c| fits(tokenizer, q, c)).take(passages).cloned().collect();
        if ps.len() == passages {
            out.push((q.clone(), ps));
        }
    }
    if out.len() < requests {
        return Err(format!("questions: {} requests of {passages} passages that fit, not {requests}", out.len()));
    }
    Ok(out)
}

/// Whether `query` and `context` make one window at the bundle's budgets
/// with nothing cut: every piece of the query fits `query_tokens`, and
/// the passage, whose windows cover all of it, needs only one.
pub fn fits_one_window(t: &Tokenizer, query: &str, context: &str, query_tokens: u32, max_seq: u32) -> bool {
    let whole = Encode {
        add_special_tokens: false,
        truncation: Truncation::None,
        max_tokens: u32::MAX,
        prompt: PromptRole::None,
    };
    let uncut = t.encode(query, whole).is_ok_and(|q| q.len() <= query_tokens as usize);
    uncut && t.encode_pair(query, context, query_tokens, max_seq, 0).is_ok_and(|w| w.len() == 1)
}

fn side(warmup: u32, mut ms: Vec<f64>, pairs: usize) -> Side {
    ms.sort_by(f64::total_cmp);
    let p50 = percentile(&ms, 50.0);
    Side {
        warmup,
        iterations: ms.len() as u32,
        p50_ms: p50,
        min_ms: ms[0],
        max_ms: ms[ms.len() - 1],
        pairs_per_second: pairs as f64 / (p50 / 1e3),
    }
}

/// Time `pass` `iterations` times after `warmup` untimed runs; the scores
/// of the last.
fn time(warmup: u32, iterations: u32, pass: &mut dyn FnMut() -> Result<Vec<f32>>) -> Result<(Vec<f64>, Vec<f32>)> {
    let mut scores = Vec::new();
    for _ in 0..warmup {
        scores = pass()?;
    }
    let mut ms = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        let t = Instant::now();
        scores = pass()?;
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    Ok((ms, scores))
}

pub fn measure(plan: &RerankPlan, p: &Provenance, recorded_at: String) -> Result<RerankRecord> {
    if plan.requests == 0 || plan.passages == 0 || plan.iterations == 0 {
        return Err("--requests, --passages and --iterations must be above 0".into());
    }
    let bytes = std::fs::read(&plan.questions).map_err(|e| format!("{}: {e}", plan.questions.display()))?;
    let text = std::str::from_utf8(&bytes).map_err(|e| format!("{}: {e}", plan.questions.display()))?;
    let bundle = Bundle::open(&plan.bundle).map_err(|e| e.message)?;
    let rerank = bundle.manifest.rerank.as_ref().ok_or("the bundle is not a rerank bundle")?;
    let (query_tokens, max_seq) = (rerank.query_tokens, rerank.max_seq);
    let tokenizer = Tokenizer::load(&bundle).map_err(|e| e.message)?;
    let fits = |t: &Tokenizer, q: &str, c: &str| fits_one_window(t, q, c, query_tokens, max_seq);
    let reqs = requests(text, &tokenizer, &fits, plan.requests, plan.passages)?;
    let pairs = plan.requests * plan.passages;

    let o = RerankSessionOptions { backend: Some(plan.device.clone()), precision: plan.precision, cpu_threads: 0 };
    let mut r = Reranker::open_with(&plan.bundle, &o).map_err(|e| e.message)?;
    let (ms, turbo_scores) = time(plan.warmup, plan.iterations, &mut || {
        let mut all = Vec::with_capacity(pairs);
        for (q, ps) in &reqs {
            let pairs: Vec<(&str, &str)> = ps.iter().map(|p| (q.as_str(), p.as_str())).collect();
            all.extend(r.score(&pairs, RerankOptions::default()).map_err(|e| e.message)?.scores);
        }
        Ok(all)
    })?;
    let turbo = side(plan.warmup, ms, pairs);
    let tei = match &plan.tei {
        Some((image, dir)) => Some(tei(image, dir, plan, &reqs, &turbo_scores, &turbo)?),
        None => None,
    };
    let d = r.device_info().map_err(|e| e.message)?;
    let id = r.identity().clone();
    Ok(RerankRecord {
        record_version: RERANK_RECORD_VERSION,
        recorded_at,
        machine: Machine { arch: field(&d.arch), host_cpu: String::new(), os: std::env::consts::OS.into() },
        device: Device {
            backend: field(&d.backend),
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
        model_id: id.model_id,
        revision: id.revision,
        manifest_sha256: id.manifest_sha256,
        questions_sha256: turbo::bundle::sha256_hex(&bytes),
        requests: plan.requests as u32,
        passages: plan.passages as u32,
        turbo,
        tei,
    })
}

/// TEI's /rerank on the same requests: its container started on the
/// measured GPU (the CPU image on the CPU), in float32, timed as the
/// library was, and stopped.
fn tei(image: &str, dir: &Path, plan: &RerankPlan, reqs: &[Request], ours: &[f32], turbo: &Side) -> Result<Tei> {
    let image = docker::check_pinned("--tei-image", image)?;
    let dir = std::fs::canonicalize(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut log = Log::default();
    docker::require_image(&mut log, image)?;
    let container = format!("turbo-bench-tei-rerank-{}", std::process::id());
    let args = |model: &str| {
        let mut a = argv(&["docker", "run", "--detach", "--rm", "--pull", "never", "--name", &container]);
        if plan.device != "cpu" {
            a.extend(argv(&["--gpus", "device=0"]));
        }
        a.extend(argv(&[
            "--publish",
            "127.0.0.1::80",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--mount",
            &format!("type=bind,src={model},dst=/model,readonly"),
            image,
            "--model-id",
            "/model",
            "--port",
            "80",
            "--dtype",
            "float32",
            "--max-client-batch-size",
            &plan.passages.to_string(),
        ]));
        a
    };
    log.run_as(&args(&dir.to_string_lossy()), args(docker::TEI_MODEL))?;
    let _running = Running { name: container.clone() };
    let port = docker::parse_port(&log.run(&argv(&["docker", "port", &container, "80/tcp"]))?)?;
    let base = format!("http://127.0.0.1:{port}");
    let start = Instant::now();
    while ureq::get(&format!("{base}/health")).call().is_err() {
        if start.elapsed() > START_TIMEOUT {
            let logs = log.run(&argv(&["docker", "logs", &container])).unwrap_or_default();
            return Err(format!("TEI did not become healthy within {START_TIMEOUT:?}:\n{logs}"));
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    let url = format!("{base}/rerank");
    let bodies: Vec<String> = reqs
        .iter()
        .map(|(q, ps)| json!({"query": q, "texts": ps, "raw_scores": true, "truncate": false}).to_string())
        .collect();
    let (ms, theirs) = time(plan.warmup, plan.iterations, &mut || {
        let mut all = Vec::new();
        for (body, (_, ps)) in bodies.iter().zip(reqs) {
            let mut resp = ureq::post(&url)
                .header("content-type", "application/json")
                .send(body.as_str())
                .map_err(|e| format!("POST {url}: {e}"))?;
            let text = resp.body_mut().read_to_string().map_err(|e| format!("POST {url}: {e}"))?;
            all.extend(scores(&text, ps.len())?);
        }
        Ok(all)
    })?;
    let side = side(plan.warmup, ms, ours.len());
    let max_abs_score_diff = ours.iter().zip(&theirs).map(|(a, b)| f64::from((a - b).abs())).fold(0.0, f64::max);
    Ok(Tei {
        image: image.to_owned(),
        commands: log.commands,
        turbo_speedup: turbo.pairs_per_second / side.pairs_per_second,
        side,
        max_abs_score_diff,
    })
}

/// /rerank's answer, `[{"index": i, "score": s}, ...]` in any order, as
/// the scores in passage order.
pub fn scores(body: &str, n: usize) -> Result<Vec<f32>> {
    let v: Vec<serde_json::Value> = serde_json::from_str(body).map_err(|e| format!("TEI /rerank: {e}"))?;
    let mut out = vec![f32::NAN; n];
    for item in &v {
        let i = item["index"].as_u64().ok_or("TEI /rerank: an item with no index")? as usize;
        let s = item["score"].as_f64().ok_or("TEI /rerank: an item with no score")?;
        *out.get_mut(i).ok_or("TEI /rerank: an index past the passages")? = s as f32;
    }
    if v.len() != n || out.iter().any(|s| s.is_nan()) {
        return Err(format!("TEI /rerank: {} scores for {n} passages", v.len()));
    }
    Ok(out)
}

/// The record's file name: machine, backend, precision, model and commit.
pub fn file_name(r: &RerankRecord) -> String {
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
        "{}.{}.rerank.{}.{}-{}.{}.json",
        slug(&r.machine.arch),
        slug(&r.device.backend),
        slug(r.precision.strip_prefix("PRECISION_").unwrap_or(&r.precision)),
        model.trim_end_matches('-'),
        r.manifest_sha256.get(..8).unwrap_or(""),
        r.library.commit.get(..12).unwrap_or("")
    )
}

/// What a committed rerank record must be: made from a pushed commit, its
/// rates the quotients of its counts and p50s, and no host path in it.
pub fn check(r: &RerankRecord) -> Result<()> {
    if r.record_version != RERANK_RECORD_VERSION {
        return Err(format!("record_version {}, expected {RERANK_RECORD_VERSION}", r.record_version));
    }
    let c = &r.library.commit;
    if c.len() != 40 || !c.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("library.commit {c} is not 40 hex"));
    }
    if r.library.pushed_to.is_empty() {
        return Err("library.pushed_to is empty: made from a commit on no branch of origin".into());
    }
    let same = |a: f64, b: f64| a.is_finite() && a > 0.0 && (a - b).abs() <= 1e-9 * a.abs();
    let pairs = f64::from(r.requests) * f64::from(r.passages);
    let mut sides = vec![("turbo", &r.turbo)];
    if let Some(t) = &r.tei {
        sides.push(("tei", &t.side));
        if !same(t.turbo_speedup, r.turbo.pairs_per_second / t.side.pairs_per_second) {
            return Err("tei.turbo_speedup is not the quotient of the two rates".into());
        }
    }
    for (name, s) in sides {
        if !same(s.pairs_per_second, pairs / (s.p50_ms / 1e3)) {
            return Err(format!("{name}: pairs_per_second is not the pairs over the p50"));
        }
    }
    let text = serde_json::to_string(r).map_err(|e| e.to_string())?.replace("\\\\", "\\");
    if let Some(home) = record::HOST_PATHS.iter().find(|h| text.contains(*h)) {
        return Err(format!("the record holds a host path ({home}...)"));
    }
    Ok(())
}

/// Write the record under `dir`, checked; a record is never replaced.
pub fn write(r: &RerankRecord, dir: &Path) -> Result<PathBuf> {
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
#[path = "rerank_tests.rs"]
mod tests;
