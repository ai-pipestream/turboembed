//! `turbo-bench`: the `libturbo` half of the matched benchmark protocol
//! (PLAN.md section 11) and its receipt writer.
//!
//! ```text
//! turbo-bench embed    --provider-lib <so> --provider <id> --bundle <dir> [--ordinal N]
//!                      [--batches 1,8,32] [--seqs 32,128,256] [--iters 30] [--warmup 5]
//!                      [--out receipt.json] [--budget earlier-receipt.json]
//! turbo-bench rerank   --provider-lib <so> --provider <id> --bundle <dir> [--docs 32] ...
//! turbo-bench generate --provider-lib <so> --provider <id> --bundle <dir> [--new-tokens 128] ...
//! turbo-bench discover [--provider-lib <so>]... [--provider-dir <dir>] [--bundle <dir>]... [--json]
//! ```
//!
//! `discover` is the survey: it loads the named provider libraries (plus
//! the built-in ones), and for every device prints the name, kind, runtime
//! and driver versions, the option features it honors (decoded from the
//! capability bits), the task x modality capability matrix with status,
//! compute dtype and measured cosine floor, and, for each bundle named,
//! whether that device can run it and why not.
//!
//! Embedding workloads run the text path (`write_text` + `run` + read) and
//! the prepared-token path (`write_tokens` + `run` + read, tokens encoded
//! once up front by the core tokenizer) for every batch x seq cell, with
//! texts built from the committed STS corpus to fill about 90% of each
//! sequence length. Reported per cell: p50/p99/mean latency, rows/s and
//! tokens/s, and the per-run H2D/D2H bytes, host allocations, and provider
//! allocations from the session counters.
//!
//! A receipt carries the machine, provider, runtime and driver versions,
//! device, bundle identity (manifest and artifact hashes), and commit. With
//! `--budget`, every cell's text-path p50 is checked against the earlier
//! receipt's p50 plus a tolerance (default 25%); a regression is reported
//! and the process exits non-zero. Budgets are set from the first run per
//! provider and then held (PLAN.md section 11).
//!
//! The direct-native reference program of each pair lives with its runtime
//! (see the provider READMEs); this tool measures the `libturbo` side only.

#![deny(missing_docs)]

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use turbo::bundle::Bundle;
use turbo::provider::{EmbedOptions, GenerateDesc, Message, ModelDesc, RerankOptions, SessionDesc, TokenBatch};
use turbo::tokenizer::{EncodeOptions, EncodeTarget, Tokenizer};
use turbo::types::{FinishReason, Modality, Task};
use turbo::{Context, ContextDesc, DeviceKind, DeviceSelector, RuntimeDesc, SelectPolicy};

#[derive(Parser)]
#[command(name = "turbo-bench", version, about = "Benchmark Turbo providers and write receipts")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Args, Clone)]
struct Common {
    /// Provider library to load (omit for a built-in provider such as `mock`).
    #[arg(long)]
    provider_lib: Option<PathBuf>,
    /// Provider id the library registers.
    #[arg(long)]
    provider: String,
    /// Device ordinal within the provider (default: its first non-CPU device, else 0).
    #[arg(long)]
    ordinal: Option<u32>,
    /// Bundle directory.
    #[arg(long)]
    bundle: PathBuf,
    /// Timed iterations per cell.
    #[arg(long, default_value_t = 30)]
    iters: u32,
    /// Untimed warm-up iterations per cell.
    #[arg(long, default_value_t = 5)]
    warmup: u32,
    /// Receipt file to write (JSON).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Earlier receipt whose p50 figures are the budget.
    #[arg(long)]
    budget: Option<PathBuf>,
    /// Allowed slowdown against the budget, as a fraction.
    #[arg(long, default_value_t = 0.25)]
    tolerance: f64,
    /// Corpus of texts (JSON lines with `text_a`/`text_b`); default: testdata/corpus/sts-pairs.jsonl.
    #[arg(long)]
    corpus: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Embedding workloads over batch x seq cells.
    Embed {
        #[command(flatten)]
        common: Common,
        /// Batch sizes.
        #[arg(long, value_delimiter = ',', default_value = "1,8,32")]
        batches: Vec<u32>,
        /// Sequence lengths.
        #[arg(long, value_delimiter = ',', default_value = "32,128,256")]
        seqs: Vec<u32>,
    },
    /// Rerank one query against `docs` documents.
    Rerank {
        #[command(flatten)]
        common: Common,
        /// Documents per query.
        #[arg(long, default_value_t = 32)]
        docs: u32,
        /// Sequence length of the session.
        #[arg(long, default_value_t = 128)]
        seq: u32,
    },
    /// Survey the providers and devices on this machine.
    Discover {
        /// Provider libraries to load; repeatable.
        #[arg(long = "provider-lib")]
        provider_libs: Vec<PathBuf>,
        /// Directory whose `libturbo_provider_*` libraries are all loaded.
        #[arg(long)]
        provider_dir: Option<PathBuf>,
        /// Bundles to check with `can_run` on every device; repeatable.
        #[arg(long = "bundle")]
        bundles: Vec<PathBuf>,
        /// Print JSON instead of the table.
        #[arg(long)]
        json: bool,
    },
    /// Generate `new_tokens` tokens from a fixed prompt.
    Generate {
        #[command(flatten)]
        common: Common,
        /// New tokens per generation.
        #[arg(long, default_value_t = 128)]
        new_tokens: u32,
        /// Prompt text (user role).
        #[arg(long, default_value = "Write a short paragraph about the history of the bicycle.")]
        prompt: String,
    },
}

// ---------------------------------------------------------------------------
// Receipt
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Latency {
    p50_ms: f64,
    p99_ms: f64,
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
    rows_per_s: f64,
    tokens_per_s: f64,
    iters: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct PerRun {
    h2d_bytes: u64,
    d2h_bytes: u64,
    host_allocs: u64,
    provider_allocs: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct EmbedCell {
    batch: u32,
    seq: u32,
    live_tokens_per_row: f64,
    text_path: Latency,
    /// Absent when the bundle has no Hugging Face tokenizer for the core to
    /// encode with; the reason is in `prepared_tokens_note`.
    prepared_tokens_path: Option<Latency>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    prepared_tokens_note: String,
    per_run: PerRun,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct RerankCell {
    docs: u32,
    seq: u32,
    text_path: Latency,
    per_run: PerRun,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GenerateCell {
    new_tokens_requested: u32,
    generated_tokens_mean: f64,
    prompt_tokens: u32,
    time_to_first_token_ms_p50: f64,
    decode_tokens_per_s_p50: f64,
    total_ms_p50: f64,
    finish_reasons: Vec<String>,
    iters: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Receipt {
    receipt_version: u32,
    kind: String,
    date: String,
    machine: Machine,
    commit: String,
    provider: ProviderId,
    device: Device,
    bundle: BundleId,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    embed: Vec<EmbedCell>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    rerank: Option<RerankCell>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    generate: Option<GenerateCell>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    budget_check: Option<BudgetCheck>,
    native_reference: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Machine {
    hostname: String,
    os: String,
    arch: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ProviderId {
    id: String,
    version: String,
    runtime_version: String,
    driver_version: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Device {
    name: String,
    kind: String,
    ordinal: u32,
    caps: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct BundleId {
    dir: String,
    model_id: String,
    manifest_sha256: String,
    artifacts: std::collections::BTreeMap<String, String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct BudgetCheck {
    budget_file: String,
    tolerance: f64,
    violations: Vec<String>,
}

fn run_cmd(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn machine() -> Machine {
    Machine {
        hostname: run_cmd("uname", &["-n"]),
        os: run_cmd("uname", &["-sr"]),
        arch: std::env::consts::ARCH.to_string(),
    }
}

fn today() -> String {
    // Date without a chrono dependency: days since the epoch to a civil date.
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

fn summarize(samples: &mut [Duration], rows: u64, tokens: u64) -> Latency {
    assert!(!samples.is_empty(), "no samples");
    samples.sort();
    let n = samples.len();
    let pct = |p: f64| -> f64 {
        let idx = ((n as f64 - 1.0) * p).round() as usize;
        samples[idx.min(n - 1)].as_secs_f64() * 1e3
    };
    let total: f64 = samples.iter().map(|d| d.as_secs_f64()).sum();
    let mean = total / n as f64;
    Latency {
        p50_ms: pct(0.5),
        p99_ms: pct(0.99),
        mean_ms: mean * 1e3,
        min_ms: samples[0].as_secs_f64() * 1e3,
        max_ms: samples[n - 1].as_secs_f64() * 1e3,
        rows_per_s: rows as f64 / mean,
        tokens_per_s: tokens as f64 / mean,
        iters: n as u32,
    }
}

/// A selected device with its context, and the identity fields a receipt needs.
struct Target {
    ctx: Arc<Context>,
    provider: ProviderId,
    device: Device,
}

fn open_target(c: &Common) -> Result<Target, String> {
    let provider_paths: Vec<String> = c.provider_lib.iter().map(|p| p.to_string_lossy().into_owned()).collect();
    let rt = turbo::create_runtime(RuntimeDesc { provider_paths, ..Default::default() })
        .map_err(|e| format!("load provider {:?}: {e}", c.provider_lib))?;
    if !rt.failures().is_empty() {
        return Err(format!("provider failures: {:?}", rt.failures()));
    }
    let devices: Vec<_> = rt.devices().into_iter().filter(|d| d.info.provider_id == c.provider).collect();
    if devices.is_empty() {
        return Err(format!("provider `{}` enumerated no devices", c.provider));
    }
    let ordinal = match c.ordinal {
        Some(o) => o,
        None => devices.iter().find(|d| d.info.kind != DeviceKind::Cpu).map(|d| d.info.ordinal).unwrap_or(0),
    };
    let idx = rt
        .select(&DeviceSelector {
            policy: SelectPolicy::Explicit,
            provider_id: c.provider.clone(),
            ordinal,
            ..Default::default()
        })
        .map_err(|e| format!("select ordinal {ordinal}: {e}"))?;
    let info = rt.device(idx).map_err(|e| e.to_string())?.info;
    eprintln!(
        "device: {} ({:?}, ordinal {ordinal}) runtime {} driver {}",
        info.name, info.kind, info.runtime_version, info.driver_version
    );
    let ctx = Context::create(rt, idx, &ContextDesc::default()).map_err(|e| format!("context: {e}"))?;
    Ok(Target {
        ctx,
        provider: ProviderId {
            id: info.provider_id.clone(),
            version: info.provider_version.clone(),
            runtime_version: info.runtime_version.clone(),
            driver_version: info.driver_version.clone(),
        },
        device: Device {
            name: info.name.clone(),
            kind: format!("{:?}", info.kind),
            ordinal,
            caps: format!("{:#x}", info.caps),
        },
    })
}

fn bundle_id(dir: &Path) -> Result<(Bundle, BundleId), String> {
    let b = Bundle::open(dir).map_err(|e| format!("bundle {}: {e}", dir.display()))?;
    let artifacts = b.manifest().artifacts.iter().map(|(k, v)| (k.clone(), v.sha256.clone())).collect();
    let id = BundleId {
        dir: dir.display().to_string(),
        model_id: b.manifest().model_id.clone(),
        manifest_sha256: b.manifest_sha256().to_string(),
        artifacts,
    };
    Ok((b, id))
}

#[derive(Deserialize)]
struct CorpusLine {
    text_a: String,
    text_b: String,
}

fn corpus(c: &Common) -> Result<Vec<String>, String> {
    let path = c
        .corpus
        .clone()
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/corpus/sts-pairs.jsonl"));
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let l: CorpusLine = serde_json::from_str(line).map_err(|e| format!("{}: {e}", path.display()))?;
        out.push(l.text_a);
        out.push(l.text_b);
    }
    if out.is_empty() {
        return Err(format!("{}: no texts", path.display()));
    }
    Ok(out)
}

/// How a workload measures its texts: the bundle's tokenizer, or a word
/// count when the bundle carries none the core can load (then a token is
/// taken as 0.75 words, the usual English WordPiece/BPE ratio, and the
/// cell says so).
enum Counter {
    Tokenizer(Arc<Tokenizer>),
    Words,
}

impl Counter {
    fn count(&self, text: &str) -> Result<u32, String> {
        match self {
            Counter::Tokenizer(t) => t.count(text, true).map_err(|e| e.to_string()),
            Counter::Words => Ok((text.split_whitespace().count() as f64 / 0.75).ceil() as u32 + 2),
        }
    }
}

/// `n` texts that each tokenize to about 90% of `seq` (never more than
/// `seq - 2`), built by concatenating corpus sentences.
fn texts_for(corpus: &[String], tok: &Counter, n: usize, seq: u32, offset: usize) -> Result<Vec<String>, String> {
    let goal = ((seq as f64) * 0.9).floor() as u32;
    let cap = seq.saturating_sub(2);
    let mut out = Vec::with_capacity(n);
    let mut cursor = offset;
    for _ in 0..n {
        let mut text = String::new();
        loop {
            let s = &corpus[cursor % corpus.len()];
            cursor += 1;
            let candidate = if text.is_empty() { s.clone() } else { format!("{text} {s}") };
            let c = tok.count(&candidate)?;
            if c > cap {
                if text.is_empty() {
                    // One sentence is already over the budget: keep it; the
                    // session truncates it, and the cell reports the live count.
                    text = candidate;
                }
                break;
            }
            text = candidate;
            if c >= goal {
                break;
            }
        }
        out.push(text);
    }
    Ok(out)
}

fn delta(a: &turbo::SessionStats, b: &turbo::SessionStats, runs: u64) -> PerRun {
    let per = |x: u64, y: u64| (y.saturating_sub(x)) / runs.max(1);
    PerRun {
        h2d_bytes: per(a.h2d_bytes, b.h2d_bytes),
        d2h_bytes: per(a.d2h_bytes, b.d2h_bytes),
        host_allocs: per(a.host_allocs, b.host_allocs),
        provider_allocs: match (a.provider_allocs, b.provider_allocs) {
            (Some(x), Some(y)) => Some(per(x, y)),
            _ => None,
        },
    }
}

// ---------------------------------------------------------------------------
// Embed
// ---------------------------------------------------------------------------

fn bench_embed(c: &Common, batches: &[u32], seqs: &[u32]) -> Result<Receipt, String> {
    let t = open_target(c)?;
    let (bundle, bid) = bundle_id(&c.bundle)?;
    let (counter, tokenizer_note) = match Tokenizer::from_bundle(&bundle) {
        Ok(t) => (Counter::Tokenizer(t), String::new()),
        Err(e) => {
            eprintln!("no core tokenizer for this bundle ({e}); texts are sized by word count and the prepared-token path is skipped");
            (Counter::Words, format!("skipped: {e}"))
        }
    };
    let model = t.ctx.load_model(&c.bundle, &ModelDesc::default()).map_err(|e| format!("load model: {e}"))?;
    let info = model.info();
    let dim = info.dim as usize;
    let corpus = corpus(c)?;
    let mut cells = Vec::new();
    for &seq in seqs {
        if seq > info.max_seq {
            eprintln!("skipping seq {seq}: the model's max_seq is {}", info.max_seq);
            continue;
        }
        for &batch in batches {
            if batch > info.max_batch {
                eprintln!("skipping batch {batch}: the model's max_batch is {}", info.max_batch);
                continue;
            }
            let session = model
                .create_session(&SessionDesc { max_batch: batch, max_seq: seq, ..Default::default() })
                .map_err(|e| format!("session {batch}x{seq}: {e}"))?;
            let texts = texts_for(&corpus, &counter, batch as usize, seq, (batch * seq) as usize)?;
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let mut out = vec![0u8; batch as usize * dim * 4];

            // Prepared tokens: encoded once by the core tokenizer, padded to seq.
            let stride = seq as usize;
            let mut ids = vec![0i32; batch as usize * stride];
            let mut mask = vec![0i32; batch as usize * stride];
            let mut lengths = vec![0u32; batch as usize];
            if let Counter::Tokenizer(tok) = &counter {
                tok.encode_into(
                    &refs,
                    &EncodeOptions { max_tokens: seq, pad_to: seq, ..Default::default() },
                    EncodeTarget {
                        ids: &mut ids,
                        mask: &mut mask,
                        types: None,
                        row_stride: stride,
                        lengths: &mut lengths,
                    },
                )
                .map_err(|e| format!("encode: {e}"))?;
            } else {
                for (i, t) in refs.iter().enumerate() {
                    lengths[i] = counter.count(t)?.min(seq);
                }
            }
            let live: u64 = lengths.iter().map(|&l| l as u64).sum();
            let live_per_row = live as f64 / batch as f64;
            let token_batch = TokenBatch { batch, seq, row_stride: seq, ids: &ids, mask: &mask, types: None };

            let mut run_text = || -> Result<(), String> {
                session.write_text(&refs, &EmbedOptions::default()).map_err(|e| format!("write_text: {e}"))?;
                let r = session.run(&Default::default()).map_err(|e| format!("run: {e}"))?;
                r.read(0, &mut out).map_err(|e| format!("read: {e}"))?;
                Ok(())
            };
            for _ in 0..c.warmup {
                run_text()?;
            }
            let before = session.stats().map_err(|e| e.to_string())?;
            let mut samples = Vec::with_capacity(c.iters as usize);
            for _ in 0..c.iters {
                let t0 = Instant::now();
                run_text()?;
                samples.push(t0.elapsed());
            }
            let after = session.stats().map_err(|e| e.to_string())?;
            let per_run = delta(&before, &after, c.iters as u64);
            let text_path = summarize(&mut samples, batch as u64, live);

            let prepared = if matches!(counter, Counter::Tokenizer(_)) {
                let mut run_tokens = || -> Result<(), String> {
                    session.write_tokens(&token_batch).map_err(|e| format!("write_tokens: {e}"))?;
                    let r = session.run(&Default::default()).map_err(|e| format!("run: {e}"))?;
                    r.read(0, &mut out).map_err(|e| format!("read: {e}"))?;
                    Ok(())
                };
                for _ in 0..c.warmup {
                    run_tokens()?;
                }
                let mut samples = Vec::with_capacity(c.iters as usize);
                for _ in 0..c.iters {
                    let t0 = Instant::now();
                    run_tokens()?;
                    samples.push(t0.elapsed());
                }
                Some(summarize(&mut samples, batch as u64, live))
            } else {
                None
            };
            let prepared_p50 = prepared
                .as_ref()
                .map(|p| format!("{:.3} ms ({:.0} rows/s)", p.p50_ms, p.rows_per_s))
                .unwrap_or_else(|| "skipped".to_string());
            eprintln!(
                "embed batch {batch:>2} seq {seq:>3}: text p50 {:.3} ms ({:.0} rows/s, {:.0} tok/s); tokens p50 {prepared_p50}; live {:.1} tok/row; h2d {} d2h {} per run",
                text_path.p50_ms, text_path.rows_per_s, text_path.tokens_per_s, live_per_row, per_run.h2d_bytes, per_run.d2h_bytes
            );
            cells.push(EmbedCell {
                batch,
                seq,
                live_tokens_per_row: live_per_row,
                text_path,
                prepared_tokens_path: prepared,
                prepared_tokens_note: tokenizer_note.clone(),
                per_run,
            });
        }
    }
    if cells.is_empty() {
        return Err("no cell fit the model's limits".to_string());
    }
    Ok(receipt(&t, bid, cells, None, None))
}

// ---------------------------------------------------------------------------
// Rerank
// ---------------------------------------------------------------------------

fn bench_rerank(c: &Common, docs: u32, seq: u32) -> Result<Receipt, String> {
    let t = open_target(c)?;
    let (_bundle, bid) = bundle_id(&c.bundle)?;
    let model = t.ctx.load_model(&c.bundle, &ModelDesc::default()).map_err(|e| format!("load model: {e}"))?;
    let info = model.info();
    if docs > info.max_batch || seq > info.max_seq {
        return Err(format!("{docs} docs x {seq} exceeds the model's limits {}x{}", info.max_batch, info.max_seq));
    }
    let corpus = corpus(c)?;
    let session = model
        .create_session(&SessionDesc { max_batch: docs, max_seq: seq, ..Default::default() })
        .map_err(|e| format!("session: {e}"))?;
    let query = corpus[0].clone();
    let doc_texts: Vec<&str> = (0..docs as usize).map(|i| corpus[(i + 1) % corpus.len()].as_str()).collect();
    let mut out = vec![0u8; docs as usize * 4];
    let mut run = || -> Result<(), String> {
        session.write_pairs(&query, &doc_texts, &RerankOptions::default()).map_err(|e| format!("write_pairs: {e}"))?;
        let r = session.run(&Default::default()).map_err(|e| format!("run: {e}"))?;
        r.read(0, &mut out).map_err(|e| format!("read: {e}"))?;
        Ok(())
    };
    for _ in 0..c.warmup {
        run()?;
    }
    let before = session.stats().map_err(|e| e.to_string())?;
    let mut samples = Vec::with_capacity(c.iters as usize);
    for _ in 0..c.iters {
        let t0 = Instant::now();
        run()?;
        samples.push(t0.elapsed());
    }
    let after = session.stats().map_err(|e| e.to_string())?;
    let per_run = delta(&before, &after, c.iters as u64);
    let text_path = summarize(&mut samples, docs as u64, 0);
    eprintln!("rerank {docs} docs seq {seq}: p50 {:.3} ms ({:.0} docs/s)", text_path.p50_ms, text_path.rows_per_s);
    Ok(receipt(&t, bid, Vec::new(), Some(RerankCell { docs, seq, text_path, per_run }), None))
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

fn bench_generate(c: &Common, new_tokens: u32, prompt: &str) -> Result<Receipt, String> {
    let t = open_target(c)?;
    let (_bundle, bid) = bundle_id(&c.bundle)?;
    let model = t.ctx.load_model(&c.bundle, &ModelDesc::default()).map_err(|e| format!("load model: {e}"))?;
    let min_supported = t.device.caps.trim_start_matches("0x").chars().count() > 0
        && (u64::from_str_radix(t.device.caps.trim_start_matches("0x"), 16).unwrap_or(0)
            & turbo::abi::TURBO_CAP_OPT_GEN_MIN_TOKENS)
            != 0;
    let desc = GenerateDesc {
        max_new_tokens: new_tokens,
        min_new_tokens: if min_supported { new_tokens } else { 0 },
        ..Default::default()
    };
    let messages = [Message { role: "user", content: prompt }];
    let mut ttft = Vec::new();
    let mut decode_rate = Vec::new();
    let mut totals = Vec::new();
    let mut generated = Vec::new();
    let mut reasons = Vec::new();
    let mut prompt_tokens = 0;
    for i in 0..(c.warmup + c.iters) {
        let g = model.create_generation(&desc).map_err(|e| format!("generation: {e}"))?;
        g.prompt(&messages).map_err(|e| format!("prompt: {e}"))?;
        let t0 = Instant::now();
        let mut first: Option<Duration> = None;
        let mut n = 0u64;
        let reason;
        loop {
            let chunk = g.step().map_err(|e| format!("step: {e}"))?;
            if first.is_none() && !chunk.tokens.is_empty() {
                first = Some(t0.elapsed());
            }
            n += chunk.tokens.len() as u64;
            prompt_tokens = chunk.prompt_tokens;
            if chunk.done {
                reason = chunk.finish_reason;
                break;
            }
        }
        let total = t0.elapsed();
        if i < c.warmup {
            continue;
        }
        let f = first.unwrap_or(total);
        ttft.push(f);
        let decode = total.saturating_sub(f).as_secs_f64();
        decode_rate.push(if n > 1 && decode > 0.0 { (n - 1) as f64 / decode } else { 0.0 });
        totals.push(total);
        generated.push(n as f64);
        reasons.push(format!("{reason:?}"));
        if reason != FinishReason::Length && min_supported {
            return Err(format!(
                "generation ended with {reason:?} after {n} tokens although min_new_tokens was honored"
            ));
        }
    }
    let p50 = |v: &mut Vec<Duration>| {
        v.sort();
        v[v.len() / 2].as_secs_f64() * 1e3
    };
    let mut rates = decode_rate.clone();
    rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cell = GenerateCell {
        new_tokens_requested: new_tokens,
        generated_tokens_mean: generated.iter().sum::<f64>() / generated.len() as f64,
        prompt_tokens,
        time_to_first_token_ms_p50: p50(&mut ttft),
        decode_tokens_per_s_p50: rates[rates.len() / 2],
        total_ms_p50: p50(&mut totals),
        finish_reasons: reasons,
        iters: c.iters,
    };
    eprintln!(
        "generate {new_tokens} tokens: ttft p50 {:.1} ms, decode {:.1} tok/s p50, total p50 {:.0} ms, generated {:.1} mean",
        cell.time_to_first_token_ms_p50, cell.decode_tokens_per_s_p50, cell.total_ms_p50, cell.generated_tokens_mean
    );
    Ok(receipt(&t, bid, Vec::new(), None, Some(cell)))
}

fn receipt(
    t: &Target,
    bundle: BundleId,
    embed: Vec<EmbedCell>,
    rerank: Option<RerankCell>,
    generate: Option<GenerateCell>,
) -> Receipt {
    Receipt {
        receipt_version: 1,
        kind: "benchmark".to_string(),
        date: today(),
        machine: machine(),
        commit: run_cmd("git", &["rev-parse", "HEAD"]),
        provider: t.provider.clone(),
        device: t.device.clone(),
        bundle,
        embed,
        rerank,
        generate,
        budget_check: None,
        native_reference: "not run; see the provider README for the direct-native program of this pair".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Budget check
// ---------------------------------------------------------------------------

fn check_budget(r: &mut Receipt, budget_path: &Path, tolerance: f64) -> Result<(), String> {
    let text = std::fs::read_to_string(budget_path).map_err(|e| format!("{}: {e}", budget_path.display()))?;
    let budget: Receipt = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", budget_path.display()))?;
    let mut violations = Vec::new();
    let over = |name: &str, got: f64, want: f64, v: &mut Vec<String>| {
        let limit = want * (1.0 + tolerance);
        if got > limit {
            v.push(format!(
                "{name}: p50 {got:.3} ms exceeds the budget {want:.3} ms + {:.0}% = {limit:.3} ms",
                tolerance * 100.0
            ));
        }
    };
    for cell in &r.embed {
        match budget.embed.iter().find(|b| b.batch == cell.batch && b.seq == cell.seq) {
            Some(b) => {
                over(
                    &format!("embed {}x{} text", cell.batch, cell.seq),
                    cell.text_path.p50_ms,
                    b.text_path.p50_ms,
                    &mut violations,
                );
                if let (Some(p), Some(bp)) = (&cell.prepared_tokens_path, &b.prepared_tokens_path) {
                    over(&format!("embed {}x{} tokens", cell.batch, cell.seq), p.p50_ms, bp.p50_ms, &mut violations);
                }
            }
            None => violations.push(format!("embed {}x{}: the budget has no such cell", cell.batch, cell.seq)),
        }
    }
    if let (Some(cell), Some(b)) = (&r.rerank, &budget.rerank) {
        over("rerank", cell.text_path.p50_ms, b.text_path.p50_ms, &mut violations);
    }
    if let (Some(cell), Some(b)) = (&r.generate, &budget.generate) {
        over("generate total", cell.total_ms_p50, b.total_ms_p50, &mut violations);
    }
    r.budget_check =
        Some(BudgetCheck { budget_file: budget_path.display().to_string(), tolerance, violations: violations.clone() });
    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// Discover
// ---------------------------------------------------------------------------

/// Names of the `TURBO_CAP_*` bits, for the survey.
const CAP_NAMES: &[(u64, &str)] = &[
    (turbo::abi::TURBO_CAP_ASYNC, "async"),
    (turbo::abi::TURBO_CAP_HOST_PTR_IMPORT, "host-pointer import"),
    (turbo::abi::TURBO_CAP_DEVICE_RESULT, "device-resident results"),
    (turbo::abi::TURBO_CAP_EXTERNAL_QUEUE, "external queue"),
    (turbo::abi::TURBO_CAP_DMABUF, "dmabuf"),
    (turbo::abi::TURBO_CAP_UNIFIED_MEMORY, "unified memory"),
    (turbo::abi::TURBO_CAP_DYNAMIC_SHAPE, "dynamic shapes"),
    (turbo::abi::TURBO_CAP_WEIGHT_SHARING, "weight sharing"),
    (turbo::abi::TURBO_CAP_DEVICE_TOKENIZE, "device tokenize"),
    (turbo::abi::TURBO_CAP_DEVICE_POSTPROCESS, "device post-processing"),
    (turbo::abi::TURBO_CAP_DETERMINISTIC, "deterministic"),
    (turbo::abi::TURBO_CAP_OPT_TRUNCATE, "opt: truncate"),
    (turbo::abi::TURBO_CAP_OPT_MAX_TOKENS, "opt: max_tokens"),
    (turbo::abi::TURBO_CAP_OPT_PROMPT_ROLE, "opt: prompt_role"),
    (turbo::abi::TURBO_CAP_OPT_NORMALIZE, "opt: normalize"),
    (turbo::abi::TURBO_CAP_OPT_POOLING_OVERRIDE, "opt: pooling override"),
    (turbo::abi::TURBO_CAP_OPT_OUTPUT_DIM, "opt: output_dim"),
    (turbo::abi::TURBO_CAP_OPT_OUTPUT_DTYPE, "opt: output_dtype"),
    (turbo::abi::TURBO_CAP_OPT_TOP_N, "opt: top_n"),
    (turbo::abi::TURBO_CAP_OPT_AGGREGATION, "opt: aggregation"),
    (turbo::abi::TURBO_CAP_OPT_RAW_SCORES, "opt: raw_scores"),
    (turbo::abi::TURBO_CAP_OPT_GEN_STRUCTURED, "gen: structured output"),
    (turbo::abi::TURBO_CAP_OPT_GEN_TOOLS, "gen: tools"),
    (turbo::abi::TURBO_CAP_OPT_GEN_N, "gen: n_sequences"),
    (turbo::abi::TURBO_CAP_OPT_GEN_LOGIT_BIAS, "gen: logit bias"),
    (turbo::abi::TURBO_CAP_OPT_GEN_PENALTIES, "gen: penalties"),
    (turbo::abi::TURBO_CAP_OPT_GEN_LOGPROBS, "gen: logprobs"),
    (turbo::abi::TURBO_CAP_OPT_GEN_STOP_STRINGS, "gen: stop strings"),
    (turbo::abi::TURBO_CAP_OPT_GEN_SEED, "gen: seed"),
    (turbo::abi::TURBO_CAP_OPT_GEN_SAMPLING, "gen: sampling"),
    (turbo::abi::TURBO_CAP_OPT_GEN_MIN_TOKENS, "gen: min_new_tokens"),
    (turbo::abi::TURBO_CAP_OPT_GEN_ECHO, "gen: echo"),
    (turbo::abi::TURBO_CAP_OPT_GEN_STOP_TOKENS, "gen: stop tokens"),
];

const TASKS: &[Task] = &[
    Task::Embed,
    Task::Rerank,
    Task::Classify,
    Task::TokenClassify,
    Task::Generate,
    Task::Tokenize,
    Task::Run,
    Task::Chunk,
];
const MODALITIES: &[Modality] = &[Modality::Text, Modality::Audio, Modality::Image, Modality::Video];

#[derive(Serialize)]
struct CapCell {
    task: String,
    modality: String,
    status: String,
    dtype: Option<String>,
    reference_dtype: Option<String>,
    cosine_floor: f32,
    deterministic: bool,
    notes: String,
}

#[derive(Serialize)]
struct BundleVerdict {
    bundle: String,
    model_id: String,
    task: String,
    modality: String,
    can_run: bool,
    reason: String,
}

#[derive(Serialize)]
struct DeviceSurvey {
    index: u32,
    provider_id: String,
    provider_version: String,
    ordinal: u32,
    kind: String,
    name: String,
    vendor: String,
    runtime_version: String,
    driver_version: String,
    memory_total: u64,
    caps: String,
    features: Vec<String>,
    capabilities: Vec<CapCell>,
    bundles: Vec<BundleVerdict>,
}

#[derive(Serialize)]
struct Survey {
    machine: Machine,
    abi_version: u32,
    provider_libraries: Vec<String>,
    load_failures: Vec<String>,
    devices: Vec<DeviceSurvey>,
}

fn discover(libs: &[PathBuf], provider_dir: Option<&Path>, bundles: &[PathBuf]) -> Result<Survey, String> {
    let mut paths: Vec<PathBuf> = libs.to_vec();
    if let Some(dir) = provider_dir {
        let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                name.starts_with("libturbo_provider_") && (name.ends_with(".so") || name.ends_with(".dylib"))
            })
            .collect();
        found.sort();
        paths.extend(found);
    }
    let provider_paths: Vec<String> = paths.iter().map(|p| p.to_string_lossy().into_owned()).collect();
    // A library that fails to load is reported, not fatal: the survey's job
    // is to say what is here, and "this library did not load because ..."
    // is part of that answer. Each library is tried on its own so one
    // failure does not hide the others.
    let mut load_failures = Vec::new();
    let mut loadable = Vec::new();
    for p in &provider_paths {
        match turbo::create_runtime(RuntimeDesc { provider_paths: vec![p.clone()], ..Default::default() }) {
            Ok(rt) => {
                for f in rt.failures() {
                    load_failures.push(format!("{p}: {}: {}", f.what, f.error));
                }
                loadable.push(p.clone());
            }
            Err(e) => load_failures.push(format!("{p}: {e}")),
        }
    }
    let rt = turbo::create_runtime(RuntimeDesc { provider_paths: loadable.clone(), ..Default::default() })
        .map_err(|e| format!("runtime: {e}"))?;
    for f in rt.failures() {
        load_failures.push(format!("{}: {}", f.what, f.error));
    }
    let opened: Vec<(PathBuf, Option<Bundle>, String)> = bundles
        .iter()
        .map(|b| match Bundle::open(b) {
            Ok(bundle) => (b.clone(), Some(bundle), String::new()),
            Err(e) => (b.clone(), None, e.to_string()),
        })
        .collect();
    let mut devices = Vec::new();
    for (index, entry) in rt.devices().into_iter().enumerate() {
        let index = index as u32;
        let info = entry.info;
        let features = CAP_NAMES.iter().filter(|(bit, _)| info.caps & bit != 0).map(|(_, n)| n.to_string()).collect();
        let mut capabilities = Vec::new();
        for &task in TASKS {
            for &modality in MODALITIES {
                let cap = rt.capability(index, task, modality).map_err(|e| e.to_string())?;
                if !cap.is_offered() {
                    continue;
                }
                capabilities.push(CapCell {
                    task: format!("{task:?}"),
                    modality: format!("{modality:?}"),
                    status: format!("{:?}", cap.status),
                    dtype: cap.dtype.map(|d| format!("{d:?}")),
                    reference_dtype: cap.reference_dtype.map(|d| format!("{d:?}")),
                    cosine_floor: cap.cosine_floor,
                    deterministic: cap.deterministic,
                    notes: cap.notes.clone(),
                });
            }
        }
        let mut verdicts = Vec::new();
        for (path, bundle, err) in &opened {
            match bundle {
                None => verdicts.push(BundleVerdict {
                    bundle: path.display().to_string(),
                    model_id: String::new(),
                    task: String::new(),
                    modality: String::new(),
                    can_run: false,
                    reason: err.clone(),
                }),
                Some(b) => {
                    let (task, modality) = (b.task(), b.modality());
                    let (can, reason) = match rt.can_run(index, path, task, modality) {
                        Ok(()) => (true, String::new()),
                        Err(e) => (false, e.to_string()),
                    };
                    verdicts.push(BundleVerdict {
                        bundle: path.display().to_string(),
                        model_id: b.manifest().model_id.clone(),
                        task: format!("{task:?}"),
                        modality: format!("{modality:?}"),
                        can_run: can,
                        reason,
                    });
                }
            }
        }
        devices.push(DeviceSurvey {
            index,
            provider_id: info.provider_id.clone(),
            provider_version: info.provider_version.clone(),
            ordinal: info.ordinal,
            kind: format!("{:?}", info.kind),
            name: info.name.clone(),
            vendor: info.vendor.clone(),
            runtime_version: info.runtime_version.clone(),
            driver_version: info.driver_version.clone(),
            memory_total: info.memory_total,
            caps: format!("{:#x}", info.caps),
            features,
            capabilities,
            bundles: verdicts,
        });
    }
    Ok(Survey {
        machine: machine(),
        abi_version: turbo::abi::TURBO_ABI_VERSION,
        provider_libraries: provider_paths,
        load_failures,
        devices,
    })
}

fn print_survey(s: &Survey) {
    println!("machine: {} ({}, {}); libturbo ABI {}", s.machine.hostname, s.machine.os, s.machine.arch, s.abi_version);
    if s.provider_libraries.is_empty() {
        println!("provider libraries: none named (built-in providers only)");
    } else {
        println!("provider libraries:");
        for p in &s.provider_libraries {
            println!("  {p}");
        }
    }
    for f in &s.load_failures {
        println!("  did not load: {f}");
    }
    println!();
    if s.devices.is_empty() {
        println!("no devices");
    }
    for d in &s.devices {
        println!("[{}] {}:{}  {}  ({}, vendor {})", d.index, d.provider_id, d.ordinal, d.name, d.kind, d.vendor);
        println!(
            "     provider {} {}; runtime {}; driver {}",
            d.provider_id, d.provider_version, d.runtime_version, d.driver_version
        );
        if d.memory_total > 0 {
            println!("     memory {:.1} GiB", d.memory_total as f64 / (1u64 << 30) as f64);
        }
        println!("     features: {}", if d.features.is_empty() { "(none)".to_string() } else { d.features.join(", ") });
        if d.capabilities.is_empty() {
            println!("     offers: nothing");
        } else {
            println!("     offers:");
            for c in &d.capabilities {
                let mut line = format!("       {:<14} x {:<6} {:<12}", c.task, c.modality, c.status);
                if let Some(dt) = &c.dtype {
                    line.push_str(&format!(" compute {dt}"));
                }
                if c.cosine_floor > 0.0 {
                    line.push_str(&format!(
                        " cosine floor {:.3} vs {}",
                        c.cosine_floor,
                        c.reference_dtype.clone().unwrap_or_default()
                    ));
                }
                if c.deterministic {
                    line.push_str(" deterministic");
                }
                if !c.notes.is_empty() {
                    line.push_str(&format!("  ({})", c.notes));
                }
                println!("{line}");
            }
        }
        for b in &d.bundles {
            if b.can_run {
                println!("     can run {} ({}, {} x {})", b.bundle, b.model_id, b.task, b.modality);
            } else {
                println!("     cannot run {}: {}", b.bundle, b.reason);
            }
        }
        println!();
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Cmd::Discover { provider_libs, provider_dir, bundles, json } = &cli.command {
        return match discover(provider_libs, provider_dir.as_deref(), bundles) {
            Ok(s) => {
                if *json {
                    println!("{}", serde_json::to_string_pretty(&s).expect("serialize survey"));
                } else {
                    print_survey(&s);
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        };
    }
    let (common, result) = match &cli.command {
        Cmd::Embed { common, batches, seqs } => (common, bench_embed(common, batches, seqs)),
        Cmd::Rerank { common, docs, seq } => (common, bench_rerank(common, *docs, *seq)),
        Cmd::Generate { common, new_tokens, prompt } => (common, bench_generate(common, *new_tokens, prompt)),
        Cmd::Discover { .. } => unreachable!("handled above"),
    };
    let mut r = match result {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let budget_result = match &common.budget {
        Some(b) => check_budget(&mut r, b, common.tolerance),
        None => Ok(()),
    };
    let json = serde_json::to_string_pretty(&r).expect("serialize receipt");
    match &common.out {
        Some(path) => {
            if let Err(e) = std::fs::write(path, format!("{json}\n")) {
                eprintln!("error: write {}: {e}", path.display());
                return ExitCode::from(2);
            }
            eprintln!("receipt written to {}", path.display());
        }
        None => println!("{json}"),
    }
    match budget_result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("budget regression:\n{e}");
            ExitCode::from(1)
        }
    }
}
