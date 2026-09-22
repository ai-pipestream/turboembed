//! `turbo-bench`: the `libturbo` half of the matched benchmark protocol
//! (PLAN.md section 11) and its receipt writer.
//!
//! ```text
//! turbo-bench embed    --provider-lib <so> --provider <id> --bundle <dir> [--ordinal N]
//!                      [--batches 1,8,32] [--seqs 32,128,256] [--iters 30] [--warmup 5]
//!                      [--out receipt.json] [--budget earlier-receipt.json]
//! turbo-bench rerank   --provider-lib <so> --provider <id> --bundle <dir> [--docs 32] ...
//! turbo-bench generate --provider-lib <so> --provider <id> --bundle <dir> [--new-tokens 128] ...
//! turbo-bench discover [--provider-lib <so>]... [--provider-dir <dir>] [--bundle <dir>]... [--json] [--strict]
//! ```
//!
//! `discover` is the survey: it loads the named provider libraries (or, with
//! none named, the built-in providers), and for every device prints the name, kind, runtime
//! and driver versions, the option features it honors (decoded from the
//! capability bits), the task x modality capability matrix with status,
//! compute dtype and measured cosine floor, and, for each bundle named,
//! whether that device can run it and why not.
//!
//! Embedding workloads run the text path (`write_text` + `run` + read) and
//! the prepared-token path (`write_tokens` + `run` + read, tokens encoded
//! once up front by the core tokenizer) for every batch x seq cell, with
//! texts built from the committed STS corpus towards 90% of each sequence
//! length (whole sentences, so the live count the cell reports is what was
//! reached). Reported per cell: p50/mean/min/max latency (p99 only from 100
//! iterations up), rows/s, tokens/s when the bundle's tokenizer counted the
//! tokens, and the per-run H2D/D2H bytes, host allocations, and provider
//! allocations from the session counters, averaged over the timed runs.
//!
//! A receipt carries the machine, provider, runtime and driver versions,
//! device (with its memory), bundle identity (manifest and artifact hashes),
//! and the commit the binary was built from (or the one `--commit` names on
//! a machine without git). With `--budget`, the earlier receipt must be of
//! the same provider, device and bundle; every cell either receipt has is
//! checked (a cell missing from the new run is a violation), and a
//! text-path p50 over the earlier p50 plus a tolerance (default 25%) is a
//! regression: exit 1. An unusable budget file is exit 2. Budgets are set
//! from the first run per provider and then held (PLAN.md section 11).
//!
//! The direct-native reference program of each pair is still to be written
//! (a provider's README will carry it once it exists); this tool measures
//! the `libturbo` side of the pair only, which is why every receipt's
//! `native_reference` field reads "not run".

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
    /// Timed iterations per cell (p99 is reported from 100 up).
    #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u32).range(1..))]
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
    /// Allowed slowdown against the budget, as a fraction (default 0.25); needs --budget.
    #[arg(long)]
    tolerance: Option<f64>,
    /// Commit to record when the binary was built from a tree without git
    /// (an rsynced copy); otherwise the build's own commit is recorded.
    #[arg(long)]
    commit: Option<String>,
}

/// The corpus option, for the workloads that read texts.
#[derive(Args, Clone)]
struct CorpusArg {
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
        #[command(flatten)]
        corpus: CorpusArg,
        /// Batch sizes; each must be within the model's max_batch.
        #[arg(long, value_delimiter = ',', default_value = "1,8,32", value_parser = clap::value_parser!(u32).range(1..))]
        batches: Vec<u32>,
        /// Sequence lengths; each must be within the model's max_seq.
        #[arg(long, value_delimiter = ',', default_value = "32,128,256", value_parser = clap::value_parser!(u32).range(2..))]
        seqs: Vec<u32>,
    },
    /// Rerank one query against `docs` documents.
    Rerank {
        #[command(flatten)]
        common: Common,
        #[command(flatten)]
        corpus: CorpusArg,
        /// Documents per query.
        #[arg(long, default_value_t = 32, value_parser = clap::value_parser!(u32).range(1..))]
        docs: u32,
        /// Sequence length of the session.
        #[arg(long, default_value_t = 128, value_parser = clap::value_parser!(u32).range(2..))]
        seq: u32,
    },
    /// Survey the providers and devices on this machine.
    Discover {
        /// Provider libraries to load; repeatable.
        #[arg(long = "provider-lib")]
        provider_libs: Vec<PathBuf>,
        /// Directory whose `libturbo_provider_*` libraries are all loaded. With any
        /// library named, the built-in providers are left out of the survey.
        #[arg(long)]
        provider_dir: Option<PathBuf>,
        /// Bundles to check with `can_run` on every device; repeatable.
        #[arg(long = "bundle")]
        bundles: Vec<PathBuf>,
        /// Print JSON instead of the table.
        #[arg(long)]
        json: bool,
        /// Exit non-zero when a named provider library fails to load or its
        /// device probe fails.
        #[arg(long)]
        strict: bool,
    },
    /// Generate `new_tokens` tokens from a fixed prompt.
    Generate {
        #[command(flatten)]
        common: Common,
        /// New tokens per generation.
        #[arg(long, default_value_t = 128, value_parser = clap::value_parser!(u32).range(1..))]
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
    /// Nearest-rank p99; absent below 100 samples, where it would be the
    /// single worst sample.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    p99_ms: Option<f64>,
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
    rows_per_s: f64,
    /// Absent when the tokens were not counted by a tokenizer (word
    /// estimates, or a rerank cell, which has no token count).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tokens_per_s: Option<f64>,
    iters: u32,
}

/// Session counters averaged over the timed runs (a counter that moved
/// less than once per run still shows as a fraction, never as zero).
#[derive(Serialize, Deserialize, Debug, Clone)]
struct PerRun {
    h2d_bytes: f64,
    d2h_bytes: f64,
    host_allocs: Option<f64>,
    provider_allocs: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct EmbedCell {
    batch: u32,
    seq: u32,
    /// Live tokens per row; how they were counted is `token_count_source`.
    live_tokens_per_row: f64,
    /// `tokenizer` (the bundle's, exact) or `word-estimate` (words / 0.75 + 2).
    #[serde(default = "default_token_count_source")]
    token_count_source: String,
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
    /// From prompt submission (the prefill included) to the first chunk
    /// with a token.
    time_to_first_token_ms_p50: f64,
    /// Tokens after the first chunk over the time after it; absent when no
    /// iteration produced a second chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    decode_tokens_per_s_p50: Option<f64>,
    /// From prompt submission to the final chunk.
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

fn default_token_count_source() -> String {
    "unrecorded".to_string()
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Device {
    name: String,
    kind: String,
    ordinal: u32,
    caps: String,
    /// Bytes, as the provider reports them (0 when it does not).
    #[serde(default)]
    memory_total: u64,
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

fn run_cmd(cmd: &str, args: &[&str]) -> Result<String, String> {
    let out = Command::new(cmd).args(args).output().map_err(|e| format!("{cmd} {}: {e}", args.join(" ")))?;
    if !out.status.success() {
        return Err(format!("{cmd} {} exited with {}", args.join(" "), out.status));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn machine() -> Result<Machine, String> {
    Ok(Machine {
        hostname: run_cmd("uname", &["-n"])?,
        os: run_cmd("uname", &["-sr"])?,
        arch: std::env::consts::ARCH.to_string(),
    })
}

/// The commit a receipt names: the build's own (from build.rs) unless the
/// run names one, and an error when neither exists. A receipt without a
/// commit is not a measurement of anything.
fn commit(c: &Common) -> Result<String, String> {
    if let Some(named) = &c.commit {
        return Ok(named.clone());
    }
    match option_env!("TURBO_BENCH_GIT_COMMIT") {
        Some(built) => Ok(built.to_string()),
        None => Err("the binary was built from a tree without git, so the receipt cannot name a commit; pass --commit <sha>".to_string()),
    }
}

fn today() -> Result<String, String> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| format!("the clock is before the Unix epoch: {e}"))?
        .as_secs();
    Ok(civil_date(secs))
}

/// The UTC civil date of an epoch second, `YYYY-MM-DD`, without a chrono
/// dependency: days since the epoch through the era/day-of-era form of the
/// proleptic Gregorian calendar.
fn civil_date(secs: u64) -> String {
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

/// Latency figures over the timed samples. `tokens` is `Some` only when a
/// tokenizer counted them; a p99 is reported from 100 samples up.
fn summarize(samples: &mut [Duration], rows: u64, tokens: Option<u64>) -> Result<Latency, String> {
    if samples.is_empty() {
        return Err("no timed samples (iters must be at least 1)".to_string());
    }
    samples.sort();
    let n = samples.len();
    let pct = |p: f64| -> f64 {
        let idx = ((n as f64 - 1.0) * p).round() as usize;
        samples[idx.min(n - 1)].as_secs_f64() * 1e3
    };
    let total: f64 = samples.iter().map(|d| d.as_secs_f64()).sum();
    let mean = total / n as f64;
    if !(mean > 0.0) {
        return Err("the mean latency is zero; the clock did not advance".to_string());
    }
    Ok(Latency {
        p50_ms: pct(0.5),
        p99_ms: if n >= 100 { Some(pct(0.99)) } else { None },
        mean_ms: mean * 1e3,
        min_ms: samples[0].as_secs_f64() * 1e3,
        max_ms: samples[n - 1].as_secs_f64() * 1e3,
        rows_per_s: rows as f64 / mean,
        tokens_per_s: tokens.map(|t| t as f64 / mean),
        iters: n as u32,
    })
}

/// A selected device with its context, and the identity fields a receipt needs.
struct Target {
    ctx: Arc<Context>,
    provider: ProviderId,
    device: Device,
    caps: u64,
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
        None => devices.iter().find(|d| d.info.kind != DeviceKind::Cpu).unwrap_or(&devices[0]).info.ordinal,
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
            memory_total: info.memory_total,
        },
        caps: info.caps,
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

fn corpus(c: &CorpusArg) -> Result<Vec<String>, String> {
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

fn delta(a: &turbo::SessionStats, b: &turbo::SessionStats, runs: u64) -> Result<PerRun, String> {
    let per = |name: &str, x: u64, y: u64| -> Result<f64, String> {
        if y < x {
            return Err(format!("session counter {name} went backwards ({x} then {y}); the provider's stats are not monotonic"));
        }
        Ok((y - x) as f64 / runs.max(1) as f64)
    };
    Ok(PerRun {
        h2d_bytes: per("h2d_bytes", a.h2d_bytes, b.h2d_bytes)?,
        d2h_bytes: per("d2h_bytes", a.d2h_bytes, b.d2h_bytes)?,
        host_allocs: match (a.host_allocs, b.host_allocs) {
            (Some(x), Some(y)) => Some(per("host_allocs", x, y)?),
            _ => None,
        },
        provider_allocs: match (a.provider_allocs, b.provider_allocs) {
            (Some(x), Some(y)) => Some(per("provider_allocs", x, y)?),
            _ => None,
        },
    })
}

// ---------------------------------------------------------------------------
// Embed
// ---------------------------------------------------------------------------

fn bench_embed(c: &Common, corpus_arg: &CorpusArg, batches: &[u32], seqs: &[u32]) -> Result<Receipt, String> {
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
    let corpus = corpus(corpus_arg)?;
    // Every requested cell is measured or the run is an error: a receipt
    // that quietly dropped cells would then pass any budget.
    if let Some(seq) = seqs.iter().find(|&&s| s > info.max_seq) {
        return Err(format!("--seqs {seq} exceeds the model's max_seq {}", info.max_seq));
    }
    if let Some(batch) = batches.iter().find(|&&b| b > info.max_batch) {
        return Err(format!("--batches {batch} exceeds the model's max_batch {}", info.max_batch));
    }
    let mut cells = Vec::new();
    for &seq in seqs {
        for &batch in batches {
            let session = model
                .create_session(&SessionDesc { max_batch: batch, max_seq: seq, ..Default::default() })
                .map_err(|e| format!("session {batch}x{seq}: {e}"))?;
            let texts = texts_for(&corpus, &counter, batch as usize, seq, batch as usize * seq as usize)?;
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
            // Only a tokenizer's count is a token count; a word estimate is
            // labeled as such and yields no tokens/s.
            let counted = matches!(counter, Counter::Tokenizer(_));
            let tokens = counted.then_some(live);
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
            let per_run = delta(&before, &after, c.iters as u64)?;
            let text_path = summarize(&mut samples, batch as u64, tokens)?;

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
                Some(summarize(&mut samples, batch as u64, tokens)?)
            } else {
                None
            };
            let prepared_p50 = prepared
                .as_ref()
                .map(|p| format!("{:.3} ms ({:.0} rows/s)", p.p50_ms, p.rows_per_s))
                .unwrap_or_else(|| "skipped".to_string());
            let tok_s = text_path.tokens_per_s.map(|t| format!("{t:.0} tok/s")).unwrap_or_else(|| "tokens estimated".to_string());
            eprintln!(
                "embed batch {batch:>2} seq {seq:>3}: text p50 {:.3} ms ({:.0} rows/s, {tok_s}); tokens p50 {prepared_p50}; live {:.1} tok/row; h2d {:.0} d2h {:.0} per run",
                text_path.p50_ms, text_path.rows_per_s, live_per_row, per_run.h2d_bytes, per_run.d2h_bytes
            );
            cells.push(EmbedCell {
                batch,
                seq,
                live_tokens_per_row: live_per_row,
                token_count_source: if counted { "tokenizer" } else { "word-estimate" }.to_string(),
                text_path,
                prepared_tokens_path: prepared,
                prepared_tokens_note: tokenizer_note.clone(),
                per_run,
            });
        }
    }
    receipt(c, &t, bid, cells, None, None)
}

// ---------------------------------------------------------------------------
// Rerank
// ---------------------------------------------------------------------------

fn bench_rerank(c: &Common, corpus_arg: &CorpusArg, docs: u32, seq: u32) -> Result<Receipt, String> {
    let t = open_target(c)?;
    let (_bundle, bid) = bundle_id(&c.bundle)?;
    let model = t.ctx.load_model(&c.bundle, &ModelDesc::default()).map_err(|e| format!("load model: {e}"))?;
    let info = model.info();
    if docs > info.max_batch || seq > info.max_seq {
        return Err(format!("{docs} docs x {seq} exceeds the model's limits {}x{}", info.max_batch, info.max_seq));
    }
    let corpus = corpus(corpus_arg)?;
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
    let per_run = delta(&before, &after, c.iters as u64)?;
    // A rerank cell counts documents, not tokens.
    let text_path = summarize(&mut samples, docs as u64, None)?;
    eprintln!("rerank {docs} docs seq {seq}: p50 {:.3} ms ({:.0} docs/s)", text_path.p50_ms, text_path.rows_per_s);
    receipt(c, &t, bid, Vec::new(), Some(RerankCell { docs, seq, text_path, per_run }), None)
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

fn bench_generate(c: &Common, new_tokens: u32, prompt: &str) -> Result<Receipt, String> {
    let t = open_target(c)?;
    let (_bundle, bid) = bundle_id(&c.bundle)?;
    let model = t.ctx.load_model(&c.bundle, &ModelDesc::default()).map_err(|e| format!("load model: {e}"))?;
    let min_supported = t.caps & turbo::abi::TURBO_CAP_OPT_GEN_MIN_TOKENS != 0;
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
    let mut prompt_tokens: Option<u32> = None;
    for i in 0..(c.warmup as u64 + c.iters as u64) {
        let g = model.create_generation(&desc).map_err(|e| format!("generation: {e}"))?;
        // `prompt` runs the prefill, so the clock starts before it: time to
        // first token is what a caller waits for.
        let t0 = Instant::now();
        g.prompt(&messages).map_err(|e| format!("prompt: {e}"))?;
        let mut first: Option<(Duration, u64)> = None; // when, and how many tokens it carried
        let mut n = 0u64;
        let reason;
        loop {
            let chunk = g.step().map_err(|e| format!("step: {e}"))?;
            if first.is_none() && !chunk.tokens.is_empty() {
                first = Some((t0.elapsed(), chunk.tokens.len() as u64));
                match prompt_tokens {
                    None => prompt_tokens = Some(chunk.prompt_tokens),
                    Some(p) if p != chunk.prompt_tokens => {
                        return Err(format!("prompt_tokens changed between iterations ({p} then {})", chunk.prompt_tokens))
                    }
                    Some(_) => {}
                }
            }
            n += chunk.tokens.len() as u64;
            if chunk.done {
                reason = chunk.finish_reason;
                break;
            }
        }
        let total = t0.elapsed();
        if i < c.warmup as u64 {
            continue;
        }
        let (f, k) = first.unwrap_or((total, n));
        ttft.push(f);
        let decode = total.saturating_sub(f).as_secs_f64();
        if n > k && decode > 0.0 {
            decode_rate.push((n - k) as f64 / decode);
        }
        totals.push(total);
        generated.push(n as f64);
        reasons.push(format!("{reason:?}"));
        if reason != FinishReason::Length && min_supported {
            return Err(format!(
                "generation ended with {reason:?} after {n} tokens although min_new_tokens was honored"
            ));
        }
    }
    if totals.is_empty() {
        return Err("no timed iterations (iters must be at least 1)".to_string());
    }
    let p50 = |v: &mut Vec<Duration>| {
        v.sort();
        v[v.len() / 2].as_secs_f64() * 1e3
    };
    let mut rates = decode_rate.clone();
    rates.sort_by(f64::total_cmp);
    let cell = GenerateCell {
        new_tokens_requested: new_tokens,
        generated_tokens_mean: generated.iter().sum::<f64>() / generated.len() as f64,
        prompt_tokens: prompt_tokens.ok_or("no iteration produced a token")?,
        time_to_first_token_ms_p50: p50(&mut ttft),
        decode_tokens_per_s_p50: if rates.is_empty() { None } else { Some(rates[rates.len() / 2]) },
        total_ms_p50: p50(&mut totals),
        finish_reasons: reasons,
        iters: c.iters,
    };
    let decode = cell.decode_tokens_per_s_p50.map(|r| format!("{r:.1} tok/s p50")).unwrap_or_else(|| "no decode phase".to_string());
    eprintln!(
        "generate {new_tokens} tokens: ttft p50 {:.1} ms, decode {decode}, total p50 {:.0} ms, generated {:.1} mean",
        cell.time_to_first_token_ms_p50, cell.total_ms_p50, cell.generated_tokens_mean
    );
    receipt(c, &t, bid, Vec::new(), None, Some(cell))
}

fn receipt(
    c: &Common,
    t: &Target,
    bundle: BundleId,
    embed: Vec<EmbedCell>,
    rerank: Option<RerankCell>,
    generate: Option<GenerateCell>,
) -> Result<Receipt, String> {
    Ok(Receipt {
        receipt_version: 1,
        kind: "benchmark".to_string(),
        date: today()?,
        machine: machine()?,
        commit: commit(c)?,
        provider: t.provider.clone(),
        device: t.device.clone(),
        bundle,
        embed,
        rerank,
        generate,
        budget_check: None,
        native_reference: "not run; see the provider README for the direct-native program of this pair".to_string(),
    })
}

// ---------------------------------------------------------------------------
// Budget check
// ---------------------------------------------------------------------------

/// Why a budget check did not pass: the file could not serve as a budget
/// at all (exit 2, nothing recorded in the receipt), or the run regressed
/// (exit 1, recorded).
enum BudgetFailure {
    Unusable(String),
    Regression(String),
}

fn check_budget(r: &mut Receipt, budget_path: &Path, tolerance: f64) -> Result<(), BudgetFailure> {
    let unusable = |m: String| BudgetFailure::Unusable(format!("budget {}: {m}", budget_path.display()));
    let text = std::fs::read_to_string(budget_path).map_err(|e| unusable(e.to_string()))?;
    let budget: Receipt = serde_json::from_str(&text).map_err(|e| unusable(e.to_string()))?;
    // A budget is held per provider, device and bundle; another run's
    // figures are not a budget for this one.
    if budget.provider.id != r.provider.id {
        return Err(unusable(format!("provider.id is `{}`, this run is `{}`", budget.provider.id, r.provider.id)));
    }
    if budget.device.name != r.device.name || budget.device.ordinal != r.device.ordinal {
        return Err(unusable(format!(
            "device is `{}` ordinal {}, this run is `{}` ordinal {}",
            budget.device.name, budget.device.ordinal, r.device.name, r.device.ordinal
        )));
    }
    if budget.bundle.manifest_sha256 != r.bundle.manifest_sha256 {
        return Err(unusable(format!(
            "bundle manifest is {}, this run's is {}",
            budget.bundle.manifest_sha256, r.bundle.manifest_sha256
        )));
    }
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
    // A cell the budget has and this run did not produce is a violation:
    // a shrunken workload is not a pass.
    for b in &budget.embed {
        if !r.embed.iter().any(|c| c.batch == b.batch && c.seq == b.seq) {
            violations.push(format!("embed {}x{}: the budget has this cell and this run did not measure it", b.batch, b.seq));
        }
    }
    match (&r.rerank, &budget.rerank) {
        (Some(cell), Some(b)) => over("rerank", cell.text_path.p50_ms, b.text_path.p50_ms, &mut violations),
        (Some(_), None) => violations.push("rerank: the budget has no rerank figure".to_string()),
        (None, Some(_)) => violations.push("rerank: the budget has a rerank figure and this run did not measure one".to_string()),
        (None, None) => {}
    }
    match (&r.generate, &budget.generate) {
        (Some(cell), Some(b)) => over("generate total", cell.total_ms_p50, b.total_ms_p50, &mut violations),
        (Some(_), None) => violations.push("generate: the budget has no generate figure".to_string()),
        (None, Some(_)) => violations.push("generate: the budget has a generate figure and this run did not measure one".to_string()),
        (None, None) => {}
    }
    r.budget_check =
        Some(BudgetCheck { budget_file: budget_path.display().to_string(), tolerance, violations: violations.clone() });
    if violations.is_empty() {
        Ok(())
    } else {
        Err(BudgetFailure::Regression(violations.join("\n")))
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
    /// Libraries that did not load (dlopen, symbol, ABI, or an id already
    /// registered by an earlier path), one line each.
    load_failures: Vec<String>,
    /// Libraries that loaded but whose device probe failed.
    probe_failures: Vec<String>,
    devices: Vec<DeviceSurvey>,
}

fn discover(libs: &[PathBuf], provider_dir: Option<&Path>, bundles: &[PathBuf]) -> Result<Survey, String> {
    let mut paths: Vec<PathBuf> = libs.to_vec();
    if let Some(dir) = provider_dir {
        let mut found = Vec::new();
        for entry in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
            let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
            let p = entry.path();
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with("libturbo_provider_") && (name.ends_with(".so") || name.ends_with(".dylib")) {
                found.push(p);
            }
        }
        found.sort();
        paths.extend(found);
    }
    let provider_paths: Vec<String> = paths.iter().map(|p| p.to_string_lossy().into_owned()).collect();
    // With libraries named (or a directory given), the survey is of those
    // libraries alone: the built-in providers stay out so a packaged copy
    // of the mock or static provider can be loaded and checked (an id can
    // register once per runtime). With nothing named, the survey is of the
    // built-in providers.
    let no_default_providers = !provider_paths.is_empty() || provider_dir.is_some();
    // One runtime, one load per library: a library that fails to load is
    // reported against its path (an id collision names the path that
    // registered it first), a library that loaded but whose device probe
    // failed is reported separately, and neither is fatal to the survey.
    let rt = turbo::create_runtime(RuntimeDesc { provider_paths: Vec::new(), no_default_providers, ..Default::default() })
        .map_err(|e| format!("runtime: {e}"))?;
    let mut load_failures = Vec::new();
    for p in &provider_paths {
        if let Err(e) = rt.load_provider(Path::new(p)) {
            load_failures.push(format!("{p}: {e}"));
        }
    }
    let probe_failures: Vec<String> = rt.failures().iter().map(|f| format!("{}: {}", f.what, f.error)).collect();
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
        machine: machine()?,
        abi_version: turbo::abi::TURBO_ABI_VERSION,
        provider_libraries: provider_paths,
        load_failures,
        probe_failures,
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
    for f in &s.probe_failures {
        println!("  loaded, device probe failed: {f}");
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
    if let Cmd::Discover { provider_libs, provider_dir, bundles, json, strict } = &cli.command {
        return match discover(provider_libs, provider_dir.as_deref(), bundles) {
            Ok(s) => {
                if *json {
                    match serde_json::to_string_pretty(&s) {
                        Ok(text) => println!("{text}"),
                        Err(e) => {
                            eprintln!("error: the survey does not serialize: {e}");
                            return ExitCode::from(2);
                        }
                    }
                } else {
                    print_survey(&s);
                }
                if *strict && (!s.load_failures.is_empty() || !s.probe_failures.is_empty()) {
                    eprintln!(
                        "error: {} provider librar{} did not load and {} loaded but failed the device probe",
                        s.load_failures.len(),
                        if s.load_failures.len() == 1 { "y" } else { "ies" },
                        s.probe_failures.len()
                    );
                    return ExitCode::from(1);
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        };
    }
    let common = match &cli.command {
        Cmd::Embed { common, .. } | Cmd::Rerank { common, .. } | Cmd::Generate { common, .. } => common,
        Cmd::Discover { .. } => unreachable!("handled above"),
    };
    // --tolerance means nothing without a budget, so naming one without the
    // other is an error rather than a silently ignored flag.
    let tolerance = match (&common.budget, common.tolerance) {
        (None, Some(_)) => {
            eprintln!("error: --tolerance needs --budget");
            return ExitCode::from(2);
        }
        (_, Some(t)) if !(t >= 0.0) => {
            eprintln!("error: --tolerance must be a fraction of 0 or more, not {t}");
            return ExitCode::from(2);
        }
        (_, t) => t.unwrap_or(0.25),
    };
    let result = match &cli.command {
        Cmd::Embed { common, corpus, batches, seqs } => bench_embed(common, corpus, batches, seqs),
        Cmd::Rerank { common, corpus, docs, seq } => bench_rerank(common, corpus, *docs, *seq),
        Cmd::Generate { common, new_tokens, prompt } => bench_generate(common, *new_tokens, prompt),
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
        Some(b) => check_budget(&mut r, b, tolerance),
        None => Ok(()),
    };
    if let Err(BudgetFailure::Unusable(m)) = &budget_result {
        // No receipt: a run against an unusable budget measured nothing
        // that can be compared, and a receipt with `budget_check: null`
        // would read as "no budget was asked for".
        eprintln!("error: {m}");
        return ExitCode::from(2);
    }
    let json = match serde_json::to_string_pretty(&r) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("error: the receipt does not serialize: {e}");
            return ExitCode::from(2);
        }
    };
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
        Err(BudgetFailure::Regression(e)) => {
            eprintln!("budget regression:\n{e}");
            ExitCode::from(1)
        }
        Err(BudgetFailure::Unusable(_)) => unreachable!("handled before the receipt was written"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A civil date computed the other way round: walk years, then months,
    /// off the day count. Independent of the era arithmetic `civil_date`
    /// uses, so the two agreeing is evidence and not a restatement.
    fn reference_date(secs: u64) -> String {
        fn leap(y: i64) -> bool {
            (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
        }
        let mut days = (secs / 86_400) as i64;
        let mut year = 1970i64;
        loop {
            let len = if leap(year) { 366 } else { 365 };
            if days < len {
                break;
            }
            days -= len;
            year += 1;
        }
        let lengths = [31, if leap(year) { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let mut month = 1;
        for len in lengths {
            if days < len {
                break;
            }
            days -= len;
            month += 1;
        }
        format!("{year:04}-{month:02}-{:02}", days + 1)
    }

    #[test]
    fn civil_date_matches_an_independent_calendar_walk() {
        // The epoch, the last second of the epoch day, the next day, a leap
        // day, a leap-year end, a century non-leap year, and 2038.
        let cases: [(u64, &str); 8] = [
            (0, "1970-01-01"),
            (86_399, "1970-01-01"),
            (86_400, "1970-01-02"),
            (951_782_400, "2000-02-29"),
            (978_220_800, "2000-12-31"),
            (1_078_012_800, "2004-02-29"),
            (1_700_000_000, "2023-11-14"),
            (2_147_483_647, "2038-01-19"),
        ];
        for (secs, expected) in cases {
            assert_eq!(civil_date(secs), expected, "civil_date({secs}) must be the UTC civil date");
            assert_eq!(
                civil_date(secs),
                reference_date(secs),
                "civil_date({secs}) disagrees with the independent calendar walk"
            );
        }
        // Every day of a leap year and the year after it, in both forms.
        let start = 1_072_915_200; // 2004-01-01T00:00:00Z
        for day in 0..731u64 {
            let secs = start + day * 86_400;
            assert_eq!(civil_date(secs), reference_date(secs), "day {day} after 2004-01-01 disagrees");
        }
    }

    #[test]
    fn today_is_the_civil_date_of_now() {
        // `today` must be `civil_date` of the wall clock and nothing else:
        // the refactor exists so the calendar is testable without the clock.
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        let stamp = today().expect("the clock is after the epoch");
        assert!(
            stamp == civil_date(now) || stamp == civil_date(now + 1),
            "today() {stamp} is not the civil date of {now} (nor the second after it)"
        );
        assert_eq!(stamp.len(), 10, "a receipt date is YYYY-MM-DD: {stamp}");
        assert_eq!(stamp.match_indices('-').count(), 2, "a receipt date is YYYY-MM-DD: {stamp}");
    }

    fn close(actual: f64, expected: f64, what: &str) {
        assert!((actual - expected).abs() < 1e-6, "{what}: expected {expected}, got {actual}");
    }

    #[test]
    fn summarize_reports_the_order_statistics_of_the_sample_set() {
        // Five samples, deliberately unsorted: summarize sorts in place and
        // the percentiles are nearest-rank on the sorted set.
        let mut samples: Vec<Duration> = [5u64, 1, 4, 2, 3].iter().map(|&ms| Duration::from_millis(ms)).collect();
        let l = summarize(&mut samples, 10, Some(400)).expect("five samples");
        assert_eq!(samples, (1..=5).map(Duration::from_millis).collect::<Vec<_>>(), "summarize must sort in place");
        close(l.p50_ms, 3.0, "p50");
        assert_eq!(l.p99_ms, None, "a p99 needs 100 samples; five would make it the single worst one");
        close(l.min_ms, 1.0, "min");
        close(l.max_ms, 5.0, "max");
        close(l.mean_ms, 3.0, "mean");
        // Throughput is rows (and tokens) per mean second, not per p50.
        close(l.rows_per_s, 10.0 / 0.003, "rows_per_s");
        close(l.tokens_per_s.expect("tokens were counted"), 400.0 / 0.003, "tokens_per_s");
        assert_eq!(l.iters, 5, "iters counts the timed samples");
    }

    #[test]
    fn summarize_percentiles_pick_the_nearest_rank() {
        // 1..=100 ms: p50 is the 51st sample and p99 the 99th, by
        // `round((n - 1) * p)`. A single sample is every statistic.
        let mut samples: Vec<Duration> = (1..=100u64).map(Duration::from_millis).collect();
        let l = summarize(&mut samples, 100, None).expect("a hundred samples");
        close(l.p50_ms, 51.0, "p50");
        close(l.p99_ms.expect("a hundred samples carry a p99"), 99.0, "p99");
        close(l.min_ms, 1.0, "min");
        close(l.max_ms, 100.0, "max");
        close(l.mean_ms, 50.5, "mean");
        assert_eq!(l.tokens_per_s, None, "no token count, no tokens/s");
        assert_eq!(l.iters, 100);

        let mut one = [Duration::from_micros(2500)];
        let l = summarize(&mut one, 1, Some(7)).expect("one sample");
        close(l.p50_ms, 2.5, "p50 of one sample");
        assert_eq!(l.p99_ms, None, "one sample has no p99");
        close(l.min_ms, 2.5, "min of one sample");
        close(l.max_ms, 2.5, "max of one sample");
        close(l.rows_per_s, 400.0, "rows_per_s of one sample");
        assert_eq!(l.iters, 1);
    }

    #[test]
    fn summarize_refuses_an_empty_sample_set() {
        let e = summarize(&mut [], 1, Some(1)).unwrap_err();
        assert!(e.contains("no timed samples"), "{e}");
    }

    #[test]
    fn counter_words_estimates_tokens_at_three_quarters_of_a_word() {
        // The fallback counter, used when the bundle carries no tokenizer
        // the core can load: ceil(words / 0.75) content tokens plus the two
        // special tokens a sequence carries.
        let count = |text: &str| Counter::Words.count(text).expect("the word counter never fails");
        assert_eq!(count(""), 2, "an empty text is still the two special tokens");
        assert_eq!(count("   \t\n  "), 2, "whitespace alone is no words");
        assert_eq!(count("hello"), 4);
        assert_eq!(count("hello world"), 5);
        assert_eq!(count("a b c"), 6);
        assert_eq!(count("a b c d"), 8, "four words are exactly 16/3 -> 6, plus 2");
        assert_eq!(count("  hello   world  "), count("hello world"), "runs of whitespace do not add words");
        assert_eq!(count("héllo wörld"), count("hello world"), "the counter splits on whitespace, not bytes");
        // Monotonic in the word count, so `texts_for` can grow a text until
        // it reaches its budget instead of looping forever.
        let mut text = String::new();
        let mut previous = count(&text);
        for i in 0..64 {
            text.push_str(&format!(" w{i}"));
            let now = count(&text);
            assert!(now >= previous, "the word counter went backwards at word {i}: {previous} then {now}");
            previous = now;
        }
        assert!(previous > 64, "64 words must estimate more than 64 tokens, got {previous}");
    }
}
