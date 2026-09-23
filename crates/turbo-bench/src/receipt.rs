//! The benchmark receipt: the JSON both `turbo-bench` (kind `benchmark`)
//! and the direct-native reference programs (kind `native`) write, so a
//! `compare` can read either side of a matched pair, plus the timing
//! summary and provenance helpers they share.

use std::process::Command;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Latency figures of one cell over its timed iterations.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Latency {
    /// Nearest-rank median, milliseconds.
    pub p50_ms: f64,
    /// Nearest-rank p99; absent below 100 samples, where it would be the
    /// single worst sample.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p99_ms: Option<f64>,
    /// Arithmetic mean, milliseconds.
    pub mean_ms: f64,
    /// Fastest sample, milliseconds.
    pub min_ms: f64,
    /// Slowest sample, milliseconds.
    pub max_ms: f64,
    /// Rows (texts, documents) per second at the mean latency.
    pub rows_per_s: f64,
    /// Tokens per second at the mean latency; absent when the tokens were
    /// not counted by a tokenizer (word estimates, or a rerank cell, which
    /// has no token count).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_per_s: Option<f64>,
    /// Timed iterations the figures summarize.
    pub iters: u32,
}

/// Session counters averaged over the timed runs (a counter that moved
/// less than once per run still shows as a fraction, never as zero).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PerRun {
    /// Bytes copied host to device per run; absent when not measured (a
    /// native receipt has no session counters).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub h2d_bytes: Option<f64>,
    /// Bytes copied device to host per run; absent when not measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub d2h_bytes: Option<f64>,
    /// Host allocations per run; absent when the provider does not count them.
    pub host_allocs: Option<f64>,
    /// Provider allocations per run; absent when the provider does not count them.
    pub provider_allocs: Option<f64>,
}

/// One embedding cell: a batch by sequence-length shape.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmbedCell {
    /// Rows per run.
    pub batch: u32,
    /// Session sequence length.
    pub seq: u32,
    /// Live tokens per row; how they were counted is `token_count_source`.
    pub live_tokens_per_row: f64,
    /// `tokenizer` (the bundle's, exact) or `word-estimate` (words / 0.75 + 2).
    #[serde(default = "default_token_count_source")]
    pub token_count_source: String,
    /// Text in, vectors out: tokenization included.
    pub text_path: Latency,
    /// Prepared token ids in, vectors out; absent when the bundle has no
    /// tokenizer the core can encode with (the reason is in
    /// `prepared_tokens_note`).
    pub prepared_tokens_path: Option<Latency>,
    /// Why `prepared_tokens_path` is absent.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub prepared_tokens_note: String,
    /// Session counters per run.
    pub per_run: PerRun,
}

/// The rerank cell: one query against `docs` documents.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RerankCell {
    /// Documents per query.
    pub docs: u32,
    /// Session sequence length.
    pub seq: u32,
    /// Query and documents in, scores out.
    pub text_path: Latency,
    /// Session counters per run.
    pub per_run: PerRun,
}

/// The generation cell: `new_tokens_requested` tokens from a fixed prompt.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerateCell {
    /// `max_new_tokens` (and `min_new_tokens` when the device honors it).
    pub new_tokens_requested: u32,
    /// Tokens produced per iteration, on average.
    pub generated_tokens_mean: f64,
    /// Prompt tokens after the chat template, from the first chunk.
    pub prompt_tokens: u32,
    /// From prompt submission (the prefill included) to the first chunk
    /// with a token.
    pub time_to_first_token_ms_p50: f64,
    /// Tokens after the first chunk over the time after it; absent when no
    /// iteration produced a second chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_tokens_per_s_p50: Option<f64>,
    /// From prompt submission to the final chunk.
    pub total_ms_p50: f64,
    /// Why each timed iteration ended.
    pub finish_reasons: Vec<String>,
    /// Timed iterations.
    pub iters: u32,
}

/// A receipt: one run of one workload on one device.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Receipt {
    /// Schema version, 1.
    pub receipt_version: u32,
    /// `benchmark` (through `libturbo`) or `native` (the runtime alone).
    pub kind: String,
    /// UTC date of the run.
    pub date: String,
    /// Where it ran.
    pub machine: Machine,
    /// The commit the tool was built from, or the one `--commit` named.
    pub commit: String,
    /// The provider (or, for a native receipt, the runtime) and its versions.
    pub provider: ProviderId,
    /// The device.
    pub device: Device,
    /// The bundle, by manifest and artifact hashes.
    pub bundle: BundleId,
    /// Embedding cells.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub embed: Vec<EmbedCell>,
    /// The rerank cell.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank: Option<RerankCell>,
    /// The generation cell.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generate: Option<GenerateCell>,
    /// The budget check, when `--budget` was given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_check: Option<BudgetCheck>,
    /// How this receipt relates to its native counterpart.
    pub native_reference: String,
}

/// The machine a receipt was taken on.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Machine {
    /// `uname -n`.
    pub hostname: String,
    /// `uname -sr`.
    pub os: String,
    /// The Rust target architecture.
    pub arch: String,
}

/// The provider (or native runtime) a receipt measured.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProviderId {
    /// Provider id, or the runtime's name for a native receipt.
    pub id: String,
    /// Provider version.
    pub version: String,
    /// Runtime version string.
    pub runtime_version: String,
    /// Driver version string.
    pub driver_version: String,
}

/// The `token_count_source` of receipts written before the field existed.
pub fn default_token_count_source() -> String {
    "unrecorded".to_string()
}

/// The device a receipt measured.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Device {
    /// Device name.
    pub name: String,
    /// Device kind (`Gpu`, `Cpu`, `Npu`, `IGpu`).
    pub kind: String,
    /// Ordinal within the provider.
    pub ordinal: u32,
    /// Capability bits, hexadecimal.
    pub caps: String,
    /// Bytes, as the provider reports them (0 when it does not).
    #[serde(default)]
    pub memory_total: u64,
}

/// The bundle a receipt measured.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BundleId {
    /// Bundle directory as given.
    pub dir: String,
    /// `model_id` from the manifest.
    pub model_id: String,
    /// SHA-256 of `bundle.json`.
    pub manifest_sha256: String,
    /// Artifact format to SHA-256.
    pub artifacts: std::collections::BTreeMap<String, String>,
}

/// The outcome of a `--budget` check.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BudgetCheck {
    /// The budget receipt's path.
    pub budget_file: String,
    /// Allowed slowdown, as a fraction.
    pub tolerance: f64,
    /// Each violation, one line.
    pub violations: Vec<String>,
}

/// Run a command and return its trimmed stdout, or why it did not run.
pub fn run_cmd(cmd: &str, args: &[&str]) -> Result<String, String> {
    let out = Command::new(cmd).args(args).output().map_err(|e| format!("{cmd} {}: {e}", args.join(" ")))?;
    if !out.status.success() {
        return Err(format!("{cmd} {} exited with {}", args.join(" "), out.status));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// This machine, by `uname`. `TURBO_BENCH_MACHINE`, when set, is recorded
/// in place of the host name, so a published receipt can name the machine
/// by what it is (`rtx4080`, `orin-nano`) instead of what it is called.
pub fn machine() -> Result<Machine, String> {
    let hostname = match std::env::var("TURBO_BENCH_MACHINE") {
        Ok(name) if !name.trim().is_empty() => name.trim().to_string(),
        _ => run_cmd("uname", &["-n"])?,
    };
    Ok(Machine { hostname, os: run_cmd("uname", &["-sr"])?, arch: std::env::consts::ARCH.to_string() })
}

/// The commit a receipt names: the one `named` gives, else the build's own
/// (captured by build.rs), else an error. A receipt without a commit is
/// not a measurement of anything, and one from a tree with uncommitted
/// changes (`<sha>-dirty`) cannot be reproduced, so that is an error too
/// unless `TURBO_BENCH_ALLOW_DIRTY=1` is set for a receipt that will not
/// be committed; `compare` never calls a comparison with a dirty side
/// SUPPORTED.
pub fn commit(named: Option<&str>) -> Result<String, String> {
    let commit = match named {
        Some(named) => named.to_string(),
        None => match option_env!("TURBO_BENCH_GIT_COMMIT") {
            Some(built) => built.to_string(),
            None => {
                return Err(
                    "the binary was built from a tree without git, so the receipt cannot name a commit; pass --commit <sha>"
                        .to_string(),
                )
            }
        },
    };
    if commit.ends_with("-dirty") && std::env::var_os("TURBO_BENCH_ALLOW_DIRTY").is_none() {
        return Err(format!(
            "the binary was built from {commit}: a tree with uncommitted changes, which nobody can reproduce; commit first and rebuild, or set TURBO_BENCH_ALLOW_DIRTY=1 for a receipt that will not be committed"
        ));
    }
    Ok(commit)
}

/// Whether a receipt's commit names a tree with uncommitted changes.
pub fn is_dirty(commit: &str) -> bool {
    commit.ends_with("-dirty")
}

/// Today's UTC date, `YYYY-MM-DD`.
pub fn today() -> Result<String, String> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| format!("the clock is before the Unix epoch: {e}"))?
        .as_secs();
    Ok(civil_date(secs))
}

/// The UTC civil date of an epoch second, `YYYY-MM-DD`, without a chrono
/// dependency: days since the epoch through the era/day-of-era form of the
/// proleptic Gregorian calendar.
pub fn civil_date(secs: u64) -> String {
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

/// The least wall time a warm-up phase runs, whatever its iteration count.
pub const WARM_UP_MIN: Duration = Duration::from_millis(500);

/// Untimed warm-up before a cell's timed samples: at least `warmup`
/// iterations and at least [`WARM_UP_MIN`] of wall time. A count alone is
/// not enough for a short cell: five iterations of a 0.4 ms cell take 2 ms,
/// and a GPU that was idle is still raising its clocks when the timed
/// samples start, which shows up as a few percent on the first cells of a
/// run and nowhere else. The native reference programs warm up by the
/// same rule, so both sides of a comparison measure a device in the same
/// state.
pub fn warm_up<E>(warmup: u32, mut run: impl FnMut() -> Result<(), E>) -> Result<(), E> {
    let t0 = Instant::now();
    let mut done = 0u32;
    while done < warmup || t0.elapsed() < WARM_UP_MIN {
        run()?;
        done += 1;
    }
    Ok(())
}

/// Latency figures over the timed samples. `tokens` is `Some` only when a
/// tokenizer counted them; a p99 is reported from 100 samples up.
pub fn summarize(samples: &mut [Duration], rows: u64, tokens: Option<u64>) -> Result<Latency, String> {
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
    if mean.is_nan() || mean <= 0.0 {
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

/// The tokens a workload fed a session, written by `turbo-bench embed
/// --dump-tokens` and read by the native reference programs so both sides
/// run the same ids: one entry per embedding cell.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenDump {
    /// Bundle directory the tokens were encoded for.
    pub bundle: String,
    /// The bundle's identity, as the libturbo receipt records it, so the
    /// native receipt names the same manifest and artifact hashes.
    pub bundle_id: BundleId,
    /// Artifact format to absolute path, for the reference program to load.
    pub artifacts: std::collections::BTreeMap<String, String>,
    /// The model's pooling and normalization from the bundle contract, so
    /// the reference computes the same vector.
    pub pooling: String,
    /// `l2` or `none`.
    pub normalize: String,
    /// The texts and their token rows, per cell.
    pub cells: Vec<TokenCell>,
}

/// The token rows of one embedding cell, padded to `seq`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenCell {
    /// Rows per run.
    pub batch: u32,
    /// Row length (and stride).
    pub seq: u32,
    /// The texts, one per row.
    pub texts: Vec<String>,
    /// Row-major `[batch, seq]` ids, padded; empty when the bundle has no
    /// tokenizer the core can load (the reference tokenizes the texts).
    pub ids: Vec<i32>,
    /// Row-major `[batch, seq]` attention mask; empty with `ids`.
    pub mask: Vec<i32>,
    /// Live tokens per row.
    pub lengths: Vec<u32>,
}
