//! text-embeddings-inference (TEI), the end-to-end reference: its HTTP
//! server in a container pinned by digest, the GPU image for a CUDA
//! device and the CPU image for the CPU, sent the same token rows. For a
//! Metal device, which no container reaches, it is TEI's router built
//! with its Metal feature and run natively, pinned by the binary's
//! SHA-256.
//!
//! TEI serves a model directory in the upstream layout (config.json,
//! tokenizer.json, model.safetensors), not a bundle: the directory
//! turbo-bundle fetched the model into. Its tokenizer and weights must be
//! the bundle's, byte for byte.
//!
//! TEI takes rows as token ids, but runs them through its tokenizer again:
//! it decodes the ids to text and encodes that text without special
//! tokens. Before timing, the tool asks TEI to do exactly that (/decode,
//! then /tokenize) and refuses to measure unless every row comes back as
//! the same ids.
//!
//! Each /embed answer carries TEI's own account of the request in its
//! headers (TIMING_HEADERS). The tool keeps the round trip as the
//! measurement and gives their percentiles in the procedure beside it.
//! None of them is the request's compute. x-total-time runs from the
//! parsed request to the response headers; x-inference-time is, for each
//! input, the time of the backend batch it ran in, and the header is their
//! mean. TEI's router queues a request's inputs one at a time and its
//! batcher takes whatever has arrived, so one request can run as several
//! batches in turn (as of v1.8.3: router/src/http/server.rs, the embed
//! handler; core/src/infer.rs; core/src/queue.rs). How many it ran as is
//! read from TEI's Prometheus /metrics before and after the timed
//! requests: te_batch_next_size, less the empty polls te_batch_next_tokens
//! shows.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use turbo::bundle::sha256_hex;
use turbo::manifest::Pooling;
use turbo::record::{Measured, ReferenceRun};
use turbo::{TURBO_DTYPE_F16, TURBO_DTYPE_F32};

use crate::Result;
use crate::api::field;
use crate::cpus::{self, Cpus};
use crate::docker::{self, Log, Running, argv};
use crate::measure::{Measurement, Rows, cosine, percentile};

pub const NAME: &str = "text-embeddings-inference";

/// TEI's default --max-batch-tokens, kept when the rows need fewer.
const DEFAULT_BATCH_TOKENS: u64 = 16384;

/// The headers TEI 1.8.3 answers /embed with that time the request, as
/// router/src/lib.rs names them (`impl From<ResponseMetadata> for
/// HeaderMap`), each whole milliseconds. x-total-time runs from the
/// handler's start, the request body already parsed, to the response
/// headers being built, so it leaves out HTTP, parsing the body and
/// writing the JSON. The others are, for a request of several inputs,
/// the mean over its inputs (router/src/http/server.rs, embed) of: the
/// time from the input's own start to its entering the queue, waiting
/// behind the request's other inputs included (x-tokenization-time); its
/// time in the queue (x-queue-time); and the duration of the backend
/// batch it ran in (x-inference-time), core/src/infer.rs.
pub const TIMING_HEADERS: [&str; 4] = ["x-total-time", "x-tokenization-time", "x-queue-time", "x-inference-time"];

/// The histograms TEI's batcher records each time it polls its queue
/// (core/src/queue.rs, as of v1.8.3): the inputs and the tokens of the
/// batch it takes, both 0 when the queue was empty and it takes none. So
/// each drain of the queue ends with an empty poll or two. Their buckets
/// are powers of two, from 1 to 4096 inputs and from 1 to 2^20 tokens
/// (router/src/prometheus.rs).
pub const BATCH_SIZE_METRIC: &str = "te_batch_next_size";
pub const BATCH_TOKENS_METRIC: &str = "te_batch_next_tokens";

/// How long the server may take to load the model and answer /health.
const START_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Debug, Clone)]
pub struct Tei {
    /// `name@sha256:<64 hex>`.
    pub image: String,
    /// The model in the upstream layout.
    pub model_dir: PathBuf,
    /// The processors the container gets, with the thread count to
    /// match (`--cpus`); None: docker's and the image's defaults.
    pub cpus: Option<Cpus>,
    /// TEI's router built natively (`--tei-bin`), run in place of an
    /// image; `image` is then empty.
    pub binary: Option<PathBuf>,
}

/// The native router, as a recorded command names it.
pub const TEI_BIN: &str = "<tei-bin>";

/// TEI's --dtype for a compute dtype, or why it has none. `gpu` is
/// whether TEI runs on a GPU: its CPU image takes float16 but computes it
/// in software, many times slower than its own float32, so an F16 row
/// there would measure the emulation rather than a reference. The native
/// router built with Metal runs on the GPU.
pub fn dtype(compute_dtype: u32, gpu: bool) -> std::result::Result<&'static str, String> {
    match compute_dtype {
        TURBO_DTYPE_F32 => Ok("float32"),
        TURBO_DTYPE_F16 if gpu => Ok("float16"),
        TURBO_DTYPE_F16 => Err("TEI on the CPU emulates float16 in software; no F16 reference runs there".into()),
        d => Err(format!("TEI has no --dtype for compute dtype {d}")),
    }
}

/// TEI's --pooling for the bundle's pooling.
pub fn pooling(p: Pooling) -> &'static str {
    match p {
        Pooling::Mean => "mean",
        Pooling::Cls => "cls",
        Pooling::Last => "last-token",
    }
}

/// The `docker run` that starts the server: detached, removed on stop,
/// the model mounted read-only, no pulls, its port published on the
/// loopback only; with `cpus`, confined to those processors with every
/// thread count the image reads set to their number.
///
/// It gives no `--auto-truncate`: through 1.8 that is a bare flag (a
/// value after it is a stray argument and the router exits), and from
/// 1.9 it takes a value. Either way it is only the default for a request
/// that says nothing, and every /embed request here says `truncate: false`.
#[allow(clippy::too_many_arguments)]
pub fn run_argv(
    image: &str,
    model_dir: &Path,
    container: &str,
    gpu: Option<u32>,
    cpus: Option<&Cpus>,
    dtype: &str,
    pooling: &str,
    batch: u32,
    seq: u32,
) -> Vec<String> {
    let mut a = argv(&["docker", "run", "--detach", "--rm", "--pull", "never", "--name", container]);
    if let Some(ordinal) = gpu {
        a.extend(argv(&["--gpus", &format!("device={ordinal}")]));
    }
    if let Some(c) = cpus {
        a.extend(c.docker_args());
    }
    a.extend(argv(&[
        "--publish",
        "127.0.0.1::80",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--mount",
        &format!("type=bind,src={},dst=/model,readonly", model_dir.display()),
        image,
        "--model-id",
        "/model",
        "--port",
        "80",
        "--dtype",
        dtype,
        "--pooling",
        pooling,
        "--max-client-batch-size",
        &batch.to_string(),
        "--max-batch-tokens",
        &(batch as u64 * seq as u64).max(DEFAULT_BATCH_TOKENS).to_string(),
    ]));
    a
}

/// The native router's argv: the loopback at `port`, the model and the
/// options run_argv gives the image.
pub fn native_argv(
    bin: &str,
    model_dir: &str,
    port: u16,
    dtype: &str,
    pooling: &str,
    batch: u32,
    seq: u32,
) -> Vec<String> {
    argv(&[
        bin,
        "--model-id",
        model_dir,
        "--hostname",
        "127.0.0.1",
        "--port",
        &port.to_string(),
        "--dtype",
        dtype,
        "--pooling",
        pooling,
        "--max-client-batch-size",
        &batch.to_string(),
        "--max-batch-tokens",
        &(batch as u64 * seq as u64).max(DEFAULT_BATCH_TOKENS).to_string(),
    ])
}

/// The native router as a record pins it: its file's SHA-256.
pub fn native_pin(bin: &Path) -> Result<String> {
    let bytes = fs::read(bin).map_err(|e| format!("--tei-bin {}: {e}", bin.display()))?;
    Ok(format!("text-embeddings-router@sha256:{}", sha256_hex(&bytes)))
}

/// A native router started by the tool, stopped when this goes.
struct Native(std::process::Child);

impl Drop for Native {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// An image's environment as `docker image inspect --format '{{json
/// .Config.Env}}'` prints it: a JSON array of `KEY=value`, or null.
pub fn parse_env(out: &str) -> Result<Vec<String>> {
    let v: Option<Vec<String>> =
        serde_json::from_str(out.trim()).map_err(|e| format!("the image's environment {out:?}: {e}"))?;
    Ok(v.unwrap_or_default())
}

/// What /info says of the server: its version, and the dtype it loaded.
#[derive(Debug, Clone, PartialEq)]
pub struct Info {
    pub version: String,
    pub model_dtype: String,
}

pub fn parse_info(body: &str) -> Result<Info> {
    let v: Value = serde_json::from_str(body).map_err(|e| format!("TEI /info: {e}"))?;
    let s = |k: &str| v[k].as_str().map(str::to_owned).ok_or(format!("TEI /info: no {k}"));
    Ok(Info { version: s("version")?, model_dtype: s("model_dtype")? })
}

/// The /embed request for the rows: each row's live ids, the bundle's
/// normalization, and no truncation.
pub fn embed_body(rows: &Rows, normalize: bool) -> String {
    let inputs: Vec<&[i32]> = (0..rows.batch as usize).map(|r| rows.live(r)).collect();
    json!({ "inputs": inputs, "normalize": normalize, "truncate": false }).to_string()
}

/// /embed's answer: one vector per row, each `dim` wide.
pub fn parse_embed(body: &str, rows: usize, dim: usize) -> Result<Vec<Vec<f32>>> {
    let v: Vec<Vec<f32>> = serde_json::from_str(body).map_err(|e| format!("TEI /embed: {e}"))?;
    if v.len() != rows || v.iter().any(|r| r.len() != dim) {
        return Err(format!("TEI /embed: {} vectors, not {rows} of {dim}", v.len()));
    }
    Ok(v)
}

/// /tokenize's answer: each input's token ids.
pub fn parse_tokenize(body: &str) -> Result<Vec<Vec<i32>>> {
    let v: Vec<Vec<Value>> = serde_json::from_str(body).map_err(|e| format!("TEI /tokenize: {e}"))?;
    v.iter()
        .map(|row| {
            row.iter()
                .map(|t| t["id"].as_i64().map(|i| i as i32).ok_or_else(|| "TEI /tokenize: a token has no id".into()))
                .collect()
        })
        .collect()
}

/// The first row whose ids TEI's round trip changes, with both.
pub fn first_changed(sent: &[Vec<i32>], back: &[Vec<i32>]) -> Option<(usize, Vec<i32>, Vec<i32>)> {
    if sent.len() != back.len() {
        return Some((sent.len().min(back.len()), Vec::new(), Vec::new()));
    }
    sent.iter().zip(back).enumerate().find(|(_, (a, b))| a != b).map(|(i, (a, b))| (i, a.clone(), b.clone()))
}

/// The files TEI will read are the bundle's: tokenizer.json has the
/// bundle tokenizer's hash, model.safetensors the loaded artifact's, and
/// config.json is there. Returns why not, which a record keeps, so the
/// directory is named by its placeholder.
pub fn check_model_dir(dir: &Path, m: &Measurement) -> std::result::Result<(), String> {
    let hash =
        |f: &str| fs::read(dir.join(f)).map(|b| sha256_hex(&b)).map_err(|e| format!("{}/{f}: {e}", docker::TEI_MODEL));
    let want_tok = field(&m.model.tokenizer_sha256);
    let want_art = field(&m.model.artifact_sha256);
    let art = m.manifest.artifacts.iter().find(|a| turbo::model::artifact_sha256(&m.manifest, a) == want_art);
    if art.is_none_or(|a| a.files.len() != 1 || a.format != turbo::manifest::Format::Safetensors) {
        return Err("the artifact the device loaded is not one safetensors file TEI can read".into());
    }
    if hash("tokenizer.json")? != want_tok {
        return Err(format!("{}/tokenizer.json is not the bundle's tokenizer ({want_tok})", docker::TEI_MODEL));
    }
    if hash("model.safetensors")? != want_art {
        return Err(format!("{}/model.safetensors is not the loaded artifact ({want_art})", docker::TEI_MODEL));
    }
    fs::metadata(dir.join("config.json")).map_err(|e| format!("{}/config.json: {e}", docker::TEI_MODEL))?;
    Ok(())
}

/// The token positions TEI computes for the rows, when that can be known
/// from outside it. As of v1.8.3 it pads each batch it forms to the
/// batch's longest input, or packs the inputs where it runs flash
/// attention (core/src/queue.rs, and each backend's `is_padded`). With
/// every row the same length L that is batch x L either way. Otherwise it
/// depends on how it split the request (at most 8 inputs a batch with
/// ONNX Runtime and 4 with candle on a CPU, each backend's
/// `max_batch_size`) and on the order the inputs reached its queue, which
/// is not fixed: unknown.
pub fn computed_tokens(rows: &Rows) -> Option<u64> {
    let first = rows.live(0).len();
    (0..rows.batch as usize).all(|r| rows.live(r).len() == first).then_some(rows.batch as u64 * first as u64)
}

/// What TEI's vectors are compared with, in a clause.
fn compared(m: &Measurement) -> &'static str {
    if (0..m.rows.batch as usize).all(|r| m.rows.whole(r, &m.reference)) {
        "min_cosine is against the bundle's reference vectors"
    } else {
        "min_cosine is against the bundle's reference vector for a row that is its whole case, and the library's \
         vector of the row alone for a row cut to seq"
    }
}

fn not_run(image: &str, log: Log, procedure: &str, why: String) -> ReferenceRun {
    ReferenceRun {
        name: NAME.into(),
        role: "end_to_end".into(),
        pinned: image.into(),
        version: String::new(),
        commands: log.commands,
        procedure: procedure.into(),
        measured: None,
        not_run: Some(why),
    }
}

/// One answer's TIMING_HEADERS, in their order, in milliseconds; None
/// where the header is missing or is not a whole number.
pub fn timing_headers(headers: &ureq::http::HeaderMap) -> [Option<u64>; 4] {
    TIMING_HEADERS.map(|h| headers.get(h).and_then(|v| v.to_str().ok()).and_then(|v| v.trim().parse().ok()))
}

/// The timed requests' round trip and TEI's headers, as the procedure
/// gives them: each header's p50 and p99 over the requests, or that TEI
/// did not send it with every answer.
pub fn timing_text(round_trip_sorted_ms: &[f64], headers: &[[Option<u64>; 4]]) -> String {
    let mut out = format!(
        "round trip (measured) p50 {:.3} ms, p99 {:.3} ms; TEI's own headers, whole ms, over the same requests:",
        percentile(round_trip_sorted_ms, 50.0),
        percentile(round_trip_sorted_ms, 99.0)
    );
    for (i, name) in TIMING_HEADERS.iter().enumerate() {
        let got: Option<Vec<f64>> = headers.iter().map(|h| h[i].map(|v| v as f64)).collect();
        let sep = if i == 0 { " " } else { ", " };
        match got {
            Some(mut v) if !v.is_empty() => {
                v.sort_by(f64::total_cmp);
                out += &format!("{sep}{name} p50 {} p99 {}", percentile(&v, 50.0), percentile(&v, 99.0));
            }
            _ => out += &format!("{sep}{name} not sent"),
        }
    }
    out
}

/// The p50 of x-total-time, in milliseconds, as timing_text wrote it
/// into a procedure; None when it is not there.
pub fn total_time_p50(procedure: &str) -> Option<f64> {
    let key = format!("{} p50 ", TIMING_HEADERS[0]);
    let rest = &procedure[procedure.find(&key)? + key.len()..];
    rest.split(' ').next()?.parse().ok()
}

/// A Prometheus histogram's samples at one reading, from the text
/// exposition format: `_count`, `_sum`, and each `_bucket`'s upper bound
/// (`le`, +Inf as infinity) with its cumulative count, in the order
/// given. A summary has no buckets.
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    pub count: f64,
    pub sum: f64,
    pub buckets: Vec<(f64, f64)>,
}

/// One sample line: its metric name, its labels (the text between the
/// braces, empty when there are none), and its value.
fn sample(line: &str) -> Option<(&str, &str, f64)> {
    let line = line.trim();
    let name_end = line.find(|c: char| c == '{' || c.is_whitespace())?;
    let (name, rest) = line.split_at(name_end);
    let (labels, rest) = match rest.strip_prefix('{') {
        Some(r) => r.split_once('}')?,
        None => ("", rest),
    };
    Some((name, labels, prom_float(rest.split_whitespace().next()?)?))
}

fn prom_float(v: &str) -> Option<f64> {
    match v {
        "+Inf" => Some(f64::INFINITY),
        "-Inf" => Some(f64::NEG_INFINITY),
        v => v.parse().ok(),
    }
}

/// The histogram `name` in a /metrics body; None when it has no `_count`
/// and `_sum` (TEI has not recorded it, or the body is something else).
/// Series of `name` with other labels than `le` are added together.
pub fn parse_histogram(text: &str, name: &str) -> Option<Histogram> {
    let (mut count, mut sum, mut buckets) = (None, None, Vec::<(f64, f64)>::new());
    for line in text.lines().filter(|l| !l.trim_start().starts_with('#')) {
        let Some((series, labels, value)) = sample(line) else { continue };
        let Some(suffix) = series.strip_prefix(name) else { continue };
        match suffix {
            "_count" => *count.get_or_insert(0.0) += value,
            "_sum" => *sum.get_or_insert(0.0) += value,
            "_bucket" => {
                let Some(le) = labels
                    .split(',')
                    .find_map(|l| l.trim().strip_prefix("le=\""))
                    .and_then(|v| v.strip_suffix('"'))
                    .and_then(prom_float)
                else {
                    continue;
                };
                match buckets.iter_mut().find(|(b, _)| *b == le) {
                    Some((_, c)) => *c += value,
                    None => buckets.push((le, value)),
                }
            }
            _ => {}
        }
    }
    buckets.sort_by(|a, b| a.0.total_cmp(&b.0));
    Some(Histogram { count: count?, sum: sum?, buckets })
}

impl Histogram {
    /// What was recorded between `before` and this reading; with no
    /// `before`, everything this one holds.
    pub fn since(&self, before: Option<&Histogram>) -> Histogram {
        let Some(b) = before else { return self.clone() };
        let earlier = |le: f64| b.buckets.iter().find(|(x, _)| *x == le).map_or(0.0, |(_, c)| *c);
        Histogram {
            count: self.count - b.count,
            sum: self.sum - b.sum,
            buckets: self.buckets.iter().map(|&(le, c)| (le, c - earlier(le))).collect(),
        }
    }
}

/// TEI's batch histograms at one reading of /metrics: None when it has
/// no BATCH_SIZE_METRIC; `tokens` None when it has no
/// BATCH_TOKENS_METRIC.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchMetrics {
    pub size: Histogram,
    pub tokens: Option<Histogram>,
}

pub fn parse_batch_metrics(text: &str) -> Option<BatchMetrics> {
    Some(BatchMetrics {
        size: parse_histogram(text, BATCH_SIZE_METRIC)?,
        tokens: parse_histogram(text, BATCH_TOKENS_METRIC),
    })
}

/// A histogram's cumulative count at the bucket `le`, if it has that one.
fn at(h: &Histogram, le: f64) -> Option<f64> {
    h.buckets.iter().find(|(b, _)| *b == le).map(|(_, c)| *c)
}

/// How TEI split the timed requests into backend batches, for the
/// procedure, from /metrics read before (`before`) and after (`after`)
/// them. The batcher's empty polls are in te_batch_next_size as batches
/// of 0 inputs; te_batch_next_tokens has them as 0 tokens, and, when
/// every row has at least 2 tokens (`min_row_tokens`), no batch of inputs
/// has fewer, so its le="1" bucket counts exactly the empty polls. The
/// figures: the inputs, the batches without the empty polls, batches per
/// request, their mean size, and how many fell in each size bucket. What
/// cannot be told is said to be unknown: the batches without the tokens
/// histogram or with a row of 1 token, and everything when /metrics was
/// not read before the timed requests though warmup requests ran.
pub fn batches_text(
    before: Option<&BatchMetrics>,
    after: Option<&BatchMetrics>,
    requests: u32,
    warmup: u32,
    min_row_tokens: usize,
) -> String {
    let Some(after) = after else {
        return format!(
            "TEI's /metrics gave no {BATCH_SIZE_METRIC} after the timed requests, so how many backend batches they \
             ran as is not known"
        );
    };
    if before.is_none() && warmup > 0 {
        return format!(
            "TEI's /metrics gave no {BATCH_SIZE_METRIC} before the timed requests, after {warmup} warmup requests, so \
             how many backend batches the timed requests ran as is not known"
        );
    }
    let size = after.size.since(before.map(|b| &b.size));
    let tokens = match (&after.tokens, before) {
        (Some(t), None) => Some(t.clone()),
        (Some(t), Some(b)) => b.tokens.as_ref().map(|bt| t.since(Some(bt))),
        (None, _) => None,
    };
    let inputs = size.sum;
    let mut out = format!("TEI's /metrics over the timed requests: {inputs} inputs in {requests} requests");
    let empty = match tokens.as_ref().and_then(|t| at(t, 1.0)) {
        None => {
            return out
                + &format!(
                    "; how many backend batches they ran as is not known: {BATCH_SIZE_METRIC} also counts the \
                     batcher's empty polls, and {BATCH_TOKENS_METRIC}, which tells them apart, was not given"
                );
        }
        Some(_) if min_row_tokens < 2 => {
            return out
                + &format!(
                    "; how many backend batches they ran as is not known: {BATCH_SIZE_METRIC} also counts the \
                     batcher's empty polls, and with a row of {min_row_tokens} token {BATCH_TOKENS_METRIC} does not \
                     tell them apart"
                );
        }
        Some(z) => z,
    };
    let batches = size.count - empty;
    if batches <= 0.0 || requests == 0 {
        return out + &format!("; {BATCH_SIZE_METRIC} recorded no backend batch");
    }
    out += &format!(
        ", {batches} backend batches ({BATCH_SIZE_METRIC}'s {} samples less the {empty} empty polls \
         {BATCH_TOKENS_METRIC} shows), {:.2} batches per request, mean batch size {:.2} inputs",
        size.count,
        batches / requests as f64,
        inputs / batches
    );
    let mut parts = Vec::new();
    let (mut prev_le, mut prev_c) = (0.0_f64, 0.0);
    for &(le, c) in &size.buckets {
        // Every bucket is cumulative, so each holds the empty polls.
        let c = c - empty;
        let n = c - prev_c;
        if n > 0.0 {
            let low = prev_le.floor() + 1.0;
            parts.push(if le.is_infinite() {
                format!("above {prev_le}: {n}")
            } else if low >= le {
                format!("{le}: {n}")
            } else {
                format!("{low} to {le}: {n}")
            });
        }
        (prev_le, prev_c) = (le, c);
    }
    if !parts.is_empty() {
        out += &format!(" (batches by size, {})", parts.join(", "));
    }
    out
}

fn post(url: &str, body: &str) -> Result<String> {
    post_timed(url, body).map(|(text, _)| text)
}

/// The answer's body and its TIMING_HEADERS.
fn post_timed(url: &str, body: &str) -> Result<(String, [Option<u64>; 4])> {
    let mut resp = ureq::post(url)
        .header("content-type", "application/json")
        .send(body)
        .map_err(|e| format!("POST {url}: {e}"))?;
    let timing = timing_headers(resp.headers());
    let text =
        resp.body_mut().with_config().limit(u64::MAX).read_to_string().map_err(|e| format!("POST {url}: {e}"))?;
    Ok((text, timing))
}

fn get(url: &str) -> Result<String> {
    let mut resp = ureq::get(url).call().map_err(|e| format!("GET {url}: {e}"))?;
    resp.body_mut().read_to_string().map_err(|e| format!("GET {url}: {e}"))
}

/// Start the server, check it runs the same rows, time `iterations`
/// /embed requests of the whole batch after `warmup` untimed ones, and
/// stop it. A thing TEI cannot do for this bundle is a record that says
/// so; a failure of docker or the server is an error.
///
/// `library` is the library's thread count when the device measured is
/// the CPU, for the procedure, which says what each side ran on.
pub fn run(
    tei: &Tei,
    m: &Measurement,
    gpu: Option<u32>,
    library: Option<usize>,
    warmup: u32,
    iterations: u32,
) -> Result<ReferenceRun> {
    // What names the program: the image's digest, or the native
    // router's SHA-256.
    let pinned = match &tei.binary {
        Some(bin) => native_pin(bin)?,
        None => docker::check_pinned("--tei-image", &tei.image)?.to_owned(),
    };
    let image = pinned.as_str();
    let procedure = format!(
        "POST /decode then /tokenize to check the rows survive TEI's re-tokenization; then POST /embed with the \
         batch's {} rows as token ids, {warmup} untimed then {iterations} timed, each timed from sending the \
         request to reading the whole response, the p50 and p99 of TEI's {} headers beside it, and TEI's /metrics read \
         before and after the timed requests for its {BATCH_SIZE_METRIC} and {BATCH_TOKENS_METRIC} histograms; {}",
        m.rows.batch,
        TIMING_HEADERS.join(", "),
        compared(m)
    );
    let mut log = Log::default();
    let dtype = match dtype(m.compute_dtype, gpu.is_some() || tei.binary.is_some()) {
        Ok(d) => d,
        Err(why) => return Ok(not_run(image, log, &procedure, why)),
    };
    let dir = fs::canonicalize(&tei.model_dir).map_err(|e| format!("{}: {e}", tei.model_dir.display()))?;
    if let Err(why) = check_model_dir(&dir, m) {
        return Ok(not_run(image, log, &procedure, why));
    }
    if let Some(bin) = &tei.binary {
        return run_native(tei, bin, image, &dir, m, dtype, procedure, library, warmup, iterations, log);
    }
    docker::require_image(&mut log, image)?;
    let env =
        parse_env(&log.run(&argv(&["docker", "image", "inspect", "--format", "{{json .Config.Env}}", image]))?)?;
    let (what, threads) = (procedure, cpus::procedure(tei.cpus.as_ref(), library, &env));

    let container = format!("turbo-bench-tei-{}", std::process::id());
    let start_argv = |dir: &Path| {
        run_argv(image, dir, &container, gpu, tei.cpus.as_ref(), dtype, pooling(m.pooling()), m.rows.batch, m.rows.seq)
    };
    log.run_as(&start_argv(&dir), start_argv(Path::new(docker::TEI_MODEL)))?;
    let _running = Running { name: container.clone() };
    let port = docker::parse_port(&log.run(&argv(&["docker", "port", &container, "80/tcp"]))?)?;
    let base = format!("http://127.0.0.1:{port}");

    let start = Instant::now();
    while get(&format!("{base}/health")).is_err() {
        let state = log.run(&argv(&["docker", "inspect", "--format", "{{.State.Running}}", &container]));
        if state.as_deref().map(str::trim) != Ok("true") || start.elapsed() > START_TIMEOUT {
            let logs = log.run(&argv(&["docker", "logs", &container])).unwrap_or_default();
            return Err(format!("TEI did not become healthy within {START_TIMEOUT:?}:\n{logs}"));
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    against(&base, image, log, what, threads, m, warmup, iterations)
}

/// TEI's router run natively on the model directory, as run() runs the
/// image: `what` is the procedure so far.
#[allow(clippy::too_many_arguments)]
fn run_native(
    tei: &Tei,
    bin: &Path,
    pinned: &str,
    dir: &Path,
    m: &Measurement,
    dtype: &str,
    what: String,
    library: Option<usize>,
    warmup: u32,
    iterations: u32,
    log: Log,
) -> Result<ReferenceRun> {
    let bin = fs::canonicalize(bin).map_err(|e| format!("--tei-bin {}: {e}", bin.display()))?;
    let threads = cpus::procedure(tei.cpus.as_ref(), library, &[]);
    let what = format!("{what}; TEI's router built natively and run on this machine, not in a container");
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map_err(|e| format!("a loopback port for TEI: {e}"))?
        .port();
    let (pool, batch, seq) = (pooling(m.pooling()), m.rows.batch, m.rows.seq);
    let run = native_argv(&bin.to_string_lossy(), &dir.to_string_lossy(), port, dtype, pool, batch, seq);
    let mut log = log;
    log.commands.push(native_argv(TEI_BIN, docker::TEI_MODEL, port, dtype, pool, batch, seq));
    let out = std::env::temp_dir().join(format!("turbo-bench-tei-{}.log", std::process::id()));
    let file = fs::File::create(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let err = file.try_clone().map_err(|e| format!("{}: {e}", out.display()))?;
    let child = std::process::Command::new(&run[0])
        .args(&run[1..])
        .env("HF_HUB_OFFLINE", "1")
        .stdout(file)
        .stderr(err)
        .spawn()
        .map_err(|e| format!("{}: {e}", run.join(" ")))?;
    let mut server = Native(child);
    let base = format!("http://127.0.0.1:{port}");
    let start = Instant::now();
    while get(&format!("{base}/health")).is_err() {
        let exited = server.0.try_wait().map_err(|e| format!("TEI: {e}"))?;
        if exited.is_some() || start.elapsed() > START_TIMEOUT {
            let logs = fs::read_to_string(&out).unwrap_or_default();
            return Err(format!("TEI did not become healthy within {START_TIMEOUT:?}:\n{logs}"));
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    let r = against(&base, pinned, log, what, threads, m, warmup, iterations);
    drop(server);
    let _ = fs::remove_file(&out);
    r
}

/// Check, warm up and time the server at `base`, as the procedure says.
#[allow(clippy::too_many_arguments)]
fn against(
    base: &str,
    image: &str,
    log: Log,
    what: String,
    threads: String,
    m: &Measurement,
    warmup: u32,
    iterations: u32,
) -> Result<ReferenceRun> {
    let procedure = format!("{what}; {threads}");
    let info = parse_info(&get(&format!("{base}/info"))?)?;

    // The rows as TEI will see them.
    let distinct: Vec<usize> = {
        let mut seen = std::collections::BTreeSet::new();
        (0..m.rows.batch as usize).filter(|&r| seen.insert(m.rows.cases[r])).collect()
    };
    let sent: Vec<Vec<i32>> = distinct.iter().map(|&r| m.rows.live(r).to_vec()).collect();
    let texts: Vec<String> = serde_json::from_str(&post(
        &format!("{base}/decode"),
        &json!({ "ids": sent, "skip_special_tokens": false }).to_string(),
    )?)
    .map_err(|e| format!("TEI /decode: {e}"))?;
    let back = parse_tokenize(&post(
        &format!("{base}/tokenize"),
        &json!({ "inputs": texts, "add_special_tokens": false }).to_string(),
    )?)?;
    if let Some((i, a, b)) = first_changed(&sent, &back) {
        let mut run = not_run(
            image,
            log,
            &procedure,
            format!(
                "TEI re-tokenizes token ids, and case {} comes back as other ids ({a:?} became {b:?}): \
                 it would not run the same rows",
                m.rows.cases[distinct[i]]
            ),
        );
        run.version = info.version;
        return Ok(run);
    }

    let body = embed_body(&m.rows, m.normalize());
    let url = format!("{base}/embed");
    for _ in 0..warmup {
        post(&url, &body)?;
    }
    let metrics = || get(&format!("{base}/metrics")).ok().and_then(|t| parse_batch_metrics(&t));
    let before = metrics();
    let mut ms = Vec::with_capacity(iterations as usize);
    let mut own = Vec::with_capacity(iterations as usize);
    let mut last = String::new();
    let started = Instant::now();
    for _ in 0..iterations {
        let t = Instant::now();
        let (text, timing) = post_timed(&url, &body)?;
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        last = text;
        own.push(timing);
    }
    let total = started.elapsed().as_secs_f64();
    let min_row_tokens = (0..m.rows.batch as usize).map(|r| m.rows.live(r).len()).min().unwrap_or(0);
    let batches = batches_text(before.as_ref(), metrics().as_ref(), iterations, warmup, min_row_tokens);
    let vectors = parse_embed(&last, m.rows.batch as usize, m.model.dim as usize)?;
    let min_cosine = vectors.iter().zip(&m.expected).map(|(v, want)| cosine(v, want)).fold(1.0, f64::min);
    ms.sort_by(f64::total_cmp);
    let procedure = format!("{what}; {}; {batches}; {threads}", timing_text(&ms, &own));
    Ok(ReferenceRun {
        name: NAME.into(),
        role: "end_to_end".into(),
        pinned: image.into(),
        version: format!("{} ({})", info.version, info.model_dtype),
        commands: log.commands,
        procedure,
        measured: Some(Measured {
            iterations: iterations as u64,
            p50_ms: percentile(&ms, 50.0),
            p99_ms: percentile(&ms, 99.0),
            rows_per_second: m.rows.batch as f64 * iterations as f64 / total,
            min_cosine: Some(min_cosine),
            computed_tokens: computed_tokens(&m.rows),
        }),
        not_run: None,
    })
}
