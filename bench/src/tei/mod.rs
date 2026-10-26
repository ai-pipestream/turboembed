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

pub mod kernels;
mod metrics;
mod runner;

use std::fs;
use std::path::{Path, PathBuf};

pub use metrics::*;
pub use runner::run;
use serde_json::{Value, json};
use turbo::bundle::sha256_hex;
use turbo::manifest::Pooling;
use turbo::{TURBO_DTYPE_F16, TURBO_DTYPE_F32};

use crate::Result;
use crate::api::field;
use crate::cpus::Cpus;
use crate::docker::{self, argv};
use crate::measure::{Measurement, Rows};

pub const NAME: &str = "text-embeddings-inference";

/// TEI's default --max-batch-tokens, kept when the rows need fewer.
const DEFAULT_BATCH_TOKENS: u64 = 16384;

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
    /// Nsight Systems for its kernel time on a GPU (`--tei-nsys`).
    pub nsys: Option<kernels::Nsys>,
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
    let inputs: Vec<&[i32]> = (0..rows.row_count()).map(|r| rows.live(r)).collect();
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
    (0..rows.row_count()).all(|r| rows.live(r).len() == first).then_some(rows.row_count() as u64 * first as u64)
}
