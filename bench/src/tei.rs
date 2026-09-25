//! text-embeddings-inference (TEI), the end-to-end reference: its HTTP
//! server in a container pinned by digest, the GPU image for a CUDA
//! device and the CPU image for the CPU, sent the same token rows.
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
use crate::docker::{self, Log, Running, argv};
use crate::measure::{Measurement, Rows, cosine, percentile};

pub const NAME: &str = "text-embeddings-inference";

/// TEI's default --max-batch-tokens, kept when the rows need fewer.
const DEFAULT_BATCH_TOKENS: u64 = 16384;

/// How long the server may take to load the model and answer /health.
const START_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Debug, Clone)]
pub struct Tei {
    /// `name@sha256:<64 hex>`.
    pub image: String,
    /// The model in the upstream layout.
    pub model_dir: PathBuf,
}

/// TEI's --dtype for a compute dtype, or why it has none: it offers
/// float32 and float16 in its GPU and CPU images.
pub fn dtype(compute_dtype: u32) -> std::result::Result<&'static str, String> {
    match compute_dtype {
        TURBO_DTYPE_F32 => Ok("float32"),
        TURBO_DTYPE_F16 => Ok("float16"),
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
/// loopback only.
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
    dtype: &str,
    pooling: &str,
    batch: u32,
    seq: u32,
) -> Vec<String> {
    let mut a = argv(&["docker", "run", "--detach", "--rm", "--pull", "never", "--name", container]);
    if let Some(ordinal) = gpu {
        a.extend(argv(&["--gpus", &format!("device={ordinal}")]));
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
/// config.json is there. Returns why not.
pub fn check_model_dir(dir: &Path, m: &Measurement) -> std::result::Result<(), String> {
    let hash =
        |f: &str| fs::read(dir.join(f)).map(|b| sha256_hex(&b)).map_err(|e| format!("{}: {e}", dir.join(f).display()));
    let want_tok = field(&m.model.tokenizer_sha256);
    let want_art = field(&m.model.artifact_sha256);
    let art = m.manifest.artifacts.iter().find(|a| turbo::model::artifact_sha256(&m.manifest, a) == want_art);
    if art.is_none_or(|a| a.files.len() != 1 || a.format != turbo::manifest::Format::Safetensors) {
        return Err("the artifact the device loaded is not one safetensors file TEI can read".into());
    }
    if hash("tokenizer.json")? != want_tok {
        return Err(format!("{}/tokenizer.json is not the bundle's tokenizer ({want_tok})", dir.display()));
    }
    if hash("model.safetensors")? != want_art {
        return Err(format!("{}/model.safetensors is not the loaded artifact ({want_art})", dir.display()));
    }
    fs::metadata(dir.join("config.json")).map_err(|e| format!("{}/config.json: {e}", dir.display()))?;
    Ok(())
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

fn post(url: &str, body: &str) -> Result<String> {
    let mut resp = ureq::post(url)
        .header("content-type", "application/json")
        .send(body)
        .map_err(|e| format!("POST {url}: {e}"))?;
    resp.body_mut().with_config().limit(u64::MAX).read_to_string().map_err(|e| format!("POST {url}: {e}"))
}

fn get(url: &str) -> Result<String> {
    let mut resp = ureq::get(url).call().map_err(|e| format!("GET {url}: {e}"))?;
    resp.body_mut().read_to_string().map_err(|e| format!("GET {url}: {e}"))
}

/// Start the server, check it runs the same rows, time `iterations`
/// /embed requests of the whole batch after `warmup` untimed ones, and
/// stop it. A thing TEI cannot do for this bundle is a record that says
/// so; a failure of docker or the server is an error.
pub fn run(tei: &Tei, m: &Measurement, gpu: Option<u32>, warmup: u32, iterations: u32) -> Result<ReferenceRun> {
    let image = docker::check_pinned("--tei-image", &tei.image)?;
    let procedure = format!(
        "POST /decode then /tokenize to check the rows survive TEI's re-tokenization; then POST /embed with the \
         batch's {} rows as token ids, {warmup} untimed then {iterations} timed, each timed from sending the \
         request to reading the whole response",
        m.rows.batch
    );
    let mut log = Log::default();
    let dtype = match dtype(m.compute_dtype) {
        Ok(d) => d,
        Err(why) => return Ok(not_run(image, log, &procedure, why)),
    };
    let dir = fs::canonicalize(&tei.model_dir).map_err(|e| format!("{}: {e}", tei.model_dir.display()))?;
    if let Err(why) = check_model_dir(&dir, m) {
        return Ok(not_run(image, log, &procedure, why));
    }
    docker::require_image(&mut log, image)?;

    let container = format!("turbo-bench-tei-{}", std::process::id());
    let cmd = run_argv(image, &dir, &container, gpu, dtype, pooling(m.pooling()), m.rows.batch, m.rows.seq);
    log.run(&cmd)?;
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
    let mut ms = Vec::with_capacity(iterations as usize);
    let mut last = String::new();
    let started = Instant::now();
    for _ in 0..iterations {
        let t = Instant::now();
        last = post(&url, &body)?;
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let total = started.elapsed().as_secs_f64();
    let vectors = parse_embed(&last, m.rows.batch as usize, m.model.dim as usize)?;
    let min_cosine = vectors
        .iter()
        .zip(&m.rows.cases)
        .map(|(v, &c)| cosine(v, &m.reference.vectors[c as usize]))
        .fold(1.0, f64::min);
    ms.sort_by(f64::total_cmp);
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
        }),
        not_run: None,
    })
}
