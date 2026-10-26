//! MLX, Apple's array framework, as the metal backend's kernel reference:
//! the same BERT encoder written on MLX (bench/reference/mlx_reference.py,
//! after bench/metal-peers/mlx_bench.py), run by the machine's Python on
//! the token rows the library measured, its vectors held against the
//! bundle's reference. It computes the padded batch, every position.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use turbo::bundle::sha256_hex;
use turbo::record::{Measured, ReferenceRun};

use crate::Result;
use crate::api::field;
use crate::docker::{self, WORK};
use crate::measure::{Measurement, cosine, percentile};

pub const NAME: &str = "mlx";
const SCRIPT: &str = include_str!("../reference/mlx_reference.py");
/// The model directory MLX reads, as a recorded command names it.
pub const MLX_MODEL: &str = "<mlx-model>";
/// The runner script, as a recorded command names it.
pub const MLX_SCRIPT: &str = "<mlx-reference>";

pub struct Mlx {
    /// The upstream model directory: config.json and the loaded
    /// artifact's model.safetensors.
    pub model: PathBuf,
    /// Where the rows and the receipt are written.
    pub work: PathBuf,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    version: String,
    library_sha256: String,
    dtype: String,
    batch: u32,
    seq: u32,
    times_ms: Vec<f64>,
    vectors: Vec<Vec<f32>>,
}

/// MLX's dtype for the session's precision: float16 at FASTEST, whose
/// numerics MLX float16 is the nearest to, else float32.
pub fn dtype(precision: u32) -> &'static str {
    if precision == turbo::TURBO_PRECISION_FASTEST { "float16" } else { "float32" }
}

/// The argv of the runner, with `script`, `model` and `work` as given.
fn argv(script: &str, model: &str, work: &str, precision: u32, warmup: u32, iterations: u32) -> Vec<String> {
    [
        "python3",
        "-I",
        script,
        "--model",
        model,
        "--dtype",
        dtype(precision),
        "--rows",
        &format!("{work}/rows.json"),
        "--warmup",
        &warmup.to_string(),
        "--iterations",
        &iterations.to_string(),
        "--out",
        &format!("{work}/receipt.json"),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// The directory holds config.json and the loaded artifact as
/// model.safetensors, the one file MLX reads the weights from.
fn check_model_dir(dir: &Path, m: &Measurement) -> Result<()> {
    let want = field(&m.model.artifact_sha256);
    let art = m.manifest.artifacts.iter().find(|a| turbo::model::artifact_sha256(&m.manifest, a) == want);
    if art.is_none_or(|a| a.files.len() != 1 || a.format != turbo::manifest::Format::Safetensors) {
        return Err("the artifact the device loaded is not one safetensors file MLX can read".into());
    }
    let got =
        std::fs::read(dir.join("model.safetensors")).map_err(|e| format!("{MLX_MODEL}/model.safetensors: {e}"))?;
    if sha256_hex(&got) != want {
        return Err(format!("{MLX_MODEL}/model.safetensors is not the loaded artifact ({want})"));
    }
    std::fs::metadata(dir.join("config.json")).map_err(|e| format!("{MLX_MODEL}/config.json: {e}"))?;
    Ok(())
}

/// Run MLX on the measurement's rows and read its receipt.
pub fn run(o: &Mlx, m: &Measurement, warmup: u32, iterations: u32) -> Result<ReferenceRun> {
    if m.backend() != "metal" {
        return Err("MLX is a reference for the metal backend only".into());
    }
    if !m.normalize() || m.pooling() != turbo::manifest::Pooling::Mean {
        return Err("MLX's encoder pools by the mean and normalizes; the bundle's contract does not".into());
    }
    check_model_dir(&o.model, m)?;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e| e.to_string())?.as_nanos();
    let root = o.work.join(format!("mlx-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    let root = root.canonicalize().map_err(|e| e.to_string())?;
    let script = root.join("mlx_reference.py");
    std::fs::write(&script, SCRIPT).map_err(|e| e.to_string())?;
    let r = &m.rows;
    let rows = serde_json::json!({ "batch": r.batch, "seq": r.seq, "ids": r.ids, "mask": r.mask, "types": r.types });
    std::fs::write(root.join("rows.json"), rows.to_string()).map_err(|e| e.to_string())?;
    let model = o.model.canonicalize().map_err(|e| format!("--mlx-model: {e}"))?;
    let path = |p: &Path| p.to_string_lossy().into_owned();
    let run = argv(&path(&script), &path(&model), &path(&root), m.precision, warmup, iterations);
    let out = Command::new(&run[0]).args(&run[1..]).output().map_err(|e| format!("MLX runner: {e}"))?;
    if !out.status.success() {
        return Err(format!("MLX runner failed ({}): {}", out.status, String::from_utf8_lossy(&out.stderr)));
    }
    let bytes = std::fs::read(root.join("receipt.json")).map_err(|e| e.to_string())?;
    let recorded = argv(MLX_SCRIPT, MLX_MODEL, WORK, m.precision, warmup, iterations);
    from_receipt(&bytes, m, iterations, recorded)
}

/// Check MLX's receipt against the rows and the precision, and make its
/// reference entry: its p50 and p99 from its own timed runs, and its
/// lowest cosine against each row's expected vector.
pub fn from_receipt(bytes: &[u8], m: &Measurement, iterations: u32, command: Vec<String>) -> Result<ReferenceRun> {
    let e: Receipt = serde_json::from_slice(bytes).map_err(|e| format!("MLX receipt: {e}"))?;
    let rows = &m.rows;
    if e.batch != rows.batch || e.seq != rows.seq || e.vectors.len() != rows.row_count() || rows.row_count() == 0 {
        return Err("MLX receipt differs from the measured rows".into());
    }
    if e.dtype != dtype(m.precision) || e.version.is_empty() {
        return Err(format!("MLX receipt: dtype {:?} where the precision asks for {}", e.dtype, dtype(m.precision)));
    }
    if e.times_ms.len() != iterations as usize
        || iterations == 0
        || e.times_ms.iter().any(|t| !t.is_finite() || *t <= 0.0)
    {
        return Err(format!("MLX receipt: {} timed runs where {iterations} were asked for", e.times_ms.len()));
    }
    let pin = format!("mlx@sha256:{}", e.library_sha256);
    docker::check_pinned("MLX's library", &pin)?;
    let mut ms = e.times_ms;
    ms.sort_by(f64::total_cmp);
    let p50 = percentile(&ms, 50.0);
    let min_cosine = e.vectors.iter().zip(&m.expected).map(|(v, want)| cosine(v, want)).fold(1.0, f64::min);
    let procedure = format!(
        "MLX {} on the machine's Python, the BERT encoder of bench/reference/mlx_reference.py with every weight \
         cast to {}, mx.compile, mean pooling and L2 in F32, on the same token rows as a padded batch; untimed \
         runs, then {} timed from the host, each waited for with mx.eval. min_cosine is against each row's \
         expected vector. Runner SHA-256 {}",
        e.version,
        dtype(m.precision),
        ms.len(),
        sha256_hex(SCRIPT.as_bytes())
    );
    Ok(ReferenceRun {
        name: NAME.into(),
        role: "kernel".into(),
        pinned: pin,
        version: e.version,
        commands: vec![command],
        procedure,
        measured: Some(Measured {
            iterations: ms.len() as u64,
            p50_ms: p50,
            p99_ms: percentile(&ms, 99.0),
            rows_per_second: rows.row_count() as f64 * 1000.0 / p50,
            min_cosine: Some(min_cosine),
            computed_tokens: Some(rows.padded_tokens()),
        }),
        not_run: None,
    })
}
