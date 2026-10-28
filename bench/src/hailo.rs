//! HailoRT's native CLI, on the same verified HEF and mixed token inputs.
//! Its throughput samples are batch-equivalent service times, not request tails.

use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use turbo::record::{Measured, ReferenceRun};

use crate::Result;
use crate::api::{Runtime, field};
use crate::measure::{Measurement, RowKind, loaded_artifact, percentile};

pub const NAME: &str = "hailort";
const SCRIPT: &str = include_str!("../reference/hailort_reference.py");

pub struct HailoRt {
    pub binary: PathBuf,
    pub work: PathBuf,
}

#[derive(Deserialize)]
struct Evidence {
    target: String,
    manifest_sha256: String,
    hef_sha256: String,
    rows_sha256: String,
    batch: u32,
    seq: u32,
    cases: Vec<u32>,
    live_tokens: u64,
    computed_tokens_per_pass: u64,
    cli_version: String,
    cli_sha256: String,
    fps_samples: Vec<f64>,
    seconds_per_process: u32,
    warmup_processes: u32,
    commands: Vec<Vec<String>>,
}

/// A run is restricted to a full 32-frame burst, matching the backend's
/// configured burst. No guessed inputs, random input fallback or other
/// device ordinal is accepted. Every fitting reference case must be present.
pub fn run(options: &HailoRt, m: &Measurement) -> Result<ReferenceRun> {
    if m.backend() != "hailo" || m.rows.batch != 32 || m.rows.kind != RowKind::Mixed {
        return Err("HailoRT reference requires hailo, batch 32, mixed inputs".into());
    }
    let artifact = loaded_artifact(&m.manifest, &field(&m.model.artifact_sha256))?;
    if artifact.fixed_seq != m.rows.seq || artifact.files.len() != 1 {
        return Err("HailoRT reference needs one HEF at its fixed sequence length".into());
    }
    let rt = Runtime::create()?;
    let mut count = 0;
    for i in 0..rt.device_count()? {
        count += usize::from(field(&rt.device_info(i)?.backend) == "hailo");
    }
    if count != 1 {
        return Err("HailoRT reference requires exactly one Hailo device to prevent an ordinal mismatch".into());
    }
    drop(rt);
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e| e.to_string())?.as_nanos();
    let root = options.work.join(format!("hailort-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    let root = root.canonicalize().map_err(|e| e.to_string())?;
    let script = root.join("reference.py");
    let output = root.join("receipt.json");
    std::fs::write(&script, SCRIPT).map_err(|e| e.to_string())?;
    let result = Command::new("python3")
        .arg(&script)
        .arg("--bundle")
        .arg(&m.bundle_dir)
        .arg("--target")
        .arg(field(&m.device.arch))
        .arg("--hailortcli")
        .arg(&options.binary)
        .arg("--work")
        .arg(root.join("inputs"))
        .arg("--out")
        .arg(&output)
        .output()
        .map_err(|e| format!("HailoRT reference launcher: {e}"))?;
    if !result.status.success() {
        return Err(format!(
            "HailoRT reference failed ({}): {}",
            result.status,
            String::from_utf8_lossy(&result.stderr)
        ));
    }
    let bytes = std::fs::read(&output).map_err(|e| e.to_string())?;
    from_evidence(&bytes, m)
}

/// Check the vendor receipt against what the library actually loaded.
/// This also serves the explicit CPU/Hailo evidence example.
pub fn from_evidence(bytes: &[u8], m: &Measurement) -> Result<ReferenceRun> {
    let e: Evidence = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    let artifact = loaded_artifact(&m.manifest, &field(&m.model.artifact_sha256))?;
    if m.backend() != "hailo"
        || artifact.files.len() != 1
        || artifact.fixed_seq != m.rows.seq
        || e.target != field(&m.device.arch)
        || e.manifest_sha256 != field(&m.model.manifest_sha256)
        || e.hef_sha256 != m.manifest.file(&artifact.files[0]).sha256
        || e.rows_sha256 != m.rows.sha256()
        || e.batch != 32
        || e.batch != m.rows.batch
        || e.seq != m.rows.seq
        || e.cases != m.rows.cases
        || e.live_tokens != m.rows.live_tokens()
        || e.computed_tokens_per_pass != m.rows.padded_tokens()
    {
        return Err("HailoRT receipt differs from the library's HEF, device or token inputs".into());
    }
    if e.fps_samples.len() < 5
        || e.seconds_per_process < 5
        || e.warmup_processes != 1
        || e.cli_version.is_empty()
        || e.commands.is_empty()
        || e.fps_samples.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err(
            "HailoRT receipt needs five positive measured samples of at least five seconds and one warmup".into()
        );
    }
    let pin = format!("hailortcli@sha256:{}", e.cli_sha256);
    crate::docker::check_pinned("hailortcli", &pin)?;
    let mut ms: Vec<f64> = e.fps_samples.iter().map(|f| 1000.0 * f64::from(e.batch) / f).collect();
    ms.sort_by(f64::total_cmp);
    let p50 = percentile(&ms, 50.0);
    Ok(ReferenceRun {
        name: NAME.into(),
        role: "kernel".into(),
        pinned: pin,
        version: e.cli_version,
        commands: e.commands,
        procedure: format!(
            "HailoRT on the same verified HEF and mixed token inputs; F32 word and attention-bias input files. \
             One untimed process, then {} processes of {} seconds. Each sample is batch * 1000 / FPS. \
             p50 and p99 describe aggregate service-time estimates across processes, not request latency tails. \
             Host gather, tokenization, pooling and normalization are excluded; CLI input conversion is included. \
             No reference output vectors are read. Runner SHA-256 {}",
            ms.len(),
            e.seconds_per_process,
            turbo::bundle::sha256_hex(SCRIPT.as_bytes())
        ),
        measured: Some(Measured {
            iterations: ms.len() as u64,
            p50_ms: p50,
            p99_ms: percentile(&ms, 99.0),
            rows_per_second: f64::from(e.batch) * 1000.0 / p50,
            min_cosine: None,
            computed_tokens: Some(m.rows.padded_tokens()),
        }),
        not_run: None,
        kernel_time: None,
    })
}
