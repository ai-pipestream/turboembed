//! Actual library measurements paired with the vendor runner's input hashes.
//! Writes evidence only: no reference is invented and no capability is promoted.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde_json::{Value, json};
use turbo_bench::api::field;
use turbo_bench::measure::{self, Plan, RowKind};
use turbo_bench::{Result, git};

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [bundle, vendor, output] = args.as_slice() else {
        return Err("usage: hailo_evidence BUNDLE VENDOR_JSON OUTPUT_JSON".into());
    };
    let vendor: Value =
        serde_json::from_slice(&std::fs::read(vendor).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let number = |key: &str| -> Result<u32> {
        vendor[key].as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| format!("missing {key}"))
    };
    let batch = number("batch")?;
    let seq = number("seq")?;
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let before = git::provenance(&repo)?;
    turbo_bench::check_build(&before.commit, turbo_bench::BUILD_COMMIT, turbo_bench::BUILD_CHANGES)?;
    let plan = Plan {
        bundle: PathBuf::from(bundle),
        device: "hailo".into(),
        precision: turbo::TURBO_PRECISION_MODEL,
        batch: Some(batch),
        seq: Some(seq),
        rows: RowKind::Mixed,
        warmup: 3,
        iterations: 20,
    };
    let mut hailo = measure::measure(&plan)?;
    if vendor["target"].as_str() != Some(field(&hailo.device.arch).as_str())
        || vendor["manifest_sha256"].as_str() != Some(field(&hailo.model.manifest_sha256).as_str())
        || vendor["rows_sha256"].as_str() != Some(hailo.rows.sha256().as_str())
    {
        return Err("vendor and library inputs or device architecture differ".into());
    }
    let artifact = measure::loaded_artifact(&hailo.manifest, &field(&hailo.model.artifact_sha256))?;
    if artifact.fixed_seq != seq || artifact.files.len() != 1 {
        return Err("measurement must use the HEF's full sequence length".into());
    }
    if vendor["hef_sha256"].as_str() != Some(hailo.manifest.file(&artifact.files[0]).sha256.as_str()) {
        return Err("vendor and library HEFs differ".into());
    }
    // A HEF computes all positions, including padding. The generic bench
    // currently counts Hailo like a packed backend; preserve the real count.
    hailo.timing.computed_tokens = Some(hailo.rows.padded_tokens());
    let cpu = measure::measure(&Plan { device: "cpu".into(), precision: turbo::TURBO_PRECISION_EXACT, ..plan })?;
    if cpu.rows.sha256() != hailo.rows.sha256() {
        return Err("CPU and Hailo token inputs differ".into());
    }
    if cpu.conformance.min_cosine < 0.9999 || cpu.conformance.max_abs_diff > 1e-4 {
        return Err("CPU Exact failed F32 conformance against the bundle reference".into());
    }
    let mut min_cosine = 1.0_f64;
    for (h, c) in hailo.vectors.iter().zip(&cpu.vectors) {
        let similarity = measure::cosine(h, c);
        if !similarity.is_finite() {
            return Err("non-finite CPU/Hailo cosine".into());
        }
        min_cosine = min_cosine.min(similarity);
    }
    if git::provenance(&repo)? != before {
        return Err("repository changed during measurement".into());
    }
    let date = turbo_bench::utc(SystemTime::now());
    let result = json!({
        "capability_record": false,
        "i8_min_cosine_floor": 0.93,
        "cpu_exact_vs_hailo_min_cosine": min_cosine,
        "floor_passed": min_cosine >= 0.93 && hailo.conformance.min_cosine >= 0.93,
        "hailo": turbo_bench::record(&hailo, &before, vec![], date.clone())?,
        "cpu": turbo_bench::record(&cpu, &before, vec![], date)?,
        "vendor": vendor,
    });
    let bytes = serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?;
    std::fs::write(output, bytes).map_err(|e| e.to_string())?;
    println!(
        "CPU Exact {:.4} ms; Hailo I8 {:.4} ms; min cosine {min_cosine:.8}",
        cpu.timing.p50_ms, hailo.timing.p50_ms
    );
    Ok(())
}
