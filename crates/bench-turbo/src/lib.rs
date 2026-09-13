//! Shared helpers for the SOLIDIFY (7) Turbo bench.
//!
//! Percentiles are **nearest rank** on a sorted sample:
//! `rank = ceil(p/100 * n)` (1-based). Numbers in receipts come from
//! live forwards, not estimates.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde_json::Value;

/// Default warmup before the timed window. Load already warms arenas;
/// this is extra steady-state before the first measured sample.
pub const DEFAULT_WARMUP: u32 = 32;

/// Default timed samples. N=200 is enough for a stable nearest-rank p99.
pub const DEFAULT_ITERS: u32 = 200;

/// Nearest-rank percentile in microseconds. `samples` must be sorted ascending.
pub fn nearest_rank_us(sorted_us: &[u64], percent: f64) -> u64 {
    assert!((0.0..=100.0).contains(&percent), "percent out of range");
    let n = sorted_us.len();
    if n == 0 {
        return 0;
    }
    let rank = ((percent / 100.0) * n as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(n - 1);
    sorted_us[idx]
}

pub fn duration_us(d: Duration) -> u64 {
    d.as_nanos().div_ceil(1000) as u64
}

pub fn summarize_latencies(mut samples_us: Vec<u64>) -> Value {
    samples_us.sort_unstable();
    let n = samples_us.len();
    let min = samples_us.first().copied().unwrap_or(0);
    let max = samples_us.last().copied().unwrap_or(0);
    let sum: u128 = samples_us.iter().map(|v| u128::from(*v)).sum();
    let mean = if n == 0 {
        0.0
    } else {
        sum as f64 / n as f64
    };
    serde_json::json!({
        "p50": nearest_rank_us(&samples_us, 50.0),
        "p99": nearest_rank_us(&samples_us, 99.0),
        "min": min,
        "max": max,
        "mean": mean,
        "n": n,
        "unit": "us",
        "percentile_method": "nearest_rank",
    })
}

pub fn workspace_root() -> PathBuf {
    let from_env = std::env::var_os("INFERSTREAM_ROOT").map(PathBuf::from);
    if let Some(p) = from_env {
        if p.join("testdata").is_dir() {
            return p;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("inferstream workspace root")
}

pub fn git_head(root: &Path) -> String {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .expect("git rev-parse");
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

pub fn hostname() -> String {
    fs::read_to_string("/etc/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

pub fn nvidia_gpu_name() -> Result<String, String> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .map_err(|e| format!("nvidia-smi: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "nvidia-smi failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if name.is_empty() {
        return Err("nvidia-smi returned no GPU name".into());
    }
    Ok(name)
}

pub fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let body = serde_json::to_string_pretty(value).map_err(|e| e.to_string())? + "\n";
    fs::write(path, body).map_err(|e| format!("{}: {e}", path.display()))?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

pub fn read_json(path: &Path) -> Result<Value, String> {
    let raw = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector length");
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt();
    assert!(na > 0.0 && nb > 0.0, "zero-norm vector");
    (dot / (na * nb)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_empty_is_zero() {
        assert_eq!(nearest_rank_us(&[], 50.0), 0);
        assert_eq!(nearest_rank_us(&[], 99.0), 0);
    }

    #[test]
    fn nearest_rank_single() {
        assert_eq!(nearest_rank_us(&[7], 50.0), 7);
        assert_eq!(nearest_rank_us(&[7], 99.0), 7);
    }

    #[test]
    fn nearest_rank_n100() {
        let s: Vec<u64> = (1..=100).collect();
        assert_eq!(nearest_rank_us(&s, 50.0), 50);
        assert_eq!(nearest_rank_us(&s, 99.0), 99);
        assert_eq!(nearest_rank_us(&s, 100.0), 100);
    }

    #[test]
    fn nearest_rank_n200_p99() {
        let s: Vec<u64> = (1..=200).collect();
        // ceil(0.99 * 200) = 198 → 1-based rank 198
        assert_eq!(nearest_rank_us(&s, 99.0), 198);
        assert_eq!(nearest_rank_us(&s, 50.0), 100);
    }
}
