//! Live NVIDIA proof: C ABI `embed("minilm", text)` via ORT CUDA IoBinding.
//!
//! Ignored by default — needs the MiniLM ONNX on disk, CUDA 13 user-space
//! libs, and a GPU. Run on krick:
//!
//! ```bash
//! export LD_LIBRARY_PATH="$(pwd)/.libs/nvidia/lib:${LD_LIBRARY_PATH:-}"
//! cargo test -p turboembed --features ort-cuda -- --ignored --nocapture
//! ```
//!
//! Or: `make test-turboembed-nvidia`
//!
//! Cosine vs `testdata/e2e/goldens/nvidia/minilm.json` must be ≥ 0.99 on the
//! fixed overlapping subset (all `parity:*` items). A CPU fallback or mock
//! vector fails this gate (and the device-buffer checks in the engine).

#![cfg(feature = "ort-cuda")]

use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};

use serde_json::Value;
use turboembed::{cosine, ffi, Arch, Engine};

const GOLDEN: &str = "testdata/e2e/goldens/nvidia/minilm.json";
const RECEIPT: &str = "testdata/receipts/turboembed/nvidia-minilm.json";
const MIN_COSINE: f32 = 0.99;

/// Fixed overlapping subset: every `parity:*` golden (short / medium / query /
/// unicode / long / inferstream). These are the texts the e2e harness always
/// embeds.
const SUBSET_PREFIX: &str = "parity:";

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/turboembed → repo root")
        .to_path_buf()
}

fn git_sha(root: &Path) -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn load_subset(path: &Path) -> (usize, Vec<(String, String, Vec<f32>)>) {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("golden unreadable at {}: {e}", path.display());
    });
    let golden: Value = serde_json::from_str(&raw).expect("golden JSON");
    assert_eq!(golden["alias"], "minilm");
    assert_eq!(golden["arch"], "nvidia");
    assert_eq!(golden["pooling"], "mean");
    assert_eq!(golden["normalize"], true);
    let dim = golden["dim"].as_u64().expect("dim") as usize;
    assert_eq!(dim, 384, "MiniLM is 384-d");

    let items = golden["items"].as_array().expect("items");
    let mut subset = Vec::new();
    for item in items {
        let id = item["id"].as_str().unwrap_or_default();
        if !id.starts_with(SUBSET_PREFIX) {
            continue;
        }
        let text = item["text"].as_str().expect("item.text").to_string();
        let vector: Vec<f32> = item["vector"]
            .as_array()
            .expect("item.vector")
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(vector.len(), dim, "{id} dim");
        subset.push((id.to_string(), text, vector));
    }
    assert!(
        !subset.is_empty(),
        "golden {GOLDEN} had no {SUBSET_PREFIX}* items"
    );
    (dim, subset)
}

fn write_receipt(
    root: &Path,
    dim: usize,
    n: usize,
    worst: f32,
    mean: f32,
    commands: &[&str],
) {
    let path = root.join(RECEIPT);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("receipt dir");
    }
    let receipt = serde_json::json!({
        "schema_version": 1,
        "crate": "turboembed",
        "alias": "minilm",
        "arch": "nvidia",
        "device": "CUDA",
        "provider": "ORT CUDA EP + IoBinding device buffers",
        "pooling": "mean",
        "normalize": true,
        "dims": dim,
        "n_texts": n,
        "subset": "parity:*",
        "golden": GOLDEN,
        "worst_cosine": worst,
        "mean_cosine": mean,
        "threshold": MIN_COSINE,
        "pass": true,
        "git_sha": git_sha(root),
        "host": hostname(),
        "commands": commands,
        "notes": "No mock. No CPU fallback. Output tensors must reside on AllocationDevice::CUDA before the host mean+L2 copy."
    });
    std::fs::write(&path, serde_json::to_string_pretty(&receipt).unwrap() + "\n")
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    eprintln!("wrote {}", path.display());
}

fn hostname() -> String {
    std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

#[test]
#[ignore = "needs MiniLM ONNX + CUDA 13 libs + GPU; see docs/turboembed.md"]
fn nvidia_minilm_ort_cuda_matches_golden() {
    let root = workspace_root();
    let golden_path = root.join(GOLDEN);
    let (dim, subset) = load_subset(&golden_path);

    let engine = Engine::open(Arch::Nvidia).unwrap_or_else(|e| {
        panic!(
            "Engine::open(nvidia) failed — this must be a real ORT CUDA session, \
             not a stub. Set LD_LIBRARY_PATH to .libs/nvidia/lib and rebuild \
             with --features ort-cuda. error: {e}"
        );
    });
    assert_eq!(engine.device().as_str(), "CUDA");

    let mut worst = 1.0_f32;
    let mut sum = 0.0_f32;
    for (id, text, expected) in &subset {
        let got = engine.embed("minilm", text).unwrap_or_else(|e| {
            panic!("embed(\"minilm\", {id:?}) failed: {e}");
        });
        assert_eq!(got.len(), dim, "{id} live dim");
        let sim = cosine(&got, expected);
        assert!(
            sim >= MIN_COSINE,
            "{id}: cosine {sim} < {MIN_COSINE} — not a real MiniLM CUDA embed \
             (mock / CPU / wrong pooling would fail here)"
        );
        worst = worst.min(sim);
        sum += sim;
        eprintln!("  {id}: cosine={sim:.6} dim={}", got.len());
    }
    let mean = sum / subset.len() as f32;
    eprintln!(
        "turboembed nvidia minilm: n={} dim={} worst={worst:.6} mean={mean:.6} device=CUDA",
        subset.len(),
        dim
    );

    // Also prove the C ABI `embed("minilm", text)` path.
    unsafe {
        let arch = CString::new("nvidia").unwrap();
        let mut err: *mut i8 = std::ptr::null_mut();
        let c_engine = ffi::turboembed_create(arch.as_ptr(), std::ptr::null(), &mut err);
        assert!(
            !c_engine.is_null(),
            "turboembed_create failed: {}",
            err_str(err)
        );
        let device = CStr::from_ptr(ffi::turboembed_device(c_engine))
            .to_str()
            .unwrap();
        assert_eq!(device, "CUDA");

        let (id, text, expected) = &subset[0];
        let alias = CString::new("minilm").unwrap();
        let text_c = CString::new(text.as_str()).unwrap();
        let mut out: *mut f32 = std::ptr::null_mut();
        let mut out_dim: usize = 0;
        let rc = ffi::turboembed_embed(
            c_engine,
            alias.as_ptr(),
            text_c.as_ptr(),
            &mut out,
            &mut out_dim,
            &mut err,
        );
        assert_eq!(rc, 0, "turboembed_embed failed: {}", err_str(err));
        assert_eq!(out_dim, dim);
        let live = std::slice::from_raw_parts(out, out_dim);
        let sim = cosine(live, expected);
        assert!(
            sim >= MIN_COSINE,
            "C ABI {id}: cosine {sim} < {MIN_COSINE}"
        );
        ffi::turboembed_free(out.cast());
        ffi::turboembed_destroy(c_engine);
    }

    let commands = [
        "export LD_LIBRARY_PATH=\"$(pwd)/.libs/nvidia/lib:${LD_LIBRARY_PATH:-}\"",
        "cargo test -p turboembed --features ort-cuda -- --ignored --nocapture",
        "make test-turboembed-nvidia",
    ];
    write_receipt(&root, dim, subset.len(), worst, mean, &commands);
}

unsafe fn err_str(ptr: *mut i8) -> String {
    if ptr.is_null() {
        return "(null)".into();
    }
    let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
    ffi::turboembed_free_str(ptr);
    s
}
