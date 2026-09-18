//! Live Intel NPU MiniLM qualification harness (Intel Core Ultra).
//!
//! `Device::OpenVinoNpu` has never been proven live: the committed
//! `testdata/receipts/turboembed/intel-npu.json` is an honest **fail**
//! receipt from Machine B (Battlemage dGPU + AMD CPU, no NPU silicon).
//! This ignored test is the pass path. Run it on a host whose `ov::Core`
//! actually lists `NPU` (Intel Core Ultra client silicon — e.g. an Intel
//! Cloud AI PC instance; see `docs/intel-cloud-npu-runbook.md`):
//!
//! ```bash
//! source /work/opt/openvino_genai/setupvars.sh   # or your install
//! make fetch-ov-genai ALIASES=minilm
//! cargo test -p turboembed --features genai --test intel_npu \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Policy (identical to GPU): a request for NPU either runs on NPU or
//! fails loud. Never CPU. Never GPU. Never the 8-d FNV mock. On a host
//! without NPU this test FAILS — it does not skip, and it does not
//! touch the committed fail receipt. On a pass it overwrites
//! `testdata/receipts/turboembed/intel-npu.json` with `pass=true`.
//!
//! The always-on fail-loud policy assertions stay in
//! `intel_genai_gpu.rs::npu_request_never_silently_uses_cpu_or_mock`
//! and `device_policy.rs`; this file only adds the live pass path.
//!
//! No OVMS. No mock-as-done. No Python.

#![cfg(feature = "genai")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use turboembed::{Device, EmbedOptions, Engine, Pooling};

unsafe extern "C" {
    fn turboembed_test_genai_available_devices(
        out: *mut std::os::raw::c_char,
        out_len: usize,
    ) -> turboembed::ffi::turboembed_status;
}

/// Same floor as the Machine B GPU/CPU receipts. Do not raise or lower.
const COSINE_FLOOR: f32 = 0.99;
const TEXT: &str = "hello world";
const ALIAS: &str = "minilm";

fn workspace_root() -> PathBuf {
    let from_build = PathBuf::from(env!("TURBOEMBED_WORKSPACE_ROOT"));
    if from_build
        .join("testdata/e2e/goldens/nvidia/minilm.json")
        .is_file()
    {
        return from_build;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
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

fn golden_vector(path: &Path) -> Vec<f32> {
    let raw = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let v: serde_json::Value =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    assert_eq!(v["text"].as_str(), Some(TEXT), "{} text", path.display());
    v["vector"]
        .as_array()
        .unwrap_or_else(|| panic!("{} missing vector", path.display()))
        .iter()
        .map(|x| x.as_f64().expect("f64") as f32)
        .collect()
}

fn sha256_file(path: &Path) -> String {
    let out = Command::new("sha256sum")
        .arg(path)
        .output()
        .expect("sha256sum");
    assert!(out.status.success(), "sha256sum {}", path.display());
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .expect("hex")
        .to_string()
}

fn git_head(root: &Path) -> String {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .expect("git rev-parse");
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn cpu_model_name() -> String {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
        })
        .unwrap_or_default()
}

fn ov_available_devices() -> Vec<String> {
    let mut buf = vec![0u8; 256];
    let st = unsafe { turboembed_test_genai_available_devices(buf.as_mut_ptr().cast(), buf.len()) };
    assert_eq!(
        st,
        turboembed::ffi::turboembed_status::TURBOEMBED_OK,
        "ov::Core available-devices probe failed"
    );
    let csv = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr().cast()) }
        .to_string_lossy()
        .into_owned();
    csv.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

fn maps_blob() -> String {
    fs::read_to_string("/proc/self/maps").expect("/proc/self/maps")
}

fn require_mapped(maps: &str, needle: &str) {
    assert!(
        maps.contains(needle),
        "/proc/self/maps must contain {needle} (live NPU CompiledModel proof). maps excerpt:\n{}",
        maps.lines()
            .filter(|l| l.contains("openvino") || l.contains("libze") || l.contains("python"))
            .take(40)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

fn forbid_mapped(maps: &str, needle: &str) {
    assert!(
        !maps
            .to_ascii_lowercase()
            .contains(&needle.to_ascii_lowercase()),
        "/proc/self/maps must not contain {needle}"
    );
}

/// Live NPU pass: create on `Device::OpenVinoNpu`, embed MiniLM with the
/// catalog pooling (mean + L2), gate cosine against the committed Intel
/// and NVIDIA goldens at the existing 0.99 floor, and overwrite
/// `testdata/receipts/turboembed/intel-npu.json` with the pass receipt.
///
/// On a host without an NPU this test fails loud (create refuses CPU/GPU
/// fallback) and leaves the committed fail receipt untouched.
#[test]
#[ignore = "needs Intel Core Ultra NPU host (Intel Cloud AI PC) + models/ov/minilm — see docs/intel-cloud-npu-runbook.md"]
fn intel_npu_minilm_live_pass_receipt() {
    let root = workspace_root();
    let model_dir = root.join("models/ov/minilm");
    assert!(
        model_dir.join("openvino_model.xml").is_file(),
        "missing {} — run `make fetch-ov-genai ALIASES=minilm`",
        model_dir.display()
    );

    let intel_golden = golden_vector(&root.join("testdata/e2e/goldens/intel/minilm.json"));
    let nvidia_golden = golden_vector(&root.join("testdata/e2e/goldens/nvidia/minilm.json"));
    assert_eq!(intel_golden.len(), 384);
    assert_eq!(nvidia_golden.len(), 384);

    // Fail loud, never fall back: a missing NPU is a test FAILURE here,
    // not a skip. The committed fail receipt stays as-is on failure.
    let engine = Engine::create(Device::OpenVinoNpu).unwrap_or_else(|e| {
        panic!(
            "turboembed_engine_create(OPENVINO_NPU) failed: {e:?} — this \
             harness needs a live Intel Core Ultra NPU (driver + \
             libopenvino_intel_npu_plugin.so). CPU/GPU/mock are not \
             success. See docs/intel-cloud-npu-runbook.md."
        );
    });
    assert_eq!(Device::OpenVinoNpu.as_str(), "openvino-npu");

    let listed = ov_available_devices();
    assert!(
        listed.iter().any(|d| d.starts_with("NPU")),
        "engine created for NPU but ov::Core does not list NPU: {listed:?}"
    );

    engine.load_model(ALIAS).unwrap_or_else(|e| {
        panic!(
            "load_model({ALIAS}) on NPU failed: {e:?} — NPU create \
             succeeded so the compile must stay on NPU, never CPU/GPU"
        );
    });

    let models = engine.list_models().expect("list_models");
    let info = models.get(0).expect("minilm row");
    assert_eq!(info.alias, ALIAS);
    assert_eq!(
        info.device,
        Device::OpenVinoNpu,
        "list_models device must be NPU — the request must never be \
         silently rerouted"
    );
    assert!(info.ready);
    assert_ne!(info.dim, 8, "FAKE: NPU path returned dim=8 (FNV mock)");
    assert_eq!(info.dim, 384, "minilm dim");

    let opts = EmbedOptions {
        pooling: Pooling::Mean,
        normalize: Some(true),
        ..Default::default()
    };
    let one = engine
        .embed_one(ALIAS, TEXT, &opts)
        .unwrap_or_else(|e| panic!("embed_one minilm on NPU failed: {e:?}"));
    assert_ne!(
        one.dim(),
        8,
        "FAKE: minilm on NPU returned dim=8 (FNV mock)"
    );
    assert_eq!(one.dim(), 384);
    assert_eq!(one.count(), 1);

    let live = one.values();
    let l2: f32 = live.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (l2 - 1.0).abs() < 1e-3,
        "live MiniLM row must be L2-normalized, got {l2}"
    );

    let cosine_intel = cosine(live, &intel_golden);
    let cosine_nvidia = cosine(live, &nvidia_golden);
    assert!(
        cosine_intel >= COSINE_FLOOR,
        "NPU cosine vs intel golden {cosine_intel} < {COSINE_FLOOR}"
    );
    assert!(
        cosine_nvidia >= COSINE_FLOOR,
        "NPU cosine vs nvidia golden {cosine_nvidia} < {COSINE_FLOOR}"
    );

    let maps = maps_blob();
    require_mapped(&maps, "libopenvino_intel_npu_plugin");
    // SOLIDIFY 5 still holds on NPU: WordPiece write-through, no
    // encode→copy tokenizer libs, no Python on the hot path.
    forbid_mapped(&maps, "libopenvino_genai");
    forbid_mapped(&maps, "libopenvino_tokenizers");
    forbid_mapped(&maps, "libpython");

    let receipt = serde_json::json!({
        "schema_version": 1,
        "alias": ALIAS,
        "device": "NPU",
        "wired": true,
        "pass": true,
        "pipeline": "WordPiece write-through + CompiledModel(\"NPU\")",
        "abi": "turboembed.h",
        "abi_version": 1,
        "host": "Intel Cloud AI PC (Machine D)",
        "chip": cpu_model_name(),
        "npu_plugin": true,
        "available_devices": listed,
        "pooling": "mean",
        "normalize": true,
        "dim": 384,
        "text": TEXT,
        "cosine": {
            "intel_golden": cosine_intel,
            "nvidia_golden": cosine_nvidia,
            "threshold": COSINE_FLOOR,
        },
        "sha": {
            "git": git_head(&root),
            "openvino_model.bin": sha256_file(&model_dir.join("openvino_model.bin")),
            "openvino_tokenizer.bin": sha256_file(&model_dir.join("openvino_tokenizer.bin")),
        },
        "maps": {
            "libopenvino_intel_npu_plugin": true,
            "libopenvino_genai": false,
            "libopenvino_tokenizers": false,
            "libpython": false,
        },
        "note": "Live Device::OpenVinoNpu MiniLM pass. Same IR, pooling, and 0.99 cosine floor as the Machine B GPU/CPU receipts. Replaces the earlier honest fail receipt from Machine B (no NPU silicon). NPU requests still fail loud when the plugin is missing — never CPU/GPU/mock.",
    });
    let receipt_dir = root.join("testdata/receipts/turboembed");
    fs::create_dir_all(&receipt_dir).expect("receipts dir");
    let receipt_path = receipt_dir.join("intel-npu.json");
    fs::write(
        &receipt_path,
        serde_json::to_string_pretty(&receipt).unwrap() + "\n",
    )
    .expect("write npu pass receipt");
    eprintln!("wrote {}", receipt_path.display());
}
