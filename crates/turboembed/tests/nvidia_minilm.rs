//! Live NVIDIA proof through the official TurboEmbed C ABI.
//!
//! Compiled only with `--features ort-cuda`. Loads MiniLM via ONNX Runtime
//! CUDA EP + IoBinding device buffers. Fails loud if the CUDA EP is missing,
//! if outputs land on CPU, if `/proc/self/maps` lacks the CUDA provider, if
//! `libpython` is mapped, or if cosine vs nvidia goldens is below 0.99.
//!
//! No mock. No CPU fallback. No Python.
//!
//! ```bash
//! export LD_LIBRARY_PATH="$(pwd)/.libs/nvidia/lib:${LD_LIBRARY_PATH:-}"
//! cargo test -p turboembed --features ort-cuda -- --include-ignored --nocapture
//! ```
//!
//! Or: `make test-turboembed-nvidia`

#![cfg(feature = "ort-cuda")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Process-wide arena / ORT counters are shared. Serialize tests that
/// create an engine so a CUDA load cannot increment allocs mid-CPU proof.
fn serialize_engine_tests() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().expect("engine test lock")
}

use serde_json::Value;
use turboembed::ffi::{
    turboembed_device, turboembed_embed_one, turboembed_embed_result, turboembed_embed_result_free,
    turboembed_engine, turboembed_engine_create, turboembed_engine_destroy, turboembed_last_error,
    turboembed_load_model, turboembed_status,
};
use turboembed::{Device, EmbedOptions, Engine, Error, Pooling};

const COSINE_FLOOR: f32 = 0.99;
const ALIAS: &str = "minilm";
const GOLDEN: &str = "testdata/e2e/goldens/nvidia/minilm.json";
const RECEIPT: &str = "testdata/receipts/turboembed/nvidia-minilm.json";
const RECEIPT_TRT: &str = "testdata/receipts/turboembed/nvidia-minilm-tensorrt.json";
const SUBSET_PREFIX: &str = "parity:";

unsafe extern "C" {
    fn turboembed_ort_hot_path_reset();
    fn turboembed_ort_arena_allocs() -> u64;
    fn turboembed_ort_external_allocs() -> u64;
    fn turboembed_ort_external_last_bytes() -> u64;
    fn turboembed_ort_d2h_bytes() -> u64;
    fn turboembed_ort_d2h_calls() -> u64;
    fn turboembed_ort_d2h_result_bytes() -> u64;
    fn turboembed_ort_result_host_bytes() -> u64;
    fn turboembed_ort_cuda_forward_allocs() -> u64;
    fn turboembed_ort_cuda_forward_h2d_bytes() -> u64;
    fn turboembed_ort_cuda_forward_h2d_calls() -> u64;
}

fn reset_hot_path() {
    unsafe { turboembed_ort_hot_path_reset() };
}

fn hot_path_snapshot() -> serde_json::Value {
    unsafe {
        serde_json::json!({
            "arena_allocs": turboembed_ort_arena_allocs(),
            "ort_gpu_external_allocs": turboembed_ort_external_allocs(),
            "cuda_forward_allocs": turboembed_ort_cuda_forward_allocs(),
            "h2d_bytes": turboembed_ort_cuda_forward_h2d_bytes(),
            "h2d_calls": turboembed_ort_cuda_forward_h2d_calls(),
            "d2h_hidden_bytes": turboembed_ort_d2h_bytes(),
            "d2h_hidden_calls": turboembed_ort_d2h_calls(),
            "d2h_result_bytes": turboembed_ort_d2h_result_bytes(),
            "result_host_bytes": turboembed_ort_result_host_bytes(),
        })
    }
}

fn assert_no_hot_path_allocs(label: &str) {
    unsafe {
        let arena = turboembed_ort_arena_allocs();
        let ext = turboembed_ort_external_allocs();
        let fwd = turboembed_ort_cuda_forward_allocs();
        let h2d = turboembed_ort_cuda_forward_h2d_bytes();
        assert_eq!(
            arena, 0,
            "{label}: turbo_buffer_alloc_counter={arena} (arena-owned slots must reuse after warmup)"
        );
        assert_eq!(
            ext, 0,
            "{label}: ORT gpu_external_alloc={ext} last_bytes={} (ORT still allocated behind the embed)",
            turboembed_ort_external_last_bytes()
        );
        assert_eq!(
            fwd, 0,
            "{label}: turbo_buffer_cuda_forward_allocs={fwd}"
        );
        assert_eq!(
            h2d, 0,
            "{label}: token H2D bytes={h2d} (PINNED mapped tokens must not H2D)"
        );
        let hidden_d2h = turboembed_ort_d2h_bytes();
        let result_d2h = turboembed_ort_d2h_result_bytes();
        assert_eq!(
            hidden_d2h, 0,
            "{label}: activation D2H bytes={hidden_d2h} (mean+L2 must stay on DEVICE)"
        );
        assert_eq!(
            result_d2h, 0,
            "{label}: result-row cudaMemcpy D2H={result_d2h} (mapped PINNED, no memcpy)"
        );
    }
}

fn workspace_root() -> PathBuf {
    let from_build = PathBuf::from(env!("TURBOEMBED_WORKSPACE_ROOT"));
    if from_build.join(GOLDEN).is_file() {
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
    let na: f64 = a.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>().sqrt();
    assert!(na > 0.0 && nb > 0.0, "zero-norm vector");
    (dot / (na * nb)) as f32
}

fn git_head(root: &Path) -> String {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .expect("git rev-parse");
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn hostname() -> String {
    fs::read_to_string("/etc/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn maps_blob() -> String {
    fs::read_to_string("/proc/self/maps").expect("/proc/self/maps")
}

fn require_mapped(maps: &str, needle: &str) {
    assert!(
        maps.contains(needle),
        "/proc/self/maps must contain {needle} (ORT CUDA proof). maps excerpt:\n{}",
        maps.lines()
            .filter(|l| {
                let s = l.to_ascii_lowercase();
                s.contains("onnx") || s.contains("cuda") || s.contains("python")
            })
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

fn load_subset(path: &Path) -> (usize, Vec<(String, String, Vec<f32>)>, Vec<f32>) {
    let raw = fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("golden unreadable at {}: {e}", path.display());
    });
    let golden: Value = serde_json::from_str(&raw).expect("golden JSON");
    assert_eq!(golden["alias"], "minilm");
    assert_eq!(golden["arch"], "nvidia");
    assert_eq!(golden["pooling"], "mean");
    assert_eq!(golden["normalize"], true);
    let dim = golden["dim"].as_u64().expect("dim") as usize;
    assert_eq!(dim, 384, "MiniLM is 384-d");

    let hello: Vec<f32> = golden["vector"]
        .as_array()
        .expect("top-level vector")
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(hello.len(), dim);
    assert_eq!(golden["text"].as_str(), Some("hello world"));

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
    (dim, subset, hello)
}

/// AUTO is host-default GPU. On NVIDIA that is CUDA — never a silent CPU EP.
#[test]
fn auto_request_never_silently_uses_cpu() {
    let _lock = serialize_engine_tests();
    match Engine::create(Device::Auto) {
        Ok(engine) => match engine.load_model(ALIAS) {
            Ok(()) => {
                let info = engine.list_models().expect("list").get(0).expect("row");
                assert_eq!(
                    info.device,
                    Device::Cuda,
                    "AUTO must resolve to CUDA, not CPU"
                );
            }
            Err(Error::Unavailable(msg)) | Err(Error::UnsupportedDevice(msg)) => {
                let lower = msg.to_ascii_lowercase();
                assert!(
                    !lower.contains("cpu ep") || lower.contains("not a fallback"),
                    "AUTO must not silently become CPU, got {msg}"
                );
            }
            Err(other) => panic!("AUTO load must succeed on CUDA or fail loud, got {other:?}"),
        },
        Err(err) => panic!("AUTO create must succeed with --features ort-cuda, got {err:?}"),
    }
}

/// Device::TensorRT must not become CUDA, CPU, or 8-d FNV.
/// Create may succeed when `--features ort-cuda`; load then registers
/// the TensorRT EP with error_on_failure. Missing libnvinfer is loud.
#[test]
fn tensorrt_never_silent_cuda_cpu_or_fnv8() {
    let _lock = serialize_engine_tests();
    match Engine::create(Device::TensorRt) {
        Err(err) => {
            assert!(
                matches!(err, Error::Unavailable(_) | Error::UnsupportedDevice(_)),
                "TensorRT must fail loud, got {err:?}"
            );
            let lower = err.to_string().to_ascii_lowercase();
            assert!(
                lower.contains("tensorrt"),
                "error must name TensorRT, got {err}"
            );
            assert!(
                lower.contains("libnvinfer") || lower.contains("fallback"),
                "error must name the TRT blocker, got {err}"
            );
            assert!(
                lower.contains("cpu"),
                "error must say CPU is not accepted, got {err}"
            );
        }
        Ok(engine) => {
            // Do not load MiniLM here — TRT engine compile is the ignored
            // receipt test. Listing must not advertise catalog aliases as FNV8.
            let models = engine.list_models().expect("list");
            for m in models.iter() {
                if m.alias != "mock-embed" && m.alias != "mock" {
                    assert_ne!(
                        m.dim, 8,
                        "FAKE: {} listed dim=8 (FNV mock) on TensorRT",
                        m.alias
                    );
                }
            }
            let err = engine
                .load_model("mock-embed")
                .expect_err("TensorRT must not load the FNV mock alias");
            assert!(
                matches!(
                    err,
                    Error::NotImplemented(_) | Error::NotFound(_) | Error::Unavailable(_)
                ),
                "TensorRT mock-embed: {err:?}"
            );
        }
    }
}

/// CUDA request must stay on CUDA — never a silent CPU EP.
/// When CUDA is present, create+load must list CUDA. When it is missing,
/// the error must name CUDA and must not succeed as CPU.
#[test]
fn cuda_request_never_silently_uses_cpu() {
    let _lock = serialize_engine_tests();
    match Engine::create(Device::Cuda) {
        Ok(engine) => match engine.load_model(ALIAS) {
            Ok(()) => {
                let info = engine.list_models().expect("list").get(0).expect("row");
                assert_eq!(
                    info.device,
                    Device::Cuda,
                    "CUDA request compiled a non-CUDA session"
                );
            }
            Err(Error::Unavailable(msg)) | Err(Error::UnsupportedDevice(msg)) => {
                let lower = msg.to_ascii_lowercase();
                if lower.contains("not found") || lower.contains("onnx model") {
                    return;
                }
                assert!(
                    lower.contains("cuda"),
                    "missing-CUDA error must name CUDA, got {msg}"
                );
                assert!(
                    lower.contains("fallback")
                        || lower.contains("not accepted")
                        || lower.contains("cpu"),
                    "missing-CUDA error must say CPU is not a fallback, got {msg}"
                );
            }
            Err(other) => panic!("CUDA load must succeed on CUDA or fail loud, got {other:?}"),
        },
        Err(Error::UnsupportedDevice(msg)) | Err(Error::Unavailable(msg)) => {
            let lower = msg.to_ascii_lowercase();
            assert!(
                lower.contains("cuda"),
                "missing-CUDA create must name CUDA, got {msg}"
            );
        }
        Err(other) => panic!("CUDA create must succeed or fail loud, got {other:?}"),
    }
}

#[test]
fn minilm_ort_cpu_matches_golden() {
    let _lock = serialize_engine_tests();
    let root = workspace_root();
    let (dim, _subset, hello) = load_subset(&root.join(GOLDEN));

    let engine = Engine::create(Device::Cpu).unwrap_or_else(|e| {
        panic!("turboembed_engine_create(CPU) failed: {e:?}");
    });
    engine.load_model(ALIAS).unwrap_or_else(|e| {
        panic!("load_model({ALIAS}) via ORT CPU EP failed: {e:?}");
    });

    let info = engine.list_models().expect("list").get(0).expect("row");
    assert_eq!(info.alias, ALIAS);
    assert_eq!(
        info.device,
        Device::Cpu,
        "explicit CPU must list CPU (not CUDA)"
    );
    assert!(info.ready);
    assert_eq!(info.dim, dim as u32);

    let opts = EmbedOptions {
        pooling: Pooling::Mean,
        normalize: Some(true),
        ..Default::default()
    };
    reset_hot_path();
    let one = engine
        .embed_one(ALIAS, "hello world", &opts)
        .unwrap_or_else(|e| panic!("embed_one hello world on CPU failed: {e:?}"));
    assert_eq!(one.dim(), dim);
    assert_no_hot_path_allocs("cpu hello world");
    let hello_cos = cosine(one.values(), &hello);
    assert!(
        hello_cos >= COSINE_FLOOR,
        "CPU cosine vs nvidia golden hello world {hello_cos} < {COSINE_FLOOR}"
    );
    let hot = hot_path_snapshot();
    drop(one);

    let cpu_engine = Engine::create(Device::Cpu).expect("create Device::Cpu again");
    cpu_engine
        .load_model(ALIAS)
        .expect("second CPU load");
    assert_eq!(
        cpu_engine.list_models().unwrap().get(0).unwrap().device,
        Device::Cpu
    );

    let receipt = serde_json::json!({
        "schema_version": 1,
        "crate": "turboembed",
        "alias": ALIAS,
        "arch": "nvidia",
        "device": "CPU",
        "provider": "ORT CPU EP (explicit Device::Cpu)",
        "abi": "turboembed.h",
        "abi_version": 1,
        "pooling": "mean",
        "normalize": true,
        "dims": dim,
        "text": "hello world",
        "hello_world_cosine": hello_cos,
        "threshold": COSINE_FLOOR,
        "pass": true,
        "git_sha": git_head(&root),
        "host": hostname(),
        "hot_path": hot,
        "notes": "Explicit CPU EP on a turbo_buffer HOST arena. Tokens and hidden states are rented HOST views bound through IoBinding. CUDA requests still fail loud if the CUDA EP is missing.",
    });
    let path = root.join("testdata/receipts/turboembed/nvidia-minilm-cpu.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("receipt dir");
    }
    fs::write(&path, serde_json::to_string_pretty(&receipt).unwrap() + "\n")
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    eprintln!("wrote {} cosine={hello_cos:.6}", path.display());
}

#[test]
#[ignore = "needs MiniLM ONNX + CUDA 13 libs + GPU; see docs/turboembed.md"]
fn minilm_ort_cuda_iobinding_matches_golden() {
    let _lock = serialize_engine_tests();
    let root = workspace_root();
    let (dim, subset, hello) = load_subset(&root.join(GOLDEN));

    let engine = Engine::create(Device::Cuda).unwrap_or_else(|e| {
        panic!(
            "turboembed_engine_create(CUDA) failed: {e:?} — \
             CUDA EP required, CPU is not success"
        );
    });
    assert_eq!(Device::Cuda.as_str(), "cuda");

    engine.load_model(ALIAS).unwrap_or_else(|e| {
        panic!(
            "load_model({ALIAS}) via ORT CUDA IoBinding failed: {e:?} — \
             not a mock, not a CPU EP"
        );
    });

    let models = engine.list_models().expect("list_models");
    let info = models.get(0).expect("minilm row");
    assert_eq!(info.alias, ALIAS);
    assert_eq!(info.device, Device::Cuda, "list_models device must be CUDA");
    assert!(info.ready);
    assert_eq!(info.dim, dim as u32, "minilm dim");

    let opts = EmbedOptions {
        pooling: Pooling::Mean,
        normalize: Some(true),
        ..Default::default()
    };

    reset_hot_path();
    let hello_live = engine
        .embed_one(ALIAS, "hello world", &opts)
        .unwrap_or_else(|e| panic!("embed_one hello world on CUDA failed: {e:?}"));
    assert_ne!(
        hello_live.dim(),
        8,
        "FAKE: minilm on CUDA returned dim=8 (FNV mock)"
    );
    assert_eq!(hello_live.dim(), dim);
    assert_eq!(hello_live.count(), 1);
    assert_eq!(hello_live.values().len(), dim);
    assert_eq!(hello_live.packed().len(), dim * 4);
    let hello_cos = cosine(hello_live.values(), &hello);
    assert!(
        hello_cos >= COSINE_FLOOR,
        "cosine vs nvidia golden hello world {hello_cos} < {COSINE_FLOOR}"
    );
    assert_no_hot_path_allocs("cuda hello world");
    let hidden_vol = 256 * 384 * std::mem::size_of::<f32>();
    let result_vol = 384 * std::mem::size_of::<f32>();
    let result_host = unsafe { turboembed_ort_result_host_bytes() };
    assert_eq!(
        result_host, result_vol as u64,
        "hello world must read the mapped 384-d row ({result_vol} bytes), got {result_host}"
    );
    assert!(
        result_host < hidden_vol as u64,
        "result host read {result_host} must be much smaller than hidden volume {hidden_vol}"
    );
    assert_eq!(
        unsafe { turboembed_ort_d2h_bytes() },
        0,
        "activation D2H must be 0 after DEVICE mean+L2"
    );
    // Return the PINNED result slab before the next embed. Holding
    // `hello_live` across the subset used to look like a hot-path leak
    // (the warmed [32, dim] slab stayed checked out).
    drop(hello_live);

    let mut worst = 1.0_f32;
    let mut sum = 0.0_f32;
    for (id, text, expected) in &subset {
        let before_arena = unsafe { turboembed_ort_arena_allocs() };
        let before_ext = unsafe { turboembed_ort_external_allocs() };
        let got = engine
            .embed_one(ALIAS, text, &opts)
            .unwrap_or_else(|e| panic!("embed_one({ALIAS}, {id:?}) failed: {e:?}"));
        assert_eq!(got.dim(), dim, "{id} live dim");
        let sim = cosine(got.values(), expected);
        assert!(
            sim >= COSINE_FLOOR,
            "{id}: cosine {sim} < {COSINE_FLOOR} — not a real MiniLM CUDA embed \
             (mock / CPU / wrong pooling would fail here)"
        );
        worst = worst.min(sim);
        sum += sim;
        let after_arena = unsafe { turboembed_ort_arena_allocs() };
        let after_ext = unsafe { turboembed_ort_external_allocs() };
        eprintln!(
            "  {id}: cosine={sim:.6} dim={} arena {before_arena}→{after_arena} \
             ext {before_ext}→{after_ext} d2h={}",
            got.dim(),
            unsafe { turboembed_ort_d2h_bytes() }
        );
        assert_eq!(
            after_arena, before_arena,
            "{id}: turbo_buffer_alloc_counter rose {before_arena}→{after_arena}"
        );
        assert_eq!(
            after_ext, before_ext,
            "{id}: ORT gpu_external_alloc rose {before_ext}→{after_ext} last_bytes={}",
            unsafe { turboembed_ort_external_last_bytes() }
        );
        drop(got);
    }
    let mean = sum / subset.len() as f32;
    assert_no_hot_path_allocs("cuda parity subset");

    let maps = maps_blob();
    require_mapped(&maps, "libonnxruntime_providers_cuda");
    require_mapped(&maps, "libcudart");
    forbid_mapped(&maps, "libpython");

    let gpu_name = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    assert!(
        !gpu_name.is_empty(),
        "nvidia-smi returned no GPU name; this is not a real NVIDIA host"
    );

    eprintln!(
        "turboembed nvidia minilm: n={} dim={} min_cosine={worst:.6} mean={mean:.6} \
         device=CUDA gpu={gpu_name}",
        subset.len(),
        dim
    );

    let commands = [
        "export LD_LIBRARY_PATH=\"$(pwd)/.libs/nvidia/lib:${LD_LIBRARY_PATH:-}\"",
        "cargo test -p turboembed --features ort-cuda -- --include-ignored --nocapture --test-threads=1",
        "make test-turboembed-nvidia",
    ];
    let receipt = serde_json::json!({
        "schema_version": 1,
        "crate": "turboembed",
        "alias": ALIAS,
        "arch": "nvidia",
        "device": "CUDA",
        "provider": "ORT CUDA EP + IoBinding device buffers",
        "abi": "turboembed.h",
        "abi_version": 1,
        "pooling": "mean",
        "normalize": true,
        "dims": dim,
        "n_texts": subset.len(),
        "subset": "parity:*",
        "golden": GOLDEN,
        "hello_world_cosine": hello_cos,
        "worst_cosine": worst,
        "mean_cosine": mean,
        "threshold": COSINE_FLOOR,
        "pass": true,
        "git_sha": git_head(&root),
        "host": hostname(),
        "gpu": gpu_name,
        "maps": {
            "libonnxruntime_providers_cuda": true,
            "libcudart": true,
            "libpython": false,
        },
        "commands": commands,
        "hot_path": hot_path_snapshot(),
        "io": {
            "tokens": "turbo_buffer PINNED mapped rent; IoBinding CUDA view via mapped device ptr",
            "hidden": "turbo_buffer DEVICE rent bound with IoBinding BindOutput",
            "result": "turbo_buffer PINNED rent (host-visible)",
            "ort_gpu_allocator": "CUDA EP gpu_external_alloc → turbo_buffer DEVICE rent",
            "d2h": "none for activations; mean+L2 on DEVICE into mapped PINNED 384-d row (result_host_bytes, no cudaMemcpy)"
        },
        "notes": "No mock. No CPU fallback. Hidden stays DEVICE. Mask-weighted mean+L2 writes the 384-d row into rented mapped PINNED. d2h_hidden_bytes must be 0. result_host_bytes is the mapped row the caller reads — much smaller than the hidden volume. Arena/ORT-external allocs after warmup must be 0."
    });
    let path = root.join(RECEIPT);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("receipt dir");
    }
    fs::write(&path, serde_json::to_string_pretty(&receipt).unwrap() + "\n")
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    eprintln!("wrote {}", path.display());
}

#[test]
#[ignore = "needs MiniLM ONNX + CUDA 13 libs + GPU; see docs/turboembed.md"]
fn minilm_c_abi_embed_one_on_cuda() {
    let _lock = serialize_engine_tests();
    let root = workspace_root();
    let (_, _, hello) = load_subset(&root.join(GOLDEN));

    unsafe {
        let mut engine: *mut turboembed_engine = std::ptr::null_mut();
        let st = turboembed_engine_create(
            turboembed_device::TURBOEMBED_DEVICE_CUDA,
            std::ptr::null(),
            &mut engine,
        );
        assert_eq!(
            st,
            turboembed_status::TURBOEMBED_OK,
            "C ABI create CUDA: {}",
            std::ffi::CStr::from_ptr(turboembed_last_error(std::ptr::null()))
                .to_string_lossy()
        );
        assert!(!engine.is_null());

        let alias = ALIAS.as_bytes();
        let st = turboembed_load_model(engine, alias.as_ptr().cast(), alias.len());
        assert_eq!(
            st,
            turboembed_status::TURBOEMBED_OK,
            "C ABI load minilm: {}",
            std::ffi::CStr::from_ptr(turboembed_last_error(engine)).to_string_lossy()
        );

        let text = b"hello world";
        let mut out: *mut turboembed_embed_result = std::ptr::null_mut();
        let st = turboembed_embed_one(
            engine,
            alias.as_ptr().cast(),
            alias.len(),
            text.as_ptr().cast(),
            text.len(),
            std::ptr::null(),
            &mut out,
        );
        assert_eq!(
            st,
            turboembed_status::TURBOEMBED_OK,
            "C ABI embed_one: {}",
            std::ffi::CStr::from_ptr(turboembed_last_error(engine)).to_string_lossy()
        );
        assert!(!out.is_null());
        assert_eq!((*out).dim, 384);
        assert_eq!((*out).count, 1);
        assert!(!(*out).values.is_null());

        let live = std::slice::from_raw_parts((*out).values, 384);
        let c = cosine(live, &hello);
        assert!(
            c >= COSINE_FLOOR,
            "C ABI cosine vs nvidia golden {c} < {COSINE_FLOOR}"
        );

        turboembed_embed_result_free(out);
        turboembed_engine_destroy(engine);
    }
}

#[test]
fn minilm_c_abi_embed_one_on_cpu() {
    let _lock = serialize_engine_tests();
    let root = workspace_root();
    let (_, _, hello) = load_subset(&root.join(GOLDEN));

    unsafe {
        let mut engine: *mut turboembed_engine = std::ptr::null_mut();
        let st = turboembed_engine_create(
            turboembed_device::TURBOEMBED_DEVICE_CPU,
            std::ptr::null(),
            &mut engine,
        );
        assert_eq!(
            st,
            turboembed_status::TURBOEMBED_OK,
            "C ABI create CPU: {}",
            std::ffi::CStr::from_ptr(turboembed_last_error(std::ptr::null())).to_string_lossy()
        );
        assert!(!engine.is_null());

        let alias = ALIAS.as_bytes();
        let st = turboembed_load_model(engine, alias.as_ptr().cast(), alias.len());
        assert_eq!(
            st,
            turboembed_status::TURBOEMBED_OK,
            "C ABI load minilm CPU: {}",
            std::ffi::CStr::from_ptr(turboembed_last_error(engine)).to_string_lossy()
        );

        let text = b"hello world";
        let mut out: *mut turboembed_embed_result = std::ptr::null_mut();
        let st = turboembed_embed_one(
            engine,
            alias.as_ptr().cast(),
            alias.len(),
            text.as_ptr().cast(),
            text.len(),
            std::ptr::null(),
            &mut out,
        );
        assert_eq!(
            st,
            turboembed_status::TURBOEMBED_OK,
            "C ABI embed_one CPU: {}",
            std::ffi::CStr::from_ptr(turboembed_last_error(engine)).to_string_lossy()
        );
        assert!(!out.is_null());
        assert_eq!((*out).dim, 384);

        let live = std::slice::from_raw_parts((*out).values, 384);
        let c = cosine(live, &hello);
        assert!(
            c >= COSINE_FLOOR,
            "C ABI CPU cosine vs nvidia golden {c} < {COSINE_FLOOR}"
        );

        turboembed_embed_result_free(out);
        turboembed_engine_destroy(engine);
    }
}

#[test]
#[ignore = "needs MiniLM ONNX + TensorRT 10 (libnvinfer.so.10) + CUDA 13; see docs/turboembed.md"]
fn minilm_ort_tensorrt_matches_golden() {
    let _lock = serialize_engine_tests();
    let root = workspace_root();
    let (dim, subset, hello) = load_subset(&root.join(GOLDEN));

    let engine = Engine::create(Device::TensorRt).unwrap_or_else(|e| {
        panic!(
            "turboembed_engine_create(TENSORRT) failed: {e:?} — \
             not a CUDA or CPU stand-in"
        );
    });
    assert_eq!(Device::TensorRt.as_str(), "tensorrt");

    engine.load_model(ALIAS).unwrap_or_else(|e| {
        panic!(
            "load_model({ALIAS}) via ORT TensorRT EP failed: {e:?} — \
             not a mock, not CUDA-only, not a CPU EP"
        );
    });

    let info = engine.list_models().expect("list").get(0).expect("row");
    assert_eq!(info.alias, ALIAS);
    assert_eq!(
        info.device,
        Device::TensorRt,
        "list_models device must be TensorRT"
    );
    assert!(info.ready);
    assert_ne!(info.dim, 8, "FAKE: minilm on TensorRT returned dim=8 (FNV mock)");
    assert_eq!(info.dim, dim as u32);

    let opts = EmbedOptions {
        pooling: Pooling::Mean,
        normalize: Some(true),
        ..Default::default()
    };
    let hello_live = engine
        .embed_one(ALIAS, "hello world", &opts)
        .unwrap_or_else(|e| panic!("embed_one hello world on TensorRT failed: {e:?}"));
    assert_eq!(hello_live.dim(), dim);
    let hello_cos = cosine(hello_live.values(), &hello);
    assert!(
        hello_cos >= COSINE_FLOOR,
        "TensorRT cosine vs nvidia golden hello world {hello_cos} < {COSINE_FLOOR}"
    );
    drop(hello_live);

    let mut worst = 1.0_f32;
    let mut sum = 0.0_f32;
    for (id, text, expected) in &subset {
        let got = engine
            .embed_one(ALIAS, text, &opts)
            .unwrap_or_else(|e| panic!("embed_one({ALIAS}, {id:?}) failed: {e:?}"));
        let sim = cosine(got.values(), expected);
        assert!(
            sim >= COSINE_FLOOR,
            "{id}: TensorRT cosine {sim} < {COSINE_FLOOR}"
        );
        worst = worst.min(sim);
        sum += sim;
        eprintln!("  {id}: cosine={sim:.6} dim={}", got.dim());
    }
    let mean = sum / subset.len() as f32;

    let maps = maps_blob();
    require_mapped(&maps, "libonnxruntime_providers_tensorrt");
    require_mapped(&maps, "libnvinfer");
    require_mapped(&maps, "libcudart");
    forbid_mapped(&maps, "libpython");

    let gpu_name = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    eprintln!(
        "turboembed nvidia minilm TensorRT: n={} dim={} min_cosine={worst:.6} \
         mean={mean:.6} gpu={gpu_name}",
        subset.len(),
        dim
    );

    let receipt = serde_json::json!({
        "schema_version": 1,
        "crate": "turboembed",
        "alias": ALIAS,
        "arch": "nvidia",
        "device": "TENSORRT",
        "provider": "ORT TensorRT EP (same MiniLM ONNX, CUDA IoBinding buffers)",
        "abi": "turboembed.h",
        "abi_version": 1,
        "pooling": "mean",
        "normalize": true,
        "dims": dim,
        "n_texts": subset.len(),
        "subset": "parity:*",
        "golden": GOLDEN,
        "hello_world_cosine": hello_cos,
        "worst_cosine": worst,
        "mean_cosine": mean,
        "threshold": COSINE_FLOOR,
        "pass": true,
        "git_sha": git_head(&root),
        "host": hostname(),
        "gpu": gpu_name,
        "maps": {
            "libonnxruntime_providers_tensorrt": true,
            "libnvinfer": true,
            "libcudart": true,
            "libpython": false,
        },
        "commands": [
            "export LD_LIBRARY_PATH=\"$(pwd)/.libs/nvidia/lib:${LD_LIBRARY_PATH:-}\"",
            "cargo test -p turboembed --features ort-cuda --test nvidia_minilm -- --ignored --nocapture minilm_ort_tensorrt_matches_golden",
        ],
        "notes": "No mock. No CUDA-only stand-in. No CPU EP. Maps must contain libnvinfer + providers_tensorrt."
    });
    let path = root.join(RECEIPT_TRT);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("receipt dir");
    }
    fs::write(&path, serde_json::to_string_pretty(&receipt).unwrap() + "\n")
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    eprintln!("wrote {}", path.display());
}
