//! Live Intel GenAI proof through the TurboEmbed C ABI.
//!
//! `--features genai` loads MiniLM via
//! `ov::genai::TextEmbeddingPipeline(models_path, device, config)` with
//! the official device strings `"GPU"` and `"CPU"`.
//! Asking for GPU when the GPU plugin is missing fails (no CPU swap).
//! Same for NPU: create fails loud if the plugin is missing (never CPU / FNV8).
//! Explicit CPU compiles `"CPU"` and must produce real embeds.
//!
//! No OVMS. No mock-as-done. No Python.

#![cfg(feature = "genai")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use std::os::raw::c_void;

use turboembed::ffi::{
    turboembed_device, turboembed_embed_one, turboembed_embed_result, turboembed_embed_result_free,
    turboembed_engine, turboembed_engine_create, turboembed_engine_destroy, turboembed_last_error,
    turboembed_load_model, turboembed_status,
};
use turboembed::{Device, EmbedOptions, Engine, Error, Pooling};

const TURBO_BUFFER_OK: i32 = 0;
const TURBO_BUFFER_DEVICE_ZE: u32 = 4;
const TURBO_BUFFER_PLACE_HOST: u32 = 1;
const TURBO_BUFFER_PLACE_SHARED: u32 = 3;

unsafe extern "C" {
    fn turboembed_test_genai_arena_info(
        engine: *const turboembed_engine,
        arena_device: *mut u32,
        token_placement: *mut u32,
        token_ids: *mut *const c_void,
        hidden: *mut *const c_void,
        owns_tokens: *mut i32,
        owns_hidden: *mut i32,
        hidden_used_arena: *mut i32,
    ) -> turboembed_status;
    fn turbo_buffer_alloc_counter_reset();
    fn turbo_buffer_alloc_counter() -> u64;
    fn turbo_buffer_ze_query(ptr: *const c_void, out: *mut u32) -> i32;
}

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

fn maps_blob() -> String {
    fs::read_to_string("/proc/self/maps").expect("/proc/self/maps")
}

fn require_mapped(maps: &str, needle: &str) {
    assert!(
        maps.contains(needle),
        "/proc/self/maps must contain {needle} (GenAI GPU proof). maps excerpt:\n{}",
        maps.lines()
            .filter(|l| l.contains("openvino") || l.contains("libze") || l.contains("python"))
            .take(40)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

struct ArenaProof {
    arena_device: u32,
    token_placement: u32,
    token_ids: *const c_void,
    hidden: *const c_void,
    owns_tokens: bool,
    owns_hidden: bool,
    hidden_used_arena: bool,
    token_ze: Option<u32>,
    result_ze: Option<u32>,
    allocs_after_warmup: u64,
}

fn arena_info(engine: &Engine) -> ArenaProof {
    let raw = engine.raw_engine();
    let mut arena_device = 0u32;
    let mut token_placement = 0u32;
    let mut token_ids: *const c_void = std::ptr::null();
    let mut hidden: *const c_void = std::ptr::null();
    let mut owns_tokens = 0i32;
    let mut owns_hidden = 0i32;
    let mut hidden_used_arena = 0i32;
    let st = unsafe {
        turboembed_test_genai_arena_info(
            raw,
            &mut arena_device,
            &mut token_placement,
            &mut token_ids,
            &mut hidden,
            &mut owns_tokens,
            &mut owns_hidden,
            &mut hidden_used_arena,
        )
    };
    assert_eq!(
        st,
        turboembed_status::TURBOEMBED_OK,
        "arena_info: {}",
        engine.last_error()
    );
    ArenaProof {
        arena_device,
        token_placement,
        token_ids,
        hidden,
        owns_tokens: owns_tokens != 0,
        owns_hidden: owns_hidden != 0,
        hidden_used_arena: hidden_used_arena != 0,
        token_ze: None,
        result_ze: None,
        allocs_after_warmup: 0,
    }
}

fn ze_place(ptr: *const c_void) -> Option<u32> {
    if ptr.is_null() {
        return None;
    }
    let mut place = 0u32;
    let st = unsafe { turbo_buffer_ze_query(ptr, &mut place) };
    if st == TURBO_BUFFER_OK {
        Some(place)
    } else {
        None
    }
}

fn prove_steady_state_zero_allocs(engine: &Engine, alias: &str, text: &str) -> u64 {
    let opts = EmbedOptions {
        pooling: Pooling::Mean,
        normalize: Some(true),
        ..Default::default()
    };
    let warm = engine
        .embed_one(alias, text, &opts)
        .expect("warmup embed for arena reuse");
    drop(warm);
    unsafe { turbo_buffer_alloc_counter_reset() };
    let again = engine
        .embed_one(alias, text, &opts)
        .expect("steady-state embed");
    let allocs = unsafe { turbo_buffer_alloc_counter() };
    assert_eq!(
        allocs, 0,
        "steady-state GenAI embed must rent token/result slabs (allocs/forward==0); \
         got {allocs} — GenAI is still private-allocating"
    );
    assert_eq!(again.dim(), 384);
    allocs
}

fn forbid_mapped(maps: &str, needle: &str) {
    assert!(
        !maps
            .to_ascii_lowercase()
            .contains(&needle.to_ascii_lowercase()),
        "/proc/self/maps must not contain {needle}"
    );
}

#[test]
fn minilm_text_embedding_pipeline_on_gpu() {
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

    let engine = Engine::create(Device::OpenVinoGpu).unwrap_or_else(|e| {
        panic!(
            "turboembed_engine_create(OPENVINO_GPU) failed: {e:?} — \
             GPU plugin required, CPU is not success"
        );
    });
    assert_eq!(Device::OpenVinoGpu.as_str(), "openvino-gpu");

    engine.load_model(ALIAS).unwrap_or_else(|e| {
        panic!("load_model({ALIAS}) via TextEmbeddingPipeline GPU failed: {e:?}");
    });

    let models = engine.list_models().expect("list_models");
    let info = models.get(0).expect("minilm row");
    assert_eq!(info.alias, ALIAS);
    assert_eq!(
        info.device,
        Device::OpenVinoGpu,
        "list_models device must be GPU"
    );
    assert!(info.ready);
    assert_eq!(info.dim, 384, "minilm dim");

    let opts = EmbedOptions {
        pooling: Pooling::Mean,
        normalize: Some(true),
        ..Default::default()
    };
    let one = engine
        .embed_one(ALIAS, TEXT, &opts)
        .unwrap_or_else(|e| panic!("embed_one minilm on GPU failed: {e:?}"));
    assert_ne!(one.dim(), 8, "FAKE: minilm on GPU returned dim=8 (FNV mock)");
    assert_eq!(one.dim(), 384);
    assert_eq!(one.count(), 1);
    assert_eq!(one.values().len(), 384);
    assert_eq!(one.packed().len(), 384 * 4);

    let live = one.values();
    let result_ptr = live.as_ptr().cast::<c_void>();
    let mut proof = arena_info(&engine);
    assert_eq!(
        proof.arena_device, TURBO_BUFFER_DEVICE_ZE,
        "GPU GenAI must rent a ZE arena, not CPU"
    );
    assert_eq!(
        proof.token_placement, TURBO_BUFFER_PLACE_SHARED,
        "GPU tokens must be ZE SHARED, not HOST"
    );
    assert!(proof.owns_tokens, "token ids must be arena-owned USM");
    assert!(proof.owns_hidden, "hidden scratch must be arena-owned USM");
    proof.token_ze = ze_place(proof.token_ids);
    proof.result_ze = ze_place(result_ptr);
    assert_eq!(
        proof.token_ze,
        Some(TURBO_BUFFER_PLACE_SHARED),
        "ze_query(token ids) must be SHARED; got {:?}",
        proof.token_ze
    );
    assert_eq!(
        proof.result_ze,
        Some(TURBO_BUFFER_PLACE_SHARED),
        "ze_query(result) must be SHARED; got {:?}",
        proof.result_ze
    );
    drop(one);
    proof.allocs_after_warmup = prove_steady_state_zero_allocs(&engine, ALIAS, TEXT);

    let one = engine
        .embed_one(ALIAS, TEXT, &opts)
        .unwrap_or_else(|e| panic!("embed_one minilm on GPU failed: {e:?}"));
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
        "cosine vs intel golden {cosine_intel} < {COSINE_FLOOR}"
    );
    assert!(
        cosine_nvidia >= COSINE_FLOOR,
        "cosine vs nvidia golden {cosine_nvidia} < {COSINE_FLOOR}"
    );

    let maps = maps_blob();
    require_mapped(&maps, "libopenvino_genai");
    require_mapped(&maps, "libopenvino_intel_gpu_plugin");
    require_mapped(&maps, "libopenvino_tokenizers");
    forbid_mapped(&maps, "libpython");

    let gpu_name = Command::new("sycl-ls")
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8(o.stdout)
                .ok()
                .and_then(|s| s.lines().find(|l| l.contains("gpu")).map(str::to_string))
        })
        .unwrap_or_default();

    let receipt = serde_json::json!({
        "schema_version": 1,
        "alias": ALIAS,
        "device": "GPU",
        "pipeline": "ov::genai::TextEmbeddingPipeline",
        "abi": "turboembed.h",
        "abi_version": 1,
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
            "libopenvino_genai": true,
            "libopenvino_intel_gpu_plugin": true,
            "libpython": false,
        },
        "gpu": gpu_name,
        "arena": {
            "device": "ZE",
            "token_placement": "SHARED",
            "result_placement": "SHARED",
            "ze_query_tokens": "SHARED",
            "ze_query_result": "SHARED",
            "owns_tokens": proof.owns_tokens,
            "owns_hidden": proof.owns_hidden,
            "hidden_set_tensor": proof.hidden_used_arena,
            "allocs_after_warmup": proof.allocs_after_warmup,
            "tokenizer_encode": "ov::genai::Tokenizer.encode still private-allocs; API has no caller buffer"
        },
        "note": "SOLIDIFY (4) Machine B. Token/hidden/result rows rented from the ZE arena as SHARED. InferRequest.set_tensor wraps USM; embed_documents is not on the hot path. GPU create without ZE SHARED fails loud. See intel-minilm-cpu.json for HOST USM.",
    });
    let receipt_dir = root.join("testdata/receipts/turboembed");
    fs::create_dir_all(&receipt_dir).expect("receipts dir");
    let receipt_path = receipt_dir.join("intel-minilm.json");
    fs::write(
        &receipt_path,
        serde_json::to_string_pretty(&receipt).unwrap() + "\n",
    )
    .expect("write receipt");
    assert!(receipt_path.is_file());
}

#[test]
fn minilm_c_abi_embed_one_on_gpu() {
    let root = workspace_root();
    let model_dir = root.join("models/ov/minilm");
    assert!(
        model_dir.join("openvino_tokenizer.xml").is_file(),
        "missing tokenizer IR at {}",
        model_dir.display()
    );

    unsafe {
        let mut engine: *mut turboembed_engine = std::ptr::null_mut();
        let st = turboembed_engine_create(
            turboembed_device::TURBOEMBED_DEVICE_OPENVINO_GPU,
            std::ptr::null(),
            &mut engine,
        );
        assert_eq!(
            st,
            turboembed_status::TURBOEMBED_OK,
            "C ABI create GPU: {}",
            std::ffi::CStr::from_ptr(turboembed_last_error(std::ptr::null())).to_string_lossy()
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

        let text = TEXT.as_bytes();
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
        let intel = golden_vector(&root.join("testdata/e2e/goldens/intel/minilm.json"));
        let c = cosine(live, &intel);
        assert!(
            c >= COSINE_FLOOR,
            "C ABI cosine vs intel golden {c} < {COSINE_FLOOR}"
        );

        turboembed_embed_result_free(out);
        turboembed_engine_destroy(engine);
    }
}

fn run_minilm_on(device: Device, ov_name: &str, plugin_needle: &str) -> (Vec<f32>, f32, f32) {
    let root = workspace_root();
    let intel_golden = golden_vector(&root.join("testdata/e2e/goldens/intel/minilm.json"));
    let nvidia_golden = golden_vector(&root.join("testdata/e2e/goldens/nvidia/minilm.json"));

    let engine = Engine::create(device).unwrap_or_else(|e| {
        panic!("create({ov_name}) failed: {e:?}");
    });
    engine
        .load_model(ALIAS)
        .unwrap_or_else(|e| panic!("load_model minilm on {ov_name}: {e:?}"));

    let info = engine.list_models().expect("list").get(0).expect("row");
    assert_eq!(info.alias, ALIAS);
    assert_eq!(
        info.device, device,
        "list_models device must be the requested {ov_name} engine (no silent swap)"
    );
    assert!(info.ready);
    assert_eq!(info.dim, 384);

    let opts = EmbedOptions {
        pooling: Pooling::Mean,
        normalize: Some(true),
        ..Default::default()
    };
    let one = engine
        .embed_one(ALIAS, TEXT, &opts)
        .unwrap_or_else(|e| panic!("embed_one on {ov_name}: {e:?}"));
    assert_ne!(
        one.dim(),
        8,
        "FAKE: minilm on {ov_name} returned dim=8 (FNV mock)"
    );
    assert_eq!(one.dim(), 384);
    let live = one.values().to_vec();
    let cosine_intel = cosine(&live, &intel_golden);
    let cosine_nvidia = cosine(&live, &nvidia_golden);
    assert!(
        cosine_intel >= COSINE_FLOOR,
        "{ov_name} cosine vs intel {cosine_intel} < {COSINE_FLOOR}"
    );
    assert!(
        cosine_nvidia >= COSINE_FLOOR,
        "{ov_name} cosine vs nvidia {cosine_nvidia} < {COSINE_FLOOR}"
    );

    let maps = maps_blob();
    require_mapped(&maps, "libopenvino_genai");
    require_mapped(&maps, plugin_needle);
    require_mapped(&maps, "libopenvino_tokenizers");
    forbid_mapped(&maps, "libpython");
    (live, cosine_intel, cosine_nvidia)
}

#[test]
fn minilm_text_embedding_pipeline_on_cpu() {
    let root = workspace_root();
    let model_dir = root.join("models/ov/minilm");
    assert!(
        model_dir.join("openvino_model.xml").is_file(),
        "missing {}",
        model_dir.display()
    );

    let (_live, cosine_intel, cosine_nvidia) =
        run_minilm_on(Device::OpenVinoCpu, "CPU", "libopenvino_intel_cpu_plugin");
    assert_eq!(Device::OpenVinoCpu.as_str(), "openvino-cpu");

    let cpu_engine = Engine::create(Device::OpenVinoCpu).expect("CPU arena engine");
    cpu_engine
        .load_model(ALIAS)
        .expect("CPU arena load minilm");
    let cpu_one = cpu_engine
        .embed_one(
            ALIAS,
            TEXT,
            &EmbedOptions {
                pooling: Pooling::Mean,
                normalize: Some(true),
                ..Default::default()
            },
        )
        .expect("CPU arena embed");
    let mut cpu_proof = arena_info(&cpu_engine);
    assert!(
        cpu_proof.owns_tokens,
        "CPU token ids must be arena-owned (ZE HOST or CPU HOST)"
    );
    assert_eq!(
        cpu_proof.token_placement, TURBO_BUFFER_PLACE_HOST,
        "CPU tokens must be HOST, not SHARED pretending to be GPU"
    );
    cpu_proof.token_ze = ze_place(cpu_proof.token_ids);
    cpu_proof.result_ze = ze_place(cpu_one.values().as_ptr().cast());
    if cpu_proof.arena_device == TURBO_BUFFER_DEVICE_ZE {
        assert_eq!(
            cpu_proof.token_ze,
            Some(TURBO_BUFFER_PLACE_HOST),
            "CPU ZE tokens must query HOST"
        );
    }
    drop(cpu_one);
    cpu_proof.allocs_after_warmup = prove_steady_state_zero_allocs(&cpu_engine, ALIAS, TEXT);

    /* ABI device=CPU (not only OPENVINO_CPU) also compiles "CPU". */
    let engine = Engine::create(Device::Cpu).expect("create Device::Cpu");
    engine
        .load_model(ALIAS)
        .unwrap_or_else(|e| panic!("Device::Cpu load_model minilm: {e:?}"));
    let one = engine
        .embed_one(
            ALIAS,
            TEXT,
            &EmbedOptions {
                pooling: Pooling::Mean,
                normalize: Some(true),
                ..Default::default()
            },
        )
        .expect("Device::Cpu embed_one");
    assert_ne!(one.dim(), 8, "FAKE: minilm on CPU returned dim=8 (FNV mock)");
    assert_eq!(one.dim(), 384);
    let intel = golden_vector(&root.join("testdata/e2e/goldens/intel/minilm.json"));
    let c = cosine(one.values(), &intel);
    assert!(c >= COSINE_FLOOR, "Device::Cpu cosine {c} < {COSINE_FLOOR}");

    let receipt = serde_json::json!({
        "schema_version": 1,
        "alias": ALIAS,
        "device": "CPU",
        "pipeline": "ov::genai::TextEmbeddingPipeline",
        "abi": "turboembed.h",
        "abi_version": 1,
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
            "libopenvino_genai": true,
            "libopenvino_intel_cpu_plugin": true,
            "libpython": false,
        },
        "arena": {
            "device": if cpu_proof.arena_device == TURBO_BUFFER_DEVICE_ZE {
                "ZE"
            } else {
                "CPU"
            },
            "token_placement": "HOST",
            "owns_tokens": cpu_proof.owns_tokens,
            "allocs_after_warmup": cpu_proof.allocs_after_warmup,
            "ze_query_tokens": cpu_proof.token_ze.map(|p| {
                if p == TURBO_BUFFER_PLACE_HOST {
                    "HOST"
                } else {
                    "OTHER"
                }
            }),
        },
        "note": "SOLIDIFY (4) Machine B CPU. Same IR as GPU. Tokens/results rented HOST (ZE USM when L0 is present). GPU requests still fail-loud if the GPU plugin or ZE SHARED is missing.",
    });
    let receipt_dir = root.join("testdata/receipts/turboembed");
    fs::create_dir_all(&receipt_dir).expect("receipts dir");
    fs::write(
        receipt_dir.join("intel-minilm-cpu.json"),
        serde_json::to_string_pretty(&receipt).unwrap() + "\n",
    )
    .expect("write cpu receipt");
}

#[test]
fn minilm_c_abi_embed_one_on_cpu() {
    let root = workspace_root();
    assert!(
        root.join("models/ov/minilm/openvino_tokenizer.xml")
            .is_file(),
        "missing tokenizer IR"
    );

    unsafe {
        let mut engine: *mut turboembed_engine = std::ptr::null_mut();
        let st = turboembed_engine_create(
            turboembed_device::TURBOEMBED_DEVICE_OPENVINO_CPU,
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

        let text = TEXT.as_bytes();
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
        let intel = golden_vector(&root.join("testdata/e2e/goldens/intel/minilm.json"));
        let c = cosine(live, &intel);
        assert!(
            c >= COSINE_FLOOR,
            "C ABI CPU cosine vs intel {c} < {COSINE_FLOOR}"
        );

        turboembed_embed_result_free(out);
        turboembed_engine_destroy(engine);
    }
}

/// NPU request must fail loud on this host (no Intel NPU / no plugin).
/// Never CPU. Never 8-d FNV mock. Writes the honest defer receipt.
#[test]
fn npu_request_never_silently_uses_cpu_or_mock() {
    let root = workspace_root();
    match Engine::create(Device::OpenVinoNpu) {
        Ok(engine) => {
            // Only valid if this host actually listed NPU. Then load must
            // stay on NPU and produce dim 384 — never FNV8 / CPU.
            engine.load_model(ALIAS).unwrap_or_else(|e| {
                panic!("NPU create succeeded so load must stay on NPU, not fall back: {e:?}");
            });
            let info = engine.list_models().expect("list").get(0).expect("row");
            assert_eq!(
                info.device,
                Device::OpenVinoNpu,
                "NPU request compiled a non-NPU pipeline"
            );
            assert_ne!(info.dim, 8, "FAKE: NPU path returned dim=8 (FNV mock)");
            assert_eq!(info.dim, 384);
            panic!(
                "NPU create+load succeeded on this host — refresh \
                 testdata/receipts/turboembed/intel-minilm-npu.json with a \
                 live cosine receipt instead of the defer file"
            );
        }
        Err(Error::UnsupportedDevice(msg)) | Err(Error::Unavailable(msg)) => {
            let lower = msg.to_ascii_lowercase();
            assert!(
                lower.contains("npu"),
                "missing-NPU error must name NPU, got {msg}"
            );
            assert!(
                lower.contains("fallback") || lower.contains("refus"),
                "missing-NPU error must say CPU is not a fallback, got {msg}"
            );
            assert!(
                !msg.contains("dim=8") && !lower.contains("fnv"),
                "NPU fail must not be the FNV mock, got {msg}"
            );

            let cpu = fs::read_to_string("/proc/cpuinfo")
                .ok()
                .and_then(|s| {
                    s.lines()
                        .find(|l| l.starts_with("model name"))
                        .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
                })
                .unwrap_or_default();
            let host = Command::new("hostname")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            let plugin = Path::new("/work/opt/openvino_genai/runtime/lib/intel64")
                .join("libopenvino_intel_npu_plugin.so")
                .is_file();
            let listed = msg
                .split("listed: [")
                .nth(1)
                .and_then(|rest| rest.split(']').next())
                .unwrap_or("")
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>();

            let receipt = serde_json::json!({
                "schema_version": 1,
                "alias": ALIAS,
                "device": "NPU",
                "wired": false,
                "pass": false,
                "pipeline": "ov::genai::TextEmbeddingPipeline",
                "abi": "turboembed.h",
                "abi_version": 1,
                "host": host,
                "cpu": cpu,
                "npu_plugin": plugin,
                "create": "UNSUPPORTED_DEVICE",
                "error": msg,
                "available_devices": listed,
                "live_probe": {
                    "listed_npu": false,
                    "constructor": "TextEmbeddingPipeline(models/ov/minilm, \"NPU\")",
                    "exception": "Device with \"NPU\" name is not registered in the OpenVINO Runtime",
                    "source": "standalone C++ ov::Core + TextEmbeddingPipeline probe on Machine B; create() uses the same Core list via require_ov_device",
                },
                "blocker": "Machine B is AMD Ryzen 9 9950X + Intel Battlemage G31 dGPU. No Intel NPU silicon, no intel-npu/accel node, no libopenvino_intel_npu_plugin.so. ov::Core lists CPU GPU only.",
                "note": "Honest defer. Device::OpenVinoNpu create fails loud and never compiles CPU or the 8-d FNV mock. Do not treat this file as a passing MiniLM receipt.",
                "sha": {
                    "git": git_head(&root),
                },
            });
            let receipt_dir = root.join("testdata/receipts/turboembed");
            fs::create_dir_all(&receipt_dir).expect("receipts dir");
            fs::write(
                receipt_dir.join("intel-npu.json"),
                serde_json::to_string_pretty(&receipt).unwrap() + "\n",
            )
            .expect("write npu defer receipt");
        }
        Err(other) => panic!(
            "NPU create must fail loud (UNSUPPORTED/UNAVAILABLE) or succeed on a real NPU, got {other:?}"
        ),
    }
}

/// GPU request must fail loud when the plugin is missing — never compile CPU.
/// When the plugin is present, create+load must stay on `"GPU"`.
#[test]
fn gpu_request_never_silently_uses_cpu() {
    match Engine::create(Device::OpenVinoGpu) {
        Ok(engine) => {
            engine
                .load_model(ALIAS)
                .unwrap_or_else(|e| panic!("GPU load must stay on GPU, not fall back: {e:?}"));
            let info = engine.list_models().expect("list").get(0).expect("row");
            assert_eq!(
                info.device,
                Device::OpenVinoGpu,
                "GPU request compiled a non-GPU pipeline"
            );
        }
        Err(Error::UnsupportedDevice(msg)) | Err(Error::Unavailable(msg)) => {
            let lower = msg.to_ascii_lowercase();
            assert!(
                lower.contains("gpu"),
                "missing-GPU error must name GPU, got {msg}"
            );
            assert!(
                lower.contains("fallback") || lower.contains("refus") || lower.contains("cpu"),
                "missing-GPU error must say CPU is not a fallback, got {msg}"
            );
        }
        Err(other) => panic!("GPU create must succeed on GPU or fail loud, got {other:?}"),
    }
}
