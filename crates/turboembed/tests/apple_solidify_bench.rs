//! Machine C FINAL SOLIDIFY bench — TurboEmbed Metal + combined receipt.
//!
//! Measures p50/p99 after warmup. Gates: allocs/forward==0, MiniLM
//! goldens in band, turbo_buffer Metal SHARED (no CPU fallback). Merges
//! the TurboRerank slice from `BENCH_RERANK_JSON` and writes
//! `testdata/receipts/bench/machine-c-metal.json`.
//!
//! ```bash
//! make bench-machine-c
//! ```

#![cfg(all(target_os = "macos", feature = "mlx-live"))]

use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use turboembed::ffi::{
    turboembed_device, turboembed_embed, turboembed_embed_options, turboembed_embed_result,
    turboembed_embed_result_free, turboembed_engine, turboembed_engine_create,
    turboembed_engine_destroy, turboembed_last_error, turboembed_list_models,
    turboembed_load_model, turboembed_model_info, turboembed_model_list_free, turboembed_pooling,
    turboembed_status, turboembed_str,
};
use turboembed::{Device, EmbedOptions, Engine, Pooling};

unsafe extern "C" {
    fn turbo_buffer_alloc_counter() -> u64;
    fn turbo_buffer_alloc_counter_reset();
    fn turbo_buffer_metal_owns(ptr: *const std::ffi::c_void) -> i32;
    fn turbo_buffer_metal_lookup(
        ptr: *const std::ffi::c_void,
        out_native: *mut *mut std::ffi::c_void,
        out_offset: *mut usize,
    ) -> i32;
}

const MINILM_DIM: usize = 384;
const NVIDIA_FLOOR: f32 = 0.97;
const APPLE_FLOOR: f32 = 0.99;
const HELLO: &[u8] = b"hello world";
const DEFAULT_WARMUP: usize = 32;
const DEFAULT_ITERS: usize = 200;

#[derive(Debug, Deserialize)]
struct GoldenDump {
    alias: String,
    pooling: String,
    normalize: bool,
    dim: u32,
    items: Vec<GoldenItem>,
}

#[derive(Debug, Deserialize)]
struct GoldenItem {
    id: String,
    text: String,
    vector: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct RerankSlice {
    alias: String,
    device: String,
    memory_path: String,
    cpu_fallback: bool,
    auto_resolves_metal: bool,
    metal_owns_tokens: bool,
    arena_owns_tokens: bool,
    metal_lookup: bool,
    gpu: String,
    warmup: u32,
    iters: u32,
    p50_us: u64,
    p99_us: u64,
    mean_us: u64,
    allocs_per_forward: u64,
    logits: Vec<f32>,
    max_abs_logit_err: f64,
    cosine_vs_golden: f64,
    berlin_in_band: bool,
    pass: bool,
}

#[derive(Debug, Serialize)]
struct Score {
    min: f32,
    mean: f32,
    floor: f32,
    worst_id: String,
}

#[derive(Debug, Serialize)]
struct CombinedReceipt {
    schema_version: u32,
    kind: &'static str,
    machine: &'static str,
    device: &'static str,
    memory_path: &'static str,
    gpu: String,
    host: String,
    arch: &'static str,
    chip: String,
    git_sha: String,
    command: &'static str,
    captured_at_unix: u64,
    pass: bool,
    gates: Gates,
    turboembed: EmbedSlice,
    turborerank: RerankOut,
    note: &'static str,
}

#[derive(Debug, Serialize)]
struct Gates {
    p50_p99_measured: bool,
    allocs_per_forward: u64,
    memory_path: &'static str,
    cpu_fallback: bool,
    berlin_in_band: bool,
    embed_goldens_in_band: bool,
}

#[derive(Debug, Serialize)]
struct EmbedSlice {
    alias: &'static str,
    device: &'static str,
    memory_path: &'static str,
    cpu_fallback: bool,
    auto_resolves_metal: bool,
    metal_owns_result: bool,
    warmup: u32,
    iters: u32,
    p50_us: u64,
    p99_us: u64,
    mean_us: u64,
    allocs_per_forward: u64,
    dim: u32,
    n_goldens: usize,
    cosine_vs_nvidia: Score,
    cosine_vs_apple: Score,
    goldens_in_band: bool,
}

#[derive(Debug, Serialize)]
struct RerankOut {
    alias: String,
    device: String,
    memory_path: String,
    cpu_fallback: bool,
    auto_resolves_metal: bool,
    metal_owns_tokens: bool,
    arena_owns_tokens: bool,
    warmup: u32,
    iters: u32,
    p50_us: u64,
    p99_us: u64,
    mean_us: u64,
    allocs_per_forward: u64,
    logits: Vec<f32>,
    max_abs_logit_err: f64,
    cosine_vs_golden: f64,
    berlin_in_band: bool,
}

fn workspace_root() -> PathBuf {
    let from_env = option_env!("INFERSTREAM_ROOT").map(PathBuf::from);
    if let Some(root) = from_env {
        if root.join("include/turboembed.h").is_file() {
            return root;
        }
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.join("../..").canonicalize().expect("workspace")
}

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|n| *n > 0 && *n <= 100_000)
        .unwrap_or(fallback)
}

fn load_dump(path: &Path) -> GoldenDump {
    let text = fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing golden {}: {e}", path.display());
    });
    serde_json::from_str(&text).unwrap_or_else(|e| {
        panic!("golden {}: {e}", path.display());
    })
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

fn l2(row: &[f32]) -> f32 {
    row.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn fail_if_fake(dim: usize, values: &[f32], where_: &str) {
    assert_ne!(
        dim, 8,
        "FAKE: {where_} returned mock-embed dim=8. turboembed_embed(minilm) must be 384-d Metal MiniLM."
    );
    assert_eq!(
        dim, MINILM_DIM,
        "FAKE: {where_} dim={dim}, MiniLM-L6 is 384. BERT pooler / stub / wrong weights."
    );
    assert_eq!(values.len(), dim, "{where_}: ragged row");
    assert!(
        !values.iter().all(|v| *v == 0.0),
        "FAKE: {where_} is an all-zero vector"
    );
    let norm = l2(values);
    assert!(
        (norm - 1.0).abs() < 1e-2,
        "FAKE: {where_} L2={norm} (mean+L2 MiniLM is ~1.0)"
    );
}

fn score_against(got: &[(&str, &[f32])], dump: &GoldenDump, floor: f32, label: &str) -> Score {
    let by_id: std::collections::HashMap<&str, &GoldenItem> =
        dump.items.iter().map(|i| (i.id.as_str(), i)).collect();
    let by_text: std::collections::HashMap<&str, &GoldenItem> =
        dump.items.iter().map(|i| (i.text.as_str(), i)).collect();
    let mut min = f32::MAX;
    let mut sum = 0.0f32;
    let mut n = 0usize;
    let mut worst = String::new();
    for (id, row) in got {
        let gold = by_id.get(id).copied().or_else(|| by_text.get(id).copied());
        let Some(gold) = gold else { continue };
        if gold.vector.len() != row.len() {
            panic!(
                "FAKE: {label} {} dim {} vs live {}",
                gold.id,
                gold.vector.len(),
                row.len()
            );
        }
        let c = cosine(row, &gold.vector);
        if c < min {
            min = c;
            worst = gold.id.clone();
        }
        sum += c;
        n += 1;
        if c < 0.2 {
            panic!(
                "FAKE / BERT pooler suspected: {label} {} cosine={c:.4}",
                gold.id
            );
        }
    }
    assert!(n > 0, "{label}: no overlapping golden texts");
    let mean = sum / n as f32;
    eprintln!("{label}: n={n} min={min:.6} mean={mean:.6} worst={worst} floor={floor}");
    assert!(
        min + f32::EPSILON >= floor,
        "{label} min cosine {min:.6} < {floor} (worst {worst}). Metal MiniLM mean+L2 must hold."
    );
    Score {
        min,
        mean,
        floor,
        worst_id: worst,
    }
}

fn percentile_us(samples_ns: &mut [u128], p: u32) -> u64 {
    assert!(
        !samples_ns.is_empty(),
        "p50/p99: no samples — do not invent"
    );
    samples_ns.sort_unstable();
    let idx = (samples_ns.len() - 1) * p as usize / 100;
    (samples_ns[idx] / 1000) as u64
}

fn mean_us(samples_ns: &[u128]) -> u64 {
    let sum: u128 = samples_ns.iter().copied().sum();
    ((sum / samples_ns.len() as u128) / 1000) as u64
}

fn hostname() -> String {
    std::process::Command::new("scutil")
        .args(["--get", "LocalHostName"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "macos".into())
}

fn chip_name() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "Apple Silicon".into())
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

fn set_workspace_env(root: &Path) {
    // Safety: test-only, before any engine create.
    unsafe { std::env::set_var("INFERSTREAM_ROOT", root) };
}

fn prove_auto_is_metal() {
    let auto = Engine::create(Device::Auto).expect("AUTO is host GPU (Metal), not CPU");
    let models = auto.list_models().expect("auto list");
    assert!(
        models.iter().all(|m| m.alias != "mock-embed"
            && m.device != Device::Cpu
            && m.device != Device::Mock),
        "FAKE: AUTO advertised CPU/mock — Metal must not fall back"
    );
    assert!(
        models.iter().any(|m| m.alias == "minilm"
            && m.device == Device::Metal
            && m.dim == MINILM_DIM as u32),
        "AUTO must resolve to Metal MiniLM dim={MINILM_DIM}"
    );
}

#[test]
#[ignore = "needs macOS Metal + models/mlx/minilm + BENCH_RERANK_JSON"]
fn apple_solidify_bench_writes_machine_c_receipt() {
    let root = workspace_root();
    set_workspace_env(&root);
    let warmup = env_usize("BENCH_WARMUP", DEFAULT_WARMUP);
    let iters = env_usize("BENCH_ITERS", DEFAULT_ITERS);

    let rerank_path = std::env::var("BENCH_RERANK_JSON").unwrap_or_else(|_| {
        root.join("native/turborerank/build/machine-c-rerank-bench.json")
            .to_string_lossy()
            .into_owned()
    });
    let rerank_text = fs::read_to_string(&rerank_path).unwrap_or_else(|e| {
        panic!(
            "TurboRerank slice missing at {rerank_path}: {e}. Run `make bench-machine-c-rerank` first — do not invent p50/p99."
        );
    });
    let rerank: RerankSlice = serde_json::from_str(&rerank_text).unwrap_or_else(|e| {
        panic!("rerank bench JSON {rerank_path}: {e}");
    });
    assert!(
        rerank.pass
            && rerank.berlin_in_band
            && rerank.allocs_per_forward == 0
            && rerank.memory_path == "SHARED"
            && !rerank.cpu_fallback
            && rerank.auto_resolves_metal
            && rerank.metal_owns_tokens
            && rerank.arena_owns_tokens
            && rerank.metal_lookup
            && rerank.device == "METAL"
            && rerank.p50_us > 0
            && rerank.p99_us >= rerank.p50_us,
        "TurboRerank Metal gates failed: {rerank:?}"
    );

    prove_auto_is_metal();

    let weights = root.join("models/mlx/minilm");
    assert!(
        weights.join("model.safetensors").is_file(),
        "FAKE: {} missing — run `make fetch-mlx ALIASES=minilm`",
        weights.display()
    );
    let nvidia = load_dump(&root.join("testdata/e2e/goldens/nvidia/minilm.json"));
    let apple = load_dump(&root.join("testdata/e2e/goldens/apple/minilm.json"));
    assert_eq!(nvidia.alias, "minilm");
    assert_eq!(nvidia.pooling, "mean");
    assert!(nvidia.normalize);
    assert_eq!(nvidia.dim, MINILM_DIM as u32);

    let mut metal_owns_result = false;
    let mut p50_us;
    let mut p99_us;
    let mut mean;
    unsafe {
        let mut raw: *mut turboembed_engine = ptr::null_mut();
        let st = turboembed_engine_create(
            turboembed_device::TURBOEMBED_DEVICE_METAL,
            ptr::null(),
            &mut raw,
        );
        assert_eq!(
            st,
            turboembed_status::TURBOEMBED_OK,
            "create Metal: {:?}",
            std::ffi::CStr::from_ptr(turboembed_last_error(ptr::null()))
        );
        assert!(!raw.is_null());

        let mut infos: *mut turboembed_model_info = ptr::null_mut();
        let mut count = 0usize;
        let st = turboembed_list_models(raw, &mut infos, &mut count);
        assert_eq!(st, turboembed_status::TURBOEMBED_OK);
        let listed = std::slice::from_raw_parts(infos, count);
        assert!(
            listed.iter().all(|m| {
                let alias = std::str::from_utf8(std::slice::from_raw_parts(
                    m.alias.ptr as *const u8,
                    m.alias.len,
                ))
                .unwrap_or("");
                alias != "mock-embed"
                    && m.device != turboembed_device::TURBOEMBED_DEVICE_CPU
                    && m.device != turboembed_device::TURBOEMBED_DEVICE_MOCK
            }),
            "FAKE: Metal engine listed CPU/mock"
        );
        turboembed_model_list_free(infos, count);

        let alias = b"minilm";
        let st = turboembed_load_model(raw, alias.as_ptr().cast(), alias.len());
        assert_eq!(
            st,
            turboembed_status::TURBOEMBED_OK,
            "load minilm: {}",
            std::ffi::CStr::from_ptr(turboembed_last_error(raw)).to_string_lossy()
        );

        let view = turboembed_str {
            ptr: HELLO.as_ptr().cast(),
            len: HELLO.len(),
        };
        let opts = turboembed_embed_options {
            pooling: turboembed_pooling::TURBOEMBED_POOLING_MEAN,
            normalize: 1,
            truncate_to: 256,
            output_format: turboembed::ffi::turboembed_output_format::TURBOEMBED_OUTPUT_TYPED,
        };

        for _ in 0..warmup {
            let mut out: *mut turboembed_embed_result = ptr::null_mut();
            let st = turboembed_embed(
                raw,
                alias.as_ptr().cast(),
                alias.len(),
                &view,
                1,
                &opts,
                &mut out,
            );
            assert_eq!(st, turboembed_status::TURBOEMBED_OK);
            fail_if_fake(
                (*out).dim as usize,
                std::slice::from_raw_parts((*out).values, (*out).dim as usize),
                "warmup embed",
            );
            assert_eq!(
                turbo_buffer_metal_owns((*out).values as *const std::ffi::c_void),
                1,
                "FAKE: result.values is not turbo_buffer Metal SHARED"
            );
            let mut native = std::ptr::null_mut();
            let mut off = 0usize;
            assert_eq!(
                turbo_buffer_metal_lookup(
                    (*out).values as *const std::ffi::c_void,
                    &mut native,
                    &mut off
                ),
                1,
                "FAKE: turbo_buffer_metal_lookup missed result.values"
            );
            metal_owns_result = true;
            turboembed_embed_result_free(out);
        }

        turbo_buffer_alloc_counter_reset();
        let mut out2: *mut turboembed_embed_result = ptr::null_mut();
        let st = turboembed_embed(
            raw,
            alias.as_ptr().cast(),
            alias.len(),
            &view,
            1,
            &opts,
            &mut out2,
        );
        assert_eq!(st, turboembed_status::TURBOEMBED_OK);
        let allocs = turbo_buffer_alloc_counter();
        assert_eq!(
            allocs, 0,
            "FAKE: allocs/forward={allocs} after warmup — Metal SHARED must reuse"
        );
        turboembed_embed_result_free(out2);

        let mut samples = Vec::with_capacity(iters);
        turbo_buffer_alloc_counter_reset();
        for _ in 0..iters {
            let mut out: *mut turboembed_embed_result = ptr::null_mut();
            let t0 = Instant::now();
            let st = turboembed_embed(
                raw,
                alias.as_ptr().cast(),
                alias.len(),
                &view,
                1,
                &opts,
                &mut out,
            );
            let elapsed = t0.elapsed().as_nanos();
            assert_eq!(st, turboembed_status::TURBOEMBED_OK, "timed embed failed");
            samples.push(elapsed);
            turboembed_embed_result_free(out);
        }
        let allocs_loop = turbo_buffer_alloc_counter();
        assert_eq!(
            allocs_loop, 0,
            "FAKE: allocs/forward={allocs_loop} during timed loop"
        );
        p50_us = percentile_us(&mut samples, 50);
        p99_us = percentile_us(&mut samples, 99);
        mean = mean_us(&samples);
        assert!(
            p50_us > 0 && p99_us >= p50_us,
            "FAKE: p50/p99 not measured (p50={p50_us} p99={p99_us})"
        );
        eprintln!(
            "turboembed Metal hello-world: warmup={warmup} iters={iters} p50_us={p50_us} p99_us={p99_us} mean_us={mean}"
        );

        turboembed_engine_destroy(raw);
    }

    let engine = Engine::create(Device::Metal).expect("Metal engine — must not fall back");
    engine
        .load_model("minilm")
        .unwrap_or_else(|e| panic!("load minilm: {e}"));
    let opts = EmbedOptions {
        pooling: Pooling::Mean,
        normalize: Some(true),
        truncate_to: Some(256),
        ..EmbedOptions::default()
    };
    let mut live: Vec<(String, Vec<f32>)> = Vec::with_capacity(nvidia.items.len());
    for chunk in nvidia.items.chunks(16) {
        let texts: Vec<&str> = chunk.iter().map(|i| i.text.as_str()).collect();
        let batch = engine
            .embed("minilm", &texts, &opts)
            .unwrap_or_else(|e| panic!("batch embed: {e}"));
        assert_eq!(batch.dim(), MINILM_DIM);
        assert_eq!(batch.count(), chunk.len());
        for (i, item) in chunk.iter().enumerate() {
            let row = batch.row(i).expect("row");
            fail_if_fake(row.len(), row, &item.id);
            live.push((item.id.clone(), row.to_vec()));
        }
    }
    let got: Vec<(&str, &[f32])> = live
        .iter()
        .map(|(id, v)| (id.as_str(), v.as_slice()))
        .collect();
    let vs_nvidia = score_against(&got, &nvidia, NVIDIA_FLOOR, "apple-mlx↔nvidia");
    let vs_apple = score_against(&got, &apple, APPLE_FLOOR, "apple-mlx↔apple-golden");

    let goldens_in_band =
        vs_nvidia.min + f32::EPSILON >= NVIDIA_FLOOR && vs_apple.min + f32::EPSILON >= APPLE_FLOOR;
    let pass =
        goldens_in_band && metal_owns_result && p50_us > 0 && rerank.pass && rerank.berlin_in_band;

    let receipt = CombinedReceipt {
        schema_version: 1,
        kind: "solidify-bench",
        machine: "Machine C",
        device: "METAL",
        memory_path: "SHARED",
        gpu: rerank.gpu.clone(),
        host: hostname(),
        arch: "apple",
        chip: chip_name(),
        git_sha: git_sha(&root),
        command: "make bench-machine-c",
        captured_at_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        pass,
        gates: Gates {
            p50_p99_measured: true,
            allocs_per_forward: 0,
            memory_path: "SHARED",
            cpu_fallback: false,
            berlin_in_band: rerank.berlin_in_band,
            embed_goldens_in_band: goldens_in_band,
        },
        turboembed: EmbedSlice {
            alias: "minilm",
            device: "METAL",
            memory_path: "SHARED",
            cpu_fallback: false,
            auto_resolves_metal: true,
            metal_owns_result,
            warmup: warmup as u32,
            iters: iters as u32,
            p50_us,
            p99_us,
            mean_us: mean,
            allocs_per_forward: 0,
            dim: MINILM_DIM as u32,
            n_goldens: live.len(),
            cosine_vs_nvidia: vs_nvidia,
            cosine_vs_apple: vs_apple,
            goldens_in_band,
        },
        turborerank: RerankOut {
            alias: rerank.alias,
            device: rerank.device,
            memory_path: rerank.memory_path,
            cpu_fallback: rerank.cpu_fallback,
            auto_resolves_metal: rerank.auto_resolves_metal,
            metal_owns_tokens: rerank.metal_owns_tokens,
            arena_owns_tokens: rerank.arena_owns_tokens,
            warmup: rerank.warmup,
            iters: rerank.iters,
            p50_us: rerank.p50_us,
            p99_us: rerank.p99_us,
            mean_us: rerank.mean_us,
            allocs_per_forward: rerank.allocs_per_forward,
            logits: rerank.logits,
            max_abs_logit_err: rerank.max_abs_logit_err,
            cosine_vs_golden: rerank.cosine_vs_golden,
            berlin_in_band: rerank.berlin_in_band,
        },
        note: "FINAL SOLIDIFY bench on Machine C. Live Metal timings — not copied from a prior receipt. TurboEmbed + TurboRerank rent turbo_buffer SHARED (MTLResourceStorageModeShared). AUTO resolves to METAL. Create without Metal fails loud (make turborerank-tests-nometal). No CPU fallback. allocs/forward==0 after warmup. Berlin atol 2e-3. Embed goldens: vs nvidia ≥0.97, vs apple ≥0.99.",
    };

    let dest = root.join("testdata/receipts/bench/machine-c-metal.json");
    fs::create_dir_all(dest.parent().unwrap()).expect("receipts/bench");
    let mut body = serde_json::to_string_pretty(&receipt).expect("receipt json");
    body.push('\n');
    fs::write(&dest, body).unwrap_or_else(|e| panic!("write {}: {e}", dest.display()));
    eprintln!("wrote {} pass={pass}", dest.display());
    assert!(
        pass,
        "SOLIDIFY bench gates failed — do not keep a failing receipt"
    );
}
