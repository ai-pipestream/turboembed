//! Live proof: Rust → `turboembed_embed("minilm")` → FP MiniLM mean+L2
//! on Apple Metal. No mock, no Python, no BERT CLS pooler.
//!
//! ```bash
//! cargo test -p turboembed --features mlx-live -- --ignored --nocapture apple_minilm
//! ```

#![cfg(all(target_os = "macos", feature = "mlx-live"))]

use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use turboembed::ffi::{
    turboembed_device, turboembed_embed, turboembed_embed_options, turboembed_embed_result,
    turboembed_embed_result_free, turboembed_engine, turboembed_engine_create,
    turboembed_engine_destroy, turboembed_last_error, turboembed_load_model, turboembed_pooling,
    turboembed_status, turboembed_str,
};
use turboembed::{Device, EmbedOptions, Engine, Error, Pooling};

const MINILM_DIM: usize = 384;
const NVIDIA_FLOOR: f32 = 0.97;
const APPLE_FLOOR: f32 = 0.99;
const BATCH: usize = 16;

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

#[derive(Debug, Serialize)]
struct Receipt {
    schema_version: u32,
    abi: &'static str,
    abi_version: u32,
    host: String,
    arch: &'static str,
    chip: String,
    device: String,
    metal: bool,
    provider: &'static str,
    alias: &'static str,
    pooling: &'static str,
    normalize: bool,
    dim: u32,
    n: usize,
    cosine_vs_nvidia: Score,
    cosine_vs_apple: Option<Score>,
    l2_ok: bool,
    fake_rejected: bool,
    weights: String,
    captured_at_unix: u64,
}

#[derive(Debug, Serialize)]
struct Score {
    min: f32,
    mean: f32,
    floor: f32,
    worst_id: String,
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
        let gold = by_id
            .get(id)
            .copied()
            .or_else(|| by_text.get(id).copied());
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
                "FAKE / BERT pooler suspected: {label} {} cosine={c:.4} (pooler scored ≈ 0 vs nvidia)",
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

fn set_workspace_env() {
    // Safety: test-only, before any engine create.
    unsafe { std::env::set_var("INFERSTREAM_ROOT", workspace_root()) };
}

#[test]
#[ignore = "needs macOS Metal + models/mlx/minilm"]
fn metal_create_lists_minilm_not_only_mock() {
    set_workspace_env();
    let engine = Engine::create(Device::Metal).expect("Metal engine — must not fall back to stub");
    let models = engine.list_models().expect("list");
    let minilm = models
        .iter()
        .find(|m| m.alias == "minilm")
        .expect("FAKE: Metal engine listed only mock-embed; minilm weights / MLX path missing");
    assert_eq!(minilm.device, Device::Metal);
    assert_eq!(minilm.dim, MINILM_DIM as u32);
    assert!(
        !models.iter().all(|m| m.alias == "mock-embed"),
        "FAKE: catalog is mock-only"
    );

    let auto = Engine::create(Device::Auto).expect("AUTO is host GPU (Metal), not CPU");
    let auto_models = auto.list_models().expect("auto list");
    assert!(
        auto_models
            .iter()
            .any(|m| m.alias == "minilm" && m.device == Device::Metal),
        "AUTO must resolve to Metal MiniLM, never a CPU fallback"
    );
}

#[test]
#[ignore = "needs macOS Metal + models/mlx/minilm + e2e goldens"]
fn apple_minilm_metal_cosine_vs_goldens() {
    set_workspace_env();
    let root = workspace_root();
    let weights = root.join("models/mlx/minilm");
    assert!(
        weights.join("model.safetensors").is_file(),
        "FAKE: {} missing — run `make fetch-mlx ALIASES=minilm`",
        weights.display()
    );
    let nvidia_path = root.join("testdata/e2e/goldens/nvidia/minilm.json");
    let apple_path = root.join("testdata/e2e/goldens/apple/minilm.json");
    let nvidia = load_dump(&nvidia_path);
    assert_eq!(nvidia.alias, "minilm");
    assert_eq!(nvidia.pooling, "mean");
    assert!(nvidia.normalize);
    assert_eq!(nvidia.dim, MINILM_DIM as u32);
    assert!(!nvidia.items.is_empty());

    // Raw C ABI — the required proof path.
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

        let alias = b"minilm";
        let st = turboembed_load_model(raw, alias.as_ptr().cast(), alias.len());
        assert_eq!(
            st,
            turboembed_status::TURBOEMBED_OK,
            "load minilm: {}",
            std::ffi::CStr::from_ptr(turboembed_last_error(raw)).to_string_lossy()
        );

        let hello = b"hello world";
        let view = turboembed_str {
            ptr: hello.as_ptr().cast(),
            len: hello.len(),
        };
        let opts = turboembed_embed_options {
            pooling: turboembed_pooling::TURBOEMBED_POOLING_MEAN,
            normalize: 1,
            truncate_to: 256,
            output_format: turboembed::ffi::turboembed_output_format::TURBOEMBED_OUTPUT_TYPED,
        };
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
        assert_eq!(
            st,
            turboembed_status::TURBOEMBED_OK,
            "turboembed_embed(minilm): {}",
            std::ffi::CStr::from_ptr(turboembed_last_error(raw)).to_string_lossy()
        );
        assert!(!out.is_null());
        let dim = (*out).dim as usize;
        let count = (*out).count as usize;
        assert_eq!(count, 1);
        let row = std::slice::from_raw_parts((*out).values, dim);
        fail_if_fake(dim, row, "raw turboembed_embed(minilm) hello world");
        let hello_vec = row.to_vec();
        turboembed_embed_result_free(out);
        turboembed_engine_destroy(raw);

        let nv_hello = nvidia
            .items
            .iter()
            .find(|i| i.text == "hello world")
            .or_else(|| nvidia.items.first())
            .expect("nvidia golden hello world");
        let c = cosine(&hello_vec, &nv_hello.vector);
        eprintln!(
            "raw FFI hello-world vs nvidia {}: cosine={c:.6}",
            nv_hello.id
        );
        assert!(
            c >= NVIDIA_FLOOR,
            "FAKE: hello-world cosine {c:.6} < {NVIDIA_FLOOR}"
        );
        let _ = opts;
    }

    let engine = Engine::create(Device::Metal).expect("Metal engine");
    engine
        .load_model("minilm")
        .unwrap_or_else(|e| panic!("load minilm: {e}"));
    let mock_err = engine.embed_one(
        "minilm",
        "hello world",
        &EmbedOptions {
            pooling: Pooling::Mean,
            normalize: Some(true),
            truncate_to: Some(256),
            ..EmbedOptions::default()
        },
    );
    let one = match mock_err {
        Ok(v) => v,
        Err(Error::NotImplemented(_)) => {
            panic!("FAKE: minilm is still NOT_IMPLEMENTED — C++ stub leaked into the macOS link");
        }
        Err(e) => panic!("embed minilm: {e}"),
    };
    fail_if_fake(one.dim(), one.values(), "safe Engine::embed_one(minilm)");

    let mut live: Vec<(String, Vec<f32>)> = Vec::with_capacity(nvidia.items.len());
    for chunk in nvidia.items.chunks(BATCH) {
        let texts: Vec<&str> = chunk.iter().map(|i| i.text.as_str()).collect();
        let batch = engine
            .embed(
                "minilm",
                &texts,
                &EmbedOptions {
                    pooling: Pooling::Mean,
                    normalize: Some(true),
                    truncate_to: Some(256),
                    ..EmbedOptions::default()
                },
            )
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

    let vs_apple = if apple_path.is_file() {
        let apple = load_dump(&apple_path);
        assert_eq!(apple.dim, MINILM_DIM as u32);
        Some(score_against(&got, &apple, APPLE_FLOOR, "apple-mlx↔apple-golden"))
    } else {
        None
    };

    let receipt = Receipt {
        schema_version: 1,
        abi: "include/turboembed.h",
        abi_version: 1,
        host: hostname(),
        arch: "apple",
        chip: chip_name(),
        device: "Device(gpu, 0)".into(),
        metal: true,
        provider: "mlx",
        alias: "minilm",
        pooling: "mean",
        normalize: true,
        dim: MINILM_DIM as u32,
        n: live.len(),
        cosine_vs_nvidia: vs_nvidia,
        cosine_vs_apple: vs_apple,
        l2_ok: true,
        fake_rejected: true,
        weights: "models/mlx/minilm".into(),
        captured_at_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    let dest = root.join("testdata/receipts/turboembed/apple-minilm.json");
    fs::create_dir_all(dest.parent().unwrap()).expect("receipts dir");
    let mut body = serde_json::to_string_pretty(&receipt).expect("receipt json");
    body.push('\n');
    fs::write(&dest, body).unwrap_or_else(|e| panic!("write {}: {e}", dest.display()));
    eprintln!("wrote {}", dest.display());
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
