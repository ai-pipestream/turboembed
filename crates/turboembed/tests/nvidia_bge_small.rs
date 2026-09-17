//! Live NVIDIA proof for the second, deliberately different model contract:
//! bge-small-en-v1.5 with CLS pooling (MiniLM is mean pooling). Same ORT
//! CUDA EP + IoBinding + DEVICE pooling path, same fail-loud device policy.
//!
//! Roadmap M4 expands model coverage one contract at a time; this test
//! qualifies CLS pooling and the Xenova bge-small tokenizer against the
//! committed NVIDIA goldens without touching the MiniLM contract.
//!
//! ```bash
//! make fetch-embeddings ALIASES=bge-small
//! export LD_LIBRARY_PATH="$(pwd)/.libs/nvidia/lib:${LD_LIBRARY_PATH:-}"
//! cargo test -p turboembed --features ort-cuda --test nvidia_bge_small -- \
//!   --ignored --nocapture
//! ```

#![cfg(feature = "ort-cuda")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use turboembed::{Device, EmbedOptions, Engine, Error, Pooling};

const ALIAS: &str = "bge-small";
const GOLDEN: &str = "testdata/e2e/goldens/nvidia/bge-small.json";
const RECEIPT: &str = "testdata/receipts/turboembed/nvidia-bge-small.json";
const COSINE_FLOOR: f32 = 0.99;
const SUBSET_PREFIX: &str = "parity:";

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
    let nb: f64 = b.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn load_subset(path: &Path) -> (usize, Vec<(String, String, Vec<f32>)>) {
    let raw = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e} — capture goldens first", path.display()));
    let golden: Value = serde_json::from_str(&raw).expect("golden JSON");
    assert_eq!(golden["alias"], ALIAS, "golden alias");
    let dim = golden["dim"].as_u64().expect("golden dim") as usize;
    let mut subset = Vec::new();
    for item in golden["items"].as_array().expect("items") {
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
        subset.push((id.to_string(), text, vector));
    }
    assert!(!subset.is_empty(), "no {SUBSET_PREFIX}* items in {GOLDEN}");
    (dim, subset)
}

#[test]
#[ignore = "needs bge-small ONNX (make fetch-embeddings ALIASES=bge-small) + CUDA 13 libs + GPU"]
fn bge_small_cls_ort_cuda_matches_golden() {
    let root = workspace_root();
    let model = root.join("models/onnx/bge-small/onnx/model.onnx");
    assert!(
        model.is_file(),
        "{} missing — run `make fetch-embeddings ALIASES=bge-small` (SHA-pinned)",
        model.display()
    );
    let (dim, subset) = load_subset(&root.join(GOLDEN));

    let engine = Engine::create(Device::Cuda).unwrap_or_else(|e| {
        panic!("engine create(CUDA) failed: {e:?} — CUDA EP required, CPU is not success")
    });
    engine
        .load_model(ALIAS)
        .unwrap_or_else(|e| panic!("load_model({ALIAS}) on CUDA failed: {e:?}"));

    let models = engine.list_models().expect("list_models");
    let info = models.get(0).expect("bge-small row");
    assert_eq!(info.alias, ALIAS);
    assert_eq!(info.device, Device::Cuda, "list_models device must be CUDA");
    assert_eq!(info.dim, dim as u32);

    // The loaded contract is CLS+L2; a mean-pooling request must be an
    // explicit NotImplemented, never a silent substitution.
    let mean_opts = EmbedOptions {
        pooling: Pooling::Mean,
        normalize: Some(true),
        ..Default::default()
    };
    match engine.embed_one(ALIAS, "hello world", &mean_opts) {
        Err(Error::NotImplemented(_)) => {}
        other => panic!("mean pooling on a CLS alias must be NotImplemented, got {other:?}"),
    }

    let opts = EmbedOptions {
        pooling: Pooling::Cls,
        normalize: Some(true),
        ..Default::default()
    };
    let mut worst = 1.0_f32;
    let mut sum = 0.0_f32;
    let mut per_text = Vec::new();
    for (id, text, expected) in &subset {
        let got = engine
            .embed_one(ALIAS, text, &opts)
            .unwrap_or_else(|e| panic!("embed_one({ALIAS}, {id:?}) failed: {e:?}"));
        assert_eq!(got.dim(), dim, "{id} dim");
        let norm: f64 = got
            .values()
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum();
        assert!(
            (norm.sqrt() - 1.0).abs() < 1e-3,
            "{id}: CLS output must be L2-normalized, norm={}",
            norm.sqrt()
        );
        let sim = cosine(got.values(), expected);
        assert!(
            sim >= COSINE_FLOOR,
            "{id}: cosine {sim} < {COSINE_FLOOR} — wrong pooling / model / device \
             would fail here (goldens are CLS+L2)"
        );
        eprintln!("  {id}: cosine={sim:.6} dim={}", got.dim());
        worst = worst.min(sim);
        sum += sim;
        per_text.push(serde_json::json!({ "id": id, "cosine": sim }));
        drop(got);
    }
    let mean_cos = sum / subset.len() as f32;

    let gpu_name = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    assert!(!gpu_name.is_empty(), "nvidia-smi returned no GPU name");
    let git_sha = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&root)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    eprintln!(
        "turboembed nvidia bge-small (CLS): n={} dim={dim} min_cosine={worst:.6} mean={mean_cos:.6} gpu={gpu_name}",
        subset.len()
    );

    let receipt = serde_json::json!({
        "schema_version": 1,
        "crate": "turboembed",
        "alias": ALIAS,
        "device": "CUDA",
        "provider": "ORT CUDA EP + IoBinding + DEVICE cls+L2",
        "pooling": "cls",
        "dims": dim,
        "gpu": gpu_name,
        "git_sha": git_sha,
        "golden": GOLDEN,
        "min_cosine": worst,
        "mean_cosine": mean_cos,
        "threshold": COSINE_FLOOR,
        "n_texts": subset.len(),
        "per_text": per_text,
        "mean_pooling_request": "NotImplemented (explicit contract, no substitution)",
        "commands": [
            "make fetch-embeddings ALIASES=bge-small",
            "export LD_LIBRARY_PATH=\"$(pwd)/.libs/nvidia/lib:${LD_LIBRARY_PATH:-}\"",
            "cargo test -p turboembed --features ort-cuda --test nvidia_bge_small -- --ignored --nocapture",
        ],
    });
    let path = root.join(RECEIPT);
    fs::create_dir_all(path.parent().unwrap()).expect("receipt dir");
    fs::write(
        &path,
        serde_json::to_string_pretty(&receipt).unwrap() + "\n",
    )
    .expect("write receipt");
    eprintln!("wrote {}", path.display());
}
