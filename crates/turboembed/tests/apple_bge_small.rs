//! Machine C checklist step 6: the Apple Metal equivalent of
//! `nvidia_bge_small.rs`. Second model contract on the MLX path:
//! bge-small-en-v1.5 with CLS pooling (MiniLM is mean pooling). Same
//! fail-loud device policy — Metal or error, never CPU. Scores the
//! `parity:*` subset of the committed apple golden and writes
//! `testdata/receipts/turboembed/apple-bge-small.json`.
//!
//! ```bash
//! make fetch-mlx ALIASES=bge-small
//! cargo test -p turboembed --features mlx-live --test apple_bge_small -- \
//!   --ignored --nocapture
//! ```

#![cfg(all(target_os = "macos", feature = "mlx-live"))]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;
use turboembed::{Device, EmbedOptions, Engine, Error, Pooling};

const ALIAS: &str = "bge-small";
const GOLDEN: &str = "testdata/e2e/goldens/apple/bge-small.json";
const RECEIPT: &str = "testdata/receipts/turboembed/apple-bge-small.json";
const COSINE_FLOOR: f32 = 0.99;
const SUBSET_PREFIX: &str = "parity:";

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

fn set_workspace_env() {
    // Safety: test-only, before any engine create.
    unsafe { std::env::set_var("INFERSTREAM_ROOT", workspace_root()) };
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
    assert_eq!(golden["pooling"], "cls", "golden pooling");
    assert_eq!(golden["normalize"], true, "golden normalize");
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

fn chip_name() -> String {
    Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "Apple Silicon".into())
}

#[test]
#[ignore = "needs macOS Metal + models/mlx/bge-small (make fetch-mlx ALIASES=bge-small)"]
fn bge_small_cls_metal_matches_golden() {
    set_workspace_env();
    let root = workspace_root();
    let model = root.join("models/mlx/bge-small/model.safetensors");
    assert!(
        model.is_file(),
        "{} missing — run `make fetch-mlx ALIASES=bge-small` (SHA-pinned)",
        model.display()
    );
    let (dim, subset) = load_subset(&root.join(GOLDEN));

    let engine = Engine::create(Device::Metal).unwrap_or_else(|e| {
        panic!("engine create(Metal) failed: {e:?} — Metal required, CPU is not success")
    });
    engine
        .load_model(ALIAS)
        .unwrap_or_else(|e| panic!("load_model({ALIAS}) on Metal failed: {e:?}"));

    let models = engine.list_models().expect("list_models");
    let info = models
        .iter()
        .find(|m| m.alias == ALIAS)
        .expect("bge-small row");
    assert_eq!(
        info.device,
        Device::Metal,
        "list_models device must be Metal"
    );
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

    let chip = chip_name();
    let git_sha = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&root)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    eprintln!(
        "turboembed apple bge-small (CLS): n={} dim={dim} min_cosine={worst:.6} mean={mean_cos:.6} chip={chip}",
        subset.len()
    );

    let receipt = serde_json::json!({
        "schema_version": 1,
        "crate": "turboembed",
        "alias": ALIAS,
        "device": "METAL",
        "provider": "mlx first-token CLS + L2 (hidden state, not the BERT NSP pooler)",
        "pooling": "cls",
        "dims": dim,
        "chip": chip,
        "git_sha": git_sha,
        "golden": GOLDEN,
        "min_cosine": worst,
        "mean_cosine": mean_cos,
        "threshold": COSINE_FLOOR,
        "n_texts": subset.len(),
        "per_text": per_text,
        "mean_pooling_request": "NotImplemented (explicit contract, no substitution)",
        "captured_at_unix": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        "commands": [
            "make fetch-mlx ALIASES=bge-small",
            "cargo test -p turboembed --features mlx-live --test apple_bge_small -- --ignored --nocapture",
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
