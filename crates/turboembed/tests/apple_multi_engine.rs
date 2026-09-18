//! Machine C checklist step 3: Metal ownership under multiple live engines.
//! The `mlx-live` analog of `intel_engine_isolation.rs` — two independent
//! `Device::Metal` engines embed concurrently, retained results survive peer
//! engine destruction, and every per-engine output matches a single-engine
//! baseline within 1e-5 absolute (the checklist gate). Writes
//! `testdata/receipts/turboembed/apple-multi-engine.json`.
//!
//! ```bash
//! make fetch-mlx ALIASES=minilm
//! cargo test -p turboembed --features mlx-live --test apple_multi_engine -- \
//!   --ignored --nocapture
//! ```

#![cfg(all(target_os = "macos", feature = "mlx-live"))]

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{mpsc, Arc, Barrier};
use std::time::{SystemTime, UNIX_EPOCH};

use turboembed::{Device, EmbedOptions, Embeddings, Engine};

const ALIAS: &str = "minilm";
const DIM: usize = 384;
const MAX_ABS: f32 = 1e-5;
const ITERATIONS: usize = 8;
const RECEIPT: &str = "testdata/receipts/turboembed/apple-multi-engine.json";

/// Fixed inputs shared by the baseline engine and both live engines:
/// ASCII, non-ASCII with an embedded NUL, and a longer sentence.
const TEXTS: &[&str] = &[
    "hello world",
    "Straße [MASK] café\0world",
    "TurboEmbed keeps native buffers reusable across engines and calls.",
];

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

fn opts() -> EmbedOptions {
    EmbedOptions {
        pooling: turboembed::Pooling::Mean,
        normalize: Some(true),
        truncate_to: Some(256),
        ..EmbedOptions::default()
    }
}

/// Asserts `got` matches `expected` within the checklist gate and returns the
/// observed worst absolute difference.
fn assert_close(got: &[f32], expected: &[f32], where_: &str) -> f32 {
    assert_eq!(got.len(), DIM, "{where_}: dim");
    assert_eq!(expected.len(), got.len(), "{where_}: baseline dim");
    let mut worst = 0.0_f32;
    for (i, (&a, &b)) in got.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "{where_}[{i}]: non-finite");
        let diff = (a - b).abs();
        assert!(
            diff <= MAX_ABS,
            "{where_}[{i}]: |{a} - {b}| = {diff} > {MAX_ABS} — per-engine output \
             must match the single-engine baseline within the checklist gate"
        );
        worst = worst.max(diff);
    }
    worst
}

/// Embeds every fixture text ITERATIONS times, comparing each output against
/// the baseline. Returns the retained first native results (they keep their
/// engine alive) and the observed worst absolute difference.
fn run_engine(engine: &Engine, baseline: &[Vec<f32>], label: &str) -> (Vec<Embeddings>, f32) {
    let opts = opts();
    let mut max_abs = 0.0_f32;
    let mut held = Vec::with_capacity(TEXTS.len());
    for (text, expected) in TEXTS.iter().zip(baseline) {
        let first = engine.embed_one(ALIAS, text, &opts).unwrap();
        max_abs = max_abs.max(assert_close(first.values(), expected, label));
        held.push(first);
        for _ in 0..ITERATIONS {
            let again = engine.embed_one(ALIAS, text, &opts).unwrap();
            max_abs = max_abs.max(assert_close(again.values(), expected, label));
        }
    }
    (held, max_abs)
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

fn git_sha(root: &std::path::Path) -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

#[test]
#[ignore = "needs macOS Metal + models/mlx/minilm (make fetch-mlx ALIASES=minilm)"]
fn independent_metal_engines_match_single_engine_baseline() {
    set_workspace_env();
    let root = workspace_root();
    assert!(
        root.join("models/mlx/minilm/model.safetensors").is_file(),
        "models/mlx/minilm missing — run `make fetch-mlx ALIASES=minilm` (SHA-pinned)"
    );

    // Single-engine baseline: the reference outputs each live engine must
    // reproduce within MAX_ABS. The engine is destroyed before the
    // multi-engine phase so nothing is shared with it.
    let baseline: Vec<Vec<f32>> = {
        let single = Engine::create(Device::Metal)
            .unwrap_or_else(|e| panic!("baseline Metal engine: {e:?} — CPU is not success"));
        single.load_model(ALIAS).unwrap();
        let opts = opts();
        TEXTS
            .iter()
            .map(|text| {
                let out = single.embed_one(ALIAS, text, &opts).unwrap();
                assert_eq!(out.dim(), DIM);
                assert!(out.values().iter().all(|x| x.is_finite()));
                out.values().to_vec()
            })
            .collect()
    };

    let first = Engine::create(Device::Metal).expect("first Metal engine");
    let second = Engine::create(Device::Metal).expect("second Metal engine");
    first.load_model(ALIAS).unwrap();
    second.load_model(ALIAS).unwrap();

    let start = Arc::new(Barrier::new(2));
    let (released, wait_release) = mpsc::channel();
    let first_baseline = baseline.clone();
    let first_start = Arc::clone(&start);
    let first_thread = std::thread::spawn(move || {
        first_start.wait();
        let (held, max_abs) = run_engine(&first, &first_baseline, "engine-1");
        drop(first);
        // The retained native results outlive their engine handle; reading
        // them after the drop must still match the baseline.
        for (row, expected) in held.iter().zip(&first_baseline) {
            assert_close(row.values(), expected, "engine-1 held after engine drop");
        }
        // Last owners: native destruction here must not affect the peer.
        drop(held);
        released.send(()).unwrap();
        max_abs
    });
    let second_baseline = baseline.clone();
    let second_thread = std::thread::spawn(move || {
        start.wait();
        let (held, mut max_abs) = run_engine(&second, &second_baseline, "engine-2");
        wait_release.recv().unwrap();
        // Peer engine destroyed: this engine and its retained results must
        // keep matching the baseline.
        let after = second.embed_one(ALIAS, TEXTS[0], &opts()).unwrap();
        max_abs = max_abs.max(assert_close(
            after.values(),
            &second_baseline[0],
            "engine-2 after peer destroy",
        ));
        drop(after);
        drop(second);
        for (row, expected) in held.iter().zip(&second_baseline) {
            assert_close(
                row.values(),
                expected,
                "engine-2 held after own engine drop",
            );
        }
        max_abs
    });
    let first_max = first_thread.join().unwrap();
    let second_max = second_thread.join().unwrap();
    let max_abs = first_max.max(second_max);

    eprintln!(
        "apple multi-engine: engines=2 texts={} iterations={ITERATIONS} \
         max_abs={max_abs:.3e} gate={MAX_ABS:.0e}",
        TEXTS.len()
    );

    let receipt = serde_json::json!({
        "schema_version": 1,
        "crate": "turboembed",
        "test": "apple_multi_engine::independent_metal_engines_match_single_engine_baseline",
        "alias": ALIAS,
        "device": "METAL",
        "provider": "mlx",
        "engines": 2,
        "iterations_per_text": ITERATIONS,
        "n_texts": TEXTS.len(),
        "dim": DIM,
        "max_abs": max_abs,
        "per_engine_max_abs": [first_max, second_max],
        "threshold": MAX_ABS,
        "retained_results_survive_peer_destruction": true,
        "pass": true,
        "git_sha": git_sha(&root),
        "chip": chip_name(),
        "captured_at_unix": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        "commands": [
            "make fetch-mlx ALIASES=minilm",
            "cargo test -p turboembed --features mlx-live --test apple_multi_engine -- --ignored --nocapture",
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
