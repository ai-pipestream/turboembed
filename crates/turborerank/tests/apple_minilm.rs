//! Live Apple Metal MiniLM-L6 CE. Compiled when build.rs found Metal.
//! Skips if weights are absent so `cargo test --workspace` stays green
//! without a fetch. `make test-turborerank-apple` fetches then runs ignored.

#![cfg(turborerank_metal)]

use std::fs;
use std::path::PathBuf;

use serde::Deserialize;
use turborerank::{
    default_model_dir, weights_present, Activation, Device, Engine, TokenBuffer, Truncation,
};

const QUERY: &str = "How many people live in Berlin?";
const REL: &str = "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.";
const MID: &str = "Berlin is well known for its museums.";
const IRREL: &str = "New York City is famous for its pizza and bagels.";
const DOCS: &[&str] = &[REL, MID, IRREL];

#[derive(Deserialize)]
struct GoldenFile {
    logits: Vec<f32>,
    sigmoid: Vec<f32>,
    texts: GoldenTexts,
}

#[derive(Deserialize)]
struct GoldenTexts {
    query: String,
    documents: Vec<String>,
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/reference_rerank/ms_marco_minilm_l6_berlin.json")
}

fn load_metal() -> Engine {
    let dir = default_model_dir();
    let engine = Engine::create_with_config(Device::Metal, Some(&dir)).expect("METAL engine");
    engine
        .load_model("ms-marco-minilm-l6")
        .unwrap_or_else(|e| panic!("Metal load MiniLM CE: {e} ({})", engine.last_error()));
    engine
}

fn reject_mock_scores(scores: &[f32]) {
    let all_equal = scores.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-8);
    assert!(!all_equal, "FAKE: all scores equal {scores:?}");
    let only_unit = scores
        .iter()
        .all(|s| (*s == 0.0) || (*s == 1.0) || (*s == 0.5));
    assert!(!only_unit, "FAKE: word-overlap mock {scores:?}");
}

#[test]
fn metal_buffer_is_shared_and_caller_writable() {
    let mut buf = TokenBuffer::alloc(Device::Metal, 2, 32).expect("MTL shared alloc");
    assert_eq!(buf.device(), Device::Metal);
    assert!(
        buf.ptr_aligned(),
        "MTL shared buffers must be 64-byte aligned"
    );
    buf.input_ids_mut()[0] = 101;
    buf.input_ids_mut()[1] = 7592;
    assert_eq!(buf.input_ids()[0], 101);
    assert_eq!(buf.input_ids()[1], 7592);
    let auto_buf = TokenBuffer::alloc(Device::Auto, 1, 16).expect("AUTO→METAL buffer");
    assert_eq!(auto_buf.device(), Device::Metal);
}

#[test]
fn metal_live_scores_when_weights_present() {
    if !weights_present() {
        return;
    }
    let engine = load_metal();
    let logits = engine
        .score(
            None,
            QUERY,
            DOCS,
            Truncation::LongestFirst,
            Activation::Identity,
            512,
        )
        .unwrap();
    reject_mock_scores(&logits);
    assert!(
        logits[0] > logits[1] && logits[1] > logits[2],
        "expected relevant > mid > irrelevant, got {logits:?}"
    );
    assert!(logits[0] - logits[2] > 2.0, "got {logits:?}");
}

#[test]
fn metal_batch_matches_one_by_one() {
    if !weights_present() {
        return;
    }
    let engine = load_metal();
    let batch = engine
        .score(
            None,
            QUERY,
            DOCS,
            Truncation::LongestFirst,
            Activation::Identity,
            512,
        )
        .unwrap();
    for (i, doc) in DOCS.iter().enumerate() {
        let one = engine
            .score(
                None,
                QUERY,
                &[*doc],
                Truncation::LongestFirst,
                Activation::Identity,
                512,
            )
            .unwrap();
        assert!(
            (one[0] - batch[i]).abs() < 1e-5,
            "row {i}: batch {} vs one {}",
            batch[i],
            one[0]
        );
    }
}

#[test]
fn metal_golden_vector() {
    if !weights_present() {
        return;
    }
    let path = golden_path();
    if !path.is_file() {
        return;
    }
    let g: GoldenFile = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(g.texts.query, QUERY);
    let engine = load_metal();
    let docs: Vec<&str> = g.texts.documents.iter().map(String::as_str).collect();
    let logits = engine
        .score(
            None,
            &g.texts.query,
            &docs,
            Truncation::LongestFirst,
            Activation::Identity,
            512,
        )
        .unwrap();
    reject_mock_scores(&logits);
    let mut max_abs = 0.0f32;
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (got, exp) in logits.iter().zip(g.logits.iter()) {
        let e = (got - exp).abs();
        if e > max_abs {
            max_abs = e;
        }
        assert!(e < 2e-3, "logit got {got} expected {exp}");
        dot += got * exp;
        na += got * got;
        nb += exp * exp;
    }
    let cosine = dot / (na.sqrt() * nb.sqrt());
    assert!(cosine > 0.999, "cosine vs HF golden {cosine}");
    eprintln!("Metal vs HF: max_abs={max_abs:.6e} cosine={cosine:.10}");
    let sig = engine
        .score(
            None,
            &g.texts.query,
            &docs,
            Truncation::LongestFirst,
            Activation::Sigmoid,
            512,
        )
        .unwrap();
    for (got, exp) in sig.iter().zip(g.sigmoid.iter()) {
        assert!((got - exp).abs() < 2e-3, "sigmoid got {got} expected {exp}");
    }
}

#[test]
fn metal_caller_pointer_forward_zero_host_alloc() {
    if !weights_present() {
        return;
    }
    let engine = load_metal();
    let mut buf = TokenBuffer::alloc(Device::Metal, 1, 64).unwrap();
    assert_eq!(buf.device(), Device::Metal);
    engine
        .pack_text(&mut buf, 0, QUERY, REL, Truncation::LongestFirst, 64)
        .unwrap();
    let scores = engine.forward(&buf, 1, Activation::Identity).unwrap();
    assert!(scores[0] > 0.0, "relevant pair should be a positive logit");
}

#[test]
fn metal_cpu_buffer_refuses_metal_forward() {
    if !weights_present() {
        return;
    }
    let engine = load_metal();
    let mut buf = TokenBuffer::alloc(Device::Cpu, 1, 64).unwrap();
    engine
        .pack_text(&mut buf, 0, QUERY, REL, Truncation::LongestFirst, 64)
        .unwrap();
    let err = engine.forward(&buf, 1, Activation::Identity).unwrap_err();
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("shared") || msg.contains("refus") || msg.contains("metal"),
        "CPU pointer into Metal forward must fail loud, got {err}"
    );
}

#[test]
fn metal_identity_matches_sigmoid_math() {
    if !weights_present() {
        return;
    }
    let engine = load_metal();
    let logits = engine
        .score(
            None,
            QUERY,
            DOCS,
            Truncation::LongestFirst,
            Activation::Identity,
            512,
        )
        .unwrap();
    let sig = engine
        .score(
            None,
            QUERY,
            DOCS,
            Truncation::LongestFirst,
            Activation::Sigmoid,
            512,
        )
        .unwrap();
    for i in 0..3 {
        let s = 1.0 / (1.0 + (-logits[i]).exp());
        assert!(
            (sig[i] - s).abs() < 1e-5,
            "sigmoid mismatch {i}: {} vs {s}",
            sig[i]
        );
    }
}

#[test]
fn metal_pack_matches_cpu_ids() {
    if !weights_present() {
        return;
    }
    let dir = default_model_dir();
    let metal = load_metal();
    let cpu = Engine::create_with_config(Device::Cpu, Some(&dir)).unwrap();
    cpu.load_model("ms-marco-minilm-l6").unwrap();
    let mut mb = TokenBuffer::alloc(Device::Metal, 1, 64).unwrap();
    let mut cb = TokenBuffer::alloc(Device::Cpu, 1, 64).unwrap();
    metal
        .pack_text(&mut mb, 0, QUERY, REL, Truncation::LongestFirst, 64)
        .unwrap();
    cpu.pack_text(&mut cb, 0, QUERY, REL, Truncation::LongestFirst, 64)
        .unwrap();
    assert_eq!(mb.input_ids(), cb.input_ids());
    assert_eq!(mb.attention_mask(), cb.attention_mask());
    assert_eq!(mb.token_type_ids(), cb.token_type_ids());
}

#[test]
fn auto_resolves_to_metal_and_matches() {
    if !weights_present() {
        return;
    }
    let dir = default_model_dir();
    let auto = Engine::create_with_config(Device::Auto, Some(&dir)).expect("AUTO");
    auto.load_model("ms-marco-minilm-l6").unwrap();
    let metal = load_metal();
    let a = auto
        .score(
            None,
            QUERY,
            DOCS,
            Truncation::LongestFirst,
            Activation::Identity,
            512,
        )
        .unwrap();
    let m = metal
        .score(
            None,
            QUERY,
            DOCS,
            Truncation::LongestFirst,
            Activation::Identity,
            512,
        )
        .unwrap();
    for i in 0..3 {
        assert!(
            (a[i] - m[i]).abs() < 1e-5,
            "AUTO {} vs METAL {}",
            a[i],
            m[i]
        );
    }
}

#[test]
fn metal_lists_catalog_alias() {
    let engine = Engine::create(Device::Metal).expect("METAL create");
    let models = engine.list_models().unwrap();
    assert!(
        models.iter().any(|m| m.alias == "ms-marco-minilm-l6"),
        "{models:?}"
    );
}

#[test]
#[ignore]
fn ignored_requires_metal_weights() {
    assert!(
        weights_present(),
        "weights missing at {} — run `make fetch-rerankers`",
        default_model_dir().display()
    );
    metal_live_scores_when_weights_present();
    metal_batch_matches_one_by_one();
    metal_golden_vector();
    metal_identity_matches_sigmoid_math();
    auto_resolves_to_metal_and_matches();
}
