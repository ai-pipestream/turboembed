//! Live Intel OpenVINO MiniLM-L6 CE. Compiled when build.rs found OpenVINO.
//! Skips if IR is absent so `cargo test --workspace` stays green without a
//! convert. `make test-turborerank-intel` verifies IR then runs ignored.

#![cfg(turborerank_openvino)]

use std::fs;
use std::path::PathBuf;

use serde::Deserialize;
use turborerank::{
    default_ov_ir_dir, ov_ir_present, Activation, Device, Engine, Error, TokenBuffer, Truncation,
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

fn load_ov(device: Device) -> Engine {
    let dir = default_ov_ir_dir();
    let engine = Engine::create_with_config(device, Some(&dir))
        .unwrap_or_else(|e| panic!("{device:?} engine: {e}"));
    engine
        .load_model("ms-marco-minilm-l6")
        .unwrap_or_else(|e| panic!("{device:?} load MiniLM CE: {e} ({})", engine.last_error()));
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

fn score_identity(engine: &Engine) -> Vec<f32> {
    engine
        .score(
            None,
            QUERY,
            DOCS,
            Truncation::LongestFirst,
            Activation::Identity,
            512,
        )
        .unwrap()
}

#[test]
fn ov_gpu_live_scores_when_ir_present() {
    if !ov_ir_present() {
        return;
    }
    let engine = load_ov(Device::OpenVinoGpu);
    let logits = score_identity(&engine);
    reject_mock_scores(&logits);
    assert!(
        logits[0] > logits[1] && logits[1] > logits[2],
        "expected relevant > mid > irrelevant, got {logits:?}"
    );
    assert!(logits[0] - logits[2] > 2.0, "got {logits:?}");
}

#[test]
fn ov_cpu_live_scores_when_ir_present() {
    if !ov_ir_present() {
        return;
    }
    let engine = load_ov(Device::OpenVinoCpu);
    let logits = score_identity(&engine);
    reject_mock_scores(&logits);
    assert!(
        logits[0] > logits[1] && logits[1] > logits[2],
        "expected relevant > mid > irrelevant, got {logits:?}"
    );
    assert!(logits[0] - logits[2] > 2.0, "got {logits:?}");
}

#[test]
fn ov_gpu_batch_matches_one_by_one() {
    if !ov_ir_present() {
        return;
    }
    let engine = load_ov(Device::OpenVinoGpu);
    let batch = score_identity(&engine);
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
            (one[0] - batch[i]).abs() < 1e-4,
            "row {i}: batch {} vs one {}",
            batch[i],
            one[0]
        );
    }
}

#[test]
fn ov_gpu_golden_vector() {
    if !ov_ir_present() {
        return;
    }
    let path = golden_path();
    if !path.is_file() {
        return;
    }
    let g: GoldenFile = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(g.texts.query, QUERY);
    let engine = load_ov(Device::OpenVinoGpu);
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
    eprintln!("OV GPU vs HF: max_abs={max_abs:.6e} cosine={cosine:.10}");
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
fn ov_cpu_golden_vector() {
    if !ov_ir_present() {
        return;
    }
    let path = golden_path();
    if !path.is_file() {
        return;
    }
    let g: GoldenFile = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    let engine = load_ov(Device::OpenVinoCpu);
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
    for (got, exp) in logits.iter().zip(g.logits.iter()) {
        let e = (got - exp).abs();
        if e > max_abs {
            max_abs = e;
        }
        assert!(e < 2e-3, "OV CPU logit got {got} expected {exp}");
    }
    eprintln!("OV CPU vs HF: max_abs={max_abs:.6e}");
}

#[test]
fn ov_gpu_usm_buffer_caller_writable() {
    if !ov_ir_present() {
        return;
    }
    let mut buf = TokenBuffer::alloc(Device::OpenVinoGpu, 2, 32).expect("Level Zero USM");
    assert_eq!(buf.device(), Device::OpenVinoGpu);
    assert!(buf.ptr_aligned(), "USM host pointers must be 64-byte aligned");
    buf.input_ids_mut()[0] = 101;
    buf.input_ids_mut()[1] = 7592;
    assert_eq!(buf.input_ids()[0], 101);
    assert_eq!(buf.input_ids()[1], 7592);
}

#[test]
fn ov_gpu_caller_pointer_forward_zero_host_alloc() {
    if !ov_ir_present() {
        return;
    }
    let engine = load_ov(Device::OpenVinoGpu);
    let mut buf = TokenBuffer::alloc(Device::OpenVinoGpu, 1, 64).unwrap();
    engine
        .pack_text(&mut buf, 0, QUERY, REL, Truncation::LongestFirst, 64)
        .unwrap();
    let scores = engine.forward(&buf, 1, Activation::Identity).unwrap();
    assert!(scores[0] > 0.0, "relevant pair should be a positive logit");
}

#[test]
fn auto_resolves_to_openvino_gpu_without_cuda() {
    if !ov_ir_present() {
        return;
    }
    if cfg!(turborerank_cuda) {
        return;
    }
    let dir = default_ov_ir_dir();
    let auto = Engine::create_with_config(Device::Auto, Some(&dir)).expect("AUTO→OV GPU");
    auto.load_model("ms-marco-minilm-l6").unwrap();
    let gpu = load_ov(Device::OpenVinoGpu);
    let a = score_identity(&auto);
    let g = score_identity(&gpu);
    for i in 0..3 {
        assert!(
            (a[i] - g[i]).abs() < 1e-4,
            "AUTO {} vs OPENVINO_GPU {}",
            a[i],
            g[i]
        );
    }
}

#[test]
fn ov_gpu_and_cpu_agree_on_golden() {
    if !ov_ir_present() {
        return;
    }
    let gpu = score_identity(&load_ov(Device::OpenVinoGpu));
    let cpu = score_identity(&load_ov(Device::OpenVinoCpu));
    for i in 0..3 {
        assert!(
            (gpu[i] - cpu[i]).abs() < 2e-2,
            "GPU {} vs OV-CPU {}",
            gpu[i],
            cpu[i]
        );
    }
}

#[test]
fn npu_still_fails_loud() {
    let err = Engine::create(Device::OpenVinoNpu).unwrap_err();
    assert!(
        matches!(err, Error::Unavailable(_) | Error::UnsupportedDevice(_)),
        "{err:?}"
    );
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("refus") || msg.contains("cpu fallback"),
        "{err}"
    );
}

#[test]
#[ignore]
fn ignored_requires_ov_ir() {
    assert!(
        ov_ir_present(),
        "OpenVINO IR missing at {} — export ONNX then `make convert-rerank-ov`",
        default_ov_ir_dir().display()
    );
    ov_gpu_live_scores_when_ir_present();
    ov_cpu_live_scores_when_ir_present();
    ov_gpu_batch_matches_one_by_one();
    ov_gpu_golden_vector();
    ov_cpu_golden_vector();
}
