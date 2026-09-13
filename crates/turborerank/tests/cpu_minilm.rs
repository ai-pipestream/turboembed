//! Live MiniLM-L6 CE scores. Skips when weights are absent so
//! `cargo test --workspace` stays offline-green. `make test-turborerank`
//! fetches then runs the ignored suite which fails if weights are missing.

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
    model: String,
    revision: String,
    activation: String,
    texts: GoldenTexts,
    logits: Vec<f32>,
    sigmoid: Vec<f32>,
    #[serde(default)]
    input_ids_row0: Option<Vec<i32>>,
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

fn load_cpu() -> Engine {
    let dir = default_model_dir();
    let engine = Engine::create_with_config(Device::Cpu, Some(&dir)).expect("CPU engine");
    engine
        .load_model("ms-marco-minilm-l6")
        .unwrap_or_else(|e| panic!("load MiniLM CE: {e} ({})", engine.last_error()));
    engine
}

fn reject_mock_scores(scores: &[f32]) {
    assert!(!scores.is_empty());
    let all_equal = scores.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-8);
    assert!(
        !all_equal,
        "FAKE: all scores equal {scores:?} — constant/mock scorer"
    );
    // FNV-ish 8-d mock is an embed trick; CE scores must not be 0/1 word-overlap.
    let only_unit = scores.iter().all(|s| (*s == 0.0) || (*s == 1.0) || (*s == 0.5));
    assert!(
        !only_unit,
        "FAKE: scores look like the word-overlap mock {scores:?}"
    );
}

#[test]
fn live_scores_when_weights_present() {
    if !weights_present() {
        return;
    }
    let engine = load_cpu();
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
    assert!(
        logits[0] - logits[2] > 2.0,
        "CE must separate Berlin pop vs NYC pizza by >2 logits, got {logits:?}"
    );

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
fn batch_matches_one_by_one_when_present() {
    if !weights_present() {
        return;
    }
    let engine = load_cpu();
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
fn golden_vector_when_present() {
    if !weights_present() {
        return;
    }
    let path = golden_path();
    if !path.is_file() {
        return;
    }
    let raw = fs::read_to_string(&path).unwrap();
    let g: GoldenFile = serde_json::from_str(&raw).unwrap();
    assert!(g.model.contains("MiniLM"));
    assert_eq!(g.revision.len(), 40);
    assert_eq!(g.texts.query, QUERY);
    assert_eq!(g.activation, "identity+sigmoid");

    let engine = load_cpu();
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
    assert_eq!(logits.len(), g.logits.len());
    for (i, (got, exp)) in logits.iter().zip(g.logits.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 2e-3,
            "logit[{i}] got {got} expected {exp}"
        );
    }
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
    for (i, (got, exp)) in sig.iter().zip(g.sigmoid.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 2e-3,
            "sigmoid[{i}] got {got} expected {exp}"
        );
    }

    if let Some(ids) = g.input_ids_row0 {
        let mut buf = TokenBuffer::alloc(Device::Cpu, 1, 64).unwrap();
        engine
            .pack_text(
                &mut buf,
                0,
                &g.texts.query,
                &g.texts.documents[0],
                Truncation::LongestFirst,
                64,
            )
            .unwrap();
        let got: Vec<i32> = buf
            .input_ids()
            .iter()
            .copied()
            .zip(buf.attention_mask().iter().copied())
            .take_while(|(_, m)| *m == 1)
            .map(|(id, _)| id)
            .collect();
        assert_eq!(got, ids, "WordPiece+pack must match HF pair encoding");
    }
}

#[test]
fn caller_pointer_forward_no_new_alloc_when_present() {
    if !weights_present() {
        return;
    }
    let engine = load_cpu();
    let mut buf = TokenBuffer::alloc(Device::Cpu, 1, 64).unwrap();
    engine
        .pack_text(&mut buf, 0, QUERY, REL, Truncation::LongestFirst, 64)
        .unwrap();
    let scores = engine.forward(&buf, 1, Activation::Identity).unwrap();
    assert!(scores[0] > 0.0, "relevant pair should be a positive logit");
}

/// `make test-turborerank` runs this. Fails if someone forgot to fetch.
#[test]
#[ignore]
fn ignored_requires_weights() {
    assert!(
        weights_present(),
        "weights missing at {} — run `make fetch-rerankers`",
        default_model_dir().display()
    );
    live_scores_when_weights_present();
    batch_matches_one_by_one_when_present();
    golden_vector_when_present();
    assert!(golden_path().is_file(), "committed golden missing");
}
