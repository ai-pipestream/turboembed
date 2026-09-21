//! Live Hailo MiniLM proofs — Raspberry Pi AI HAT+ only.
//!
//! Run on a provisioned Pi (see docs/hailo-embed.md):
//!   cargo test -p turboembed --features hailo --test hailo_minilm \
//!     -- --include-ignored --nocapture --test-threads=1
//!
//! Reference vectors are the FP32 ONNX goldens in
//! testdata/reference_embeddings/ort_cuda_minilm_*.json. The Hailo encoder
//! is INT8 and runs attention unmasked over zero padding (see
//! docs/hailo-embed.md), so the gate is a provisional cosine floor of 0.97
//! (the same floor the e2e harness uses for quantized-vs-float parity);
//! the first Pi receipt should record the measured values.

#![cfg(feature = "hailo")]

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, OnceLock};

use serde::Deserialize;
use turboembed::{Device, EmbedOptions, Engine};

fn engine_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    // A panicking test must not cascade-poison the others: recover the lock.
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

#[derive(Deserialize)]
struct Golden {
    model: String,
    text: String,
    pooling: String,
    normalize: bool,
    dim: u32,
    vector: Vec<f32>,
}

fn goldens() -> Vec<(String, Golden)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/reference_embeddings");
    let mut out = Vec::new();
    for name in [
        "ort_cuda_minilm_short.json",
        "ort_cuda_minilm_medium.json",
        "ort_cuda_minilm_unicode.json",
        "ort_cuda_minilm_empty.json",
        "ort_cuda_minilm_long_truncation.json",
    ] {
        let path = dir.join(name);
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let g: Golden = serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {name}: {e}"));
        out.push((name.to_string(), g));
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn load_minilm() -> Engine {
    let engine = Engine::create(Device::Hailo).expect("create hailo engine");
    engine
        .load_model("minilm")
        .unwrap_or_else(|e| panic!("load minilm on hailo: {e}"));
    engine
}

/// INT8 HEFs compress the vector space: per-text cosine vs the FP32 golden
/// varies widely on out-of-calibration texts (0.70 short / 0.53 medium /
/// 0.32 mixed-script unicode, pi5ai1 2026-09-20) while *ranking* stays
/// near-parity (Spearman 0.937 vs 0.9438 FP32 on the same 96-pair STS
/// corpus). So individual cosines are reported, not gated; the quality gate
/// is the Spearman floor in `sts_spearman_matches_fp32_reference`. The
/// floors below are pure garbage-detectors (a broken layout/tokenizer
/// produces cos ≈ 0).
const COSINE_MEAN_TRIPWIRE: f64 = 0.5;
const COSINE_PER_TEXT_TRIPWIRE: f64 = 0.25;

#[test]
#[ignore = "requires a Raspberry Pi AI HAT+ with models/hailo/minilm"]
fn minilm_matches_fp32_reference_above_floor() {
    let _g = engine_lock();
    let engine = load_minilm();
    let mut cosines = Vec::new();
    for (name, golden) in goldens() {
        assert_eq!(golden.model, "minilm-l6-v2");
        assert_eq!(golden.dim, 384);
        assert_eq!(golden.pooling, "mean");
        assert!(golden.normalize);
        let emb = engine
            .embed_one("minilm", &golden.text, &EmbedOptions::default())
            .unwrap_or_else(|e| panic!("embed {name}: {e}"));
        assert_eq!(emb.dim(), 384, "{name}");
        assert_eq!(emb.count(), 1, "{name}");
        let cos = cosine(emb.values(), &golden.vector);
        // Receipt-style line for docs/hailo-embed.md receipts.
        eprintln!("hailo golden {name}: cosine={cos:.6}");
        assert!(
            cos >= COSINE_PER_TEXT_TRIPWIRE,
            "{name}: cosine {cos:.6} collapsed below {COSINE_PER_TEXT_TRIPWIRE}"
        );
        cosines.push(cos);
    }
    let mean = cosines.iter().sum::<f64>() / cosines.len() as f64;
    eprintln!("hailo golden mean cosine: {mean:.4}");
    assert!(
        mean >= COSINE_MEAN_TRIPWIRE,
        "mean cosine {mean:.4} below tripwire {COSINE_MEAN_TRIPWIRE}"
    );
}

#[test]
#[ignore = "requires a Raspberry Pi AI HAT+ with models/hailo/minilm"]
fn sts_spearman_matches_fp32_reference() {
    let _g = engine_lock();
    let engine = load_minilm();
    let corpus =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/corpus/sts-pairs.jsonl");
    let raw = std::fs::read_to_string(&corpus).expect("sts corpus read");
    let mut labels = Vec::new();
    let mut scores = Vec::new();
    for line in raw.lines() {
        let v: serde_json::Value = serde_json::from_str(line).expect("pair json");
        let a = engine
            .embed_one(
                "minilm",
                v["text_a"].as_str().unwrap(),
                &EmbedOptions::default(),
            )
            .expect("embed a");
        let b = engine
            .embed_one(
                "minilm",
                v["text_b"].as_str().unwrap(),
                &EmbedOptions::default(),
            )
            .expect("embed b");
        labels.push(v["score"].as_f64().unwrap());
        scores.push(cosine(a.values(), b.values()));
    }
    let rho = spearman(&labels, &scores);
    eprintln!(
        "hailo minilm STS pairs: n={} spearman={rho:.4}",
        labels.len()
    );
    // FP32 ORT CPU measures 0.9438 on this corpus; gate the Hailo lane at
    // 0.85 so ranking regressions fail while INT8 noise does not.
    assert!(rho >= 0.85, "spearman {rho:.4} below 0.85 quality floor");
}

fn spearman(xs: &[f64], ys: &[f64]) -> f64 {
    fn ranks(v: &[f64]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|a, b| v[*a].partial_cmp(&v[*b]).unwrap());
        let mut r = vec![0.0; v.len()];
        let mut i = 0;
        while i < v.len() {
            let mut j = i;
            while j + 1 < v.len() && v[idx[j + 1]] == v[idx[i]] {
                j += 1;
            }
            let avg = (i + j) as f64 / 2.0 + 1.0;
            for k in i..=j {
                r[idx[k]] = avg;
            }
            i = j + 1;
        }
        r
    }
    let (rx, ry) = (ranks(xs), ranks(ys));
    let n = xs.len() as f64;
    let mx = rx.iter().sum::<f64>() / n;
    let my = ry.iter().sum::<f64>() / n;
    let (mut cov, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for i in 0..rx.len() {
        let dx = rx[i] - mx;
        let dy = ry[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

#[test]
#[ignore = "requires a Raspberry Pi AI HAT+ with models/hailo/minilm"]
fn batch_rows_match_embed_one_bitwise() {
    let _g = engine_lock();
    let engine = load_minilm();
    let texts = ["hello world", "Berlin is in Germany", "🦀 rust"];
    let batch = engine
        .embed("minilm", &texts, &EmbedOptions::default())
        .expect("batch embed");
    assert_eq!(batch.count(), 3);
    for (i, text) in texts.iter().enumerate() {
        let single = engine
            .embed_one("minilm", text, &EmbedOptions::default())
            .expect("embed_one");
        assert_eq!(
            batch.row(i).unwrap(),
            single.values(),
            "batch row {i} must equal the single-text call"
        );
    }
}

#[test]
#[ignore = "requires a Raspberry Pi AI HAT+ with models/hailo/minilm"]
fn repeated_calls_are_bitwise_deterministic() {
    let _g = engine_lock();
    let engine = load_minilm();
    let first = engine
        .embed_one("minilm", "determinism probe", &EmbedOptions::default())
        .expect("embed 1");
    for _ in 0..4 {
        let again = engine
            .embed_one("minilm", "determinism probe", &EmbedOptions::default())
            .expect("embed n");
        assert_eq!(first.values(), again.values());
    }
}

#[test]
#[ignore = "requires a Raspberry Pi AI HAT+ with models/hailo/minilm"]
fn truncation_and_empty_and_options() {
    let _g = engine_lock();
    let engine = load_minilm();
    let long = "museum ".repeat(1500);
    let emb = engine
        .embed_one("minilm", &long, &EmbedOptions::default())
        .expect("long text must truncate to the HEF sequence length");
    assert_eq!(emb.dim(), 384);
    assert!(emb.values().iter().all(|v| v.is_finite()));

    let empty = engine
        .embed_one("minilm", "", &EmbedOptions::default())
        .expect("empty string embeds");
    assert_eq!(empty.dim(), 384);
    assert!(empty.values().iter().all(|v| v.is_finite()));

    // Options are honored host-side: CLS pooling and no-normalize must
    // change the vector; an invalid truncate floor must fail loud.
    let mut opts = EmbedOptions::default();
    opts.normalize = Some(false);
    let raw = engine
        .embed_one("minilm", "hello world", &opts)
        .expect("normalize=false");
    let l2: f64 = raw
        .values()
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    assert!(
        (l2 - 1.0).abs() > 1e-3,
        "normalize=false should not be unit-norm (l2={l2})"
    );

    opts.normalize = None;
    opts.truncate_to = Some(2);
    let err = engine
        .embed_one("minilm", "hello world", &opts)
        .expect_err("truncate_to below the specials floor must fail");
    assert!(
        matches!(err, turboembed::Error::InvalidArgument(_)),
        "{err:?}"
    );
}
