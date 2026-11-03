//! `BundleSequenceScorer` on a device session against the CPU (#989): a
//! sequence-classification bundle of testdata/tiny-bert-bundle's encoder
//! and a head made for the test, scored over the same texts on both. The
//! device is the one TURBO_TEST_DEVICE names (`levelzero`, `cuda`,
//! `metal`; the CPU when it is not set) at the precision
//! TURBO_TEST_PRECISION names (model, or exact). Each logit within 1e-4 of
//! the CPU's at f32 and 1e-2 at f16, and the label each text picks the same
//! wherever the CPU's margin is over twice the bound, those texts picking
//! at least two labels. With TURBO_TEST_REQUIRE_CUDA,
//! TURBO_TEST_REQUIRE_LEVELZERO or TURBO_TEST_REQUIRE_METAL set, the test
//! fails unless it runs there.

use std::collections::BTreeSet;

use turbo_analysis::Execution;
use turbo_analysis_embed::{BundleSequenceScorer, Device, EncoderOptions, Precision};
use turbo_doccat::{DlInferenceOptions, DlSequenceScorer};

mod nli_support;

use nli_support::make_nli_bundle_with;

/// The backend a TURBO_TEST_REQUIRE_* variable requires, if any.
fn required() -> Option<&'static str> {
    let set = |v: &str| std::env::var(v).is_ok_and(|v| !v.is_empty() && v != "0");
    [
        ("TURBO_TEST_REQUIRE_CUDA", "cuda"),
        ("TURBO_TEST_REQUIRE_LEVELZERO", "levelzero"),
        ("TURBO_TEST_REQUIRE_METAL", "metal"),
    ]
    .into_iter()
    .find(|(v, _)| set(v))
    .map(|(_, b)| b)
}

/// The device options the environment names, and the CPU's at the same
/// precision.
fn options() -> (EncoderOptions, EncoderOptions) {
    let device = match std::env::var("TURBO_TEST_DEVICE").as_deref() {
        Err(_) | Ok("cpu") => Device::Cpu,
        Ok(d) => Device::Backend(d.to_owned()),
    };
    let precision = match std::env::var("TURBO_TEST_PRECISION").as_deref() {
        Err(_) | Ok("model") => Precision::Model,
        Ok("exact") => Precision::Exact,
        Ok(p) => panic!("TURBO_TEST_PRECISION={p}: model or exact"),
    };
    if let Some(b) = required() {
        assert!(!matches!(device, Device::Cpu), "{b} is required but TURBO_TEST_DEVICE names no device");
    }
    let cpu = EncoderOptions { device: Device::Cpu, precision, ..EncoderOptions::default() };
    (EncoderOptions { device, ..cpu.clone() }, cpu)
}

/// The session's bound: 1e-2 at f16, else 1e-4. A required backend is
/// checked here.
fn bound(e: &Execution) -> f64 {
    if let Some(b) = required() {
        assert_eq!(e.backend, b, "the session runs on the backend required");
    }
    if e.precision.contains("f16") { 1e-2 } else { 1e-4 }
}

/// The label a logit row picks, and its margin over the next.
fn decision(row: &[f32]) -> (usize, f32) {
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
    (order[0], row[order[0]] - row[order[1]])
}

#[test]
fn the_bundle_sequence_scorer_scores_on_the_device_as_on_the_cpu() {
    let (device, cpu) = options();
    let labels = ["billing", "hardware", "search"];
    let texts: Vec<String> = [
        "Invoice billing charges the monthly subscription to the credit card.",
        "GPU tensor cores accelerate matrix multiplication on device memory.",
        "Approximate nearest neighbor search queries the vector index.",
        "Café naïve résumé",
        "Rain fell all night over the quiet harbour.",
        "She signed the contract on Monday and paid the first invoice.",
        "A very long sentence about nothing in particular that goes on and on for a while, \
         repeating itself, and then stops.",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let o = DlInferenceOptions::default();
    // The tiny encoder's pooled state moves by about 1e-3 between inputs,
    // so a fixed head picks one label for every text. The head is centred
    // on these texts instead: a CPU pass reads three pooled dimensions, and
    // each label's logit is its dimension less the texts' mean, times
    // SCALE, so the label follows the input. SCALE keeps an f32 device's
    // 1e-7 inside the bound.
    const SCALE: f32 = 200.0;
    let dims = [21, 22, 23];
    let rows = |scale: f32| (0..3 * 32).map(|k| if k % 32 == dims[k / 32] { scale } else { 0.0 }).collect();
    let probe = make_nli_bundle_with("seq-probe", &labels, Some((rows(1.0), vec![0.0; 3])), |_| {});
    let pooled = BundleSequenceScorer::open(&probe.0, &cpu).unwrap().score_chunks(&texts, &o).unwrap();
    let mean: Vec<f32> = (0..3).map(|d| pooled.iter().map(|r| r[d]).sum::<f32>() / texts.len() as f32).collect();
    let bias = mean.iter().map(|m| -SCALE * m).collect();
    let dir = make_nli_bundle_with("seq-device", &labels, Some((rows(SCALE), bias)), |_| {});
    let (want, got) =
        (BundleSequenceScorer::open(&dir.0, &cpu).unwrap(), BundleSequenceScorer::open(&dir.0, &device).unwrap());
    let execution = DlSequenceScorer::execution(&got).expect("the session's execution");
    let bound = bound(&execution);
    let (w, g) = (want.score_chunks(&texts, &o).unwrap(), got.score_chunks(&texts, &o).unwrap());
    assert_eq!(w.len(), texts.len());
    assert_eq!(g.len(), texts.len());
    let (mut worst, mut picked) = (0f64, BTreeSet::new());
    for (i, (w, g)) in w.iter().zip(&g).enumerate() {
        assert_eq!(w.len(), g.len(), "text {i}: logits");
        for (a, b) in w.iter().zip(g) {
            let d = (*a as f64 - *b as f64).abs();
            assert!(d <= bound, "text {i}: {b} against {a}");
            worst = worst.max(d);
        }
        let ((label, margin), (theirs, _)) = (decision(w), decision(g));
        println!("text {i}: {} by {margin}", labels[label]);
        // A margin inside twice the bound cannot hold a device to a label.
        if margin as f64 > 2.0 * bound {
            assert_eq!(labels[label], labels[theirs], "text {i}: the label picked");
            picked.insert(label);
        }
    }
    assert!(picked.len() >= 2, "the decided texts pick one label only: {picked:?}");
    println!(
        "bundle sequence scorer on {} {} ({}): logits {worst:e}, labels picked {:?}",
        execution.backend,
        execution.device,
        execution.precision,
        picked.iter().map(|&l| labels[l]).collect::<Vec<_>>()
    );
}
