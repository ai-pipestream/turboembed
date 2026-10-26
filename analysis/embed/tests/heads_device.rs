//! `DenseClassifier` and `BundleNliScorer` on a device session against
//! the CPU (#989): the dense head fit once on the CPU sentence vectors of
//! testdata/tiny-bert-bundle and read over the device's, and the NLI
//! scorer of the tiny NLI bundle on both. The device is the one
//! TURBO_TEST_DEVICE names (`levelzero`, `cuda`, `metal`; the CPU when it
//! is not set) at the precision TURBO_TEST_PRECISION names (model, or
//! exact). Each vector, probability and logit within 1e-4 of the CPU's at
//! f32 and 1e-2 at f16, the predicted tags the same, and the NLI label
//! the same for each pair whose CPU margin is over twice the bound, those
//! pairs picking at least two labels. With
//! TURBO_TEST_REQUIRE_CUDA, TURBO_TEST_REQUIRE_LEVELZERO or
//! TURBO_TEST_REQUIRE_METAL set, the test
//! fails unless it runs there. `heads.rs` holds `ClassifierHead`, the
//! centroids and `LearnedAspectSentiment` to the same device.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo_analysis::{Analyzer, Document, Execution, Key, Provenance, Span, Spans, TagSet, ViewId};
use turbo_analysis_embed::dense::*;
use turbo_analysis_embed::heads::HiddenActivation;
use turbo_analysis_embed::{
    BundleNliScorer, Device, EncoderOptions, Precision, SentenceVectors, SubwordAnnotator, VectorAnnotator,
    VectorOptions,
};
use turbo_analysis_layers::{Sentences, Vectors};
use turbo_doccat::NliScorer;

mod nli_support;

use nli_support::make_nli_bundle_with;

fn bundle() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

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

fn close(want: &[f64], got: &[f64], bound: f64, what: &str, worst: &mut f64) {
    assert_eq!(want.len(), got.len(), "{what}: length");
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        let d = (w - g).abs();
        assert!(d <= bound, "{what}[{i}]: {g} against {w}");
        *worst = worst.max(d);
    }
}

const CORPUS: [(&str, [f64; 3]); 8] = [
    ("Vector search indexes dense passage embeddings by cosine similarity.", [1., 0., 0.]),
    ("Approximate nearest neighbor search queries the vector index.", [1., 0., 0.]),
    ("GPU tensor cores accelerate matrix multiplication on device memory.", [0., 1., 0.]),
    ("PCIe bandwidth limits batch throughput on the accelerator card.", [0., 1., 0.]),
    ("GPU clusters accelerate dense vector search and nearest neighbor retrieval.", [1., 1., 0.]),
    ("Invoice billing charges the monthly subscription to the credit card.", [0., 0., 1.]),
    ("Monthly cloud GPU billing charges per hardware accelerator hour.", [0., 1., 1.]),
    ("General status update with no specialized topic.", [0., 0., 0.]),
];

/// The corpus's sentence vectors on the session `o` opens, and where it
/// ran.
fn sentence_vectors(o: &EncoderOptions) -> (Vectors<Sentences>, Execution) {
    let options = VectorOptions { encoder: o.clone(), ..Default::default() };
    let vectors = Arc::new(VectorAnnotator::open(&bundle(), &options).unwrap());
    let execution = vectors.execution().clone();
    let (mut text, mut spans) = (String::new(), Spans::new(ViewId::SOURCE));
    for (s, _) in CORPUS {
        if !text.is_empty() {
            text.push(' ');
        }
        let start = text.len() as u32;
        text.push_str(s);
        spans.push(Span::new(start, text.len() as u32));
    }
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test")).unwrap();
    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(SubwordAnnotator::open(&bundle()).unwrap())
        .add(SentenceVectors(vectors))
        .build()
        .unwrap();
    let mut docs = [doc];
    pipe.run(&mut docs).remove(0).unwrap();
    let v = docs[0].get::<Vectors<Sentences>>().unwrap();
    (Vectors::new(v.dim, v.data.clone(), v.empty.clone(), v.model.clone()), execution)
}

#[test]
fn a_dense_classifier_reads_device_vectors_as_the_cpus() {
    let (device, cpu) = options();
    let (want, _) = sentence_vectors(&cpu);
    let (got, execution) = sentence_vectors(&device);
    let bound = bound(&execution);
    let targets: Vec<Vec<f64>> = CORPUS.iter().map(|(_, y)| y.to_vec()).collect();
    let mut worst = (0f64, 0f64);
    for (architecture, output) in [
        (Architecture::Logistic, OutputKind::Softmax),
        (Architecture::Mlp { hidden: 16, activation: HiddenActivation::Gelu }, OutputKind::Sigmoid),
    ] {
        let trainer = DenseClassifierTrainer {
            architecture,
            output,
            params: AdamParameters {
                seed: 7,
                learning_rate: 0.02,
                weight_decay: 1e-4,
                batch_size: 4,
                epochs: 120,
                patience: None,
            },
        };
        let tags = TagSet::new("domain", &["search", "hardware", "billing"]);
        // Softmax reads each row as a distribution: the labels shared, or
        // all three for a sentence with none.
        let fit_on: Vec<Vec<f64>> = match output {
            OutputKind::Softmax => targets
                .iter()
                .map(|r| {
                    let n = r.iter().sum::<f64>();
                    if n == 0.0 { vec![1.0 / 3.0; 3] } else { r.iter().map(|v| v / n).collect() }
                })
                .collect(),
            _ => targets.clone(),
        };
        let model = trainer.fit(&want, &fit_on, tags, None).unwrap().model;
        let gold: Vec<BTreeSet<usize>> = targets
            .iter()
            .map(|r| r.iter().enumerate().filter(|(_, v)| **v >= 0.5).map(|(i, _)| i).collect())
            .collect();
        let model = match output {
            OutputKind::Sigmoid => model.calibrate_thresholds(&want, &gold, &ThresholdCalibrator::default()).unwrap(),
            _ => model,
        };
        for (i, (w, g)) in want.data.chunks(want.dim as usize).zip(got.data.chunks(got.dim as usize)).enumerate() {
            let as64 = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
            close(&as64(w), &as64(g), bound, &format!("sentence {i} vector"), &mut worst.0);
            let (pw, pg) = (model.probabilities(w).unwrap(), model.probabilities(g).unwrap());
            close(&pw, &pg, bound, &format!("{output:?} sentence {i} probabilities"), &mut worst.1);
            assert_eq!(model.predicted_tags(w).unwrap(), model.predicted_tags(g).unwrap(), "sentence {i} tags");
        }
    }
    println!(
        "dense classifier on {} {} ({}): vectors {:e}, probabilities {:e}",
        execution.backend, execution.device, execution.precision, worst.0, worst.1
    );
}

/// The label a logit row picks, and its margin over the next.
fn decision(row: &[f32]) -> (usize, f32) {
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
    (order[0], row[order[0]] - row[order[1]])
}

#[test]
fn the_bundle_nli_scorer_scores_on_the_device_as_on_the_cpu() {
    let (device, cpu) = options();
    let labels = ["contradiction", "entailment", "neutral"];
    let pairs = [
        ("The quick brown fox jumps over the lazy dog.", "This example is about animals."),
        ("Café naïve résumé", "This example is about food."),
        ("GPU tensor cores accelerate matrix multiplication.", "This example is about hardware."),
        ("Invoice billing charges the monthly subscription.", "This example is not about money."),
        ("Rain fell all night over the quiet harbour.", "The weather was dry."),
        ("She signed the contract on Monday.", "A document was signed."),
    ];
    // The tiny encoder's pooled state moves by about 1e-3 between inputs,
    // so a fixed head picks one label for every pair. The head is centred
    // on these pairs instead: a CPU pass reads three pooled dimensions,
    // and each label's logit is its dimension less the pairs' mean, times
    // SCALE, so the label follows the input. SCALE keeps an f32 device's
    // 1e-7 inside the bound.
    const SCALE: f32 = 200.0;
    let dims = [21, 22, 23];
    let rows = |scale: f32| (0..3 * 32).map(|k| if k % 32 == dims[k / 32] { scale } else { 0.0 }).collect();
    let probe = make_nli_bundle_with("probe", &labels, Some((rows(1.0), vec![0.0; 3])), |_| {});
    let pooled = BundleNliScorer::open(&probe.0, &cpu).unwrap().score_pairs(&pairs).unwrap();
    let mean: Vec<f32> = (0..3).map(|d| pooled.iter().map(|r| r[d]).sum::<f32>() / pairs.len() as f32).collect();
    let bias = mean.iter().map(|m| -SCALE * m).collect();
    let dir = make_nli_bundle_with("device", &labels, Some((rows(SCALE), bias)), |_| {});
    let (want, got) = (BundleNliScorer::open(&dir.0, &cpu).unwrap(), BundleNliScorer::open(&dir.0, &device).unwrap());
    let execution = got.info().execution.clone();
    let bound = bound(&execution);
    let (w, g) = (want.score_pairs(&pairs).unwrap(), got.score_pairs(&pairs).unwrap());
    let (mut worst, mut picked) = (0f64, BTreeSet::new());
    for (i, (w, g)) in w.iter().zip(&g).enumerate() {
        let as64 = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
        close(&as64(w), &as64(g), bound, &format!("pair {i} logits"), &mut worst);
        let ((label, margin), (theirs, _)) = (decision(w), decision(g));
        println!("pair {i}: {} by {margin}", labels[label]);
        // A margin inside twice the bound cannot hold a device to a label.
        if margin as f64 > 2.0 * bound {
            assert_eq!(labels[label], labels[theirs], "pair {i}: the label picked");
            picked.insert(label);
        }
    }
    assert!(picked.len() >= 2, "the decided pairs pick one label only: {picked:?}");
    println!(
        "bundle nli scorer on {} {} ({}): logits {worst:e}, labels picked {:?}",
        execution.backend,
        execution.device,
        execution.precision,
        picked.iter().map(|&l| labels[l]).collect::<Vec<_>>()
    );
}
