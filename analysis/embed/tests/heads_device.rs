//! `DenseClassifier` and `BundleNliScorer` on a device session against
//! the CPU (#989): the dense head fit once on the CPU sentence vectors of
//! testdata/tiny-bert-bundle and read over the device's, and the NLI
//! scorer of the tiny NLI bundle on both. The device is the one
//! TURBO_TEST_DEVICE names (`levelzero`, `cuda`, `metal`; the CPU when it
//! is not set) at the precision TURBO_TEST_PRECISION names (model, or
//! exact). Each vector, probability and logit within 1e-4 of the CPU's at
//! f32 and 1e-2 at f16, the predicted tags the same. With
//! TURBO_TEST_REQUIRE_CUDA or TURBO_TEST_REQUIRE_LEVELZERO set, the test
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

use nli_support::make_nli_bundle;

fn bundle() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

/// The backend a TURBO_TEST_REQUIRE_* variable requires, if any.
fn required() -> Option<&'static str> {
    let set = |v: &str| std::env::var(v).is_ok_and(|v| !v.is_empty() && v != "0");
    [("TURBO_TEST_REQUIRE_CUDA", "cuda"), ("TURBO_TEST_REQUIRE_LEVELZERO", "levelzero")]
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

#[test]
fn the_bundle_nli_scorer_scores_on_the_device_as_on_the_cpu() {
    let (device, cpu) = options();
    let dir = make_nli_bundle("device", &["contradiction", "entailment", "neutral"], |_| {});
    let (want, got) = (BundleNliScorer::open(&dir.0, &cpu).unwrap(), BundleNliScorer::open(&dir.0, &device).unwrap());
    let execution = got.info().execution.clone();
    let bound = bound(&execution);
    let pairs = [
        ("The quick brown fox jumps over the lazy dog.", "This example is about animals."),
        ("Café naïve résumé", "This example is about food."),
        ("GPU tensor cores accelerate matrix multiplication.", "This example is about hardware."),
        ("Invoice billing charges the monthly subscription.", "This example is not about money."),
    ];
    let (w, g) = (want.score_pairs(&pairs).unwrap(), got.score_pairs(&pairs).unwrap());
    let mut worst = 0f64;
    for (i, (w, g)) in w.iter().zip(&g).enumerate() {
        let as64 = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
        close(&as64(w), &as64(g), bound, &format!("pair {i} logits"), &mut worst);
    }
    println!(
        "bundle nli scorer on {} {} ({}): logits {worst:e}",
        execution.backend, execution.device, execution.precision
    );
}
