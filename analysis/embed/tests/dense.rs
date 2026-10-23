use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use turbo_analysis::{
    Analyzer, Annotator, Document, Key, Layer, ModelRef, Provenance, Span, Spans, Tag, TagSet, ViewId,
};
use turbo_analysis_embed::{
    Device, EncoderOptions, Precision, SentenceVectors, SubwordAnnotator, VectorAnnotator, VectorOptions,
    dense::*,
    heads::{ClassifierHead, HeadMode, HiddenActivation},
};
use turbo_analysis_layers::{Sentences, Vectors};
fn space() -> ModelRef {
    ModelRef { id: "encoder".into(), revision: "1".into(), sha256: "abc".into() }
}
fn vectors(data: Vec<f32>, dim: u32) -> Vectors<Sentences> {
    Vectors::new(dim, data, vec![], space())
}
fn tags() -> Arc<TagSet> {
    TagSet::new("labels", &["no", "yes"])
}
fn trainer(architecture: Architecture) -> DenseClassifierTrainer {
    DenseClassifierTrainer {
        architecture,
        output: OutputKind::Softmax,
        params: AdamParameters { epochs: 600, batch_size: 4, learning_rate: 0.03, ..Default::default() },
    }
}
fn xor() -> (Vectors<Sentences>, Vec<Vec<f64>>) {
    (vectors(vec![-1., -1., -1., 1., 1., -1., 1., 1.], 2), vec![vec![1., 0.], vec![0., 1.], vec![0., 1.], vec![1., 0.]])
}
fn accuracy(m: &DenseClassifierModel, v: &Vectors<Sentences>, y: &[Vec<f64>]) -> usize {
    v.data
        .chunks_exact(v.dim as usize)
        .zip(y)
        .filter(|(x, y)| {
            let p = m.probabilities(x).unwrap();
            (p[1] > p[0]) == (y[1] > y[0])
        })
        .count()
}
#[test]
fn logistic_separates_and_seed_is_reproducible() {
    let v = vectors(vec![-2., -1., 1., 2.], 1);
    let y = vec![vec![1., 0.], vec![1., 0.], vec![0., 1.], vec![0., 1.]];
    let t = trainer(Architecture::Logistic);
    let a = t.fit(&v, &y, tags(), None).unwrap();
    let b = t.fit(&v, &y, tags(), None).unwrap();
    assert_eq!(a.model.to_bytes(), b.model.to_bytes());
    assert_eq!(accuracy(&a.model, &v, &y), 4);
    assert!(a.training_loss < 0.03);
}
#[test]
fn relu_and_gelu_fit_xor_where_logistic_cannot() {
    let (v, y) = xor();
    let linear = trainer(Architecture::Logistic).fit(&v, &y, tags(), None).unwrap();
    assert!(accuracy(&linear.model, &v, &y) < 4);
    for activation in [HiddenActivation::Relu, HiddenActivation::Gelu] {
        let t = trainer(Architecture::Mlp { hidden: 8, activation });
        let fit = t.fit(&v, &y, tags(), None).unwrap();
        assert_eq!(accuracy(&fit.model, &v, &y), 4, "{activation:?}");
        assert!(fit.training_loss < 0.03);
        assert_eq!(fit.model.to_bytes(), t.fit(&v, &y, tags(), None).unwrap().model.to_bytes());
    }
}
#[test]
fn sigmoid_handles_binary_and_independent_targets() {
    let v = vectors(vec![-2., -1., 1., 2.], 1);
    let mut t = trainer(Architecture::Logistic);
    t.output = OutputKind::Sigmoid;
    let y = vec![vec![0., 1.], vec![0., 1.], vec![1., 1.], vec![1., 1.]];
    let fit = t.fit(&v, &y, tags(), None).unwrap();
    for (x, y) in v.data.chunks(1).zip(&y) {
        for (p, y) in fit.model.probabilities(x).unwrap().iter().zip(y) {
            assert_eq!(*p >= 0.5, *y == 1.);
        }
    }
    let binary: Vec<_> = y.iter().map(|y| vec![y[0]]).collect();
    let fit = t.fit(&v, &binary, TagSet::new("binary", &["yes"]), None).unwrap();
    assert!(fit.model.probabilities(&[2.]).unwrap()[0] > 0.9);
    assert!(fit.model.probabilities(&[-2.]).unwrap()[0] < 0.1);
}
#[test]
fn validation_stops_and_restores_best_checkpoint() {
    let (v, y) = xor();
    let mut t = trainer(Architecture::Mlp { hidden: 8, activation: HiddenActivation::Relu });
    t.params.patience = Some(2);
    let opposite: Vec<_> = y.iter().map(|r| vec![r[1], r[0]]).collect();
    let fit = t.fit(&v, &y, tags(), Some((&v, &opposite))).unwrap();
    assert!(fit.epochs < 600);
    assert!(fit.best_epoch < fit.epochs);
    let ann = DenseClassifierAnnotator::<Sentences>::new(fit.model);
    assert!(ann.annotate((&v,)).unwrap().0.check().is_ok());
}
#[test]
fn temperature_codec_and_provenance_cover_calibration() {
    let head =
        ClassifierHead::linear(space(), 1, tags(), HeadMode::SingleLabel, vec![10., -10.], vec![0., 0.]).unwrap();
    let model = DenseClassifierModel::new(head).unwrap();
    let v = vectors(vec![1., -1.], 1);
    let y = vec![vec![0., 1.], vec![1., 0.]];
    let calibrated = model.calibrate_temperature(&v, &y).unwrap();
    assert!(calibrated.temperature() > 1.);
    assert!(calibrated.probabilities(&[1.]).unwrap()[1] > model.probabilities(&[1.]).unwrap()[1]);
    let bytes = calibrated.to_bytes();
    assert_eq!(DenseClassifierModel::from_bytes(&bytes).unwrap(), calibrated);
    for i in 0..bytes.len() {
        assert!(DenseClassifierModel::from_bytes(&bytes[..i]).is_err(), "prefix {i}");
    }
    let mut tail = bytes.clone();
    tail.push(0);
    assert!(DenseClassifierModel::from_bytes(&tail).is_err());
    assert_ne!(
        DenseClassifierAnnotator::<Sentences>::new(model.clone()).config_hash(),
        DenseClassifierAnnotator::<Sentences>::new(calibrated).config_hash()
    );
    for t in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(model.clone().with_temperature(t).is_err());
    }
    assert!(model.probabilities(&[f32::NAN]).is_err());
    assert!(model.probabilities(&[]).is_err());
}
#[test]
fn encoder_mismatch_and_bad_vectors_are_rejected() {
    let (mut v, y) = xor();
    let model = trainer(Architecture::Logistic).fit(&v, &y, tags(), None).unwrap().model;
    let ann = DenseClassifierAnnotator::<Sentences>::new(model);
    for field in [0, 1, 2] {
        v.model = space();
        match field {
            0 => v.model.id.push('x'),
            1 => v.model.revision.push('x'),
            _ => v.model.sha256.push('x'),
        };
        assert!(ann.annotate((&v,)).is_err());
    }
    v.model = space();
    v.dim = 1;
    assert!(ann.annotate((&v,)).is_err());
    v.dim = 2;
    v.data[0] = f32::INFINITY;
    assert!(ann.annotate((&v,)).is_err());
    v.data[0] = 0.;
    v.data.pop();
    assert!(ann.annotate((&v,)).is_err());
}
#[test]
fn malformed_training_returns_errors() {
    let (v, y) = xor();
    let base = trainer(Architecture::Logistic);
    for value in [f64::NAN, f64::INFINITY, 0., -1.] {
        let mut t = base.clone();
        t.params.learning_rate = value;
        assert!(t.fit(&v, &y, tags(), None).is_err());
    }
    for which in 0..5 {
        let mut t = base.clone();
        match which {
            0 => t.params.batch_size = 0,
            1 => t.params.epochs = 0,
            2 => t.params.patience = Some(1),
            3 => t.params.weight_decay = f64::NAN,
            _ => t.architecture = Architecture::Mlp { hidden: u32::MAX, activation: HiddenActivation::Relu },
        };
        assert!(t.fit(&v, &y, tags(), None).is_err());
    }
    let mut bad = y.clone();
    bad[0] = vec![1.];
    assert!(base.fit(&v, &bad, tags(), None).is_err());
    bad = y.clone();
    bad[0][0] = f64::NAN;
    assert!(base.fit(&v, &bad, tags(), None).is_err());
    bad = y.clone();
    bad[0] = vec![1., 1.];
    assert!(base.fit(&v, &bad, tags(), None).is_err());
    assert!(base.fit(&v, &y, TagSet::new("labels", &["a", "a"]), None).is_err());
    let mut t = base;
    t.params.weight_decay = f64::MAX;
    assert!(t.fit(&v, &y, tags(), None).is_err());
}
#[test]
fn hostile_codec_counts_do_not_allocate_from_untrusted_dimensions() {
    let mut bytes = b"TDENSE01".to_vec();
    bytes.extend(1f64.to_le_bytes());
    bytes.extend(b"THEAD001");
    bytes.extend([0, 0, 1]);
    for _ in 0..4 {
        bytes.extend(0u32.to_le_bytes());
    }
    let offset = bytes.len();
    bytes.extend(u32::MAX.to_le_bytes());
    assert!(DenseClassifierModel::from_bytes(&bytes).is_err());
    bytes.truncate(offset);
    bytes.extend(1u32.to_le_bytes());
    bytes.extend(1u32.to_le_bytes());
    bytes.push(b'a');
    bytes.extend(u32::MAX.to_le_bytes());
    bytes.extend(u32::MAX.to_le_bytes());
    assert!(DenseClassifierModel::from_bytes(&bytes).is_err());
}

#[test]
fn sigmoid_threshold_calibration_and_tdense02_codec_work() {
    let ts = TagSet::new("topics", &["billing", "shipping", "refund"]);
    let head = ClassifierHead::linear(
        space(),
        3,
        ts.clone(),
        HeadMode::MultiLabel,
        vec![1., 0., 0., 0., 1., 0., 0., 0., 1.],
        vec![0., 0., 0.],
    )
    .unwrap();
    let model = DenseClassifierModel::new(head).unwrap();
    assert_eq!(model.thresholds(), &[0.5, 0.5, 0.5]);
    assert!(model.to_bytes().starts_with(b"TDENSE01"));

    // Label 0 has noisy negatives around logit 0.4 (p ~ 0.60) and true positives around logit 1.5 (p ~ 0.82).
    // Label 1 is well-separated at 0.5.
    // Label 2 is calibrated lower: true positives sit around logit -0.8 (p ~ 0.31), negatives around -2.5 (p ~ 0.08).
    let dev_v = vectors(
        vec![
            1.5, -2.0, -2.5, 1.4, -2.0, -2.5, 0.4, 2.0, -2.5, 0.5, -2.0, -0.7, 0.3, -2.0, -0.8, -2.0, 1.8, -0.6, -2.0,
            -2.0, -2.5,
        ],
        3,
    );
    let dev_y = vec![
        vec![1., 0., 0.],
        vec![1., 0., 0.],
        vec![0., 1., 0.],
        vec![0., 0., 1.],
        vec![0., 0., 1.],
        vec![0., 1., 1.],
        vec![0., 0., 0.],
    ];
    let dev_gold: Vec<BTreeSet<usize>> =
        dev_y.iter().map(|r| r.iter().enumerate().filter(|(_, v)| **v >= 0.5).map(|(i, _)| i).collect()).collect();

    let cal = ThresholdCalibrator::default();
    let tuned = model.clone().calibrate_thresholds(&dev_v, &dev_gold, &cal).unwrap();
    let tuned_from_y = model.clone().calibrate_thresholds_from_targets(&dev_v, &dev_y, &cal).unwrap();
    assert_eq!(tuned.thresholds(), tuned_from_y.thresholds());
    assert!(tuned.thresholds()[0] > 0.65, "{:?}", tuned.thresholds());
    assert!(tuned.thresholds()[2] < 0.35, "{:?}", tuned.thresholds());

    let def_report = model.evaluate_multilabel(&dev_v, &dev_gold).unwrap();
    let tuned_report = tuned.evaluate_multilabel(&dev_v, &dev_gold).unwrap();
    assert!(tuned_report.micro.f1 > def_report.micro.f1, "{tuned_report:?} vs {def_report:?}");
    assert!(tuned_report.r#macro.f1 > def_report.r#macro.f1, "{tuned_report:?} vs {def_report:?}");
    assert!((tuned_report.micro.f1 - 1.0).abs() < 1e-12);
    assert!((tuned_report.r#macro.f1 - 1.0).abs() < 1e-12);

    // Annotator writes selected multi-label tags and abstains with Tag::NONE when none clear thresholds.
    let ann = DenseClassifierAnnotator::<Sentences>::new(tuned.clone());
    let (cats,) = ann.annotate((&dev_v,)).unwrap();
    assert!(cats.check().is_ok());
    assert_eq!(cats.selected(0), &[Tag(0)]);
    assert_eq!(cats.selected(5), &[Tag(1), Tag(2)]);
    assert_eq!(cats.selected(6), &[]);
    assert_eq!(cats.category[6], Tag::NONE);

    // TDENSE02 round-trip, truncation, trailing bytes, and provenance invalidation.
    let bytes = tuned.to_bytes();
    assert!(bytes.starts_with(b"TDENSE02"));
    assert_eq!(DenseClassifierModel::from_bytes(&bytes).unwrap(), tuned);
    for i in 0..bytes.len() {
        assert!(DenseClassifierModel::from_bytes(&bytes[..i]).is_err(), "prefix {i}");
    }
    let mut tail = bytes.clone();
    tail.push(0);
    assert!(DenseClassifierModel::from_bytes(&tail).is_err());
    // An all-default (0.5) TDENSE02 payload is refused so from_bytes -> to_bytes preserves bytes and provenance.
    let mut all_default_v2 = b"TDENSE02".to_vec();
    all_default_v2.extend(model.temperature().to_le_bytes());
    all_default_v2.extend(3u32.to_le_bytes());
    for _ in 0..3 {
        all_default_v2.extend(0.5f64.to_le_bytes());
    }
    all_default_v2.extend(model.head().to_bytes());
    assert!(DenseClassifierModel::from_bytes(&all_default_v2).is_err());
    assert_ne!(
        DenseClassifierAnnotator::<Sentences>::new(model.clone()).config_hash(),
        DenseClassifierAnnotator::<Sentences>::new(tuned).config_hash()
    );

    // Reject invalid thresholds and Softmax threshold overrides.
    for bad in [vec![0.5, 0.5], vec![0.0, 0.5, 0.5], vec![1.0, 0.5, 0.5], vec![f64::NAN, 0.5, 0.5]] {
        assert!(model.clone().with_thresholds(bad).is_err());
    }
    let softmax_head =
        ClassifierHead::linear(space(), 1, tags(), HeadMode::SingleLabel, vec![1., -1.], vec![0., 0.]).unwrap();
    let softmax_model = DenseClassifierModel::new(softmax_head).unwrap();
    assert!(softmax_model.clone().with_thresholds(vec![0.5, 0.5]).is_err());
    assert!(softmax_model.clone().with_thresholds(vec![0.3, 0.7]).is_err());
    assert!(
        softmax_model
            .calibrate_thresholds(&vectors(vec![1., -1.], 1), &[BTreeSet::from([0]), BTreeSet::new()], &cal)
            .is_err()
    );
}

#[test]
fn tiny_bert_bundle_multilabel_sigmoid_head_calibrates_and_annotates() {
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let vec_ann = Arc::new(
        VectorAnnotator::open(
            &bundle,
            &VectorOptions {
                encoder: EncoderOptions { device: Device::Cpu, precision: Precision::Exact, ..Default::default() },
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let make_doc = |texts: &[&str]| -> Document {
        let mut full = String::new();
        let mut spans = Spans::new(ViewId::SOURCE);
        for s in texts {
            if !full.is_empty() {
                full.push(' ');
            }
            let start = full.len() as u32;
            full.push_str(s);
            spans.push(Span::new(start, full.len() as u32));
        }
        let mut doc = Document::new(full);
        doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test")).unwrap();
        doc
    };
    let corpus: [(&str, [f64; 3]); 12] = [
        ("Vector search indexes dense passage embeddings by cosine similarity.", [1., 0., 0.]),
        ("Semantic retrieval ranks candidate documents for each query vector.", [1., 0., 0.]),
        ("Approximate nearest neighbor search queries the vector index.", [1., 0., 0.]),
        ("GPU tensor cores accelerate matrix multiplication on device memory.", [0., 1., 0.]),
        ("Hardware accelerators execute quantized int8 neural network graphs.", [0., 1., 0.]),
        ("PCIe bandwidth limits batch throughput on the accelerator card.", [0., 1., 0.]),
        ("GPU clusters accelerate dense vector search and nearest neighbor retrieval.", [1., 1., 0.]),
        ("Hardware accelerators speed up semantic embedding search indices.", [1., 1., 0.]),
        ("Invoice billing charges the monthly subscription to the credit card.", [0., 0., 1.]),
        ("Account billing issued a refund for the duplicate invoice charge.", [0., 0., 1.]),
        ("Monthly cloud GPU billing charges per hardware accelerator hour.", [0., 1., 1.]),
        ("General status update with no specialized topic.", [0., 0., 0.]),
    ];
    let texts: Vec<&str> = corpus.iter().map(|(s, _)| *s).collect();
    let targets: Vec<Vec<f64>> = corpus.iter().map(|(_, y)| y.to_vec()).collect();
    let gold: Vec<BTreeSet<usize>> =
        targets.iter().map(|r| r.iter().enumerate().filter(|(_, v)| **v >= 0.5).map(|(i, _)| i).collect()).collect();
    let embed_pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(SubwordAnnotator::open(&bundle).unwrap())
        .add(SentenceVectors(vec_ann))
        .build()
        .unwrap();
    let mut docs = [make_doc(&texts)];
    embed_pipe.run(&mut docs).remove(0).unwrap();
    let vecs = docs[0].get::<Vectors<Sentences>>().unwrap();
    let ts = TagSet::new("domain", &["search", "hardware", "billing"]);
    let trainer = DenseClassifierTrainer {
        architecture: Architecture::Mlp { hidden: 16, activation: HiddenActivation::Gelu },
        output: OutputKind::Sigmoid,
        params: AdamParameters {
            seed: 7,
            learning_rate: 0.02,
            weight_decay: 1e-4,
            batch_size: 4,
            epochs: 250,
            patience: None,
        },
    };
    let fit = trainer.fit(vecs, &targets, ts, None).unwrap();
    let calibrated = fit.model.clone().calibrate_thresholds(vecs, &gold, &ThresholdCalibrator::default()).unwrap();
    let base_eval = fit.model.evaluate_multilabel(vecs, &gold).unwrap();
    let cal_eval = calibrated.evaluate_multilabel(vecs, &gold).unwrap();
    assert!(cal_eval.micro.f1 >= base_eval.micro.f1, "{cal_eval:?} vs {base_eval:?}");
    assert!(cal_eval.micro.f1 > 0.90, "micro F1: {}", cal_eval.micro.f1);
    assert!(cal_eval.r#macro.f1 > 0.90, "macro F1: {}", cal_eval.r#macro.f1);

    let (cats,) = DenseClassifierAnnotator::<Sentences>::new(calibrated).annotate((vecs,)).unwrap();
    assert!(cats.check().is_ok());
    assert_eq!(cats.len(), 12);
    assert_eq!(cats.selected(0), &[Tag(0)]);
    assert_eq!(cats.selected(6), &[Tag(0), Tag(1)]);
    assert_eq!(cats.selected(11), &[]);
    assert_eq!(cats.category[11], Tag::NONE);
}
