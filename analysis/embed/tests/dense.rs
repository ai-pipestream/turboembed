use std::sync::Arc;
use turbo_analysis::{Annotator, Layer, ModelRef, TagSet};
use turbo_analysis_embed::{
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
