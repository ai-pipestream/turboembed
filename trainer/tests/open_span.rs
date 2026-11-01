//! The open-label span matcher in training: its gradients against central
//! finite differences in f64 (BiLSTM adapter, leaky ReLU, span projection,
//! and label projection included), its refusals, and a run on synthetic
//! spans that generalizes to both seen and held-out label vectors and
//! agrees with `turbo::head::OpenSpanMatcher`.

use turbo::manifest::{Adapter, Bilstm, Decode, HeadKind, HeadSpec, Scheme, SpanShape, WordPooling};
use turbo_trainer::fit_open_span::open_span_f1;
use turbo_trainer::open_span::{OpenSpans, Work};
use turbo_trainer::rng::Rng;
use turbo_trainer::span::GoldSpan;
use turbo_trainer::{Config, OpenSpanExample, fit_open_spans};

const HIDDEN: usize = 4;

fn spec(decode: Decode, adapter: u32) -> HeadSpec {
    HeadSpec {
        name: "open_ner".into(),
        layer: "turboembed:entities".into(),
        kind: HeadKind::OpenSpan,
        labels: ["person", "org"].map(String::from).to_vec(),
        scheme: Scheme::Tags,
        word_pooling: WordPooling::First,
        adapter: (adapter > 0).then_some(Adapter { bilstm: Bilstm { hidden: adapter, layers: 1 } }),
        decode,
        biaffine: None,
        pos_input: None,
        hash_embed: None,
        layer_mix: None,
        span: Some(SpanShape { max_words: 3, width_dim: 3 }),
        coref: None,
        relation: None,
        license_class: None,
        artifact: "weights".into(),
        tensor_prefix: "open_ner".into(),
    }
}

fn gs(first: u32, last: u32, label: u32) -> GoldSpan {
    GoldSpan { first, last, label }
}

#[test]
fn gradients_match_finite_differences() {
    let s = spec(Decode::SpansNested, 3);
    let mut rng = Rng::new(7);
    let mut model = OpenSpans::<f64>::new(&s, HIDDEN, &mut rng).unwrap();
    // Perturb biases slightly so no pre-activation sits at the 0.0 kink of leaky ReLU.
    for t in 0..model.params().len() {
        for v in model.param_mut(t) {
            if *v == 0.0 {
                *v = rng.uniform(0.2);
            }
        }
    }
    let features: Vec<f64> = (0..5 * HIDDEN).map(|_| rng.uniform(1.0)).collect();
    let label_features: Vec<f64> = (0..3 * HIDDEN).map(|_| rng.uniform(1.0)).collect();
    let gold = [gs(0, 1, 0), gs(1, 1, 2), gs(2, 4, 1)];
    let mut grads = model.zero_grads();
    let (_, count) = model.loss(&features, &label_features, &gold, &mut grads, &mut Work::default()).unwrap();
    assert_eq!(count, 5 + 4 + 3);
    let eps = 1e-6;
    for (t, tensor) in grads.iter().enumerate() {
        for (i, &analytic) in tensor.iter().enumerate() {
            let at = |d: f64| {
                let mut m = model.clone();
                m.param_mut(t)[i] += d;
                let mut g = m.zero_grads();
                m.loss(&features, &label_features, &gold, &mut g, &mut Work::default()).unwrap().0
            };
            let numeric = (at(eps) - at(-eps)) / (2.0 * eps);
            assert!(
                (numeric - analytic).abs() < 1e-5 * (1.0 + numeric.abs()),
                "{} [{i}]: {analytic} vs {numeric}",
                model.names()[t]
            );
        }
    }
}

#[test]
fn bad_gold_spans_and_label_features_are_refused_and_leave_gradients_alone() {
    let s = spec(Decode::SpansNested, 0);
    let model = OpenSpans::<f32>::new(&s, HIDDEN, &mut Rng::new(1)).unwrap();
    let features = vec![0.5f32; 4 * HIDDEN];
    let labels = vec![0.25f32; 2 * HIDDEN];
    let mut grads = model.zero_grads();
    for bad in [[gs(0, 3, 0)], [gs(2, 4, 0)], [gs(1, 0, 0)], [gs(0, 0, 2)]] {
        assert!(model.loss(&features, &labels, &bad, &mut grads, &mut Work::default()).is_err(), "{bad:?}");
    }
    assert!(model.loss(&features, &labels, &[gs(0, 0, 0), gs(0, 0, 1)], &mut grads, &mut Work::default()).is_err());
    assert!(model.loss(&features, &[], &[], &mut grads, &mut Work::default()).is_err());
    assert!(model.loss(&features, &[0.1; HIDDEN + 1], &[], &mut grads, &mut Work::default()).is_err());
    assert!(model.loss(&features, &[f32::NAN; HIDDEN], &[], &mut grads, &mut Work::default()).is_err());
    assert!(model.loss(&[f32::NAN; HIDDEN], &labels, &[], &mut grads, &mut Work::default()).is_err());
    assert!(grads.iter().flatten().all(|&g| g == 0.0));
}

/// Synthetic dataset where each example supplies two caller label directions
/// in the 3D entity subspace `span(e_1, e_2, e_3)` (while `e_0` is the
/// outside word direction). Seen examples use two label directions; held-out
/// zero-shot examples use new label directions in the same subspace that were
/// never seen during training.
fn data(seed: u64, n: usize, label_dirs: &[[f32; HIDDEN]; 2]) -> Vec<OpenSpanExample> {
    let mut rng = Rng::new(seed);
    let mut label_features = Vec::with_capacity(2 * HIDDEN);
    for dir in label_dirs {
        label_features.extend_from_slice(dir);
    }
    (0..n)
        .map(|_| {
            let (mut features, mut spans, mut at) = (Vec::new(), Vec::new(), 0u32);
            for _ in 0..2 + rng.below(3) {
                let label = rng.below(2) as u32;
                let len = 1 + rng.below(2) as u32;
                let dir = &label_dirs[label as usize];
                for _ in 0..len {
                    for d in dir {
                        features.push(d + rng.uniform(0.1) as f32);
                    }
                }
                // Outside separator word along e_0.
                features.extend([1.0 + rng.uniform(0.1) as f32, 0.0, 0.0, 0.0]);
                spans.push(gs(at, at + len - 1, label));
                at += len + 1;
            }
            OpenSpanExample { features, label_features: label_features.clone(), spans }
        })
        .collect()
}

#[test]
fn a_trained_open_span_matcher_generalizes_to_seen_and_unseen_label_vectors() {
    let s = spec(Decode::SpansFlat, 8);
    let seen_a: [[f32; HIDDEN]; 2] = [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
    let seen_b: [[f32; HIDDEN]; 2] = [[0.0, 0.0, 0.0, 1.0], [0.0, -1.0, 0.0, 0.0]];
    let mut train_set = data(1, 100, &seen_a);
    train_set.extend(data(2, 100, &seen_b));
    let dev_seen = data(3, 40, &seen_a);
    // Held-out zero-shot label vectors: diagonal combinations never passed in train_set.
    let s2 = std::f32::consts::FRAC_1_SQRT_2;
    let unseen_dirs: [[f32; HIDDEN]; 2] = [[0.0, s2, s2, 0.0], [0.0, 0.0, -s2, s2]];
    let test_unseen = data(4, 40, &unseen_dirs);

    let c = Config { epochs: 50, batch: 8, lr: 2e-2, patience: 0, threads: 1, ..Config::default() };
    let (trained, report) = fit_open_spans(&s, HIDDEN, &train_set, &dev_seen, &c).unwrap();
    let best = report.dev[report.best_epoch];
    assert!(best > 0.9, "seen dev F1: {report:?}");

    let core = trained.to_core().unwrap();
    assert_eq!(open_span_f1(&core, &dev_seen).unwrap(), best);

    let unseen_f1 = open_span_f1(&core, &test_unseen).unwrap();
    assert!(unseen_f1 > 0.85, "zero-shot unseen label F1 was {unseen_f1}");

    let (again, _) = fit_open_spans(&s, HIDDEN, &train_set, &dev_seen, &c).unwrap();
    assert_eq!(again.params(), trained.params(), "the same seed gives the same bytes");
}

fn steep() -> OpenSpans<f32> {
    let mut model = OpenSpans::<f32>::new(&spec(Decode::SpansNested, 1), 1, &mut Rng::new(5)).unwrap();
    for t in 0..model.params().len() {
        let name = model.names()[t].clone();
        let w = model.param_mut(t);
        match name.as_str() {
            "span.weight" | "label.bias" => w.fill(5e18),
            _ => w.fill(0.0),
        }
    }
    model
}

#[test]
fn an_adapter_gradient_past_f32_is_refused_and_leaves_gradients_alone() {
    let model = steep();
    let mut grads = model.zero_grads();
    assert!(model.loss(&[1e38], &[1.0], &[gs(0, 0, 0)], &mut grads, &mut Work::default()).is_err());
    assert!(grads.iter().flatten().all(|&g| g == 0.0));
    model.loss(&[1e-3], &[1.0], &[gs(0, 0, 0)], &mut grads, &mut Work::default()).unwrap();
    assert!(grads.iter().flatten().all(|g| g.is_finite()));
}

#[test]
fn a_sum_past_f32_is_refused_before_any_gradient_is_stored() {
    let model = steep();
    let mut local = model.zero_grads();
    model.loss(&[1e-3], &[1.0], &[gs(0, 0, 0)], &mut local, &mut Work::default()).unwrap();
    let mut grads: Vec<Vec<f32>> =
        local.iter().map(|t| t.iter().map(|&g| if g < 0.0 { -f32::MAX } else { f32::MAX }).collect()).collect();
    let before = grads.clone();
    assert!(model.loss(&[1e-3], &[1.0], &[gs(0, 0, 0)], &mut grads, &mut Work::default()).is_err());
    assert_eq!(grads, before, "nothing is stored when one sum overflows");
}
