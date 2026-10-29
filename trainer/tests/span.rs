//! The span classifier in training: its gradients against central finite
//! differences in f64 (adapter included), its refusals, and a run on
//! synthetic spans whose weights the core decodes as the trainer scored.

use turbo::manifest::{Adapter, Bilstm, Decode, HeadKind, HeadSpec, Scheme, SpanShape, WordPooling};
use turbo_trainer::fit_span::span_f1;
use turbo_trainer::rng::Rng;
use turbo_trainer::span::{GoldSpan, Spans, Work};
use turbo_trainer::{Config, SpanExample, fit_spans};

const HIDDEN: usize = 4;

fn spec(decode: Decode, adapter: u32) -> HeadSpec {
    HeadSpec {
        name: "mentions".into(),
        layer: "turboembed:mentions".into(),
        kind: HeadKind::SpanClassifier,
        labels: ["O", "X", "Y"].map(String::from).to_vec(),
        scheme: Scheme::Tags,
        word_pooling: WordPooling::First,
        adapter: (adapter > 0).then_some(Adapter { bilstm: Bilstm { hidden: adapter, layers: 1 } }),
        decode,
        biaffine: None,
        pos_input: None,
        hash_embed: None,
        span: Some(SpanShape { max_words: 3, width_dim: 2 }),
        coref: None,
        license_class: None,
        artifact: "weights".into(),
        tensor_prefix: "mentions".into(),
    }
}

fn gs(first: u32, last: u32, label: u32) -> GoldSpan {
    GoldSpan { first, last, label }
}

#[test]
fn gradients_match_finite_differences() {
    let s = spec(Decode::SpansNested, 3);
    let mut rng = Rng::new(3);
    let model = Spans::<f64>::new(&s, HIDDEN, &mut rng).unwrap();
    let features: Vec<f64> = (0..5 * HIDDEN).map(|_| rng.uniform(1.0)).collect();
    let gold = [gs(0, 1, 1), gs(1, 1, 2), gs(2, 4, 1)];
    let mut grads = model.zero_grads();
    let (_, count) = model.loss(&features, &gold, &mut grads, &mut Work::default()).unwrap();
    assert_eq!(count, 5 + 4 + 3);
    let eps = 1e-6;
    for (t, tensor) in grads.iter().enumerate() {
        for (i, &analytic) in tensor.iter().enumerate() {
            let at = |d: f64| {
                let mut m = model.clone();
                m.param_mut(t)[i] += d;
                let mut g = m.zero_grads();
                m.loss(&features, &gold, &mut g, &mut Work::default()).unwrap().0
            };
            let numeric = (at(eps) - at(-eps)) / (2.0 * eps);
            assert!(
                (numeric - analytic).abs() < 1e-6 * (1.0 + numeric.abs()),
                "{} [{i}]: {analytic} vs {numeric}",
                model.names()[t]
            );
        }
    }
}

#[test]
fn bad_gold_spans_are_refused_and_leave_gradients_alone() {
    let s = spec(Decode::SpansNested, 0);
    let model = Spans::<f32>::new(&s, HIDDEN, &mut Rng::new(1)).unwrap();
    let features = vec![0.5f32; 4 * HIDDEN];
    let mut grads = model.zero_grads();
    for bad in [[gs(0, 3, 1)], [gs(2, 4, 1)], [gs(1, 0, 1)], [gs(0, 0, 0)], [gs(0, 0, 3)]] {
        assert!(model.loss(&features, &bad, &mut grads, &mut Work::default()).is_err(), "{bad:?}");
    }
    assert!(model.loss(&features, &[gs(0, 0, 1), gs(0, 0, 2)], &mut grads, &mut Work::default()).is_err());
    assert!(grads.iter().flatten().all(|&g| g == 0.0));
    assert!(model.loss(&[f32::NAN; HIDDEN], &[], &mut grads, &mut Work::default()).is_err());
}

/// Words of class 0 (outside), 1 (X) and 2 (Y), noisy unit vectors; a
/// run of one class 1 to 3 words long is a gold span of its class, and a
/// class-0 word separates runs.
fn data(seed: u64, n: usize) -> Vec<SpanExample> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let (mut features, mut spans, mut at) = (Vec::new(), Vec::new(), 0u32);
            for _ in 0..2 + rng.below(4) {
                let class = 1 + rng.below(2);
                let len = 1 + rng.below(3) as u32;
                for c in std::iter::repeat_n(class, len as usize).chain([0]) {
                    for j in 0..HIDDEN {
                        features.push((if j == c { 1.0 } else { 0.0 } + rng.uniform(0.2)) as f32);
                    }
                }
                spans.push(gs(at, at + len - 1, class as u32));
                at += len + 1;
            }
            SpanExample { features, spans }
        })
        .collect()
}

#[test]
fn a_trained_classifier_finds_the_runs_and_the_core_agrees() {
    let s = spec(Decode::SpansFlat, 8);
    let c = Config { epochs: 60, batch: 8, lr: 3e-2, patience: 0, threads: 1, ..Config::default() };
    let (trained, report) = fit_spans(&s, HIDDEN, &data(1, 160), &data(2, 40), &c).unwrap();
    let best = report.dev[report.best_epoch];
    assert!(best > 0.9, "{report:?}");
    let core = trained.to_core().unwrap();
    assert_eq!(span_f1(&core, &data(2, 40)).unwrap(), best);
    let (again, _) = fit_spans(&s, HIDDEN, &data(1, 160), &data(2, 40), &c).unwrap();
    assert_eq!(again.params(), trained.params(), "the same seed gives the same bytes");
}

/// A zero BiLSTM of width 1 over one feature, and output weights of 1e38
/// for X and -1e38 for Y: the adapter's input weight gradients are the
/// output gradient (about 1e38) times the feature.
fn steep() -> Spans<f32> {
    let mut model = Spans::<f32>::new(&spec(Decode::SpansNested, 1), 1, &mut Rng::new(5)).unwrap();
    for t in 0..model.params().len() {
        let out = model.names()[t].ends_with("out.weight");
        let w = model.param_mut(t);
        let cols = w.len() / 3;
        for (k, v) in w.iter_mut().enumerate() {
            *v = if out { [0.0, 1e38, -1e38][k / cols] } else { 0.0 };
        }
    }
    model
}

#[test]
fn an_adapter_gradient_past_f32_is_refused_and_leaves_gradients_alone() {
    let model = steep();
    let mut grads = model.zero_grads();
    assert!(model.loss(&[1e38], &[gs(0, 0, 1)], &mut grads, &mut Work::default()).is_err());
    assert!(grads.iter().flatten().all(|&g| g == 0.0));
    // The same weights over a feature of 1e-3 add finite gradients.
    model.loss(&[1e-3], &[gs(0, 0, 1)], &mut grads, &mut Work::default()).unwrap();
    assert!(grads.iter().flatten().all(|g| g.is_finite()));
    assert!(grads.iter().flatten().any(|g| g.abs() > 1e33));
}

#[test]
fn a_sum_past_f32_is_refused_before_any_gradient_is_stored() {
    let model = steep();
    let mut local = model.zero_grads();
    model.loss(&[1e-3], &[gs(0, 0, 1)], &mut local, &mut Work::default()).unwrap();
    // Stored gradients of f32::MAX with each local gradient's sign: the
    // large ones would overflow when added.
    let mut grads: Vec<Vec<f32>> =
        local.iter().map(|t| t.iter().map(|&g| if g < 0.0 { -f32::MAX } else { f32::MAX }).collect()).collect();
    let before = grads.clone();
    assert!(model.loss(&[1e-3], &[gs(0, 0, 1)], &mut grads, &mut Work::default()).is_err());
    assert_eq!(grads, before, "nothing is stored when one sum overflows");
}
