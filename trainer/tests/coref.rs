//! The coreference ranker in training: its gradients against central
//! finite differences in f64 (adapter included), its refusals, and a run
//! on synthetic documents whose weights the core decodes as the trainer
//! scored.

use turbo::manifest::{Adapter, Bilstm, CorefShape, Decode, HeadKind, HeadSpec, Scheme, SpanShape, WordPooling};
use turbo_trainer::coref::{Ranker, Work};
use turbo_trainer::fit_coref::antecedent_accuracy;
use turbo_trainer::rng::Rng;
use turbo_trainer::{Config, CorefExample, fit_coref};

const HIDDEN: usize = 5;

fn spec(adapter: u32, antecedents: u32) -> HeadSpec {
    HeadSpec {
        name: "coref".into(),
        layer: "turboembed:coref".into(),
        kind: HeadKind::CorefRanker,
        labels: vec!["coref".into()],
        scheme: Scheme::Tags,
        word_pooling: WordPooling::First,
        adapter: (adapter > 0).then_some(Adapter { bilstm: Bilstm { hidden: adapter, layers: 1 } }),
        decode: Decode::Antecedent,
        biaffine: None,
        pos_input: None,
        hash_embed: None,
        span: Some(SpanShape { max_words: 2, width_dim: 2 }),
        coref: Some(CorefShape { dim: 4, antecedents }),
        license_class: None,
        artifact: "weights".into(),
        tensor_prefix: "coref".into(),
    }
}

#[test]
fn gradients_match_finite_differences() {
    let s = spec(2, 2);
    let mut rng = Rng::new(5);
    let model = Ranker::<f64>::new(&s, HIDDEN, &mut rng).unwrap();
    let mut model = model;
    // Nonzero distance scores and biases, so every path carries gradient.
    for i in 0..model.params().len() {
        for v in model.param_mut(i) {
            *v += rng.uniform(0.3);
        }
    }
    let features: Vec<f64> = (0..6 * HIDDEN).map(|_| rng.uniform(1.0)).collect();
    let mentions = [(0, 0), (0, 1), (2, 2), (3, 4), (5, 5)];
    let chains = [0, 1, 0, 1, 0];
    let mut grads = model.zero_grads();
    let (_, m) = model.loss(&features, &mentions, &chains, &mut grads, &mut Work::default()).unwrap();
    assert_eq!(m, 5);
    let eps = 1e-6;
    for (t, tensor) in grads.iter().enumerate() {
        for (i, &analytic) in tensor.iter().enumerate() {
            let at = |d: f64| {
                let mut m = model.clone();
                m.param_mut(t)[i] += d;
                let mut g = m.zero_grads();
                m.loss(&features, &mentions, &chains, &mut g, &mut Work::default()).unwrap().0
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
fn bad_documents_are_refused_and_leave_gradients_alone() {
    let model = Ranker::<f32>::new(&spec(0, 2), HIDDEN, &mut Rng::new(1)).unwrap();
    let features = vec![0.5f32; 3 * HIDDEN];
    let mut grads = model.zero_grads();
    let mut w = Work::default();
    assert!(model.loss(&features, &[(0, 0), (1, 1)], &[0], &mut grads, &mut w).is_err(), "a chain each");
    assert!(model.loss(&features, &[(1, 1), (0, 0)], &[0, 0], &mut grads, &mut w).is_err(), "order");
    assert!(model.loss(&features, &[(0, 2)], &[0], &mut grads, &mut w).is_err(), "too long");
    assert!(model.loss(&features, &[(2, 3)], &[0], &mut grads, &mut w).is_err(), "outside");
    assert!(grads.iter().flatten().all(|&g| g == 0.0));
    assert_eq!(model.loss(&features, &[], &[], &mut grads, &mut w).unwrap(), (0.0, 0));
}

/// Documents of one-word mentions separated by filler words; each
/// mention's features carry its entity (one of three, on axes 0 to 2)
/// and filler words sit on axis 4, so mentions of one entity corefer.
fn data(seed: u64, n: usize) -> Vec<CorefExample> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let (mut features, mut mentions, mut chains) = (Vec::new(), Vec::new(), Vec::new());
            let mut at = 0u32;
            for _ in 0..3 + rng.below(5) {
                let entity = rng.below(3);
                for (axis, mention) in [(entity, true), (4, false)] {
                    for j in 0..HIDDEN {
                        features.push((if j == axis { 1.0 } else { 0.0 } + rng.uniform(0.15)) as f32);
                    }
                    if mention {
                        mentions.push((at, at));
                        chains.push(entity as u32);
                    }
                    at += 1;
                }
            }
            CorefExample { features, mentions, chains }
        })
        .collect()
}

#[test]
fn a_trained_ranker_links_the_mentions_of_one_entity() {
    let s = spec(0, 8);
    let c = Config { epochs: 40, batch: 8, lr: 2e-2, patience: 0, threads: 1, ..Config::default() };
    let (trained, report) = fit_coref(&s, HIDDEN, &data(1, 120), &data(2, 30), &c).unwrap();
    let best = report.dev[report.best_epoch];
    assert!(best > 0.95, "{report:?}");
    assert_eq!(antecedent_accuracy(&trained.to_core().unwrap(), &data(2, 30)).unwrap(), best);
    let (again, _) = fit_coref(&s, HIDDEN, &data(1, 120), &data(2, 30), &c).unwrap();
    assert_eq!(again.params(), trained.params(), "the same seed gives the same bytes");
}

#[test]
fn an_adapter_gradient_past_f32_is_refused_and_leaves_gradients_alone() {
    // Codex's case: hidden 1, a BiLSTM of width 1 whose candidate biases
    // are 1 and other weights 0, mention weights of 1 and a pair weight
    // of 1e10. Over features of 3e38 the loss is finite but the adapter's
    // input weight gradients are not.
    let mut s = spec(1, 2);
    s.coref = Some(CorefShape { dim: 1, antecedents: 2 });
    let mut model = Ranker::<f32>::new(&s, 1, &mut Rng::new(9)).unwrap();
    for t in 0..model.params().len() {
        let name = model.names()[t].clone();
        let w = model.param_mut(t);
        w.fill(0.0);
        if name.starts_with("lstm.") && name.ends_with(".b") {
            w[2] = 1.0;
        } else if name == "mention.weight" {
            w.fill(1.0);
        } else if name == "pair.weight" {
            w.fill(1e10);
        }
    }
    let (mentions, chains) = ([(0, 0), (1, 1)], [0, 1]);
    let mut grads = model.zero_grads();
    let mut w = Work::default();
    assert!(model.loss(&[3e38, 3e38], &mentions, &chains, &mut grads, &mut w).is_err());
    assert!(grads.iter().flatten().all(|&g| g == 0.0), "nothing is stored");
    // The same weights over features of normal scale add finite gradients.
    model.loss(&[1.0, 1.0], &mentions, &chains, &mut grads, &mut w).unwrap();
    assert!(grads.iter().flatten().all(|g| g.is_finite()));
    // At features of 1e25 the largest gradient is about 1.6e34: finite,
    // but stored sums of f32::MAX with its sign would overflow.
    let mut big = model.zero_grads();
    model.loss(&[1e25, 1e25], &mentions, &chains, &mut big, &mut w).unwrap();
    assert!(big.iter().flatten().any(|g| g.abs() > 1e33));
    let mut full: Vec<Vec<f32>> =
        big.iter().map(|t| t.iter().map(|&g| if g < 0.0 { -f32::MAX } else { f32::MAX }).collect()).collect();
    let before = full.clone();
    assert!(model.loss(&[1e25, 1e25], &mentions, &chains, &mut full, &mut w).is_err());
    assert_eq!(full, before);
}
