//! The relation classifier in training: its gradients against central
//! finite differences in f64 (adapter, mention pooling and type
//! embeddings included), its refusals, and a run on synthetic entity
//! pairs, some holding two relations, whose weights the core decodes as
//! the trainer scored, falling in loss and the same bytes for the same
//! seed.

use turbo::head::relation::Mention;
use turbo::manifest::{Adapter, Bilstm, Decode, HeadKind, HeadSpec, RelationShape, Scheme, WordPooling};
use turbo_trainer::fit_relation::{relation_counts, relation_f1};
use turbo_trainer::relation::{GoldRelation, Relations, Work};
use turbo_trainer::rng::Rng;
use turbo_trainer::{Config, RelationExample, fit_relations};

const HIDDEN: usize = 4;

fn spec(adapter: u32, types: &[&str], type_dim: u32) -> HeadSpec {
    HeadSpec {
        name: "relations".into(),
        layer: "turboembed:relations".into(),
        kind: HeadKind::RelationClassifier,
        labels: ["none", "works_for", "rival_of", "leads"].map(String::from).to_vec(),
        scheme: Scheme::Tags,
        word_pooling: WordPooling::First,
        adapter: (adapter > 0).then_some(Adapter { bilstm: Bilstm { hidden: adapter, layers: 1 } }),
        decode: Decode::Relations,
        biaffine: None,
        pos_input: None,
        hash_embed: None,
        span: None,
        relation: Some(RelationShape {
            mlp: 5,
            entity_labels: types.iter().map(|t| t.to_string()).collect(),
            type_dim,
        }),
        license_class: None,
        artifact: "weights".into(),
        tensor_prefix: "relations".into(),
    }
}

fn mention(first: u32, last: u32, entity: u32) -> Mention {
    Mention { first, last, entity }
}

fn gr(head: u32, tail: u32, label: u32) -> GoldRelation {
    GoldRelation { head, tail, label }
}

#[test]
fn gradients_match_finite_differences() {
    let s = spec(3, &["PER", "ORG"], 2);
    let mut rng = Rng::new(3);
    let mut model = Relations::<f64>::new(&s, HIDDEN, &mut rng).unwrap();
    // The biaffine starts at zero, which would leave the MLPs without a
    // gradient: give every tensor random weights.
    for t in 0..model.params().len() {
        model.param_mut(t).iter_mut().for_each(|v| *v = rng.uniform(0.8));
    }
    let features: Vec<f64> = (0..6 * HIDDEN).map(|_| rng.uniform(1.0)).collect();
    // Entity 0 is two mentions; pair (0, 1) holds two relations.
    let mentions = [mention(0, 1, 0), mention(2, 2, 1), mention(3, 4, 2), mention(5, 5, 0)];
    let types = [0, 1, 0];
    let gold = [gr(0, 1, 1), gr(0, 1, 3), gr(2, 1, 2), gr(1, 0, 2)];
    let mut grads = model.zero_grads();
    let (_, pairs) = model.loss(&features, &mentions, &types, &gold, &mut grads, &mut Work::default()).unwrap();
    assert_eq!(pairs, 6);
    let eps = 1e-6;
    for (t, tensor) in grads.iter().enumerate() {
        assert!(tensor.iter().any(|&g| g != 0.0), "{} has no gradient", model.names()[t]);
        for (i, &analytic) in tensor.iter().enumerate() {
            let at = |d: f64| {
                let mut m = model.clone();
                m.param_mut(t)[i] += d;
                let mut g = m.zero_grads();
                m.loss(&features, &mentions, &types, &gold, &mut g, &mut Work::default()).unwrap().0
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
fn bad_mentions_and_gold_relations_are_refused_and_leave_gradients_alone() {
    let model = Relations::<f32>::new(&spec(0, &["PER"], 1), HIDDEN, &mut Rng::new(1)).unwrap();
    let features = vec![0.5f32; 4 * HIDDEN];
    let ok = [mention(0, 0, 0), mention(1, 2, 1)];
    let mut grads = model.zero_grads();
    let mut w = Work::default();
    for bad in [mention(2, 1, 1), mention(0, 4, 1), mention(0, 0, 2)] {
        assert!(model.loss(&features, &[ok[0], bad], &[0, 0], &[], &mut grads, &mut w).is_err(), "{bad:?}");
    }
    // An entity of no mention, and a type the head lacks.
    assert!(model.loss(&features, &ok[..1], &[0, 0], &[], &mut grads, &mut w).is_err());
    assert!(model.loss(&features, &ok, &[0, 1], &[], &mut grads, &mut w).is_err());
    for bad in [[gr(0, 0, 1)], [gr(0, 2, 1)], [gr(0, 1, 0)], [gr(0, 1, 4)]] {
        assert!(model.loss(&features, &ok, &[0, 0], &bad, &mut grads, &mut w).is_err(), "{bad:?}");
    }
    assert!(model.loss(&features, &ok, &[0, 0], &[gr(0, 1, 1), gr(0, 1, 1)], &mut grads, &mut w).is_err());
    assert!(model.loss(&[f32::NAN; HIDDEN], &[], &[], &[], &mut grads, &mut w).is_err());
    assert!(grads.iter().flatten().all(|&g| g == 0.0));
    assert_eq!(model.loss(&features, &ok[..1], &[0], &[], &mut grads, &mut w).unwrap(), (0.0, 0));
    // Two relations on one pair are a gold set, not a refusal.
    model.loss(&features, &ok, &[0, 0], &[gr(0, 1, 1), gr(0, 1, 3)], &mut grads, &mut w).unwrap();
}

/// Entities of a PER (type 0) or an ORG (type 1), each one or two
/// mentions of one or two words, each word a noisy vector of its type,
/// its side (+1 or -1) and, for a PER, whether it is senior, with filler
/// words between them. A PER works for an ORG of its side and is a rival
/// of an ORG of the other, and a senior PER also leads an ORG of its
/// side, so the labels turn on both entities at once and some pairs hold
/// two.
fn data(seed: u64, n: usize) -> Vec<RelationExample> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let (mut features, mut mentions, mut types, mut facts) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut at = 0u32;
            for e in 0..2 + rng.below(3) as u32 {
                let (kind, side) = (rng.below(2) as u32, if rng.below(2) == 0 { 1.0 } else { -1.0 });
                let senior = kind == 0 && rng.below(2) == 0;
                for _ in 0..1 + rng.below(2) {
                    let len = 1 + rng.below(2) as u32;
                    for _ in 0..len {
                        let v = [f64::from(kind == 0), f64::from(kind == 1), side, f64::from(senior)];
                        features.extend(v.iter().map(|x| (x + rng.uniform(0.2)) as f32));
                    }
                    features.extend((0..HIDDEN).map(|_| rng.uniform(0.2) as f32));
                    mentions.push(mention(at, at + len - 1, e));
                    at += len + 1;
                }
                types.push(kind);
                facts.push((kind, side, senior));
            }
            let mut relations = Vec::new();
            for (a, &(ka, sa, senior)) in facts.iter().enumerate() {
                for (b, &(kb, sb, _)) in facts.iter().enumerate() {
                    if ka == 0 && kb == 1 {
                        relations.push(gr(a as u32, b as u32, if sa == sb { 1 } else { 2 }));
                        if senior && sa == sb {
                            relations.push(gr(a as u32, b as u32, 3));
                        }
                    }
                }
            }
            RelationExample { features, mentions, types, relations }
        })
        .collect()
}

#[test]
fn a_trained_classifier_finds_the_relations_and_the_core_agrees() {
    let s = spec(0, &["PER", "ORG"], 2);
    let c = Config { epochs: 40, batch: 8, lr: 2e-2, patience: 0, threads: 1, ..Config::default() };
    let (trained, report) = fit_relations(&s, HIDDEN, &data(1, 200), &data(2, 50), &c).unwrap();
    let best = report.dev[report.best_epoch];
    assert!(best > 0.95, "{report:?}");
    assert!(report.train_loss.last().unwrap() < &(report.train_loss[0] / 4.0), "{:?}", report.train_loss);
    let core = trained.to_core().unwrap();
    assert_eq!(relation_f1(&core, &data(2, 50)).unwrap(), best);
    let counts = relation_counts(&core, &data(2, 50)).unwrap();
    assert_eq!(counts[0].found + counts[0].gold, 0, "none is never a relation");
    assert!(counts[1..].iter().all(|c| c.f1() > 0.9), "{counts:?}");
    let threads = Config { threads: 4, ..c };
    let (again, _) = fit_relations(&s, HIDDEN, &data(1, 200), &data(2, 50), &threads).unwrap();
    assert_eq!(again.params(), trained.params(), "the same seed gives the same bytes on any thread count");
}

#[test]
fn an_untyped_head_with_an_adapter_trains_and_round_trips_its_tensors() {
    let s = spec(4, &[], 0);
    let strip = |d: Vec<RelationExample>| -> Vec<RelationExample> {
        d.into_iter()
            .map(|mut e| {
                e.types.iter_mut().for_each(|t| *t = 0);
                e
            })
            .collect()
    };
    let c = Config { epochs: 30, batch: 8, lr: 2e-2, patience: 0, threads: 2, ..Config::default() };
    let (trained, report) = fit_relations(&s, HIDDEN, &strip(data(5, 160)), &strip(data(6, 40)), &c).unwrap();
    assert!(report.dev[report.best_epoch] > 0.9, "{report:?}");
    let names: Vec<String> = trained.tensors().into_iter().map(|(n, _)| n).collect();
    assert!(names.iter().all(|n| n != "relation.type") && names.iter().any(|n| n.starts_with("lstm.0.")));
    let rebuilt = Relations::<f32>::with_params(&s, HIDDEN, trained.params().to_vec()).unwrap();
    let core = rebuilt.to_core().unwrap();
    assert_eq!(relation_f1(&core, &strip(data(6, 40))).unwrap(), report.dev[report.best_epoch]);
}
