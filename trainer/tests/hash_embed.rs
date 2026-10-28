//! A token tagger with hashed word embeddings (`hash_embed`) in training:
//! its gradients, the hash tables' among them, against central finite
//! differences in f64; a task only word identity solves, learned with the
//! tables and missed without them, unseen words placed by their suffix;
//! and the trained head scoring the same through the core.

use turbo::head::tagger::Work as CoreWork;
use turbo::manifest::{Adapter, Bilstm, Decode, HashEmbed, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_trainer::fit::score;
use turbo_trainer::rng::Rng;
use turbo_trainer::tagger::Work;
use turbo_trainer::{Config, Example, Tagger, fit};

const EPS: f64 = 1e-6;

fn spec(adapter: Option<(u32, u32)>, decode: Decode, hash: HashEmbed) -> HeadSpec {
    let labels: &[&str] = if decode == Decode::Crf { &["O", "B-X", "I-X"] } else { &["N", "V", "D"] };
    HeadSpec {
        name: "t".into(),
        layer: "turboembed:test".into(),
        kind: HeadKind::TokenTagger,
        labels: labels.iter().map(|s| s.to_string()).collect(),
        scheme: if decode == Decode::Crf { Scheme::Bio } else { Scheme::Tags },
        word_pooling: WordPooling::First,
        adapter: adapter.map(|(hidden, layers)| Adapter { bilstm: Bilstm { hidden, layers } }),
        decode,
        biaffine: None,
        hash_embed: Some(hash),
        artifact: "weights".into(),
        tensor_prefix: "t".into(),
    }
}

#[test]
fn hashed_gradients_match_finite_differences() {
    let small = HashEmbed { dim: 2, rows: [5, 3, 4, 3] };
    let forms = ["Ann", "met", "met", "Bob"];
    let cases = [
        (spec(None, Decode::Argmax, small), vec![0u32, 1, 1, 0]),
        (spec(Some((2, 1)), Decode::Argmax, small), vec![0, 1, 2, 0]),
        (spec(Some((2, 1)), Decode::Crf, small), vec![1, 2, 0, 1]),
    ];
    let hidden = 3;
    for (k, (s, gold)) in cases.into_iter().enumerate() {
        let mut rng = Rng::new(40 + k as u64);
        let mut model = Tagger::<f64>::new(&s, hidden, &mut rng).unwrap();
        for i in 0..model.params().len() {
            model.param_mut(i).iter_mut().for_each(|v| *v += rng.uniform(0.3));
        }
        let x: Vec<f64> = (0..gold.len() * hidden).map(|_| rng.uniform(1.0)).collect();
        let mut grads = model.zero_grads();
        model.loss_forms(&x, Some(&forms), &gold, &mut grads, &mut Work::default()).unwrap();
        assert!(model.loss(&x, &gold, &mut model.zero_grads(), &mut Work::default()).is_err(), "forms are required");
        for (i, g) in grads.iter().enumerate() {
            let name = model.names()[i].clone();
            for (j, &gj) in g.iter().enumerate() {
                let at = |d: f64| {
                    let mut m = model.clone();
                    m.param_mut(i)[j] += d;
                    m.loss_forms(&x, Some(&forms), &gold, &mut m.zero_grads(), &mut Work::default()).unwrap()
                };
                let numeric = (at(EPS) - at(-EPS)) / (2.0 * EPS);
                let tol = 1e-6 * (1.0 + numeric.abs().max(gj.abs()));
                assert!((numeric - gj).abs() <= tol, "case {k} {name}[{j}]: numeric {numeric}, analytic {gj}");
            }
        }
        assert!(model.names().iter().any(|n| n == "hash.suffix"));
    }
}

const HIDDEN: usize = 4;
const NOUNS: [&str; 6] = ["dog", "cat", "tree", "house", "river", "stone"];
const VERBS: [&str; 6] = ["running", "eating", "singing", "walking", "jumping", "reading"];

/// Sentences whose tags only the words' text tells: every encoder feature
/// is noise. Nouns are N, `-ing` verbs V, `the` D.
fn data(seed: u64, n: usize, nouns: &[&str], verbs: &[&str]) -> Vec<Example> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let len = 3 + rng.below(5);
            let (mut features, mut labels, mut forms) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..len {
                let (form, tag) = match rng.below(3) {
                    0 => (nouns[rng.below(nouns.len())], 0),
                    1 => (verbs[rng.below(verbs.len())], 1),
                    _ => ("the", 2),
                };
                features.extend((0..HIDDEN).map(|_| rng.uniform(1.0) as f32));
                labels.push(tag);
                forms.push(form.to_string());
            }
            Example { features, labels, forms }
        })
        .collect()
}

#[test]
fn word_identity_is_learned_through_the_tables_and_reaches_unseen_words() {
    let hash = HashEmbed { dim: 8, rows: [64, 16, 32, 16] };
    let s = spec(None, Decode::Argmax, hash);
    let (train, dev) = (data(1, 200, &NOUNS[..4], &VERBS[..4]), data(2, 40, &NOUNS[..4], &VERBS[..4]));
    let c = Config { epochs: 30, lr: 2e-2, patience: 5, ..Config::default() };
    let (model, report) = fit(&s, HIDDEN, &train, &dev, &c).unwrap();
    let best = report.dev[report.best_epoch];
    assert!(best >= 0.99, "dev accuracy {best}: {report:?}");
    let core = model.to_token_tagger().unwrap();
    assert!(core.reads_forms());
    assert_eq!(score(&core, &dev).unwrap(), best);

    // Words never seen in training: the -ing suffix still marks a verb.
    let unseen = data(3, 40, &NOUNS[4..], &VERBS[4..]);
    let verbs: Vec<Example> = unseen
        .into_iter()
        .map(|mut e| {
            e.labels.iter_mut().zip(&e.forms).for_each(|(l, f)| *l = if f.ends_with("ing") { 1 } else { *l });
            e
        })
        .collect();
    let (mut scores, mut tags, mut work) = (Vec::new(), Vec::new(), CoreWork::default());
    let (mut right, mut total) = (0, 0);
    for e in &verbs {
        let forms = e.forms().unwrap();
        let ids: Vec<u32> = (0..forms.len() as u32).collect();
        core.scores_forms(&e.features, &ids, Some(&forms), &mut scores, &mut work).unwrap();
        core.decode(&scores, forms.len(), &mut tags).unwrap();
        for ((t, l), f) in tags.iter().zip(&e.labels).zip(&forms) {
            if f.ends_with("ing") {
                right += (t == l) as usize;
                total += 1;
            }
        }
    }
    assert!(right as f64 >= 0.9 * total as f64, "unseen -ing verbs: {right} of {total}");

    // The same data without the tables: the features are noise.
    let mut plain = s.clone();
    plain.hash_embed = None;
    let (_, r) = fit(&plain, HIDDEN, &train, &dev, &c).unwrap();
    assert!(r.dev[r.best_epoch] < 0.7, "without word identity {:?}", r.dev);
}

#[test]
fn a_head_that_reads_forms_refuses_examples_without_them() {
    let s = spec(None, Decode::Argmax, HashEmbed { dim: 2, rows: [4; 4] });
    let mut train = data(4, 5, &NOUNS, &VERBS);
    train[2].forms.pop();
    let e = fit(&s, HIDDEN, &train, &train, &Config { epochs: 1, ..Config::default() }).unwrap_err();
    assert!(e.message.contains("forms"), "{e:?}");
}

#[test]
fn the_trained_head_scores_the_same_through_the_core() {
    let s = spec(Some((3, 1)), Decode::Crf, HashEmbed { dim: 3, rows: [7, 5, 6, 5] });
    let mut rng = Rng::new(11);
    let model = Tagger::<f32>::new(&s, HIDDEN, &mut rng).unwrap();
    let core = model.to_token_tagger().unwrap();
    let forms = ["Ann", "met", "Bob", "in", "Paris"];
    let features: Vec<f32> = (0..forms.len() * HIDDEN).map(|_| rng.uniform(1.0) as f32).collect();
    let mut w = Work::default();
    model.forward_forms(&features, Some(&forms), &mut w).unwrap();
    let ids: Vec<u32> = (0..forms.len() as u32).collect();
    let mut scores = Vec::new();
    core.scores_forms(&features, &ids, Some(&forms), &mut scores, &mut CoreWork::default()).unwrap();
    let diff = w.scores().iter().zip(&scores).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(diff <= 1e-5, "trainer and core differ by {diff}");
}
