//! The edit-tree lemmatizer in training: loss gradients against central
//! finite differences in f64 (with and without the adapter and the tag
//! input), its scores against the core's, tree mining, and toy runs that
//! learn suffix rules and need the tags to tell two rules apart.

use turbo::head::edit_tree::EditTree;
use turbo::head::lemmatizer::Work as CoreWork;
use turbo::manifest::{Adapter, Bilstm, Decode, HeadKind, HeadSpec, PosInput, Scheme, WordPooling};
use turbo_trainer::fit_lemma::accuracy;
use turbo_trainer::lemma::Work;
use turbo_trainer::rng::Rng;
use turbo_trainer::{Config, LemmaExample, Lemmatizer, fit_lemmatizer, mine_trees};

const EPS: f64 = 1e-6;

fn spec(labels: Vec<String>, adapter: Option<(u32, u32)>, tags: Option<u32>) -> HeadSpec {
    HeadSpec {
        name: "lemma".into(),
        layer: "turboembed:lemmas".into(),
        kind: HeadKind::EditTreeLemmatizer,
        labels,
        scheme: Scheme::Tags,
        word_pooling: WordPooling::First,
        adapter: adapter.map(|(hidden, layers)| Adapter { bilstm: Bilstm { hidden, layers } }),
        decode: Decode::EditTree,
        biaffine: None,
        pos_input: tags.map(|labels| PosInput { head: "pos".into(), labels, dim: 3 }),
        hash_embed: None,
        artifact: "weights".into(),
        tensor_prefix: "lemma".into(),
    }
}

fn trees(pairs: &[(&str, &str)]) -> Vec<String> {
    pairs.iter().map(|(f, l)| EditTree::mine(f, l).label()).collect()
}

#[test]
fn loss_gradients_match_finite_differences() {
    let labels = trees(&[("walked", "walk"), ("cats", "cats"), ("running", "run"), ("was", "be")]);
    let hidden = 4;
    let gold = [Some(0u32), None, Some(3), Some(1)];
    for (k, (adapter, tags)) in
        [(None, None), (Some((2, 1)), None), (None, Some(3)), (Some((2, 1)), Some(3))].into_iter().enumerate()
    {
        let s = spec(labels.clone(), adapter, tags);
        let mut rng = Rng::new(50 + k as u64);
        let mut model = Lemmatizer::<f64>::new(&s, hidden, &mut rng).unwrap();
        for i in 0..model.params().len() {
            model.param_mut(i).iter_mut().for_each(|v| *v += rng.uniform(0.4));
        }
        let x: Vec<f64> = (0..gold.len() * hidden).map(|_| rng.uniform(1.0)).collect();
        let t: Vec<u32> = if tags.is_some() { vec![2, 0, 1, 2] } else { Vec::new() };
        let mut grads = model.zero_grads();
        model.loss(&x, &t, &gold, &mut grads, &mut Work::default()).unwrap();
        for (i, g) in grads.iter().enumerate() {
            let name = model.names()[i].clone();
            for (j, &analytic) in g.iter().enumerate() {
                let f = |v: f64| {
                    let mut m = model.clone();
                    m.param_mut(i)[j] = v;
                    m.loss(&x, &t, &gold, &mut m.zero_grads(), &mut Work::default()).unwrap()
                };
                let p = model.params()[i][j];
                let numeric = (f(p + EPS) - f(p - EPS)) / (2.0 * EPS);
                let tol = 1e-6 * (1.0 + numeric.abs().max(analytic.abs()));
                assert!((numeric - analytic).abs() <= tol, "case {k} {name}[{j}]: {numeric} vs {analytic}");
            }
        }
    }
}

#[test]
fn trainer_scores_are_the_cores() {
    let labels = trees(&[("walked", "walk"), ("cats", "cats"), ("was", "be")]);
    let s = spec(labels, Some((3, 1)), Some(4));
    let hidden = 5;
    let mut rng = Rng::new(9);
    let model = Lemmatizer::<f32>::new(&s, hidden, &mut rng).unwrap();
    let core = model.to_core().unwrap();
    let x: Vec<f32> = (0..3 * hidden).map(|_| rng.uniform(1.0) as f32).collect();
    let tags = [3, 0, 1];
    let mut w = Work::default();
    model.forward(&x, &tags, &mut w).unwrap();
    let mut scores = Vec::new();
    core.scores(&x, &[0, 1, 2], Some(&tags), &mut scores, &mut CoreWork::default()).unwrap();
    for (a, b) in w.scores().iter().zip(&scores) {
        assert!((a - b).abs() <= 1e-5, "{a} {b}");
    }
    assert!(model.forward(&x, &[0, 1], &mut w).is_err(), "a tag a word");
    assert!(model.forward(&x, &[0, 1, 4], &mut w).is_err(), "a tag past the table");
}

#[test]
fn trees_are_mined_by_frequency_then_label_and_rare_ones_dropped() {
    let pairs = [("walked", "walk"), ("jumped", "jump"), ("cats", "cat"), ("dogs", "dog"), ("was", "be")];
    let all = mine_trees(pairs.iter().copied(), 1);
    assert_eq!(all.len(), 3);
    assert_eq!(all[2], "(s 'was' 'be')");
    assert!(all[0] < all[1], "equal counts go by label: {all:?}");
    assert_eq!(mine_trees(pairs.iter().copied(), 2).len(), 2);
}

#[test]
fn a_pair_whose_tree_empties_its_word_is_not_mined() {
    // A word with an empty lemma, and "walked" -> "walk" whose suffix
    // tree would empty "ed": the first gives no tree, the second keeps
    // its own.
    let pairs = [("ed", ""), ("ed", ""), ("walked", "walk")];
    let all = mine_trees(pairs.iter().copied(), 1);
    assert_eq!(all.len(), 1, "{all:?}");
    let tree = EditTree::parse(&all[0]).unwrap();
    assert_eq!(tree.apply("walked").as_deref(), Some("walk"));
    assert_eq!(tree.apply("ed"), None, "the core refuses an empty lemma");
}

const STEMS: [&str; 12] =
    ["walk", "jump", "talk", "look", "cook", "play", "work", "call", "pull", "kick", "lift", "mix"];

/// A word of class c: 0 "-ed" to the stem, 1 "-ing" to the stem, 2 a
/// plain stem, 3 "-s" whose lemma drops the s under tag 0 (a plural noun)
/// and keeps it under tag 1 (a word that ends in s). Features say the
/// class and nothing of the tag.
fn data(seed: u64, n: usize, hidden: usize) -> Vec<LemmaExample> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let len = 2 + rng.below(6);
            let mut e = LemmaExample { features: Vec::new(), tags: Vec::new(), forms: Vec::new(), lemmas: Vec::new() };
            for _ in 0..len {
                let (c, stem) = (rng.below(4), STEMS[rng.below(STEMS.len())]);
                let tag = rng.below(2) as u32;
                let (form, lemma) = match (c, tag) {
                    (0, _) => (format!("{stem}ed"), stem.to_string()),
                    (1, _) => (format!("{stem}ing"), stem.to_string()),
                    (2, _) => (stem.to_string(), stem.to_string()),
                    (_, 0) => (format!("{stem}s"), stem.to_string()),
                    _ => (format!("{stem}s"), format!("{stem}s")),
                };
                for j in 0..hidden {
                    e.features.push((if j == c { 1.0 } else { 0.0 }) + rng.uniform(0.2) as f32);
                }
                e.tags.push(tag);
                e.forms.push(form);
                e.lemmas.push(lemma);
            }
            e
        })
        .collect()
}

fn without_tags(v: Vec<LemmaExample>) -> Vec<LemmaExample> {
    v.into_iter().map(|e| LemmaExample { tags: Vec::new(), ..e }).collect()
}

#[test]
fn a_toy_lemmatizer_learns_suffix_rules_and_needs_its_tags_for_the_ambiguous_one() {
    let hidden = 6;
    let (train, dev) = (data(1, 300, hidden), data(2, 60, hidden));
    let pairs: Vec<(&str, &str)> =
        train.iter().flat_map(|e| e.forms.iter().zip(&e.lemmas).map(|(f, l)| (f.as_str(), l.as_str()))).collect();
    let labels = mine_trees(pairs, 3);
    let c = Config { epochs: 20, lr: 2e-2, ..Config::default() };
    let tagged = spec(labels.clone(), None, Some(2));
    let (model, report) = fit_lemmatizer(&tagged, hidden, train.clone(), dev.clone(), &c).unwrap();
    let best = report.dev[report.best_epoch];
    assert!(best >= 0.99, "dev lemma accuracy {best}: {report:?}");
    assert_eq!(accuracy(&model.to_core().unwrap(), &dev).unwrap(), best);

    let plain = spec(labels, None, None);
    let (_, r) = fit_lemmatizer(&plain, hidden, without_tags(train), without_tags(dev), &c).unwrap();
    let blind = r.dev[r.best_epoch];
    assert!(blind < 0.95, "without tags the -s rule is a coin flip: {blind}");
}

#[test]
fn two_runs_and_any_thread_count_give_identical_bytes() {
    let hidden = 6;
    let (train, dev) = (data(3, 40, hidden), data(4, 10, hidden));
    let pairs: Vec<(&str, &str)> =
        train.iter().flat_map(|e| e.forms.iter().zip(&e.lemmas).map(|(f, l)| (f.as_str(), l.as_str()))).collect();
    let s = spec(mine_trees(pairs, 1), Some((4, 1)), Some(2));
    let c = Config { epochs: 3, ..Config::default() };
    let bytes = |m: &Lemmatizer<f32>| -> Vec<u8> {
        m.tensors().iter().flat_map(|(_, v)| v.iter().flat_map(|x| x.to_le_bytes())).collect()
    };
    let (a, ra) = fit_lemmatizer(&s, hidden, train.clone(), dev.clone(), &c).unwrap();
    let (b, rb) = fit_lemmatizer(&s, hidden, train.clone(), dev.clone(), &c).unwrap();
    assert_eq!(bytes(&a), bytes(&b));
    // Everything but the wall time.
    assert_eq!((&ra.train_loss, &ra.dev, ra.best_epoch), (&rb.train_loss, &rb.dev, rb.best_epoch));
    // One thread and four give the same bytes.
    for threads in [1, 4] {
        let (t, _) = fit_lemmatizer(&s, hidden, train.clone(), dev.clone(), &Config { threads, ..c }).unwrap();
        assert_eq!(bytes(&t), bytes(&a), "{threads} threads");
    }
}

#[test]
fn wrongly_shaped_gradients_are_refused_with_every_buffer_untouched() {
    let hidden = 3;
    let labels: Vec<String> = ["(m 0 0 () ())", "(m 0 1 () (s 's' ''))"].map(String::from).to_vec();
    for (adapter, tags) in [(None, None), (Some((2, 1)), Some(2)), (Some((2, 2)), None)] {
        let s = spec(labels.clone(), adapter, tags);
        let model = Lemmatizer::<f32>::new(&s, hidden, &mut Rng::new(9)).unwrap();
        let before = model.params().to_vec();
        let lens: Vec<usize> = before.iter().map(Vec::len).collect();
        let full = |lens: &[usize]| lens.iter().map(|&n| vec![7.25f32; n]).collect::<Vec<_>>();
        let mut wrong = vec![Vec::new(), lens.iter().map(|_| Vec::new()).collect(), full(&lens[..lens.len() - 1])];
        for i in 0..lens.len() {
            let mut g = full(&lens);
            g[i].push(7.25);
            wrong.push(g);
        }
        let x = [0.5f32, -0.25, 1.0, 0.75, 0.0, -1.0];
        let t: Vec<u32> = if tags.is_some() { vec![0, 1] } else { Vec::new() };
        for mut g in wrong {
            let snapshot = g.clone();
            let r = model.loss(&x, &t, &[Some(0), Some(1)], &mut g, &mut Work::default());
            assert!(r.is_err(), "{adapter:?} {tags:?}: {r:?}");
            assert_eq!(g, snapshot);
            assert_eq!(model.params(), &before[..]);
        }
        let mut g = model.zero_grads();
        assert!(model.loss(&x, &t, &[Some(0), Some(1)], &mut g, &mut Work::default()).unwrap().is_finite());
    }
}
