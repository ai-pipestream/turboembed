//! Training a biaffine parser on synthetic sentences: it learns a fixed
//! grammar well past a trivial baseline, two runs give identical bytes,
//! and bad sentences are refused.

use turbo::manifest::{Adapter, Biaffine, Bilstm, Decode, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_trainer::fit_parse::attachment;
use turbo_trainer::rng::Rng;
use turbo_trainer::{Config, ParseExample, fit_parser};

const HIDDEN: usize = 6;
const LABELS: [&str; 5] = ["root", "nsubj", "obj", "det", "conj"];

fn spec(decode: Decode) -> HeadSpec {
    HeadSpec {
        name: "deps".into(),
        layer: "turboembed:dependencies".into(),
        kind: HeadKind::BiaffineParser,
        labels: LABELS.map(String::from).to_vec(),
        scheme: Scheme::Tags,
        word_pooling: WordPooling::First,
        adapter: Some(Adapter { bilstm: Bilstm { hidden: 8, layers: 1 } }),
        decode,
        biaffine: Some(Biaffine { arc: 16, rel: 8 }),
        pos_input: None,
        hash_embed: None,
        span: None,
        coref: None,
        license_class: None,
        artifact: "weights".into(),
        tensor_prefix: "deps".into(),
    }
}

/// Sentences of verbs (class 0), nouns (1) and determiners (2), the
/// first word a noun or determiner and at least one verb. The first verb
/// is on the root and other verbs hang from it (conj); a noun hangs from
/// the nearest verb, as nsubj before it and obj after; a determiner from
/// the next word, a noun.
fn data(seed: u64, count: usize) -> Vec<ParseExample> {
    let mut rng = Rng::new(seed);
    (0..count)
        .map(|_| {
            let len = 3 + rng.below(6);
            let mut classes = Vec::new();
            while classes.len() < len {
                match rng.below(3) {
                    0 => classes.push(0),
                    1 => classes.push(1),
                    _ if classes.len() + 2 <= len => classes.extend([2, 1]),
                    _ => classes.push(1),
                }
            }
            if !classes.contains(&0) {
                let at = 1 + rng.below(len - 1);
                classes[at] = 0;
                if classes[at - 1] == 2 {
                    classes[at - 1] = 1;
                }
            }
            let verbs: Vec<usize> = (0..len).filter(|&i| classes[i] == 0).collect();
            let (mut heads, mut rels) = (Vec::new(), Vec::new());
            for (i, &c) in classes.iter().enumerate() {
                let (h, r) = match c {
                    0 if i == verbs[0] => (0, 0),
                    0 => (verbs[0] + 1, 4),
                    1 => {
                        let v = *verbs.iter().min_by_key(|&&v| (v.abs_diff(i), v)).unwrap();
                        (v + 1, if i < v { 1 } else { 2 })
                    }
                    _ => (i + 2, 3),
                };
                heads.push(h as u32);
                rels.push(r);
            }
            let mut features = Vec::new();
            for &c in &classes {
                for j in 0..HIDDEN {
                    features.push((if j == c { 1.0 } else { 0.0 } + rng.uniform(0.2)) as f32);
                }
            }
            ParseExample { features, heads, rels }
        })
        .collect()
}

#[test]
fn a_toy_grammar_is_learned_and_runs_repeat_exactly() {
    let (train, dev) = (data(1, 300), data(2, 80));
    let c = Config { epochs: 25, batch: 8, lr: 5e-3, patience: 0, threads: 3, ..Config::default() };
    let (model, report) = fit_parser(&spec(Decode::Mst), HIDDEN, &train, &dev, &c).unwrap();
    let a = attachment(&model.to_core().unwrap(), &dev).unwrap();
    // Every word on the first word: the trivial tree's score.
    let trivial = dev.iter().flat_map(|e| e.heads.iter()).filter(|&&h| h == 1).count() as f64 / a.words as f64;
    println!("UAS {:.3} LAS {:.3} trivial {trivial:.3} best epoch {}", a.uas(), a.las(), report.best_epoch);
    assert!(a.uas() >= 0.97 && a.las() >= 0.97 && trivial < 0.3, "UAS {} LAS {}", a.uas(), a.las());
    assert!((report.dev[report.best_epoch] - a.las()).abs() < 1e-12);
    // Another thread count, the same bytes.
    let (again, _) = fit_parser(&spec(Decode::Mst), HIDDEN, &train, &dev, &Config { threads: 1, ..c }).unwrap();
    assert_eq!(again.tensors(), model.tensors());
}

#[test]
fn bad_sentences_are_refused() {
    let mut e = data(3, 1).remove(0);
    let c = Config::default();
    let s = spec(Decode::Eisner);
    assert!(fit_parser(&s, HIDDEN, &[], &[], &c).is_err());
    e.heads[0] = 1;
    assert!(fit_parser(&s, HIDDEN, std::slice::from_ref(&e), &[], &c).is_err(), "word 1 on itself");
    let mut e = data(3, 1).remove(0);
    e.rels[0] = 9;
    assert!(fit_parser(&s, HIDDEN, std::slice::from_ref(&e), &[], &c).is_err());
    let mut e = data(3, 1).remove(0);
    e.features.pop();
    assert!(fit_parser(&s, HIDDEN, std::slice::from_ref(&e), &[], &c).is_err());
}
