//! Training runs on synthetic features: a toy tagger reaches a fixed
//! score, two runs give identical bytes, and the trained weights score
//! the same through the trainer and through the core's inference path.

use turbo::head::tagger::Work as CoreWork;
use turbo::manifest::{Adapter, Bilstm, Decode, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_trainer::fit::score;
use turbo_trainer::rng::Rng;
use turbo_trainer::tagger::Work;
use turbo_trainer::{Config, Example, Tagger, fit};

const HIDDEN: usize = 8;

fn spec(labels: &[&str], scheme: Scheme, adapter: Option<(u32, u32)>, decode: Decode) -> HeadSpec {
    HeadSpec {
        name: "toy".into(),
        layer: "turboembed:toy".into(),
        kind: HeadKind::TokenTagger,
        labels: labels.iter().map(|s| s.to_string()).collect(),
        scheme,
        word_pooling: WordPooling::First,
        adapter: adapter.map(|(hidden, layers)| Adapter { bilstm: Bilstm { hidden, layers } }),
        decode,
        biaffine: None,
        pos_input: None,
        hash_embed: None,
        artifact: "weights".into(),
        tensor_prefix: "toy".into(),
    }
}

/// A word of class `c` (0..4): a unit vector on axis c, plus noise.
fn word(rng: &mut Rng, c: usize, out: &mut Vec<f32>) {
    for j in 0..HIDDEN {
        let v = if j == c { 1.0 } else { 0.0 };
        out.push((v + rng.uniform(0.35)) as f32);
    }
}

/// Part-of-speech-like data: the tag is the word's class, except that a
/// class-3 word after a class-0 word takes tag 0, which only context
/// (the adapter) can tell.
fn tags_data(seed: u64, n: usize) -> Vec<Example> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let len = 3 + rng.below(8);
            let (mut features, mut labels) = (Vec::new(), Vec::new());
            let mut prev = usize::MAX;
            for _ in 0..len {
                let c = rng.below(4);
                word(&mut rng, c, &mut features);
                labels.push(if c == 3 && prev == 0 { 0 } else { c as u32 });
                prev = c;
            }
            Example { features, labels, forms: Vec::new() }
        })
        .collect()
}

/// Entity-like data under BIO: class 1 words are in X spans, class 2 in Y
/// spans, class 0 outside; a run's first word is B, the rest I, which
/// only the CRF's transitions or the adapter can tell.
fn bio_data(seed: u64, n: usize) -> Vec<Example> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let len = 4 + rng.below(8);
            let (mut features, mut labels) = (Vec::new(), Vec::new());
            let mut prev = 0;
            for _ in 0..len {
                let c = rng.below(3);
                word(&mut rng, c, &mut features);
                // O, B-X, I-X, B-Y, I-Y
                labels.push(match (c, prev == c) {
                    (0, _) => 0,
                    (1, false) => 1,
                    (1, true) => 2,
                    (_, false) => 3,
                    (_, true) => 4,
                });
                prev = c;
            }
            Example { features, labels, forms: Vec::new() }
        })
        .collect()
}

fn bytes(t: &Tagger<f32>) -> Vec<u8> {
    t.tensors().iter().flat_map(|(_, v)| v.iter().flat_map(|x| x.to_le_bytes())).collect()
}

#[test]
fn a_toy_tagger_needs_its_adapter_and_learns_with_it() {
    let s = spec(&["A", "B", "C", "D"], Scheme::Tags, Some((8, 1)), Decode::Argmax);
    let (train, dev) = (tags_data(1, 300), tags_data(2, 60));
    let c = Config { epochs: 25, lr: 1e-2, patience: 5, ..Config::default() };
    let (model, report) = fit(&s, HIDDEN, &train, &dev, &c).unwrap();
    let best = report.dev[report.best_epoch];
    assert!(best >= 0.97, "dev accuracy {best}, {report:?}");
    assert_eq!(score(&model.to_token_tagger().unwrap(), &dev).unwrap(), best);
    assert!(report.train_loss.first() > report.train_loss.last(), "{report:?}");

    // Without the adapter, the context rule is out of reach.
    let flat = spec(&["A", "B", "C", "D"], Scheme::Tags, None, Decode::Argmax);
    let (_, r) = fit(&flat, HIDDEN, &train, &dev, &c).unwrap();
    assert!(r.dev[r.best_epoch] < best, "flat {:?} vs adapter {best}", r.dev);
}

#[test]
fn a_toy_crf_tagger_learns_bio_spans() {
    let s = spec(&["O", "B-X", "I-X", "B-Y", "I-Y"], Scheme::Bio, None, Decode::Crf);
    let (train, dev) = (bio_data(3, 300), bio_data(4, 60));
    let c = Config { epochs: 20, lr: 2e-2, ..Config::default() };
    let (_, report) = fit(&s, HIDDEN, &train, &dev, &c).unwrap();
    assert!(report.dev[report.best_epoch] >= 0.97, "dev span F1 {report:?}");
}

#[test]
fn two_runs_give_identical_bytes_and_another_seed_does_not() {
    let s = spec(&["O", "B-X", "I-X", "B-Y", "I-Y"], Scheme::Bio, Some((4, 1)), Decode::Crf);
    let (train, dev) = (bio_data(5, 40), bio_data(6, 10));
    let c = Config { epochs: 3, ..Config::default() };
    let (a, ra) = fit(&s, HIDDEN, &train, &dev, &c).unwrap();
    let (b, rb) = fit(&s, HIDDEN, &train, &dev, &c).unwrap();
    assert_eq!(bytes(&a), bytes(&b));
    // Everything but the wall time.
    assert_eq!((&ra.train_loss, &ra.dev, ra.best_epoch), (&rb.train_loss, &rb.dev, rb.best_epoch));
    assert_eq!(ra.epoch_seconds.len(), ra.dev.len());
    let (other, _) = fit(&s, HIDDEN, &train, &dev, &Config { seed: 2, ..c }).unwrap();
    assert_ne!(bytes(&a), bytes(&other));
}

#[test]
fn trained_weights_score_the_same_through_the_core() {
    let s = spec(&["O", "B-X", "I-X", "L-X", "U-X"], Scheme::Bilou, Some((6, 2)), Decode::Crf);
    let mut rng = Rng::new(9);
    let model = Tagger::<f32>::new(&s, HIDDEN, &mut rng).unwrap();
    let core = model.to_token_tagger().unwrap();
    let mut features = Vec::new();
    for c in [0, 1, 2, 1, 0, 3, 2] {
        word(&mut rng, c, &mut features);
    }
    let mut work = Work::default();
    let n = model.forward(&features, &mut work).unwrap();
    let words: Vec<u32> = (0..n as u32).collect();
    let (mut scores, mut cw) = (Vec::new(), CoreWork::default());
    core.scores(&features, &words, &mut scores, &mut cw).unwrap();
    let ours = work.scores();
    assert_eq!(ours.len(), scores.len());
    for (a, b) in ours.iter().zip(&scores) {
        assert!((a - b).abs() <= 1e-4, "trainer {a} core {b}");
    }
}

#[test]
fn ill_fitting_examples_are_refused() {
    let s = spec(&["A", "B"], Scheme::Tags, None, Decode::Argmax);
    let good = Example { features: vec![0.0; 2 * HIDDEN], labels: vec![0, 1], forms: Vec::new() };
    let short = Example { features: vec![0.0; HIDDEN], labels: vec![0, 1], forms: Vec::new() };
    let label = Example { features: vec![0.0; HIDDEN], labels: vec![2], forms: Vec::new() };
    let c = Config::default();
    assert!(fit(&s, HIDDEN, &[short], &[], &c).is_err());
    assert!(fit(&s, HIDDEN, std::slice::from_ref(&good), &[label], &c).is_err());
    assert!(fit(&s, HIDDEN, &[], &[], &c).is_err());
    assert!(fit(&s, HIDDEN, &[good], &[], &Config { batch: 0, ..c }).is_err());
    // A gold sequence the scheme forbids cannot be a CRF target.
    let bio = spec(&["O", "B-X", "I-X"], Scheme::Bio, None, Decode::Crf);
    let bad = Example { features: vec![0.0; 2 * HIDDEN], labels: vec![0, 2], forms: Vec::new() };
    assert!(fit(&bio, HIDDEN, &[bad], &[], &c).is_err());
}
