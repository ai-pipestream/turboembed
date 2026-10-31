//! Seeded dropout in the encoder (#1519): the mask keeps the
//! expectation and is the same bits at the same place in a run; a
//! training forward's gradients match finite differences under the same
//! masks; inference drops nothing; the bytes do not depend on threads
//! or lanes; and the toy task is still learnt.

use turbo::head::tagger::NO_WORD;
use turbo::manifest::{Decode, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_trainer::encoder::tuned::Work as TunedWork;
use turbo_trainer::encoder::{Dropout, Encoder, Geometry, Work, dropout};
use turbo_trainer::rng::Rng;
use turbo_trainer::{Config, Draw, Tuned, TunedExample, fit_tuned, tuned_gradients};

const EPS: f64 = 1e-6;

fn spec() -> HeadSpec {
    HeadSpec {
        name: "pos".into(),
        layer: "turboembed:test".into(),
        kind: HeadKind::TokenTagger,
        labels: ["A", "B", "C"].map(String::from).to_vec(),
        scheme: Scheme::Tags,
        word_pooling: WordPooling::Mean,
        adapter: None,
        decode: Decode::Argmax,
        biaffine: None,
        pos_input: None,
        hash_embed: None,
        license_class: None,
        span: None,
        coref: None,
        artifact: "weights".into(),
        tensor_prefix: "pos".into(),
    }
}

/// Two layers over two heads, layer 1 local with one key each side.
fn geometry(hidden: usize, intermediate: usize) -> Geometry {
    Geometry {
        layers: 2,
        hidden,
        heads: 2,
        intermediate,
        eps: 1e-5,
        window: 1,
        global_every: 2,
        max_positions: 16,
        global_theta: 160_000.0,
        local_theta: 10_000.0,
    }
}

fn rates(seed: u64) -> Dropout {
    Dropout { embedding: 0.2, attention: 0.3, mlp: 0.25, seed }
}

fn example() -> TunedExample {
    TunedExample {
        ids: vec![0, 3, 7, 7, 1, 9, 2],
        words: vec![NO_WORD, 0, 1, 1, 2, 3, NO_WORD],
        labels: vec![0, 2, 1, 0],
        forms: Vec::new(),
    }
}

/// Checks `analytic` against the central difference of `f` in each
/// coordinate of `x`.
fn check(what: &str, x: &mut [f64], analytic: &[f64], f: &mut dyn FnMut(&[f64]) -> f64) {
    assert_eq!(x.len(), analytic.len(), "{what}");
    for i in 0..x.len() {
        let keep = x[i];
        x[i] = keep + EPS;
        let up = f(x);
        x[i] = keep - EPS;
        let down = f(x);
        x[i] = keep;
        let numeric = (up - down) / (2.0 * EPS);
        let tol = 1e-6 * (1.0 + numeric.abs().max(analytic[i].abs()));
        assert!((numeric - analytic[i]).abs() <= tol, "{what}[{i}]: numeric {numeric}, analytic {}", analytic[i]);
    }
}

#[test]
fn the_mask_keeps_the_expectation_and_follows_its_key() {
    let n = 4096;
    for p in [0.1, 0.3, 0.5] {
        let mut x = vec![1.0f64; n];
        dropout::forward(&mut x, p, dropout::key(1, 2, 3, 4));
        let dropped = x.iter().filter(|v| **v == 0.0).count() as f64;
        // Four standard deviations of a binomial count.
        let slack = 4.0 * (n as f64 * p * (1.0 - p)).sqrt();
        assert!((dropped - n as f64 * p).abs() <= slack, "p {p}: {dropped} dropped");
        assert!(x.iter().all(|v| *v == 0.0 || (*v - 1.0 / (1.0 - p)).abs() < 1e-12), "p {p}: the rest are scaled");
    }
    let mut a = vec![1.0f32; 64];
    let mut b = a.clone();
    dropout::forward(&mut a, 0.5, dropout::key(7, 8, 1, 2));
    dropout::forward(&mut b, 0.5, dropout::key(7, 8, 1, 2));
    assert_eq!(a, b, "the same key gives the same mask");
    for key in [dropout::key(7, 9, 1, 2), dropout::key(7, 8, 0, 2), dropout::key(7, 8, 1, 1), dropout::key(6, 8, 1, 2)]
    {
        let mut c = vec![1.0f32; 64];
        dropout::forward(&mut c, 0.5, key);
        assert_ne!(a, c, "another place, layer, branch or seed gives another mask");
    }
    let mut none = vec![2.0f32; 8];
    dropout::forward(&mut none, 0.0, dropout::key(1, 1, 1, 1));
    assert_eq!(none, vec![2.0; 8], "a rate of 0 drops nothing");
}

#[test]
fn rates_outside_the_unit_are_refused() {
    let e = Encoder::<f32>::random(geometry(8, 6), 12, &mut Rng::new(1)).unwrap();
    assert!(e.dropout().is_none());
    for bad in [
        Dropout { embedding: 1.0, ..rates(1) },
        Dropout { attention: -0.1, ..rates(1) },
        Dropout { mlp: f64::NAN, ..rates(1) },
    ] {
        assert!(e.clone().with_dropout(bad).is_err(), "{bad:?}");
    }
    let e = e.with_dropout(Dropout { embedding: 0.0, attention: 0.0, mlp: 0.0, seed: 1 }).unwrap();
    assert_eq!(e.dropout().map(|d| d.seed), Some(1));
}

#[test]
fn a_training_forward_drops_by_its_place_and_inference_drops_nothing() {
    let mut rng = Rng::new(21);
    let plain = Encoder::<f32>::random(geometry(8, 6), 12, &mut rng).unwrap();
    let e = plain.clone().with_dropout(rates(5)).unwrap();
    let ids = [3u32, 7, 1, 7, 9];
    let run = |e: &Encoder<f32>, at: Option<u64>| {
        let mut w = Work::default();
        match at {
            Some(at) => e.forward_at(&ids, at, &mut w).unwrap(),
            None => e.forward(&ids, &mut w).unwrap(),
        }
        w.out
    };
    assert_eq!(run(&e, None), run(&plain, None), "inference is the plain encoder's");
    assert_eq!(run(&e, Some(3)), run(&e, Some(3)), "the same place gives the same states");
    assert_ne!(run(&e, Some(3)), run(&e, Some(4)), "another place gives other states");
    assert_ne!(run(&e, Some(3)), run(&e, None), "training drops");
    let other = plain.with_dropout(rates(6)).unwrap();
    assert_ne!(run(&e, Some(3)), run(&other, Some(3)), "another seed gives other states");
}

/// Every trainable weight against finite differences, the forward at
/// `at` both times, through the encoder alone and under the head.
#[test]
fn gradients_match_finite_differences_under_the_masks() {
    let mut rng = Rng::new(23);
    let e = Encoder::<f64>::random(geometry(8, 6), 12, &mut rng).unwrap().with_dropout(rates(9)).unwrap();
    let ids = [3u32, 7, 1, 7, 9];
    let r: Vec<f64> = (0..ids.len() * 8).map(|_| rng.uniform(1.0)).collect();
    let at = 11;
    let mut w = Work::default();
    e.forward_at(&ids, at, &mut w).unwrap();
    let mut g = e.grads();
    e.backward(&w, &r, &mut g).unwrap();
    let mut params = e.params().to_vec();
    for s in 0..params.len() {
        let mut p = std::mem::take(&mut params[s]);
        let analytic = g[s].clone();
        check(&format!("encoder slot {s}"), &mut p, &analytic, &mut |p| {
            let mut m = e.clone();
            let mut all = params.clone();
            all[s] = p.to_vec();
            m.set_params(all).unwrap();
            let mut w = Work::default();
            m.forward_at(&ids, at, &mut w).unwrap();
            w.out.iter().zip(&r).map(|(a, b)| a * b).sum()
        });
        params[s] = p;
    }
    let t = Tuned::new(e, &spec(), &mut rng).unwrap();
    let ex = example();
    let mut g = t.zero_grads();
    t.loss_at(&ex, Draw::at(at), &mut g, &mut TunedWork::default()).unwrap();
    let mut params = t.params();
    for s in 0..params.len() {
        let mut p = std::mem::take(&mut params[s]);
        let analytic = g[s].clone();
        check(&format!("tuned slot {s}"), &mut p, &analytic, &mut |p| {
            let mut m = t.clone();
            let mut all = params.clone();
            all[s] = p.to_vec();
            m.set_params(all).unwrap();
            m.loss_at(&ex, Draw::at(at), &mut m.zero_grads(), &mut TunedWork::default()).unwrap()
        });
        params[s] = p;
    }
    let mut plain = t.zero_grads();
    let undropped = t.loss(&ex, &mut plain, &mut TunedWork::default()).unwrap();
    let mut again = t.zero_grads();
    let dropped = t.loss_at(&ex, Draw::at(at), &mut again, &mut TunedWork::default()).unwrap();
    assert_ne!(undropped, dropped, "the loss drops with the masks");
    assert_eq!(plain.iter().map(Vec::len).collect::<Vec<_>>(), again.iter().map(Vec::len).collect::<Vec<_>>());
}

/// Sentences of `len` words from `words`, every third word split in two
/// subwords, each word labelled 1 when exactly one of the two markers is
/// in the sentence.
fn toy(rng: &mut Rng, count: usize, len: usize, words: std::ops::Range<u32>, markers: [u32; 2]) -> Vec<TunedExample> {
    (0..count)
        .map(|_| {
            let mut tokens: Vec<u32> = (0..len).map(|_| words.start + rng.below(words.len()) as u32).collect();
            let present = [rng.below(2) == 1, rng.below(2) == 1];
            for (m, p) in markers.iter().zip(present) {
                if p {
                    let at = rng.below(len);
                    tokens[at] = *m;
                }
            }
            let (mut ids, mut map) = (vec![0], vec![NO_WORD]);
            for (i, &t) in tokens.iter().enumerate() {
                ids.push(t);
                map.push(i as u32);
                if i % 3 == 2 {
                    ids.push(words.start + rng.below(words.len()) as u32);
                    map.push(i as u32);
                }
            }
            ids.push(2);
            map.push(NO_WORD);
            let label = (present[0] != present[1]) as u32;
            TunedExample { ids, words: map, labels: vec![label; len], forms: Vec::new() }
        })
        .collect()
}

#[test]
fn the_bytes_do_not_depend_on_threads_or_lanes_under_dropout() {
    let mut rng = Rng::new(5);
    let e = Encoder::<f32>::random(geometry(16, 24), 64, &mut rng).unwrap().with_dropout(rates(2)).unwrap();
    let t = Tuned::new(e, &spec(), &mut rng).unwrap();
    let data = toy(&mut Rng::new(6), 7, 4, 3..60, [60, 61]);
    let (want, want_loss) = tuned_gradients(&t, &data, 1, 0).unwrap();
    for (threads, lanes) in [(4, 0), (3, 1), (2, 3), (8, 2)] {
        let (got, loss) = tuned_gradients(&t, &data, threads, lanes).unwrap();
        assert_eq!(loss.to_bits(), want_loss.to_bits(), "{threads} threads, {lanes} lanes");
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()), "{threads} threads, {lanes} lanes: {i}");
        }
    }
    // Each example at its own place: one example after another, with
    // the masks of the first epoch, sums to the same loss.
    let mut w = TunedWork::default();
    let mut one = t.zero_grads();
    let loss: f64 = data.iter().zip(0..).map(|(e, at)| t.loss_at(e, Draw::at(at), &mut one, &mut w).unwrap()).sum();
    assert!((loss - want_loss).abs() <= 1e-9 * want_loss.abs());
}

#[test]
fn fine_tuning_with_dropout_learns_the_toy_task_and_scores_without_it() {
    let mut rng = Rng::new(7);
    let (train, dev) = (toy(&mut rng, 256, 4, 3..60, [60, 61]), toy(&mut rng, 64, 4, 3..60, [60, 61]));
    let e = Encoder::<f32>::random(geometry(16, 24), 64, &mut rng).unwrap();
    let e = e.with_dropout(Dropout { embedding: 0.1, attention: 0.1, mlp: 0.1, seed: 7 }).unwrap();
    let c = Config { epochs: 8, batch: 4, lr: 1e-2, warmup: 0.1, patience: 0, lanes: 2, ..Config::default() };
    let (t, report) = fit_tuned(e.clone(), &spec(), &train, &dev, &c).unwrap();
    let best = report.dev[report.best_epoch];
    println!("with dropout: dev {best} after {} epochs", report.dev.len());
    assert!(best >= 0.85, "dev {best}: {:?}", report.dev);
    assert_eq!(t.score(&dev).unwrap(), best, "dev is scored without dropout, so it is repeatable");
    let again = Config { epochs: 2, threads: 1, lanes: 0, ..c };
    let (a, ra) = fit_tuned(e.clone(), &spec(), &train, &dev, &again).unwrap();
    let (b, rb) = fit_tuned(e, &spec(), &train, &dev, &Config { threads: 3, lanes: 3, ..again }).unwrap();
    assert_eq!(a.params(), b.params());
    assert_eq!((ra.train_loss, ra.dev), (rb.train_loss, rb.dev));
}

/// A place 2^23 apart on the layer below once gave the same stream,
/// when the fields were XORed before one mix.
#[test]
fn no_place_aliases_a_layer_or_a_branch() {
    assert_ne!(dropout::key(7, 0, 1, 1), dropout::key(7, 1 << 23, 0, 1));
    assert_ne!(dropout::key(7, 0, 0, 1), dropout::key(7, 1 << 39, 0, 0));
    let mut seen = std::collections::HashSet::new();
    for at in 0..64u64 {
        for layer in [0u32, 1, 2, 3, u32::MAX] {
            for branch in 0..3u32 {
                assert!(seen.insert(dropout::key(3, at, layer, branch)), "{at} {layer} {branch}");
            }
        }
    }
}
