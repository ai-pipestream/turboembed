//! The encoder under a token tagger head (#1519): its gradients against
//! central finite differences in f64, through both poolings, hashed
//! words and adapters on the top layer alone; the same bytes whatever
//! the threads and lanes; a toy task learnt through the shared loop; and
//! what is refused.

use turbo::head::tagger::NO_WORD;
use turbo::manifest::{Decode, HashEmbed, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_trainer::encoder::tuned::Work;
use turbo_trainer::encoder::{Encoder, Geometry, Lora};
use turbo_trainer::rng::Rng;
use turbo_trainer::{Config, Tuned, TunedExample, fit_tuned, tuned_gradients};

const EPS: f64 = 1e-6;

fn spec(pooling: WordPooling, hash: Option<HashEmbed>) -> HeadSpec {
    HeadSpec {
        name: "pos".into(),
        layer: "turboembed:test".into(),
        kind: HeadKind::TokenTagger,
        labels: ["A", "B", "C"].map(String::from).to_vec(),
        scheme: Scheme::Tags,
        word_pooling: pooling,
        adapter: None,
        decode: Decode::Argmax,
        biaffine: None,
        pos_input: None,
        hash_embed: hash,
        layer_mix: None,
        license_class: None,
        span: None,
        relation: None,
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

/// `<s>`, four words of which the second has two subwords, `</s>`.
fn example() -> TunedExample {
    TunedExample {
        ids: vec![0, 3, 7, 7, 1, 9, 2],
        words: vec![NO_WORD, 0, 1, 1, 2, 3, NO_WORD],
        labels: vec![0, 2, 1, 0],
        forms: ["Ann", "running", "the", "stone"].map(String::from).to_vec(),
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

/// Every weight the optimizer steps, against finite differences.
fn check_tuned(t: &Tuned<f64>, e: &TunedExample, what: &str) {
    let mut w = Work::default();
    let mut g = t.zero_grads();
    t.loss(e, &mut g, &mut w).unwrap();
    let mut params = t.params();
    for s in 0..params.len() {
        let mut p = std::mem::take(&mut params[s]);
        let analytic = g[s].clone();
        check(&format!("{what} slot {s}"), &mut p, &analytic, &mut |p| {
            let mut m = t.clone();
            let mut all = params.clone();
            all[s] = p.to_vec();
            m.set_params(all).unwrap();
            m.loss(e, &mut m.zero_grads(), &mut Work::default()).unwrap()
        });
        params[s] = p;
    }
}

#[test]
fn tuned_gradients_match_finite_differences_under_both_poolings() {
    for (pooling, hash) in [
        (WordPooling::First, None),
        (WordPooling::Mean, None),
        (WordPooling::Mean, Some(HashEmbed { dim: 2, rows: [16, 8, 8, 8] })),
    ] {
        let mut rng = Rng::new(31);
        let e = Encoder::<f64>::random(geometry(8, 6), 12, &mut rng).unwrap();
        let t = Tuned::new(e, &spec(pooling, hash), &mut rng).unwrap();
        assert_eq!(t.params().len(), 3 + 6 * 2 + 2 + if hash.is_some() { 4 } else { 0 });
        check_tuned(&t, &example(), &format!("{pooling:?}"));
    }
}

#[test]
fn adapters_on_the_top_layer_alone_take_the_gradient() {
    let mut rng = Rng::new(33);
    let e = Encoder::<f64>::random(geometry(8, 6), 12, &mut rng).unwrap();
    let base = e.params().to_vec();
    let e = e.with_lora_top(Lora { rank: 2, alpha: 4.0 }, 1, &mut rng).unwrap();
    assert_eq!(e.lora_from(), 1);
    assert_eq!(e.params().len(), base.len() + 8);
    assert_eq!(e.grads()[..base.len()].iter().map(Vec::len).sum::<usize>(), 0, "frozen slots hold no gradient");
    let mut t = Tuned::new(e, &spec(WordPooling::First, None), &mut rng).unwrap();
    // B starts at zero, which would hide A's gradient: move both first.
    let params: Vec<Vec<f64>> = t.params().iter().map(|p| p.iter().map(|_| rng.uniform(0.5)).collect()).collect();
    t.set_params(params).unwrap();
    assert_eq!(t.params().len(), 8 + 2);
    check_tuned(&t, &example(), "top layer");
    // The lower layer's weights merge unchanged; the top layer's move.
    let merged = t.encoder().merged();
    assert_eq!(merged[..3 + 6], base[..3 + 6]);
    assert_ne!(merged[3 + 6 + 1], base[3 + 6 + 1]);
    let whole = Encoder::<f64>::random(geometry(8, 6), 12, &mut rng).unwrap();
    assert!(whole.clone().with_lora_top(Lora { rank: 2, alpha: 4.0 }, 0, &mut rng).is_err());
    assert!(whole.with_lora_top(Lora { rank: 2, alpha: 4.0 }, 3, &mut rng).is_err());
}

/// Sentences of `len` words from `words`, every third word split in two
/// subwords, each word labelled 1 when exactly one of the two markers is
/// in the sentence: what a head over a frozen random encoder cannot read
/// off, and attention can.
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

fn small() -> Tuned<f32> {
    let mut rng = Rng::new(5);
    let e = Encoder::<f32>::random(geometry(16, 24), 64, &mut rng).unwrap();
    Tuned::new(e, &spec(WordPooling::First, None), &mut rng).unwrap()
}

#[test]
fn the_bytes_do_not_depend_on_threads_or_lanes() {
    let t = small();
    let data = toy(&mut Rng::new(6), 7, 4, 3..60, [60, 61]);
    let (want, want_loss) = tuned_gradients(&t, &data, 1, 0).unwrap();
    for (threads, lanes) in [(4, 0), (3, 1), (2, 3), (8, 2)] {
        let (got, loss) = tuned_gradients(&t, &data, threads, lanes).unwrap();
        assert_eq!(loss.to_bits(), want_loss.to_bits(), "{threads} threads, {lanes} lanes");
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()), "{threads} threads, {lanes} lanes: {i}");
        }
    }
    // Within rounding of one example after another into one buffer.
    let mut one = t.zero_grads();
    let mut w = Work::default();
    let mut loss = 0.0;
    for e in &data {
        loss += t.loss(e, &mut one, &mut w).unwrap();
    }
    assert!((loss - want_loss).abs() <= 1e-9 * want_loss.abs());
    for (i, (a, b)) in one.iter().zip(&want).enumerate() {
        let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(b.is_empty() || b.iter().any(|v| *v != 0.0), "slot {i} has a gradient");
        for (j, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= 1e-5 * (1.0 + scale), "slot {i}[{j}]: {x} vs {y}");
        }
    }
}

#[test]
fn fine_tuning_through_the_loop_learns_the_toy_task() {
    let mut rng = Rng::new(7);
    let (train, dev) = (toy(&mut rng, 256, 4, 3..60, [60, 61]), toy(&mut rng, 64, 4, 3..60, [60, 61]));
    let e = Encoder::<f32>::random(geometry(16, 24), 64, &mut rng).unwrap();
    // 0.977 as written, against 0.5 for the head alone; the bar leaves
    // room for a platform's exp or sqrt to differ in the last place.
    let c = Config { epochs: 8, batch: 4, lr: 1e-2, warmup: 0.1, patience: 0, lanes: 2, ..Config::default() };
    let (t, report) = fit_tuned(e, &spec(WordPooling::First, None), &train, &dev, &c).unwrap();
    let best = report.dev[report.best_epoch];
    println!("through the loop: dev {best} after {} epochs", report.dev.len());
    assert!(best >= 0.85, "dev {best}: {:?}", report.dev);
    assert_eq!(t.score(&dev).unwrap(), best);
    // The same run on one thread, all at once, gives the same bytes.
    let e = Encoder::<f32>::random(geometry(16, 24), 64, &mut Rng::new(7).clone()).unwrap();
    let again = Config { epochs: 2, threads: 1, lanes: 0, ..c };
    let (a, ra) = fit_tuned(e.clone(), &spec(WordPooling::First, None), &train, &dev, &again).unwrap();
    let (b, rb) =
        fit_tuned(e, &spec(WordPooling::First, None), &train, &dev, &Config { threads: 3, lanes: 3, ..again }).unwrap();
    assert_eq!(a.params(), b.params());
    assert_eq!((ra.train_loss, ra.dev), (rb.train_loss, rb.dev));
}

#[test]
fn ill_fitting_examples_are_refused_by_index() {
    let mut rng = Rng::new(9);
    let s = spec(WordPooling::First, None);
    let data = toy(&mut rng, 4, 3, 3..60, [60, 61]);
    let c = Config { epochs: 1, batch: 2, ..Config::default() };
    let e = || Encoder::<f32>::random(geometry(8, 6), 64, &mut Rng::new(1)).unwrap();
    let mut bad = data.clone();
    bad[2].words.pop();
    let err = fit_tuned(e(), &s, &bad, &data, &c).unwrap_err();
    assert!(err.message.contains("example 2"), "{err:?}");
    let mut bad = data.clone();
    bad[1].labels[0] = 3;
    assert!(fit_tuned(e(), &s, &bad, &data, &c).is_err());
    let mut bad = data.clone();
    bad[0].ids[1] = 64;
    assert!(fit_tuned(e(), &s, &bad, &data, &c).is_err());
    let mut bad = data.clone();
    bad[3].words = vec![NO_WORD; bad[3].ids.len()];
    assert!(fit_tuned(e(), &s, &data, &bad, &c).is_err(), "a dev sentence of no words");
    assert!(
        fit_tuned(e(), &spec(WordPooling::First, Some(HashEmbed { dim: 2, rows: [8; 4] })), &data, &data, &c).is_err()
    );
    let t = Tuned::new(e(), &s, &mut rng).unwrap();
    assert!(t.loss(&data[0], &mut vec![vec![0.0; 1]; 2], &mut Work::default()).is_err());
}
