//! bfloat16 compute emulated on the single (#1519): the rounding, the
//! states and gradients it gives against the single's, the shadow
//! weights following the master, loss scaling that is exact for a power
//! of two and skips a step that overflows, and the toy task learnt under
//! both.

use turbo::head::tagger::NO_WORD;
use turbo::manifest::{Decode, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_trainer::adamw::{AdamW, Settings};
use turbo_trainer::encoder::tuned::Work as TunedWork;
use turbo_trainer::encoder::{Compute, Encoder, Geometry, Lora, Work};
use turbo_trainer::rng::Rng;
use turbo_trainer::{Config, Draw, LossScale, Real, Scaler, Tuned, TunedExample, fit_tuned, tuned_gradients};

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
        layer_mix: None,
        license_class: None,
        span: None,
        relation: None,
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

fn example() -> TunedExample {
    TunedExample {
        ids: vec![0, 3, 7, 7, 1, 9, 2],
        words: vec![NO_WORD, 0, 1, 1, 2, 3, NO_WORD],
        labels: vec![0, 2, 1, 0],
        forms: Vec::new(),
    }
}

#[test]
fn rounding_goes_to_the_nearest_bfloat16_ties_to_even() {
    for (v, want) in [
        (1.0f32, 1.0f32),
        (1.0 + 1.0 / 256.0, 1.0),
        (1.0 + 3.0 / 256.0, 1.0 + 4.0 / 256.0),
        (1.0 + 1.0 / 256.0 + 1.0 / 65536.0, 1.0 + 2.0 / 256.0),
        (-1.234_567, -1.234_375),
        (f32::MAX, f32::INFINITY),
        (f32::from_bits(0x0001_0000), f32::from_bits(0x0001_0000)),
        (1e-40, f32::from_bits(0x0001_0000)),
        (0.0, 0.0),
        (f32::NEG_INFINITY, f32::NEG_INFINITY),
    ] {
        assert_eq!(v.bf16().to_bits(), want.to_bits(), "{v}");
        assert_eq!(v.bf16(), v.bf16().bf16(), "{v}: rounding is idempotent");
        assert_eq!((v as f64).bf16(), want as f64, "{v} as a double");
    }
    assert!(f32::NAN.bf16().is_nan());
    assert!(f64::NAN.bf16().is_nan());
    assert_eq!((-0.0f32).bf16().to_bits(), (-0.0f32).to_bits());
    // A double just past a tie rounds up, where a single would have made
    // it the tie first and rounded to even.
    let past = 1.0 + 1.0 / 256.0 + 2f64.powi(-30);
    assert_eq!(past.bf16(), 1.0 + 2.0 / 256.0);
    assert_eq!((past as f32).bf16(), 1.0);
    assert_eq!(1e39f64.bf16(), f64::INFINITY);
    let top = f32::from_bits(0x7F7F_0000);
    assert_eq!(top.bf16(), top, "the largest bfloat16");
    assert_eq!((top as f64).bf16(), top as f64);
    assert_eq!((top as f64 * (1.0 + 2f64.powi(-9))).bf16(), top as f64, "short of the tie");
    assert_eq!((top as f64 * (1.0 + 2f64.powi(-8))).bf16(), f64::INFINITY, "the tie goes up");
    // Subnormals sit on the grid of 2^-133.
    assert_eq!((2f64.powi(-134) * 3.0).bf16(), 2f64.powi(-133) * 2.0);
    assert_eq!(2f64.powi(-140).bf16(), 0.0);
}

/// Every value of `x` is its own rounding.
fn all_bf16<R: Real>(x: &[R]) -> bool {
    x.iter().all(|v| v.bf16() == *v)
}

/// The largest |a - b| over the largest |b|.
fn relative(a: &[f32], b: &[f32]) -> f64 {
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
    a.iter().zip(b).map(|(a, b)| (a - b).abs() as f64).fold(0.0, f64::max) / scale
}

#[test]
fn bf16_states_are_bfloat16_and_near_the_single_ones() {
    let mut rng = Rng::new(41);
    let plain = Encoder::<f32>::random(geometry(16, 24), 64, &mut rng).unwrap();
    assert_eq!(plain.compute(), Compute::F32);
    let half = plain.clone().with_bf16();
    assert_eq!(half.compute(), Compute::Bf16);
    assert_eq!(half.params(), plain.params(), "the master weights are the single's");
    let ids = [3u32, 7, 1, 7, 9, 40, 2];
    let (mut a, mut b) = (Work::default(), Work::default());
    plain.forward(&ids, &mut a).unwrap();
    half.forward(&ids, &mut b).unwrap();
    assert!(all_bf16(&b.out));
    assert!(!all_bf16(&a.out));
    let err = relative(&b.out, &a.out);
    println!("bf16 states: largest error {err:.4} of the largest state");
    // 0.006 as written, over two layers.
    assert!(err < 0.02, "{err}");
    assert!(err > 0.0, "bf16 compute rounds");
    // The same bytes again, and the same under dropout at a place.
    let mut c = Work::default();
    half.forward(&ids, &mut c).unwrap();
    assert_eq!(b.out, c.out);
}

#[test]
fn bf16_gradients_point_where_the_single_ones_do() {
    let mut rng = Rng::new(43);
    let e = Encoder::<f32>::random(geometry(16, 24), 64, &mut rng).unwrap();
    let t = Tuned::new(e, &spec(), &mut rng).unwrap();
    let mut half = t.clone();
    let (enc, head) = half.into_parts();
    half = Tuned::assemble(enc.with_bf16(), head).unwrap();
    let ex = example();
    let (mut ga, mut gb) = (t.zero_grads(), half.zero_grads());
    let la = t.loss(&ex, &mut ga, &mut TunedWork::default()).unwrap();
    let lb = half.loss(&ex, &mut gb, &mut TunedWork::default()).unwrap();
    println!("bf16 loss {lb} against {la}");
    assert!((la - lb).abs() < 0.02 * la.abs(), "{lb} against {la}");
    for (s, (a, b)) in ga.iter().zip(&gb).enumerate().filter(|(_, (a, _))| !a.is_empty()) {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let cosine = dot / (na * nb);
        println!("slot {s}: cosine {cosine:.5}, norm ratio {:.4}", nb / na);
        // 0.99998 and 0.2 percent as written.
        assert!(cosine > 0.999, "slot {s}: cosine {cosine}");
        assert!((nb / na - 1.0).abs() < 0.02, "slot {s}: norms {nb} against {na}");
        assert!(!all_bf16(b) || b.iter().all(|v| *v == 0.0), "slot {s}: the weight gradients stay single");
    }
}

#[test]
fn the_shadow_follows_the_master_weights() {
    let mut rng = Rng::new(47);
    let e = Encoder::<f32>::random(geometry(8, 6), 12, &mut rng).unwrap().with_bf16();
    let mut t = Tuned::new(e, &spec(), &mut rng).unwrap();
    let ids = [3u32, 7, 1, 7, 9];
    let states = |e: &Encoder<f32>| {
        let mut w = Work::default();
        e.forward(&ids, &mut w).unwrap();
        w.out
    };
    let before = states(t.encoder());
    let shapes: Vec<usize> = t.params().iter().map(Vec::len).collect();
    let mut opt = AdamW::new(Settings { lr: 1e-2, total: 10, ..Settings::default() }, &shapes).unwrap();
    let mut g = t.zero_grads();
    t.loss(&example(), &mut g, &mut TunedWork::default()).unwrap();
    t.step(&mut opt, &mut g, &t.decays()).unwrap();
    let master = t.encoder().params().to_vec();
    assert!(!all_bf16(&master[3 + 1]), "the master weights stay single");
    // The shadow is the stepped master rounded: a fresh encoder over the
    // master gives the same states, and they moved.
    let fresh = Encoder::<f32>::assemble(geometry(8, 6), 12, master.clone()).unwrap().with_bf16();
    let after = states(t.encoder());
    assert_eq!(after, states(&fresh));
    assert_ne!(after, before);
    // set_params refreshes it too, and a slot keeps its length.
    let mut e = t.encoder().clone();
    let mut moved = master.clone();
    moved[3 + 1].iter_mut().for_each(|v| *v *= 1.5);
    e.set_params(moved.clone()).unwrap();
    assert_eq!(states(&e), states(&Encoder::<f32>::assemble(geometry(8, 6), 12, moved).unwrap().with_bf16()));
    assert!(e.put_slot(3 + 1, vec![0.0; 3]).is_err());
    assert!(e.put_slot(99, Vec::new()).is_err());
    let taken = e.take_slot(2);
    e.put_slot(2, taken).unwrap();
    assert_eq!(states(&e).len(), ids.len() * 8);
    // Adapters added to a bf16 encoder are rounded as read too.
    let base = Encoder::<f32>::random(geometry(8, 6), 12, &mut rng).unwrap().with_bf16();
    let adapted = base.clone().with_lora(Lora { rank: 2, alpha: 4.0 }, &mut rng).unwrap();
    assert_eq!(states(&adapted), states(&base), "a fresh adapter changes nothing");
    assert_eq!(adapted.compute(), Compute::Bf16);
}

#[test]
fn loss_scaling_by_a_power_of_two_is_exact() {
    let mut rng = Rng::new(53);
    let e = Encoder::<f32>::random(geometry(16, 24), 64, &mut rng).unwrap();
    let single = Tuned::new(e.clone(), &spec(), &mut rng).unwrap();
    let (enc, head) = single.clone().into_parts();
    let half = Tuned::assemble(enc.with_bf16(), head).unwrap();
    let ex = example();
    for (t, what) in [(&single, "single"), (&half, "bf16")] {
        let mut plain = t.zero_grads();
        let loss = t.loss_at(&ex, Draw::at(3), &mut plain, &mut TunedWork::default()).unwrap();
        let mut scaled = t.zero_grads();
        let same = t.loss_at(&ex, Draw { at: 3, scale: 1024.0 }, &mut scaled, &mut TunedWork::default()).unwrap();
        assert_eq!(loss, same, "{what}: the loss is not scaled");
        for (s, (a, b)) in plain.iter().zip(&scaled).enumerate() {
            assert!(a.iter().zip(b).all(|(a, b)| (a * 1024.0).to_bits() == b.to_bits()), "{what} slot {s}");
        }
        // Scaling adds this example's gradients to what is there, within
        // the rounding of the sums taken in another order.
        let mut onto = t.zero_grads();
        onto.iter_mut().flatten().for_each(|g| *g = 1.0);
        t.loss_at(&ex, Draw { at: 3, scale: 2.0 }, &mut onto, &mut TunedWork::default()).unwrap();
        for (s, (a, b)) in plain.iter().zip(&onto).enumerate() {
            let near = |(a, b): (&f32, &f32)| (1.0 + a * 2.0 - b).abs() <= 1e-5 * (1.0 + a.abs());
            assert!(a.iter().zip(b).all(near), "{what} slot {s} onto");
        }
    }
    // Through the loop: a run under a scale that never overflows gives
    // the bytes of the unscaled run, in both computes.
    let data = toy(&mut Rng::new(6), 24, 4, 3..60, [60, 61]);
    for bf16 in [false, true] {
        let e = if bf16 { e.clone().with_bf16() } else { e.clone() };
        let c = Config { epochs: 2, batch: 4, lr: 1e-2, lanes: 2, ..Config::default() };
        let (a, ra) = fit_tuned(e.clone(), &spec(), &data, &data, &c).unwrap();
        let scale = LossScale { initial: 1024.0, growth: 2.0, backoff: 0.5, interval: 3 };
        let (b, rb) = fit_tuned(e, &spec(), &data, &data, &Config { loss_scale: Some(scale), ..c }).unwrap();
        assert_eq!(a.params(), b.params(), "bf16 {bf16}");
        assert_eq!(ra.train_loss, rb.train_loss);
        assert_eq!(rb.skipped_steps, 0);
    }
}

#[test]
fn the_scaler_backs_off_on_overflow_and_grows_after_good_steps() {
    let mut s = Scaler::new(LossScale { initial: 8.0, growth: 2.0, backoff: 0.5, interval: 2 });
    assert_eq!(s.scale(), 8.0);
    assert!(s.update(true));
    assert_eq!(s.scale(), 8.0);
    assert!(s.update(true));
    assert_eq!(s.scale(), 16.0, "grown after two finite steps");
    assert!(!s.update(false));
    assert_eq!((s.scale(), s.skipped()), (8.0, 1));
    assert!(s.update(true));
    assert!(!s.update(false), "the count of finite steps starts over");
    assert_eq!(s.scale(), 4.0);
    assert!(s.update(true));
    assert!(s.update(true));
    assert_eq!(s.scale(), 8.0);
    let mut top = Scaler::new(LossScale { initial: f32::MAX, growth: 2.0, backoff: 0.5, interval: 1 });
    assert!(top.update(true));
    assert_eq!(top.scale(), f32::MAX, "a scale never grows past the single");
    for bad in [
        LossScale { initial: 0.0, ..LossScale::default() },
        LossScale { initial: f32::INFINITY, ..LossScale::default() },
        LossScale { growth: 0.5, ..LossScale::default() },
        LossScale { backoff: 1.0, ..LossScale::default() },
        LossScale { backoff: 0.0, ..LossScale::default() },
        LossScale { interval: 0, ..LossScale::default() },
    ] {
        assert!(bad.check().is_err(), "{bad:?}");
        let c = Config { epochs: 1, loss_scale: Some(bad), ..Config::default() };
        let e = Encoder::<f32>::random(geometry(8, 6), 64, &mut Rng::new(1)).unwrap();
        assert!(fit_tuned(e, &spec(), &[example()], &[example()], &c).is_err(), "{bad:?}");
    }
    assert!(LossScale::default().check().is_ok());
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
fn a_step_that_overflows_is_skipped_and_the_run_goes_on() {
    let mut rng = Rng::new(7);
    let (train, dev) = (toy(&mut rng, 32, 4, 3..60, [60, 61]), toy(&mut rng, 16, 4, 3..60, [60, 61]));
    let e = Encoder::<f32>::random(geometry(16, 24), 64, &mut rng).unwrap();
    let scale = LossScale { initial: f32::MAX, growth: 2.0, backoff: 0.5, interval: 4 };
    let c = Config { epochs: 2, batch: 4, lr: 1e-2, loss_scale: Some(scale), ..Config::default() };
    let (t, report) = fit_tuned(e.clone(), &spec(), &train, &dev, &c).unwrap();
    println!("overflow: {} steps skipped", report.skipped_steps);
    assert!(report.skipped_steps > 0, "a scale at the single's largest overflows");
    assert!(report.skipped_steps < 16, "the scale backs off until steps go through");
    assert!(t.params().iter().flatten().all(|v| v.is_finite()));
    // Without a scaler the same gradients are refused outright.
    let none = Config { loss_scale: None, ..c };
    assert_eq!(fit_tuned(e, &spec(), &train, &dev, &none).unwrap().1.skipped_steps, 0);
}

#[test]
fn the_toy_task_is_learnt_under_bf16_compute_with_loss_scaling() {
    let mut rng = Rng::new(7);
    let (train, dev) = (toy(&mut rng, 256, 4, 3..60, [60, 61]), toy(&mut rng, 64, 4, 3..60, [60, 61]));
    let e = Encoder::<f32>::random(geometry(16, 24), 64, &mut rng).unwrap().with_bf16();
    let c = Config {
        epochs: 8,
        batch: 4,
        lr: 1e-2,
        warmup: 0.1,
        patience: 0,
        lanes: 2,
        loss_scale: Some(LossScale::default()),
        ..Config::default()
    };
    let (t, report) = fit_tuned(e.clone(), &spec(), &train, &dev, &c).unwrap();
    let best = report.dev[report.best_epoch];
    println!("bf16 through the loop: dev {best} after {} epochs, {} skipped", report.dev.len(), report.skipped_steps);
    assert!(best >= 0.85, "dev {best}: {:?}", report.dev);
    assert_eq!(t.score(&dev).unwrap(), best);
    assert_eq!(t.encoder().compute(), Compute::Bf16);
    // The same bytes whatever the threads and lanes.
    let (want, want_loss) = tuned_gradients(&t, &train[..7], 1, 0).unwrap();
    for (threads, lanes) in [(4, 0), (3, 1), (2, 3)] {
        let (got, loss) = tuned_gradients(&t, &train[..7], threads, lanes).unwrap();
        assert_eq!(loss.to_bits(), want_loss.to_bits());
        assert!(got.iter().zip(&want).all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())));
    }
    let again = Config { epochs: 2, threads: 1, lanes: 0, ..c };
    let (a, ra) = fit_tuned(e.clone(), &spec(), &train, &dev, &again).unwrap();
    let (b, rb) = fit_tuned(e, &spec(), &train, &dev, &Config { threads: 3, lanes: 3, ..again }).unwrap();
    assert_eq!(a.params(), b.params());
    assert_eq!((ra.train_loss, ra.dev), (rb.train_loss, rb.dev));
}
