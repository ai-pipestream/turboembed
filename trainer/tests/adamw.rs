//! The optimizer's schedule, clipping and decoupled decay, by hand.

use turbo_trainer::adamw::{AdamW, Settings};

#[test]
fn the_rate_warms_up_then_decays_to_zero() {
    let s = Settings { lr: 1.0, warmup: 4, total: 12, ..Settings::default() };
    let o = AdamW::<f64>::new(s, &[1]);
    let rates: Vec<f64> = (1..=12).map(|t| o.rate(t)).collect();
    assert_eq!(&rates[..4], [0.25, 0.5, 0.75, 1.0]);
    assert_eq!(rates[7], 0.5);
    assert_eq!(rates[11], 0.0);
    let flat = AdamW::<f64>::new(Settings { lr: 0.1, ..Settings::default() }, &[1]);
    assert_eq!(flat.rate(1000), 0.1);
}

#[test]
fn a_first_step_moves_each_weight_by_the_rate_and_clips_by_global_norm() {
    let s = Settings { lr: 0.1, weight_decay: 0.0, clip: 1.0, ..Settings::default() };
    let mut o = AdamW::<f64>::new(s, &[2, 1]);
    let mut params = vec![vec![1.0, 1.0], vec![1.0]];
    let mut grads = vec![vec![3.0, -4.0], vec![0.0]];
    let norm = o.step(&mut params, &mut grads, &[true, false]);
    assert_eq!(norm, 5.0);
    assert!((grads[0][0] - 0.6).abs() < 1e-12 && (grads[0][1] + 0.8).abs() < 1e-12);
    // Adam's first step is lr * sign(g), up to eps.
    assert!((params[0][0] - 0.9).abs() < 1e-6 && (params[0][1] - 1.1).abs() < 1e-6);
    assert_eq!(params[1][0], 1.0);
}

#[test]
fn decay_applies_only_to_marked_tensors() {
    let s = Settings { lr: 0.5, weight_decay: 0.1, clip: 0.0, ..Settings::default() };
    let mut o = AdamW::<f64>::new(s, &[1, 1]);
    let mut params = vec![vec![2.0], vec![2.0]];
    let mut grads = vec![vec![0.0], vec![0.0]];
    o.step(&mut params, &mut grads, &[true, false]);
    assert!((params[0][0] - 1.9).abs() < 1e-12);
    assert_eq!(params[1][0], 2.0);
}
