//! The optimizer's schedule, clipping and decoupled decay, by hand.

use turbo_trainer::adamw::{AdamW, Settings};

#[test]
fn the_rate_warms_up_then_decays_and_every_step_of_the_schedule_moves() {
    let s = Settings { lr: 1.0, warmup: 4, total: 12, ..Settings::default() };
    let o = AdamW::<f64>::new(s, &[1]).unwrap();
    let rates: Vec<f64> = (1..=13).map(|t| o.rate(t)).collect();
    assert_eq!(&rates[..4], [0.25, 0.5, 0.75, 1.0]);
    assert_eq!(&rates[4..], [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]);
    // One step with no warmup takes the whole rate; so does a two-step
    // schedule's first, and its second half of it.
    let one = AdamW::<f64>::new(Settings { lr: 0.1, total: 1, ..Settings::default() }, &[1]).unwrap();
    assert_eq!(one.rate(1), 0.1);
    let two = AdamW::<f64>::new(Settings { lr: 0.1, total: 2, ..Settings::default() }, &[1]).unwrap();
    assert_eq!([two.rate(1), two.rate(2)], [0.1, 0.05]);
    let flat = AdamW::<f64>::new(Settings { lr: 0.1, ..Settings::default() }, &[1]).unwrap();
    assert_eq!(flat.rate(1000), 0.1);
}

#[test]
fn settings_that_would_not_train_and_mismatched_tensors_are_refused() {
    for bad in [
        Settings { lr: -0.1, ..Settings::default() },
        Settings { lr: 0.0, ..Settings::default() },
        Settings { lr: f64::NAN, ..Settings::default() },
        Settings { beta1: 1.0, ..Settings::default() },
        Settings { weight_decay: -0.01, ..Settings::default() },
        Settings { clip: f64::INFINITY, ..Settings::default() },
        Settings { warmup: 3, total: 3, ..Settings::default() },
        // A zero eps divides 0 by 0 on a zero gradient.
        Settings { eps: 0.0, ..Settings::default() },
        Settings { eps: f64::NAN, ..Settings::default() },
    ] {
        assert!(AdamW::<f64>::new(bad, &[1]).is_err(), "{bad:?}");
    }
    let mut o = AdamW::<f64>::new(Settings::default(), &[2]).unwrap();
    assert!(o.step(&mut [vec![1.0]], &mut [vec![1.0, 1.0]], &[true]).is_err());
    assert!(o.step(&mut [vec![1.0, 1.0]], &mut [vec![1.0]], &[true]).is_err());
    assert!(o.step(&mut [vec![1.0, 1.0]], &mut [vec![1.0, 1.0]], &[]).is_err());
    assert!(o.step(&mut [vec![1.0, 1.0]], &mut [vec![f64::NAN, 1.0]], &[true]).is_err());
}

#[test]
fn a_first_step_moves_each_weight_by_the_rate_and_clips_by_global_norm() {
    let s = Settings { lr: 0.1, weight_decay: 0.0, clip: 1.0, ..Settings::default() };
    let mut o = AdamW::<f64>::new(s, &[2, 1]).unwrap();
    let mut params = vec![vec![1.0, 1.0], vec![1.0]];
    let mut grads = vec![vec![3.0, -4.0], vec![0.0]];
    let norm = o.step(&mut params, &mut grads, &[true, false]).unwrap();
    assert_eq!(norm, 5.0);
    assert!((grads[0][0] - 0.6).abs() < 1e-12 && (grads[0][1] + 0.8).abs() < 1e-12);
    // Adam's first step is lr * sign(g), up to eps.
    assert!((params[0][0] - 0.9).abs() < 1e-6 && (params[0][1] - 1.1).abs() < 1e-6);
    assert_eq!(params[1][0], 1.0);
}

#[test]
fn decay_applies_only_to_marked_tensors() {
    let s = Settings { lr: 0.5, weight_decay: 0.1, clip: 0.0, ..Settings::default() };
    let mut o = AdamW::<f64>::new(s, &[1, 1]).unwrap();
    let mut params = vec![vec![2.0], vec![2.0]];
    let mut grads = vec![vec![0.0], vec![0.0]];
    o.step(&mut params, &mut grads, &[true, false]).unwrap();
    assert!((params[0][0] - 1.9).abs() < 1e-12);
    assert_eq!(params[1][0], 2.0);
}

#[test]
fn a_zero_gradient_never_writes_nan_and_settings_stay_as_checked() {
    let mut o = AdamW::<f32>::new(Settings::default(), &[1]).unwrap();
    let mut p = vec![vec![1.0f32]];
    o.step(&mut p, &mut [vec![0.0]], &[false]).unwrap();
    assert!(p[0][0].is_finite());
    // The settings are read-only after construction.
    assert_eq!(o.settings().eps, Settings::default().eps);
}

#[test]
fn the_longest_schedule_does_not_overflow() {
    let s = Settings { lr: 1.0, warmup: 0, total: u64::MAX, ..Settings::default() };
    let o = AdamW::<f64>::new(s, &[1]).unwrap();
    assert_eq!(o.rate(1), 1.0);
    assert!(o.rate(u64::MAX) > 0.0);
    let w = AdamW::<f64>::new(Settings { warmup: u64::MAX - 1, ..s }, &[1]).unwrap();
    assert_eq!(w.rate(u64::MAX), 1.0);
}
