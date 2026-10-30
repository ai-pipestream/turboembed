//! Early stopping: never armed during warmup, only a gain of more than
//! the tolerance resets patience, every better score is kept, and
//! patience counts only after warmup.

use turbo_trainer::{Config, EarlyStop};

/// The epoch a run of `scores` stops after, or None if it runs them all,
/// and the best epoch kept.
fn run(c: &Config, scores: &[f64]) -> (Option<usize>, usize) {
    let (mut stop, mut best) = (EarlyStop::new(c), 0);
    for (e, &s) in scores.iter().enumerate() {
        let (better, done) = stop.update(e, s);
        if better {
            best = e;
        }
        if done {
            return (Some(e), best);
        }
    }
    (None, best)
}

#[test]
fn a_plateau_during_warmup_does_not_stop_the_run() {
    // 40 epochs, 10% warmup: the first 4 epochs are warmup. The case the
    // 4080 run hit: dev best at epoch 1 and a hair under it after, which
    // stopped the run at epoch 4 under the old rule.
    let c = Config { epochs: 40, patience: 3, warmup: 0.1, tolerance: 0.0, ..Config::default() };
    let mut scores = vec![0.90, 0.921, 0.920, 0.920, 0.920, 0.920, 0.920];
    assert_eq!(run(&c, &scores).0, Some(6), "patience counts from the end of warmup, not epoch 1");
    scores[5] = 0.925;
    assert_eq!(run(&c, &scores), (None, 5));
}

#[test]
fn a_gain_inside_the_tolerance_is_kept_but_does_not_reset_patience() {
    let c = Config { epochs: 20, patience: 2, warmup: 0.0, tolerance: 1e-3, ..Config::default() };
    let scores = [0.5, 0.9, 0.9005, 0.9008, 0.8];
    assert_eq!(run(&c, &scores), (Some(3), 3), "creeping gains stop the run; the best of them is kept");
    let strict = Config { tolerance: 0.0, ..c };
    assert_eq!(run(&strict, &scores), (None, 3));
    // Gains that add up past the tolerance count from the last one that
    // counted, not from each epoch before.
    assert_eq!(run(&c, &[0.5, 0.9, 0.9006, 0.9012, 0.8, 0.8]), (Some(5), 3));
}

#[test]
fn a_flat_or_jittering_dev_score_stops_the_run() {
    // The 4080 run B case: dev flat at about 0.9733 from epoch 22 of 60,
    // moving by less than the tolerance, ran on under the old rule.
    let c = Config { epochs: 60, patience: 6, warmup: 0.1, ..Config::default() };
    let mut scores: Vec<f64> = (0..22).map(|e| 0.90 + 0.0033 * e as f64).collect();
    let flat = [0.9733, 0.9731, 0.9736, 0.9734, 0.9729, 0.9737, 0.9735, 0.9733, 0.9738, 0.9732];
    scores.extend(flat);
    let (stopped, best) = run(&c, &scores);
    assert_eq!(stopped, Some(28), "six epochs without a gain past 0.001 after epoch 22");
    assert_eq!(scores[best], 0.9737);
    let p = Config { epochs: 10, patience: 1, warmup: 0.0, ..Config::default() };
    assert_eq!(run(&p, &[0.9, 0.9, 0.9, 0.9]), (Some(1), 0));
    assert_eq!(run(&Config { tolerance: 0.0, ..p }, &[0.9, 0.91, 0.92, 0.92]), (Some(3), 2));
}

#[test]
fn zero_patience_never_stops() {
    let c = Config { epochs: 10, patience: 0, warmup: 0.0, ..Config::default() };
    assert_eq!(run(&c, &[0.9, 0.1, 0.1, 0.1, 0.1]).0, None);
}
