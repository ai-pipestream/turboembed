//! Early stopping: never armed during warmup, a score within tolerance of
//! the best holds the run going, and patience counts only after warmup.

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
fn a_score_within_tolerance_holds_but_only_a_better_one_is_kept() {
    let c = Config { epochs: 20, patience: 2, warmup: 0.0, tolerance: 1e-3, ..Config::default() };
    let scores = [0.5, 0.9, 0.8995, 0.8999, 0.8992, 0.9001, 0.8, 0.8];
    assert_eq!(run(&c, &scores), (Some(7), 5));
    let strict = Config { tolerance: 0.0, ..c };
    assert_eq!(run(&strict, &scores), (Some(3), 1));
}

#[test]
fn an_equal_score_holds_level() {
    let c = Config { epochs: 10, patience: 1, warmup: 0.0, tolerance: 0.0, ..Config::default() };
    assert_eq!(run(&c, &[0.9, 0.9, 0.9, 0.9]), (None, 0));
}

#[test]
fn zero_patience_never_stops() {
    let c = Config { epochs: 10, patience: 0, warmup: 0.0, ..Config::default() };
    assert_eq!(run(&c, &[0.9, 0.1, 0.1, 0.1, 0.1]).0, None);
}
