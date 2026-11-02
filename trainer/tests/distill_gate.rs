//! Release gates reject undefined measurements instead of passing NaN comparisons.
use turbo_trainer::{DistillReport, StudentGate, ThroughputComparison};

fn report() -> DistillReport {
    DistillReport {
        teacher_val_accuracy: 1.0,
        student_val_accuracy: 1.0,
        teacher_test_accuracy: 1.0,
        student_test_accuracy: 1.0,
        accuracy_drop: 0.0,
        within_margin: true,
        epochs_run: 1,
        best_epoch: 0,
        final_loss: 0.0,
        train_loss: vec![0.0],
        val_scores: vec![1.0],
    }
}
fn timing() -> ThroughputComparison {
    ThroughputComparison {
        tokens: 1,
        sentences: 1,
        teacher_seconds: 1.0,
        student_seconds: 1.0,
        teacher_tokens_per_sec: 1.0,
        student_tokens_per_sec: 1.0,
        teacher_sentences_per_sec: 1.0,
        student_sentences_per_sec: 1.0,
        speedup_ratio: 1.0,
    }
}
fn refused(r: &DistillReport, t: ThroughputComparison, drop: f64, speed: f64) {
    let gate = StudentGate::evaluate(r, t, drop, speed);
    assert!(!gate.passed, "invalid gate must not report passed: {gate:?}");
    assert!(gate.verify().is_err(), "invalid gate must not verify: {gate:?}");
}
#[test]
fn nonfinite_thresholds_and_measurements_fail_closed() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        refused(&report(), timing(), 0.02, bad);
        refused(&report(), timing(), bad, 1.0);
        for field in 0..7 {
            let mut t = timing();
            *match field {
                0 => &mut t.teacher_seconds,
                1 => &mut t.student_seconds,
                2 => &mut t.teacher_tokens_per_sec,
                3 => &mut t.student_tokens_per_sec,
                4 => &mut t.teacher_sentences_per_sec,
                5 => &mut t.student_sentences_per_sec,
                _ => &mut t.speedup_ratio,
            } = bad;
            refused(&report(), t, 0.02, 1.0);
        }
    }
}
#[test]
fn empty_or_zero_measurements_and_nonfinite_accuracy_are_refused() {
    let mut t = timing();
    t.tokens = 0;
    refused(&report(), t, 0.02, 0.0);
    t = timing();
    t.sentences = 0;
    refused(&report(), t, 0.02, 0.0);
    t = timing();
    t.teacher_seconds = 0.0;
    refused(&report(), t, 0.02, 0.0);
    for field in 0..3 {
        let mut r = report();
        *match field {
            0 => &mut r.teacher_test_accuracy,
            1 => &mut r.student_test_accuracy,
            _ => &mut r.accuracy_drop,
        } = f64::NAN;
        refused(&r, timing(), 0.02, 1.0);
    }
}
#[test]
fn finite_gates_accept_boundaries_and_refuse_measured_failures() {
    let g = StudentGate::evaluate(&report(), timing(), 0.0, 1.0);
    assert!(g.passed);
    g.verify().unwrap();
    refused(&report(), timing(), 0.02, 1.1);
    let mut r = report();
    r.student_test_accuracy = 0.9;
    r.accuracy_drop = 0.1;
    refused(&r, timing(), 0.02, 1.0);
    r.teacher_test_accuracy = 0.8;
    r.student_test_accuracy = 1.0;
    r.accuracy_drop = -0.2;
    let g = StudentGate::evaluate(&r, timing(), 0.0, 0.0);
    assert!(g.passed);
    g.verify().unwrap();
}
