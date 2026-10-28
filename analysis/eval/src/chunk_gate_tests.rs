use super::*;
use crate::record::tests::sample;
use crate::record::{QuerySpanAt10, SpanMetrics};
use crate::spans::Integrity;

/// A record of `set` with 40 queries at nDCG `ndcg`, and, when `evidence`
/// is given, span recall and containment (and split, short) per query.
fn rec(set: &str, ndcg: f64, evidence: Option<(f64, f64, f64, f64)>) -> EvalRecord {
    let scores: Vec<f64> = (0..40).map(|i| (ndcg + 0.01 * (i % 5) as f64).min(1.0)).collect();
    let mut r = sample(set, &scores);
    let span = r.metrics.span.take().unwrap();
    r.metrics.span = evidence.map(|(recall, contained, split, short)| SpanMetrics {
        integrity: Some(Integrity { split, short_documents: short }),
        per_query: (0..40)
            .map(|i| QuerySpanAt10 {
                query: format!("q{i:02}"),
                recall: (recall + 0.02 * (i % 3) as f64).min(1.0),
                contained: (contained + 0.02 * (i % 4) as f64).min(1.0),
            })
            .collect(),
        ..span
    });
    r
}

#[test]
fn whole_evidence_at_no_ranking_cost_passes() {
    // nDCG@10 0.2 points under the reference on qasper: above the floor.
    let cand = [rec("qasper", 0.378, Some((0.23, 0.21, 0.08, 0.0))), rec("scifact", 0.69, None)];
    let windows = [rec("qasper", 0.38, Some((0.17, 0.07, 0.51, 0.0))), rec("scifact", 0.69, None)];
    let r = chunk_gate(&cand, &windows, &[]).unwrap();
    assert!(r.pass, "{r}");
    assert_eq!(r.evidence.len(), 1);
    assert!(r.to_string().starts_with("evidence qasper:"), "{r}");
}

#[test]
fn a_ranking_loss_past_the_floor_fails() {
    let cand = [rec("qasper", 0.37, Some((0.23, 0.21, 0.08, 0.0))), rec("scifact", 0.60, None)];
    let windows = [rec("qasper", 0.37, Some((0.17, 0.07, 0.51, 0.0))), rec("scifact", 0.70, None)];
    let r = chunk_gate(&cand, &windows, &[]).unwrap();
    assert!(!r.pass);
    assert!(r.floors.iter().any(|f| f.set == "scifact" && !f.pass));
}

#[test]
fn the_floor_is_against_the_best_baseline() {
    let cand = [rec("qasper", 0.37, Some((0.23, 0.21, 0.08, 0.0)))];
    let windows = [rec("qasper", 0.37, Some((0.17, 0.07, 0.51, 0.0)))];
    let fixed = [rec("qasper", 0.45, Some((0.20, 0.08, 0.40, 0.0)))];
    let r = chunk_gate(&cand, &windows, &[&fixed]).unwrap();
    assert!(!r.pass, "{r}");
    assert!((r.floors[0].diff.mean - -8.0).abs() < 1e-6);
}

#[test]
fn evidence_must_beat_the_reference_and_split_less() {
    let windows = [rec("qasper", 0.37, Some((0.17, 0.07, 0.51, 0.0)))];
    // Same evidence: the interval is not above zero.
    let same = [rec("qasper", 0.37, Some((0.17, 0.07, 0.30, 0.0)))];
    assert!(!chunk_gate(&same, &windows, &[]).unwrap().pass);
    // Better evidence but more splits.
    let splits = [rec("qasper", 0.37, Some((0.23, 0.21, 0.60, 0.0)))];
    assert!(!chunk_gate(&splits, &windows, &[]).unwrap().pass);
}

#[test]
fn mostly_short_documents_are_reported_not_gated() {
    let cand = [rec("conteb-squad", 0.89, Some((0.97, 0.97, 0.01, 0.9)))];
    let windows = [rec("conteb-squad", 0.89, Some((0.97, 0.97, 0.01, 0.9)))];
    let r = chunk_gate(&cand, &windows, &[]).unwrap();
    // No gated evidence set: the gate cannot pass on these alone.
    assert!(!r.pass);
    assert!(!r.evidence[0].gated);
    assert!(r.to_string().contains("reported, mostly short documents"), "{r}");
}

#[test]
fn records_without_integrity_are_refused() {
    let mut old = rec("qasper", 0.37, Some((0.23, 0.21, 0.08, 0.0)));
    old.metrics.span.as_mut().unwrap().integrity = None;
    let windows = [rec("qasper", 0.37, Some((0.17, 0.07, 0.51, 0.0)))];
    let e = chunk_gate(&[old], &windows, &[]).unwrap_err();
    assert!(e.contains("before #172's gate"), "{e}");
}
