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

#[test]
fn a_candidate_cannot_excuse_a_set_by_keeping_its_documents_whole() {
    // The candidate keeps 90% of judged documents in one or two chunks and
    // loses the evidence; the reference splits them (10% short). The set
    // stays gated, and the gate fails.
    let cand = [rec("qasper", 0.37, Some((0.10, 0.05, 0.60, 0.9))), rec("gutenqa", 0.9, Some((0.70, 0.66, 0.05, 0.0)))];
    let windows =
        [rec("qasper", 0.37, Some((0.17, 0.07, 0.51, 0.1))), rec("gutenqa", 0.9, Some((0.60, 0.57, 0.09, 0.0)))];
    let r = chunk_gate(&cand, &windows, &[]).unwrap();
    let q = r.evidence.iter().find(|e| e.set == "qasper").unwrap();
    assert!(q.gated && !q.pass, "{r}");
    assert!(!r.pass, "{r}");
}

#[test]
fn a_candidate_missing_a_set_is_refused() {
    let cand = [rec("qasper", 0.37, Some((0.23, 0.21, 0.08, 0.0)))];
    let windows = [rec("qasper", 0.37, Some((0.17, 0.07, 0.51, 0.0))), rec("scifact", 0.69, None)];
    let e = chunk_gate(&cand, &windows, &[]).unwrap_err();
    assert!(e.contains("the candidate holds sets"), "{e}");
}

#[test]
fn a_set_given_twice_is_refused() {
    let one = rec("qasper", 0.37, Some((0.23, 0.21, 0.08, 0.0)));
    let windows = [rec("qasper", 0.37, Some((0.17, 0.07, 0.51, 0.0)))];
    let e = chunk_gate(&[one.clone(), one.clone()], &windows, &[]).unwrap_err();
    assert!(e.contains("holds set qasper twice"), "{e}");
    let fixed = [windows[0].clone(), windows[0].clone()];
    let e = chunk_gate(&[one], &windows, &[&fixed]).unwrap_err();
    assert!(e.contains("baseline 1 holds set qasper twice"), "{e}");
}

#[test]
fn a_candidate_without_evidence_on_an_evidence_set_is_refused() {
    let windows = [rec("qasper", 0.37, Some((0.17, 0.07, 0.51, 0.0)))];
    let e = chunk_gate(&[rec("qasper", 0.37, None)], &windows, &[]).unwrap_err();
    assert!(e.contains("one record scores evidence"), "{e}");
    let mut empty = rec("qasper", 0.37, Some((0.23, 0.21, 0.08, 0.0)));
    empty.metrics.span.as_mut().unwrap().per_query.clear();
    let e = chunk_gate(&[empty], &windows, &[]).unwrap_err();
    assert!(e.contains("no per-query evidence scores"), "{e}");
}

#[test]
fn records_with_bad_evidence_fields_fail_their_check() {
    let ok = rec("qasper", 0.37, Some((0.23, 0.21, 0.08, 0.0)));
    let check = |edit: &dyn Fn(&mut SpanMetrics)| {
        let mut r = ok.clone();
        // The means the per-query scores must reproduce.
        let s = r.metrics.span.as_mut().unwrap();
        let n = s.per_query.len() as f64;
        s.mean.at_10.recall = s.per_query.iter().map(|q| q.recall).sum::<f64>() / n;
        s.mean.at_10.contained = s.per_query.iter().map(|q| q.contained).sum::<f64>() / n;
        s.queries = s.per_query.len() as u32;
        edit(s);
        r.check()
    };
    assert!(check(&|_| {}).is_ok());
    assert!(
        check(&|s| s.integrity = Some(Integrity { split: 1.5, short_documents: 0.0 })).unwrap_err().contains("split")
    );
    assert!(check(&|s| s.per_query[3].contained = -0.1).unwrap_err().contains("contained"));
    assert!(check(&|s| s.per_query.swap(0, 1)).unwrap_err().contains("is not before"));
    assert!(
        check(&|s| {
            s.per_query.pop();
        })
        .unwrap_err()
        .contains("per_query has")
    );
    assert!(check(&|s| s.mean.at_10.recall += 0.1).unwrap_err().contains("per query mean"));
}
