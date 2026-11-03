use turbo_analysis::Span;
use turbo_langspan::gold::{Scores, parse};

#[test]
fn punctuation_boundaries_match_each_gold_boundary_at_most_once() {
    let gold = parse("## example\n[eng]Hello...[zho]世界").unwrap().remove(0);
    let mut scores = Scores::default();
    scores.add(&gold, &[(Span::new(0, 5), "eng"), (Span::new(5, 8), "und"), (Span::new(8, 14), "zho")]);
    assert_eq!(scores.gold_boundaries, 1);
    assert_eq!(scores.predicted_boundaries, 2);
    assert_eq!(scores.matched_boundaries, 1);
    assert_eq!(scores.boundary_recall(), 1.0);
    assert_eq!(scores.boundary_precision(), 0.5);
    assert_eq!(scores.exact_boundaries, 1);
    assert_eq!(scores.exact_documents, 0);
}

#[test]
fn repeated_predictions_cannot_hide_a_missing_gold_boundary() {
    let gold = parse("## example\n[eng]Hello...[deu]Welt...[fra]bonjour").unwrap().remove(0);
    let mut scores = Scores::default();
    scores.add(&gold, &[(Span::new(0, 5), "eng"), (Span::new(5, 8), "deu"), (Span::new(8, 22), "fra")]);
    assert_eq!(scores.exact_documents, 0);
    assert_eq!(scores.matched_boundaries, 1);
    assert_eq!(scores.boundary_recall(), 0.5);
    assert_eq!(scores.boundary_precision(), 0.5);
}

#[test]
fn repeated_gold_boundaries_remain_distinct_and_perfect_predictions_still_match() {
    let gold = parse("## example\n[eng]Hello[und]...[zho]世界").unwrap().remove(0);
    let mut partial = Scores::default();
    partial.add(&gold, &[(Span::new(0, 8), "eng"), (Span::new(8, 14), "zho")]);
    assert_eq!(partial.matched_boundaries, 1);
    assert_eq!(partial.boundary_recall(), 0.5);
    assert_eq!(partial.boundary_precision(), 1.0);
    assert_eq!(partial.exact_documents, 0);

    let mut perfect = Scores::default();
    perfect.add(&gold, &[(Span::new(0, 5), "eng"), (Span::new(5, 8), "und"), (Span::new(8, 14), "zho")]);
    assert_eq!(perfect.matched_boundaries, 2);
    assert_eq!(perfect.exact_boundaries, 2);
    assert_eq!(perfect.boundary_recall(), 1.0);
    assert_eq!(perfect.boundary_precision(), 1.0);
    assert_eq!(perfect.exact_documents, 1);
}
