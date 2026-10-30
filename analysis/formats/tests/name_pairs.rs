//! The repeated-prediction name pairs, scored by hand.
//!
//! Seven reference/prediction pairs of name samples in the shape of the
//! stripped native reader's basic fixture. Each prediction matches at most
//! one reference, so the third pair's two equal empty `p` names make one
//! true positive, not two, and recall never exceeds 1.

use turbo_analysis::Span;
use turbo_formats::evaluate::{LabeledSpan, NameFinderEvaluator, count_true_positives};
use turbo_formats::{Field, Name, NameSample};

/// A sample of `tokens` with the `names` `(start, end, kind)` over them.
/// Spans do not take part in scoring, so a zero span serves every field.
fn sample(tokens: &[&str], names: &[(u32, u32, &'static str)]) -> NameSample<'static> {
    let tokens = tokens.iter().map(|text| Field::made((*text).to_owned(), Span::default())).collect();
    let names = names.iter().map(|&(start, end, kind)| Name::new(start, end, Some(kind)).unwrap()).collect();
    NameSample::new(tokens, names, false, Span::default(), 1).unwrap()
}

/// The seven pairs of the basic fixture: `(tokens, names)` for the
/// reference, then for its prediction.
fn pairs() -> Vec<(NameSample<'static>, NameSample<'static>)> {
    let pairs = [
        // John lives in Paris; the prediction mistypes the location.
        (
            sample(&["John", "lives", "in", "Paris", "."], &[(0, 1, "person"), (3, 4, "location")]),
            sample(&["John", "lives", "in", "Paris", "."], &[(0, 1, "person"), (3, 4, "organization")]),
        ),
        // Identical single names.
        (sample(&["a", "b", "c"], &[(1, 2, "x")]), sample(&["a", "b", "c"], &[(1, 2, "x")])),
        // The prediction repeats the reference's empty name.
        (sample(&["x"], &[(0, 0, "p")]), sample(&["x"], &[(0, 0, "p"), (0, 0, "p")])),
        // The reference has no names; the prediction finds one.
        (sample(&["x", "y"], &[]), sample(&["x", "y"], &[(0, 1, "q")])),
        // Neither has names.
        (sample(&["no", "names", "here"], &[]), sample(&["no", "names", "here"], &[])),
        // Two typed names against one spanning both tokens.
        (sample(&["x", "y"], &[(0, 1, "a"), (1, 2, "b")]), sample(&["x", "y"], &[(0, 2, "a")])),
        // Two equal names in a different order and split.
        (sample(&["x", "y", "z"], &[(0, 1, "a"), (1, 2, "a")]), sample(&["x", "z", "y"], &[(0, 1, "a"), (2, 3, "a")])),
    ];
    pairs.into_iter().collect()
}

#[test]
fn the_name_pairs_count_a_repeated_prediction_once() {
    let pairs = pairs();
    // (true positives, predictions, references) after each pair.
    let running = [(1, 2, 2), (2, 3, 3), (3, 5, 4), (3, 6, 4), (3, 6, 4), (3, 7, 6), (4, 9, 8)];
    // (true positives, predictions, references) of each pair alone.
    let alone = [(1, 2, 2), (1, 1, 1), (1, 2, 1), (0, 1, 0), (0, 0, 0), (0, 1, 2), (1, 2, 2)];
    assert_eq!(pairs.len(), running.len());
    let mut e = NameFinderEvaluator::default();
    for (i, (r, p)) in pairs.iter().enumerate() {
        e.evaluate(r, p.names()).unwrap();
        let f = e.fmeasure();
        let (tp, selected, target) = running[i];
        assert_eq!((f.true_positives(), f.selected(), f.target()), (tp, selected, target), "pair {i}");
        assert!(f.recall() <= 1.0 && f.precision() <= 1.0, "pair {i}");
        let refs: Vec<LabeledSpan<'_>> = r.names().iter().map(LabeledSpan::from).collect();
        let preds: Vec<LabeledSpan<'_>> = p.names().iter().map(LabeledSpan::from).collect();
        let (tp, selected, target) = alone[i];
        assert_eq!(count_true_positives(&refs, &preds), tp, "pair {i}");
        assert_eq!((preds.len() as u64, refs.len() as u64), (selected, target), "pair {i}");
    }
}
