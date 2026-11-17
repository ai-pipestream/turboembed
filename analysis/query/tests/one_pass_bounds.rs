//! Numeric bound cues in front of a money or quantity value (`#2339`): a
//! negated bound flips the comparison and keeps the equal case, and the
//! filter's span takes in the whole cue, so no cue word is left behind as a
//! free-text term.

use turbo_query::{OnePassQueryAnalyzer, QueryComparisonOp};

fn bound(query: &str) -> (String, QueryComparisonOp, Vec<String>) {
    let analyzer = OnePassQueryAnalyzer::builder().build().unwrap();
    let plan = analyzer.analyze(query).unwrap();
    assert_eq!(plan.filters.len(), 1, "{query}: {:?}", plan.filters);
    let residual = plan.residual_terms.iter().map(|t| t.term.clone()).collect();
    (plan.filters[0].surface.clone(), plan.filters[0].comparison, residual)
}

#[test]
fn a_negated_bound_flips_the_comparison_and_keeps_equality() {
    for (query, surface, op) in [
        ("laptops no less than $50", "no less than $50", QueryComparisonOp::Gte),
        ("laptops not less than $50", "not less than $50", QueryComparisonOp::Gte),
        ("laptops no fewer than $50", "no fewer than $50", QueryComparisonOp::Gte),
        ("laptops no more than $50", "no more than $50", QueryComparisonOp::Lte),
        ("laptops not more than $50", "not more than $50", QueryComparisonOp::Lte),
        ("laptops no greater than $50", "no greater than $50", QueryComparisonOp::Lte),
        ("drones no less than 500 g", "no less than 500 g", QueryComparisonOp::Gte),
    ] {
        let (got_surface, got_op, residual) = bound(query);
        assert_eq!((got_surface.as_str(), got_op), (surface, op), "{query}");
        assert!(!residual.iter().any(|t| t == "no" || t == "not"), "{query}: {residual:?}");
    }
}

#[test]
fn a_plain_bound_is_unchanged() {
    for (query, surface, op) in [
        ("laptops less than $50", "less than $50", QueryComparisonOp::Lt),
        ("laptops more than $50", "more than $50", QueryComparisonOp::Gt),
        ("laptops at least $50", "at least $50", QueryComparisonOp::Gte),
        ("laptops at most $50", "at most $50", QueryComparisonOp::Lte),
        ("laptops under $50", "under $50", QueryComparisonOp::Lt),
        // A "no" that is not next to the bound does not negate it.
        ("no laptops less than $50", "less than $50", QueryComparisonOp::Lt),
    ] {
        let (got_surface, got_op, _) = bound(query);
        assert_eq!((got_surface.as_str(), got_op), (surface, op), "{query}");
    }
}
