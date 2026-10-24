//! Numeric query cue widening (#1049): long unit words,
//! lowercase ISO currency codes, and 4-token scaled amounts
//! reach the money and quantity extractors.

use turbo_query::{FilterKind, QueryAnalysisProfile};

fn analyze(q: &str) -> Vec<FilterKind> {
    QueryAnalysisProfile::default().analyze(q).unwrap().filters.into_iter().map(|f| f.kind).collect()
}

#[test]
fn long_unit_words_yield_quantity_filters() {
    for q in
        ["runs of 10 kilometers", "laptops under 2 kilograms", "files over 100 megabytes", "sleeps of 500 milliseconds"]
    {
        let kinds = analyze(q);
        assert!(kinds.iter().any(|k| matches!(k, FilterKind::Quantity(_))), "no quantity filter in {q:?}: {kinds:?}");
    }
}

#[test]
fn lowercase_iso_codes_yield_money_filters() {
    for q in ["flights under 500 usd", "hotels under 50 eur", "trains under 25 gbp", "rent over 100000 jpy"] {
        let kinds = analyze(q);
        assert!(kinds.iter().any(|k| matches!(k, FilterKind::Money(_))), "no money filter in {q:?}: {kinds:?}");
    }
}

#[test]
fn four_token_scaled_amounts_yield_money_filters() {
    let kinds = analyze("budgets over 10 million US dollars");
    assert!(kinds.iter().any(|k| matches!(k, FilterKind::Money(_))), "no money filter: {kinds:?}");
}
