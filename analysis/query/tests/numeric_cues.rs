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

#[test]
fn ordinary_words_that_spell_iso_codes_yield_no_money_filter() {
    for q in [
        "top 10 laptops",
        "top 10 laptops under 500",
        "all 3 seasons",
        "10 cup coffee maker",
        "pen 2 pack",
        "try 3 tips",
        "10 mad minutes",
        "usd 500 laptops",
        "top 10 php frameworks",
        "3 php books",
        "10 cop shows",
        "5 ron howard films",
        "10 cad files",
        "top 5 cad programs",
        "2 cad drawings",
        "best 3 cad models",
    ] {
        let kinds = analyze(q);
        assert!(!kinds.iter().any(|k| matches!(k, FilterKind::Money(_))), "invented money filter in {q:?}: {kinds:?}");
    }
}
