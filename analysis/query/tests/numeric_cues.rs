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

#[test]
fn money_filters_distinguish_hourly_daily_and_plain_amounts_across_languages() {
    let profile = QueryAnalysisProfile::default();

    let hourly = profile.analyze("contractors at USD 5 per hour").unwrap();
    let daily = profile.analyze("contractors at USD 5 per day").unwrap();
    let plain = profile.analyze("contractors at USD 5").unwrap();

    let (_, m_hour) = hourly.money_filters().next().expect("hourly money filter");
    let (_, m_day) = daily.money_filters().next().expect("daily money filter");
    let (_, m_plain) = plain.money_filters().next().expect("plain money filter");

    assert_eq!((m_hour.currency.as_str(), m_hour.decimal_text.as_str(), m_hour.per_unit()), ("USD", "5", Some("h")));
    assert_eq!((m_day.currency.as_str(), m_day.decimal_text.as_str(), m_day.per_unit()), ("USD", "5", Some("d")));
    assert_eq!((m_plain.currency.as_str(), m_plain.decimal_text.as_str(), m_plain.per_unit()), ("USD", "5", None));
    assert!(m_hour.is_rate());
    assert!(m_day.is_rate());
    assert!(!m_plain.is_rate());
    assert_ne!(m_hour, m_day);
    assert_ne!(m_hour, m_plain);
    assert_ne!(m_day, m_plain);

    assert_eq!(hourly.money_filters_with_per(Some("h")).count(), 1);
    assert_eq!(hourly.money_filters_with_per(Some("d")).count(), 0);
    assert_eq!(hourly.money_filters_with_per(None).count(), 0);
    assert_eq!(plain.money_filters_with_per(None).count(), 1);

    for (query, expected_surface, expected_curr, expected_val, expected_per) in [
        ("jobs paying $5 per hour", "$5 per hour", "USD", "5", Some("h")),
        ("jobs paying $5/day", "$5/day", "USD", "5", Some("d")),
        ("jobs paying $5 a day", "$5 a day", "USD", "5", Some("d")),
        ("jobs paying 50 cents an hour", "50 cents an hour", "USD", "0.5", Some("h")),
    ] {
        let out = profile.analyze(query).unwrap();
        let (filter, money) = out.money_filters().next().unwrap_or_else(|| panic!("no money filter in {query:?}"));
        assert_eq!(filter.surface, expected_surface, "{query:?}");
        assert_eq!(&query[filter.span.range()], expected_surface, "{query:?}");
        assert_eq!(money.currency.as_str(), expected_curr, "{query:?}");
        assert_eq!(money.decimal_text, expected_val, "{query:?}");
        assert_eq!(money.per_unit(), expected_per, "{query:?}");
    }

    let es_profile = QueryAnalysisProfile::builder().with_default_language("es").build().unwrap();
    for (query, expected_surface, expected_curr, expected_val, expected_per) in [
        ("trabajos a 5 dólares por hora", "5 dólares por hora", "USD", "5", Some("h")),
        ("trabajos a 5 dólares por día", "5 dólares por día", "USD", "5", Some("d")),
        ("trabajos a 5 dólares", "5 dólares", "USD", "5", None),
    ] {
        let out = es_profile.analyze(query).unwrap();
        let (filter, money) = out.money_filters().next().unwrap_or_else(|| panic!("no money filter in {query:?}"));
        assert_eq!(filter.surface, expected_surface, "{query:?}");
        assert_eq!(&query[filter.span.range()], expected_surface, "{query:?}");
        assert_eq!(money.currency.as_str(), expected_curr, "{query:?}");
        assert_eq!(money.decimal_text, expected_val, "{query:?}");
        assert_eq!(money.per_unit(), expected_per, "{query:?}");
    }

    let nl_profile = QueryAnalysisProfile::builder().with_default_language("nl").build().unwrap();
    for (query, expected_surface, expected_curr, expected_val, expected_per) in [
        ("werk voor 5 euro per uur", "5 euro per uur", "EUR", "5", Some("h")),
        ("werk voor 5 euro per dag", "5 euro per dag", "EUR", "5", Some("d")),
        ("werk voor 5 euro", "5 euro", "EUR", "5", None),
    ] {
        let out = nl_profile.analyze(query).unwrap();
        let (filter, money) = out.money_filters().next().unwrap_or_else(|| panic!("no money filter in {query:?}"));
        assert_eq!(filter.surface, expected_surface, "{query:?}");
        assert_eq!(&query[filter.span.range()], expected_surface, "{query:?}");
        assert_eq!(money.currency.as_str(), expected_curr, "{query:?}");
        assert_eq!(money.decimal_text, expected_val, "{query:?}");
        assert_eq!(money.per_unit(), expected_per, "{query:?}");
    }
}

#[test]
fn english_queries_with_dia_or_dag_preserve_english_money_and_quantity_mentions() {
    let profile = QueryAnalysisProfile::default();

    let pipe = profile.analyze("steel pipe 10 inch dia under twenty dollars").unwrap();
    let (qty_filter, _) = pipe.quantity_filters().next().expect("quantity in pipe query");
    assert_eq!(qty_filter.surface, "10 inch");
    let (money_filter, money) = pipe.money_filters().next().expect("money in pipe query");
    assert_eq!(money_filter.surface, "twenty dollars");
    assert_eq!(money.currency.as_str(), "USD");
    assert_eq!(money.decimal_text, "20");
    assert_eq!(money.per_unit(), None);

    let rate = profile.analyze("twenty-five dollars per hour dag").unwrap();
    let (rate_filter, rate_money) = rate.money_filters().next().expect("money in dag query");
    assert_eq!(rate_filter.surface, "twenty-five dollars per hour");
    assert_eq!(rate_money.currency.as_str(), "USD");
    assert_eq!(rate_money.decimal_text, "25");
    assert_eq!(rate_money.per_unit(), Some("h"));
}
