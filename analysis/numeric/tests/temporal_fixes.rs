//! Tests for default temporal extraction fixes (#934), OpenNLP parity mode,
//! and `DocumentDateSelector` fallback to grammar `Times`.

mod temporal_common;

use temporal_common::{absolute, day, relative, rows};
use turbo_analysis::{Analyzer, Document, Ix, Span, ViewId};
use turbo_analysis_layers::{Modifier, NumberNotation, TimeGrain, TimeKind, TimeMention, Times};
use turbo_grammar::TimeAnnotator;
use turbo_numeric::money::{
    CONVERTED_MONEY, Currency, DatedMoneyConverter, Decimal, EcbRates, MoneyExtractor, cmp_values,
};
use turbo_numeric::{Date, DocumentDate, DocumentDateSelector, Granularity, TemporalExtractor, Temporals};

use Granularity::{Day, Month, Quarter, Week, Year};

fn found_default(text: &str, reference: Option<Date>) -> Vec<temporal_common::Found> {
    let mut t = Temporals::new(ViewId::SOURCE);
    match reference {
        Some(d) => TemporalExtractor::with_reference(d).extract(text, &mut t),
        None => TemporalExtractor::new().extract(text, &mut t),
    }
    rows(&t)
}

fn found_parity(text: &str, reference: Option<Date>) -> Vec<temporal_common::Found> {
    let mut t = Temporals::new(ViewId::SOURCE);
    let extractor = match reference {
        Some(d) => TemporalExtractor::with_reference(d),
        None => TemporalExtractor::new(),
    }
    .with_opennlp_parity(true);
    extractor.extract(text, &mut t);
    rows(&t)
}

#[test]
fn years_across_four_digit_range_are_extracted_by_default() {
    for (text, expected, gran) in [
        ("0001-01-01", "0001-01-01", Day),
        ("0999-12-31", "0999-12-31", Day),
        ("3000-01-01", "3000-01-01", Day),
        ("9999-12-31", "9999-12-31", Day),
        ("14 July 0999", "0999-07-14", Day),
        ("July 14, 3000", "3000-07-14", Day),
        ("January 0999", "0999-01", Month),
        ("January 3000", "3000-01", Month),
        ("Q1 0999", "0999-Q1", Quarter),
        ("Q4 3000", "3000-Q4", Quarter),
    ] {
        assert_eq!(found_default(text, None), [absolute(0, text.len(), expected, gran)], "{text}");
        assert_eq!(found_parity(text, None), [], "parity should reject {text}");
    }
}

#[test]
fn dateline_accepts_four_digit_years_outside_opennlp_range_by_default() {
    let text = "Dated 0999-12-31, amended yesterday.";
    assert_eq!(found_default(text, None), [absolute(6, 16, "0999-12-31", Day), relative(26, 35, "0999-12-30", Day)]);
    assert_eq!(found_parity(text, None), []);
}

#[test]
fn multi_digit_counts_over_two_digits_are_extracted_by_default() {
    let r = Some(day("2026-07-17"));
    for (text, expected, gran) in [
        ("100 days ago", "2026-04-08", Day),
        ("in 100 days", "2026-10-25", Day),
        ("in 365 days", "2027-07-17", Day),
        ("120 weeks ago", "2024-W13", Week),
        ("in 120 months", "2036-07", Month),
        ("1000 years ago", "1026", Year),
    ] {
        assert_eq!(found_default(text, r), [relative(0, text.len(), expected, gran)], "{text}");
        assert_eq!(found_parity(text, r), [], "parity should reject {text}");
    }
}

#[test]
fn plural_units_after_next_last_and_this_are_extracted_by_default() {
    let r = Some(day("2026-07-17"));
    for (text, expected, gran) in [
        ("last days", "2026-07-16", Day),
        ("this days", "2026-07-17", Day),
        ("next days", "2026-07-18", Day),
        ("last weeks", "2026-W28", Week),
        ("this weeks", "2026-W29", Week),
        ("next weeks", "2026-W30", Week),
        ("last months", "2026-06", Month),
        ("this months", "2026-07", Month),
        ("next months", "2026-08", Month),
        ("last quarters", "2026-Q2", Quarter),
        ("this quarters", "2026-Q3", Quarter),
        ("next quarters", "2026-Q4", Quarter),
        ("last years", "2025", Year),
        ("this years", "2026", Year),
        ("next years", "2027", Year),
    ] {
        assert_eq!(found_default(text, r), [relative(0, text.len(), expected, gran)], "{text}");
        assert_eq!(found_parity(text, r), [], "parity should reject {text}");
    }
}

#[test]
fn supplementary_plane_letters_are_treated_as_letters() {
    assert_eq!(found_default("\u{10400}July 2026", None), []);
    assert_eq!(found_default("July\u{10400} 2026", None), []);
    assert_eq!(found_default("\u{10400} July 2026", None), [absolute(5, 14, "2026-07", Month)]);
    assert_eq!(found_parity("\u{10400} July 2026", None), [absolute(5, 14, "2026-07", Month)]);
}

#[test]
fn spanish_document_dates_via_grammar_times_and_converts_money() {
    let rates = EcbRates::parse("Date,USD,\n2026-07-14,1.2000,\n").unwrap();
    let usd = Currency::parse("USD").unwrap();
    let analyzer = Analyzer::builder()
        .add(TemporalExtractor::new())
        .add(TimeAnnotator::spanish())
        .add(DocumentDateSelector)
        .add(MoneyExtractor::new(NumberNotation::LatinEu))
        .step(DatedMoneyConverter::new(rates, usd).bound())
        .build()
        .unwrap();

    let mut docs = vec![Document::new("Madrid, 14 de julio de 2026. El comprador pag\u{f3} 2.500 EUR.")];
    for res in analyzer.run(&mut docs) {
        res.unwrap();
    }
    let doc = &docs[0];

    assert_eq!(doc.get::<Temporals>().unwrap().spans.len(), 0);
    let doc_date = doc.get::<DocumentDate>().unwrap();
    assert_eq!(doc_date.date, Some(day("2026-07-14")));
    assert_eq!(doc_date.mention, None);

    let converted = doc.get_key(&CONVERTED_MONEY).unwrap();
    assert_eq!(converted.spans.len(), 1);
    assert_eq!(converted.currency, [usd]);
    let a = converted.amount(Ix::new(0));
    assert_eq!(cmp_values(a, Decimal::parse("3000").unwrap().as_ref()), std::cmp::Ordering::Equal);
}

#[test]
fn dutch_document_dates_via_grammar_times_and_converts_money() {
    let rates = EcbRates::parse("Date,USD,\n2026-07-14,1.1000,\n").unwrap();
    let usd = Currency::parse("USD").unwrap();
    let analyzer = Analyzer::builder()
        .add(TimeAnnotator::dutch())
        .add(DocumentDateSelector)
        .add(MoneyExtractor::new(NumberNotation::LatinEu))
        .step(DatedMoneyConverter::new(rates, usd).bound())
        .build()
        .unwrap();

    let mut docs = vec![Document::new("Amsterdam, 14 juli 2026. De koper betaalde 1.200 EUR.")];
    for res in analyzer.run(&mut docs) {
        res.unwrap();
    }
    let doc = &docs[0];

    let doc_date = doc.get::<DocumentDate>().unwrap();
    assert_eq!(doc_date.date, Some(day("2026-07-14")));
    assert_eq!(doc_date.mention, None);

    let converted = doc.get_key(&CONVERTED_MONEY).unwrap();
    assert_eq!(converted.spans.len(), 1);
    assert_eq!(converted.currency, [usd]);
    let a = converted.amount(Ix::new(0));
    assert_eq!(cmp_values(a, Decimal::parse("1320").unwrap().as_ref()), std::cmp::Ordering::Equal);
}

#[test]
fn select_from_times_ignores_non_date_mentions_and_prefers_temporals() {
    let mut times = Times::new(ViewId::SOURCE);
    // Unresolved date is skipped.
    times.push(TimeMention {
        span: Span::new(0, 4),
        kind: TimeKind::Date,
        grain: TimeGrain::Day,
        modifier: Modifier::default(),
        start: None,
        end: None,
        duration_ms: None,
        ambiguous: false,
        timex: "",
    });
    // Duration is skipped even with a resolved value.
    times.push(TimeMention {
        span: Span::new(5, 10),
        kind: TimeKind::Duration,
        grain: TimeGrain::Day,
        modifier: Modifier::default(),
        start: Some(86_400),
        end: None,
        duration_ms: Some(86_400_000),
        ambiguous: false,
        timex: "P1D",
    });
    // Time-of-day is skipped.
    times.push(TimeMention {
        span: Span::new(11, 16),
        kind: TimeKind::Time,
        grain: TimeGrain::Hour,
        modifier: Modifier::default(),
        start: Some(36_000),
        end: None,
        duration_ms: None,
        ambiguous: false,
        timex: "T10",
    });
    // Resolved DateTime on 2026-07-14T10:00:00.
    times.push(TimeMention {
        span: Span::new(17, 36),
        kind: TimeKind::DateTime,
        grain: TimeGrain::Second,
        modifier: Modifier::default(),
        start: Some(day("2026-07-14").epoch_day() * 86_400 + 36_000),
        end: None,
        duration_ms: None,
        ambiguous: false,
        timex: "2026-07-14T10:00:00",
    });

    assert_eq!(
        DocumentDateSelector.select_from_times(&times),
        DocumentDate { date: Some(day("2026-07-14")), mention: None }
    );
}
