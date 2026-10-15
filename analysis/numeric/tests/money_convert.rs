//! The conversion step: the cases of OpenNLP's
//! `MoneyConversionAnnotatorTest`, `MoneyConversionContractTest` and the
//! money part of `NumericExtractionExampleTest`.
//!
//! The document-date cases run the temporal extractor and the document
//! date selector before [`DatedMoneyConverter`], as the Java runs
//! `TemporalAnnotator` and `DocumentDateAnnotator`. The Java also checks
//! that a provider keeps the span and the currency, and that a document
//! date layer holds at most one date; here a provider returns only the
//! amount, so it cannot change either, and a `DocumentDate` holds one
//! date or none.

mod money_common;

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use money_common::{cur, day, dec, load, same};
use turbo_analysis::{Analyzer, Annotator, Document, Ix, Layer, Span, ViewId};
use turbo_analysis_layers::{DecimalRef, DocumentDate, MoneyMention, Temporals};
use turbo_numeric::NumberNotation::LatinUs;
use turbo_numeric::money::{
    CONVERTED_MONEY, Currency, Date, DatedMoneyConverter, Decimal, EcbRates, FxRates, Money, MoneyConverter,
    MoneyExtractor, ScaleRangeError,
};
use turbo_numeric::{DocumentDateSelector, TemporalExtractor};

const FIXTURE: &str = "Date,USD,AUD,\n2026-07-10,1.2000,1.8000,\n";

fn australian() -> MoneyExtractor {
    MoneyExtractor::new(LatinUs).with_symbol('$', cur("AUD")).unwrap()
}

/// Runs extraction and conversion into USD as of `on` over `text`.
fn analyze(extractor: MoneyExtractor, rates: EcbRates, on: &str, text: &str) -> Document {
    let analyzer = Analyzer::builder()
        .add(extractor)
        .step(MoneyConverter::new(rates, cur("USD"), day(on)).bound())
        .build()
        .unwrap();
    let mut docs = vec![Document::new(text)];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    docs.pop().unwrap()
}

fn amount(m: &Money, i: u32) -> Decimal {
    let a = m.amount(Ix::new(i));
    Decimal { negative: a.negative, digits: a.digits.into(), scale: a.scale as i32 }
}

#[test]
fn regional_symbols_feed_conversion() {
    let doc = analyze(australian(), load(FIXTURE), "2026-07-10", "the house sold for $900,000 last week");
    let money = doc.get::<Money>().unwrap();
    assert_eq!(money.len(), 1);
    assert_eq!(money.currency[0], cur("AUD"));
    let converted = doc.get_key(&CONVERTED_MONEY).unwrap();
    assert_eq!(converted.len(), 1);
    assert_eq!(converted.currency[0], cur("USD"));
    // 900000 AUD * (1.2 / 1.8) = 600000 USD
    assert!(same(&dec("600000"), &amount(converted, 0)));
    assert_eq!(converted.spans.get(Ix::new(0)), money.spans.get(Ix::new(0)));
}

#[test]
fn unconvertible_mentions_are_omitted() {
    let doc = analyze(
        MoneyExtractor::default(),
        load(FIXTURE),
        "2026-07-10",
        "we spent $10 in transit and CHF 20 at the airport",
    );
    assert_eq!(doc.get::<Money>().unwrap().len(), 2);
    let converted = doc.get_key(&CONVERTED_MONEY).unwrap();
    // No CHF column in the fixture, so only the USD mention converts.
    assert_eq!(converted.len(), 1);
    assert!(same(&dec("10"), &amount(converted, 0)));
}

/// Runs the temporal extractor, the document date selector, `extractor`
/// and conversion into USD as of the document date over `text`.
fn analyze_dated(extractor: MoneyExtractor, rates: EcbRates, text: &str) -> Document {
    let analyzer = Analyzer::builder()
        .add(TemporalExtractor::new())
        .add(DocumentDateSelector)
        .add(extractor)
        .step(DatedMoneyConverter::new(rates, cur("USD")).bound())
        .build()
        .unwrap();
    let mut docs = vec![Document::new(text)];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    docs.pop().unwrap()
}

#[test]
fn the_document_date_anchors_the_conversion() {
    let doc = analyze_dated(australian(), load(FIXTURE), "Sydney, 2026-07-10. The house sold for $900,000 at auction.");
    let converted = doc.get_key(&CONVERTED_MONEY).unwrap();
    assert_eq!(converted.len(), 1);
    assert_eq!(converted.currency[0], cur("USD"));
    assert!(same(&dec("600000"), &amount(converted, 0)));
}

#[test]
fn no_document_date_converts_nothing() {
    let doc = analyze_dated(MoneyExtractor::default(), load(FIXTURE), "the undated draft mentions $10");
    assert_eq!(doc.get::<Money>().unwrap().len(), 1);
    assert_eq!(doc.get_key(&CONVERTED_MONEY).unwrap().len(), 0);
}

/// `testMissingRequiredLayersFailLoud`, document-date mode: a pipeline
/// without the document date layer is refused when it is built.
#[test]
fn a_missing_document_date_layer_is_refused() {
    let built = Analyzer::builder()
        .add(MoneyExtractor::default())
        .step(DatedMoneyConverter::new(load(FIXTURE), cur("USD")).bound())
        .build();
    let e = built.err().expect("a dated converter with no document date to read");
    assert!(e.to_string().contains("turbo:document-date"), "{e}");
}

/// `testMissingRequiredLayersFailLoud`: a pipeline without the money layer
/// is refused when it is built.
#[test]
fn a_missing_money_layer_is_refused() {
    let built =
        Analyzer::builder().step(MoneyConverter::new(load(FIXTURE), cur("USD"), day("2026-07-10")).bound()).build();
    let e = built.err().expect("a converter with no money layer to read");
    assert!(e.to_string().contains("turbo:money"), "{e}");
}

/// A provider whose `convert` gives a fixed answer and counts its calls.
struct Scripted {
    answer: Result<Option<Decimal>, ScaleRangeError>,
    calls: AtomicU32,
}

impl FxRates for Scripted {
    type Error = ScaleRangeError;

    fn rate(&self, _: Currency, _: Currency, _: Date) -> Result<Option<Decimal>, ScaleRangeError> {
        panic!("the conversion override must be used")
    }

    fn convert(
        &self,
        amount: DecimalRef<'_>,
        from: Currency,
        to: Currency,
        as_of: Date,
        out: &mut Decimal,
    ) -> Result<bool, ScaleRangeError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        assert_eq!((amount.digits, from, to, as_of), ("10", cur("EUR"), cur("USD"), day("2026-07-14")));
        match &self.answer {
            Ok(Some(d)) => {
                *out = d.clone();
                Ok(true)
            }
            Ok(None) => Ok(false),
            Err(e) => Err(*e),
        }
    }
}

/// "EUR 10", as one mention in the money layer.
fn input() -> Money {
    let mut m = Money::new(ViewId::SOURCE);
    m.push(MoneyMention { span: Span::new(0, 6), negative: false, coefficient: "10", scale: 0, currency: cur("EUR") });
    m
}

fn scripted(answer: Result<Option<Decimal>, ScaleRangeError>) -> MoneyConverter<Arc<Scripted>> {
    MoneyConverter::new(Arc::new(Scripted { answer, calls: AtomicU32::new(0) }), cur("USD"), day("2026-07-14"))
}

fn dated(rates: Arc<Scripted>) -> DatedMoneyConverter<Arc<Scripted>> {
    DatedMoneyConverter::new(rates, cur("USD"))
}

fn calls(rates: &Scripted) -> u32 {
    rates.calls.load(Ordering::Relaxed)
}

/// `testEmptyRequiredValues`: no amounts, or no document date, and the
/// provider is not asked.
#[test]
fn empty_amounts_and_a_missing_date_do_not_query_the_provider() {
    let c = scripted(Ok(None));
    let mut out = Money::new(ViewId::SOURCE);
    c.convert(&Money::new(ViewId::SOURCE), &mut out).unwrap();
    assert_eq!(out.len(), 0);
    let rates = Arc::new(Scripted { answer: Ok(None), calls: AtomicU32::new(0) });
    let dated_on = DocumentDate { date: Some(day("2026-07-14")), mention: None };
    dated(rates.clone()).convert(&Money::new(ViewId::SOURCE), &dated_on, &mut out).unwrap();
    dated(rates.clone()).convert(&input(), &DocumentDate::default(), &mut out).unwrap();
    assert_eq!((calls(&rates), out.len()), (0, 0));
    let (empty,) = dated(rates.clone()).annotate((&input(), &DocumentDate::default())).unwrap();
    assert_eq!((calls(&rates), empty.len()), (0, 0));
}

/// The dated converter asks for the document's date, once per mention.
#[test]
fn the_dated_converter_uses_the_document_date() {
    let rates = Arc::new(Scripted { answer: Ok(Some(dec("12"))), calls: AtomicU32::new(0) });
    let date = DocumentDate { date: Some(day("2026-07-14")), mention: None };
    let (out,) = dated(rates.clone()).annotate((&input(), &date)).unwrap();
    assert_eq!((calls(&rates), out.len()), (1, 1));
    assert_eq!((amount(&out, 0), out.currency[0]), (dec("12"), cur("USD")));
}

/// `testFixedDateIgnoresDocumentDates`: the fixed-date converter reads no
/// document date, so one in the pipeline changes nothing.
#[test]
fn a_fixed_date_ignores_the_document_date() {
    let rates = Arc::new(Scripted { answer: Ok(None), calls: AtomicU32::new(0) });
    let analyzer = Analyzer::builder()
        .add(TemporalExtractor::new())
        .add(DocumentDateSelector)
        .add(MoneyExtractor::default())
        .step(MoneyConverter::new(rates.clone(), cur("USD"), day("2026-07-14")).bound())
        .build()
        .unwrap();
    let mut docs = vec![Document::new("EUR 10 on 2026-07-13")];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    assert_eq!(calls(&rates), 1);
    assert!(docs[0].get::<Temporals>().is_some());
    assert_eq!(docs[0].get_key(&CONVERTED_MONEY).unwrap().len(), 0);
}

#[test]
fn valid_results_keep_negative_and_zero_amounts_unrounded() {
    for value in ["0.0000", "-12345678901234567890.123456789"] {
        let c = scripted(Ok(Some(dec(value))));
        let mut out = Money::new(ViewId::SOURCE);
        c.convert(&input(), &mut out).unwrap();
        assert_eq!(out.check(), Ok(()));
        assert_eq!(out.len(), 1);
        assert_eq!(amount(&out, 0), dec(value));
        assert_eq!(out.spans.get(Ix::new(0)), Span::new(0, 6));
        assert_eq!(out.currency[0], cur("USD"));
    }
}

#[test]
fn provider_failures_propagate() {
    let c = scripted(Err(ScaleRangeError));
    let mut out = Money::new(ViewId::SOURCE);
    assert_eq!(c.convert(&input(), &mut out), Err(ScaleRangeError));
    let e = c.annotate((&input(),)).err().unwrap();
    assert!(e.to_string().contains("outside the supported decimal scale range"), "{e}");
}

#[test]
fn a_provider_is_called_once_per_mention() {
    let rates = Arc::new(Scripted { answer: Ok(None), calls: AtomicU32::new(0) });
    let c = MoneyConverter::new(rates.clone(), cur("USD"), day("2026-07-14"));
    let mut out = Money::new(ViewId::SOURCE);
    c.convert(&input(), &mut out).unwrap();
    assert_eq!((rates.calls.load(Ordering::Relaxed), out.len()), (1, 0));
}

// NumericExtractionExampleTest.

const TEXT: &str = "Berlin, 14 July 2026. The seller asked for \u{20AC}2.5 million, but the buyer paid $2,400,000 after a week of talks.";

fn span_of(fragment: &str) -> Span {
    assert_eq!(TEXT.matches(fragment).count(), 1, "{fragment:?} occurs once");
    let start = TEXT.find(fragment).unwrap();
    Span::new(start as u32, (start + fragment.len()) as u32)
}

/// The dateline is the only temporal mention and dates the document.
#[test]
fn the_dateline_dates_the_document() {
    let doc = analyze_dated(MoneyExtractor::default(), load("Date,USD,\n2026-07-13,1.2000,\n"), TEXT);
    let t = doc.get::<Temporals>().unwrap();
    assert_eq!(t.len(), 1);
    assert_eq!(t.spans.get(Ix::new(0)), span_of("14 July 2026"));
    assert_eq!(t.value(Ix::new(0)).to_string(), "2026-07-14");
    let date = doc.get::<DocumentDate>().unwrap();
    assert_eq!((date.date, date.mention), (Some(day("2026-07-14")), Some(Ix::new(0))));
}

#[test]
fn both_amounts_are_extracted_and_converted() {
    let doc = analyze_dated(MoneyExtractor::default(), load("Date,USD,\n2026-07-13,1.2000,\n"), TEXT);
    let money = doc.get::<Money>().unwrap();
    assert_eq!(money.len(), 2);
    assert_eq!(money.spans.get(Ix::new(0)), span_of("\u{20AC}2.5 million"));
    assert!(same(&dec("2500000"), &amount(money, 0)));
    assert_eq!(money.currency[0], cur("EUR"));
    assert_eq!(money.spans.get(Ix::new(1)), span_of("$2,400,000"));
    assert!(same(&dec("2400000"), &amount(money, 1)));
    assert_eq!(money.currency[1], cur("USD"));

    let converted = doc.get_key(&CONVERTED_MONEY).unwrap();
    assert_eq!(converted.len(), 2);
    // 2500000 EUR * 1.2000 USD per EUR = 3000000 USD
    assert_eq!(converted.spans.get(Ix::new(0)), span_of("\u{20AC}2.5 million"));
    assert!(same(&dec("3000000"), &amount(converted, 0)));
    assert_eq!(converted.spans.get(Ix::new(1)), span_of("$2,400,000"));
    assert!(same(&dec("2400000"), &amount(converted, 1)));
    assert!(converted.currency.iter().all(|&c| c == cur("USD")));
    // The dollar amount keeps its own digits and scale.
    assert_eq!(amount(converted, 1), amount(money, 1));
}

// Edge cases of the port.

#[test]
fn converted_scales_can_be_below_zero() {
    let rates = load("Date,USD\n2026-07-14,1.0856\n");
    let text = "USD 123456789012345678901234 and EUR 99999999999999999999";
    let doc = analyze(MoneyExtractor::default(), rates, "2026-07-14", text);
    let converted = doc.get_key(&CONVERTED_MONEY).unwrap();
    assert_eq!(converted.check(), Ok(()));
    // The dollar amount is kept; the euro amount is rounded to 16 digits.
    assert_eq!(amount(converted, 0).to_plain_string(), "123456789012345678901234");
    let euros = amount(converted, 1);
    assert_eq!((euros.digits.as_str(), euros.scale), ("1085600000000000", -5));
    assert_eq!(euros.to_plain_string(), "108560000000000000000");
}

#[test]
fn the_configuration_names_target_and_date() {
    let a = MoneyConverter::new(load(FIXTURE), cur("USD"), day("2026-07-10"));
    let b = MoneyConverter::new(load(FIXTURE), cur("USD"), day("2026-07-11"));
    let c = MoneyConverter::new(load(FIXTURE), cur("AUD"), day("2026-07-10"));
    assert_ne!(a.config_hash(), b.config_hash());
    assert_ne!(a.config_hash(), c.config_hash());
    assert_eq!((a.target(), a.as_of()), (cur("USD"), day("2026-07-10")));
}

/// `NumericManualExampleTest.testExtendedPackListing`: the German pipeline
/// (`NumericPacks.annotators(Locale.GERMANY)`: the default symbol table
/// and the European notation) with a document-dated conversion added.
#[test]
fn the_extended_pack_converts_as_of_the_dateline() {
    let text = "2026-07-14: paid EUR 10 for 3 kg yesterday.";
    let analyzer = Analyzer::builder()
        .add(turbo_numeric::QuantityExtractor::new(turbo_numeric::NumberNotation::LatinEu))
        .add(TemporalExtractor::new())
        .add(DocumentDateSelector)
        .add(MoneyExtractor::new(turbo_numeric::NumberNotation::LatinEu))
        .step(DatedMoneyConverter::new(load("Date,USD\n2026-07-14,1.25\n"), cur("USD")).bound())
        .build()
        .unwrap();
    let mut docs = vec![Document::new(text)];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    let doc = &docs[0];
    let (source, converted) = (doc.get::<Money>().unwrap(), doc.get_key(&CONVERTED_MONEY).unwrap());
    assert_eq!((source.len(), converted.len()), (1, 1));
    assert_eq!((source.currency[0], converted.currency[0]), (cur("EUR"), cur("USD")));
    assert!(same(&dec("10"), &amount(source, 0)));
    assert!(same(&dec("12.5"), &amount(converted, 0)));
    assert_eq!(source.spans.get(Ix::new(0)), Span::new(17, 23));
    assert_eq!(converted.spans.get(Ix::new(0)), Span::new(17, 23));
    assert_eq!(doc.get::<DocumentDate>().unwrap().date, Some(day("2026-07-14")));
    let t = doc.get::<Temporals>().unwrap();
    assert_eq!((t.len(), t.value(Ix::new(1)).to_string()), (2, "2026-07-13".to_string()));
    let q = doc.get::<turbo_numeric::Quantities>().unwrap();
    assert_eq!((q.len(), q.unit(Ix::new(0))), (1, "kg"));
}
