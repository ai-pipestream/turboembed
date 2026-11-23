//! Explicit presence on the search projection's numeric and temporal
//! values (#2517 F13): a true zero (1970-01-01, $0.00, 0 kg) stays a
//! present zero through the wire, a value no f64 or date can hold is
//! absent with its exact decimal kept, and the validator refuses half a
//! range and a non-finite float.

use prost::Message;
use turbo_analysis::{Analyzer, Bound, Document, ViewId};
use turbo_analysis_layers::NumberNotation;
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::{SearchDocumentProjection, SearchTemporalRange};
use turbo_kserve::results::disclosure::Disclosure;
use turbo_kserve::results::search_projection::{
    document_to_search_projection_bound, round_trip_search_projection, validate_search_projection,
};
use turbo_numeric::money::MoneyExtractor;
use turbo_numeric::{QuantityExtractor, TemporalExtractor};

const TEXT: &str = "Paid $0.00 then $1e400 for 0 kg and 1e400 kg on 1970-01-01.";

fn projected() -> SearchDocumentProjection {
    let analyzer = Analyzer::builder()
        .step(Bound::new(MoneyExtractor::new(NumberNotation::LatinUs)))
        .step(Bound::new(QuantityExtractor::new(NumberNotation::LatinUs)))
        .step(Bound::new(TemporalExtractor::new()))
        .build()
        .unwrap();
    let mut docs = vec![Document::new(TEXT.to_owned())];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    // A debug build asserts the projection validates; a non-finite f64
    // used to fail that assertion here.
    document_to_search_projection_bound(
        "doc-f13",
        &docs[0],
        ViewId::SOURCE,
        OffsetMode::Utf8,
        None,
        &[0u8; 32],
        &Disclosure::FULL,
    )
    .unwrap()
}

/// The projection after a protobuf encode and decode.
fn over_the_wire(p: &SearchDocumentProjection) -> SearchDocumentProjection {
    SearchDocumentProjection::decode(p.encode_to_vec().as_slice()).unwrap()
}

#[test]
fn true_zeros_stay_present_and_overflow_is_absent_on_the_wire() {
    let sent = projected();
    validate_search_projection(&sent).unwrap();
    let got = round_trip_search_projection(&over_the_wire(&sent)).unwrap();
    let nt = got.numeric_temporal.unwrap();

    let money: Vec<(&str, Option<f64>)> = nt.money.iter().map(|m| (m.plain_amount.as_str(), m.amount_f64)).collect();
    assert!(money.contains(&("0.00", Some(0.0))), "{money:?}");
    let huge = nt.money.iter().find(|m| m.plain_amount.len() > 300).unwrap_or_else(|| panic!("{money:?}"));
    assert_eq!(huge.amount_f64, None);
    assert!(huge.plain_amount.starts_with('1') && huge.plain_amount.bytes().skip(1).all(|b| b == b'0'));

    let qty: Vec<(&str, Option<f64>)> = nt.quantities.iter().map(|q| (q.plain_value.as_str(), q.value_f64)).collect();
    assert!(qty.contains(&("0", Some(0.0))), "{qty:?}");
    let huge = nt.quantities.iter().find(|q| q.plain_value.len() > 300).unwrap_or_else(|| panic!("{qty:?}"));
    assert_eq!(huge.value_f64, None);

    let day = nt.temporals.iter().find(|t| t.iso == "1970-01-01").unwrap_or_else(|| panic!("{:?}", nt.temporals));
    assert_eq!((day.start_epoch_seconds, day.end_epoch_seconds), (Some(0), Some(86_400)));
}

fn temporal(start: Option<i64>, end: Option<i64>) -> SearchTemporalRange {
    SearchTemporalRange {
        span_start: 0,
        span_end: 1,
        granularity: "DAY".into(),
        origin: "ABSOLUTE".into(),
        iso: "1970-01-01".into(),
        start_epoch_seconds: start,
        end_epoch_seconds: end,
    }
}

#[test]
fn the_validator_takes_both_bounds_or_neither() {
    let base = projected();
    for (start, end, ok) in [
        (None, None, true),
        (Some(0), Some(0), true),
        (Some(0), None, false),
        (None, Some(0), false),
        (Some(1), Some(0), false),
    ] {
        let mut p = base.clone();
        p.numeric_temporal.as_mut().unwrap().temporals = vec![temporal(start, end)];
        assert_eq!(validate_search_projection(&p).is_ok(), ok, "{start:?}..{end:?}");
    }
}

#[test]
fn the_validator_refuses_a_present_non_finite_float() {
    let base = projected();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut p = base.clone();
        p.numeric_temporal.as_mut().unwrap().money[0].amount_f64 = Some(bad);
        assert!(validate_search_projection(&p).is_err(), "money {bad}");
        let mut p = base.clone();
        p.numeric_temporal.as_mut().unwrap().quantities[0].value_f64 = Some(bad);
        assert!(validate_search_projection(&p).is_err(), "quantity {bad}");
    }
}
