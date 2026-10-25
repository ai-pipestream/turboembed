//! Helpers the money tests share.

#![allow(dead_code)]

use std::cmp::Ordering;

use turbo_analysis::{Layer, Span, ViewId};
use turbo_numeric::money::{Currency, Date, Decimal, EcbRates, Money, MoneyExtractor, cmp_values};
use turbo_numeric::{Quantities, QuantityExtractor};

/// The span of the whole of `text`, in bytes.
pub fn whole(text: &str) -> Span {
    Span::new(0, text.len() as u32)
}

/// One extracted money mention: its span, its amount and its currency.
#[derive(Debug)]
pub struct FoundMoney {
    pub span: Span,
    pub amount: Decimal,
    pub currency: Currency,
}

/// Every money mention `x` finds in `text`, after checking the layer.
pub fn extract_money(x: &MoneyExtractor, text: &str) -> Vec<FoundMoney> {
    let mut m = Money::new(ViewId::SOURCE);
    x.scan(text, &mut m);
    assert_eq!(m.check(), Ok(()), "{text:?}");
    m.spans
        .iter()
        .map(|(i, span)| {
            let a = m.amount(i);
            let amount = Decimal { negative: a.negative, digits: a.digits.into(), scale: a.scale as i32 };
            FoundMoney { span, amount, currency: m.currency[i.get()] }
        })
        .collect()
}

/// The single money mention `x` finds in `text`.
pub fn single_money_with(x: &MoneyExtractor, text: &str) -> FoundMoney {
    let mut all = extract_money(x, text);
    assert_eq!(all.len(), 1, "expected one mention in {text:?}: {all:?}");
    all.pop().unwrap()
}

/// The single money mention the default extractor finds in `text`.
pub fn single_money(text: &str) -> FoundMoney {
    single_money_with(&MoneyExtractor::default(), text)
}

/// Asserts `x` finds no money mention in `text`.
pub fn none_money(x: &MoneyExtractor, text: &str) {
    let found = extract_money(x, text);
    assert!(found.is_empty(), "{text:?}: {found:?}");
}

/// One extracted quantity mention: its span, its value and its unit.
#[derive(Debug)]
pub struct FoundQuantity {
    pub span: Span,
    pub value: Decimal,
    pub unit: String,
}

/// Every quantity mention `x` finds in `text`, after checking the layer.
pub fn extract_quantity(x: &QuantityExtractor, text: &str) -> Vec<FoundQuantity> {
    let mut q = Quantities::new(ViewId::SOURCE);
    x.scan(text, &mut q);
    assert_eq!(q.check(), Ok(()), "{text:?}");
    q.spans
        .iter()
        .map(|(i, span)| {
            let v = q.value(i);
            let value = Decimal { negative: v.negative, digits: v.digits.into(), scale: v.scale as i32 };
            FoundQuantity { span, value, unit: q.unit(i).into() }
        })
        .collect()
}

/// The single quantity mention `x` finds in `text`.
pub fn single_quantity_with(x: &QuantityExtractor, text: &str) -> FoundQuantity {
    let mut all = extract_quantity(x, text);
    assert_eq!(all.len(), 1, "expected one mention in {text:?}: {all:?}");
    all.pop().unwrap()
}

/// The single quantity mention the default extractor finds in `text`.
pub fn single_quantity(text: &str) -> FoundQuantity {
    single_quantity_with(&QuantityExtractor::default(), text)
}

/// Asserts `x` finds no quantity mention in `text`.
pub fn none_quantity_with(x: &QuantityExtractor, text: &str) {
    let found = extract_quantity(x, text);
    assert!(found.is_empty(), "{text:?}: {found:?}");
}

/// Asserts the default extractor finds no quantity mention in `text`.
pub fn none_quantity(text: &str) {
    none_quantity_with(&QuantityExtractor::default(), text);
}

/// Asserts the quantity mention's value, at any scale, and unit.
#[track_caller]
pub fn assert_quantity(m: &FoundQuantity, value: &str, unit: &str) {
    assert!(same(&dec(value), &m.value), "expected {value}, got {}", m.value);
    assert_eq!(m.unit, unit);
}

/// A currency code from a literal.
pub fn cur(code: &str) -> Currency {
    Currency::parse(code).unwrap_or_else(|| panic!("{code:?} is no currency code"))
}

/// A date from an ISO literal.
pub fn day(s: &str) -> Date {
    Date::parse(s).unwrap_or_else(|| panic!("{s:?} is no date"))
}

/// A decimal from a `BigDecimal` literal.
pub fn dec(s: &str) -> Decimal {
    Decimal::parse(s).unwrap_or_else(|e| panic!("{s:?}: {e}"))
}

/// A decimal from an unscaled value and a scale, as
/// `new BigDecimal(new BigInteger(coefficient), scale)`.
pub fn unscaled(coefficient: &str, scale: i32) -> Decimal {
    let (negative, digits) = match coefficient.strip_prefix('-') {
        Some(d) => (true, d),
        None => (false, coefficient),
    };
    let digits = digits.trim_start_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    Decimal { negative: negative && digits != "0", digits: digits.into(), scale }
}

/// Whether two decimals have the same value: `compareTo == 0`.
pub fn same(a: &Decimal, b: &Decimal) -> bool {
    cmp_values(a.as_ref(), b.as_ref()) == Ordering::Equal
}

/// Asserts that a rate is there and has the value `expected`, at any scale.
pub fn assert_rate(expected: &str, actual: Option<Decimal>) {
    let actual = actual.unwrap_or_else(|| panic!("no rate, expected {expected}"));
    assert!(same(&dec(expected), &actual), "expected {expected}, got {actual}");
}

/// Loads a history from text.
pub fn load(csv: &str) -> EcbRates {
    EcbRates::parse(csv).unwrap_or_else(|e| panic!("{csv:?}: {e}"))
}
