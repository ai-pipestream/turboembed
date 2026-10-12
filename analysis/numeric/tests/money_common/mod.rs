//! Helpers the money tests share.

#![allow(dead_code)]

use std::cmp::Ordering;

use turbo_numeric::money::{Currency, Date, Decimal, EcbRates, cmp_values};

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
