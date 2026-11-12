//! The ECB rate table and its arithmetic: the cases of OpenNLP's
//! `EcbFxRatesTest`, `EcbFxRatesArithmeticTest`, `EcbFxRatesScaleTest`,
//! `EcbFxRatesPaddingTest`, `EcbFxRatesValidationTest` and
//! `FxRatesContractTest`, then edge cases of the port.
//!
//! The Java checks that a null or blank code or a null date is refused;
//! the Rust types hold neither, so those cases have no counterpart. The
//! Java's thread-safety case shares one table between threads, as here.

mod money_common;

use std::cell::Cell;

use money_common::{assert_rate, cur, day, dec, load, same, unscaled};
use turbo_numeric::money::{Currency, Date, Decimal, EcbError, EcbRates, FxRates, MAX_STALENESS_DAYS, ScaleRangeError};

const FIXTURE: &str = "Date,USD,JPY,GBP,\n2026-07-10,1.2000,160.00,0.8500,\n2026-07-09,1.1000,155.00,N/A,\n2026-06-01,1.0000,150.00,0.9000,\n\n";

fn rates() -> EcbRates {
    load(FIXTURE)
}

fn rate(r: &EcbRates, from: &str, to: &str, on: &str) -> Option<Decimal> {
    r.rate(cur(from), cur(to), day(on)).unwrap()
}

/// Converts `amount` and returns the result, `None` when there is no rate.
fn convert(
    r: &impl FxRates<Error = ScaleRangeError>,
    amount: &Decimal,
    from: &str,
    to: &str,
    on: Date,
) -> Option<Decimal> {
    let mut out = Decimal::default();
    r.convert(amount.as_ref(), cur(from), cur(to), on, &mut out).unwrap().then_some(out)
}

fn try_convert(
    r: &EcbRates,
    amount: &Decimal,
    from: &str,
    to: &str,
    on: Date,
) -> Result<Option<Decimal>, ScaleRangeError> {
    let mut out = Decimal::default();
    Ok(r.convert(amount.as_ref(), cur(from), cur(to), on, &mut out)?.then_some(out))
}

// EcbFxRatesTest

#[test]
fn euro_base_rate_on_a_reference_day() {
    assert_rate("1.2000", rate(&rates(), "EUR", "USD", "2026-07-10"));
    assert_rate("0.8333333333333333", rate(&rates(), "USD", "EUR", "2026-07-10"));
}

#[test]
fn cross_rate_goes_through_the_euro_base() {
    assert_rate("133.3333333333333", rate(&rates(), "USD", "JPY", "2026-07-10"));
}

#[test]
fn weekend_falls_back_to_the_previous_reference_day() {
    assert_rate("1.2000", rate(&rates(), "EUR", "USD", "2026-07-12"));
}

#[test]
fn stale_rates_are_absent_beyond_the_limit() {
    assert!(rate(&rates(), "EUR", "USD", "2026-07-01").is_none());
    assert!(rate(&rates(), "EUR", "USD", "2026-05-01").is_none());
}

#[test]
fn staleness_limit_is_inclusive() {
    assert_rate("1.2000", rate(&rates(), "EUR", "USD", "2026-07-17"));
    assert!(rate(&rates(), "EUR", "USD", "2026-07-18").is_none());
}

#[test]
fn not_available_cells_and_unknown_currencies_are_absent() {
    assert!(rate(&rates(), "EUR", "GBP", "2026-07-09").is_none());
    assert!(rate(&rates(), "EUR", "CHF", "2026-07-10").is_none());
}

#[test]
fn identity_rate() {
    assert_rate("1", rate(&rates(), "EUR", "EUR", "2026-07-10"));
    assert_rate("1", rate(&rates(), "USD", "USD", "2026-07-10"));
}

#[test]
fn convert_keeps_the_amount() {
    let converted = convert(&rates(), &dec("10"), "EUR", "USD", day("2026-07-10")).unwrap();
    assert!(same(&dec("12"), &converted));
}

#[test]
fn malformed_content_fails_loud() {
    assert!(matches!(EcbRates::parse("no header here"), Err(EcbError::NotAHistory { .. })));
    assert!(matches!(EcbRates::parse("Date,USD,\nyesterday,1.1,\n"), Err(EcbError::InvalidDate { .. })));
    assert!(matches!(EcbRates::parse("Date,USD,\n"), Err(EcbError::NoRows)));
    assert!(matches!(EcbRates::parse(""), Err(EcbError::NotAHistory { header: None })));
}

#[test]
fn malformed_rate_cell_fails_loud_with_row_context() {
    let e = EcbRates::parse("Date,USD,JPY,\n2026-07-10,1.08x,160.00,\n").unwrap_err();
    assert_eq!(e.to_string(), "not a reference history rate for USD: 1.08x in row: 2026-07-10,1.08x,160.00,");
}

// EcbFxRatesArithmeticTest

const DATE: &str = "2026-07-14";

#[test]
fn latest_date_lookup() {
    for age in [0, 1, 6, 7, 8, 99] {
        let date = Date::MAX.plus_days(-age).unwrap();
        let r = load(&format!("Date,USD\n{date},1.25\n"));
        let found = r.rate(cur("EUR"), cur("USD"), Date::MAX).unwrap();
        let converted =
            convert(&r, &Decimal { negative: false, digits: "10".into(), scale: 0 }, "EUR", "USD", Date::MAX);
        if age <= MAX_STALENESS_DAYS {
            assert_rate("1.25", found);
            assert!(same(&dec("12.5"), &converted.unwrap()), "age {age}");
        } else {
            assert!(found.is_none() && converted.is_none(), "age {age}");
        }
    }
}

#[test]
fn earliest_date_lookup() {
    let r = load(&format!("Date,USD\n{},1.25\n", Date::MIN));
    let at = |d: Date| r.rate(cur("EUR"), cur("USD"), d).unwrap();
    assert!(at(Date::MIN).is_some());
    assert!(at(Date::MIN.plus_days(7).unwrap()).is_some());
    assert!(at(Date::MIN.plus_days(8).unwrap()).is_none());
    let future = load(&format!("Date,USD\n{},1.25\n", Date::MIN.plus_days(1).unwrap()));
    assert!(future.rate(cur("EUR"), cur("USD"), Date::MIN).unwrap().is_none());
}

#[test]
fn identity_conversion_preserves_amount() {
    let r = load("Date,USD\n2026-07-14,1.2345\n");
    for amount in [
        "12345678901234567890.123456789",
        "-12345678901234567890.123456789",
        "0.00000000000000000000000001",
        "0.000000",
        "1E-2147483647",
    ] {
        let source = dec(amount);
        assert_eq!(convert(&r, &source, "USD", "USD", day(DATE)), Some(source.clone()), "{amount}");
    }
}

#[test]
fn identity_requires_usable_quote() {
    let r = load("Date,USD,JPY\n2026-07-14,1.25,N/A\n");
    let one = dec("1");
    assert!(convert(&r, &one, "USD", "USD", day("2026-07-13")).is_none());
    assert!(convert(&r, &one, "USD", "USD", day("2026-07-22")).is_none());
    assert!(convert(&r, &one, "JPY", "JPY", day(DATE)).is_none());
    assert!(rate(&r, "JPY", "JPY", DATE).is_none());
}

#[test]
fn newest_available_date_is_selected() {
    let r = load("Date,USD,JPY\n2026-07-14,1.25,N/A\n2026-07-10,1.2,150\n2026-07-17,1.3,160\n");
    assert_rate("1.25", rate(&r, "EUR", "USD", "2026-07-15"));
    assert!(rate(&r, "EUR", "JPY", "2026-07-15").is_none());
}

/// `java.util.Random`, for the generated cases, so they are the Java's.
struct JavaRandom(u64);

impl JavaRandom {
    fn new(seed: i64) -> JavaRandom {
        JavaRandom((seed as u64 ^ 0x5_DEEC_E66D) & ((1 << 48) - 1))
    }

    fn next(&mut self, bits: u32) -> i32 {
        self.0 = (self.0.wrapping_mul(0x5_DEEC_E66D).wrapping_add(0xB)) & ((1 << 48) - 1);
        (self.0 >> (48 - bits)) as i64 as i32
    }

    fn next_long(&mut self) -> i64 {
        (i64::from(self.next(32)) << 32).wrapping_add(i64::from(self.next(32)))
    }

    /// `Random.nextInt(int bound)`.
    fn next_int(&mut self, bound: i32) -> i32 {
        let mut r = self.next(31);
        let m = bound - 1;
        if bound & m == 0 {
            return ((i64::from(bound) * i64::from(r)) >> 31) as i32;
        }
        let mut u = r;
        loop {
            r = u % bound;
            if u.wrapping_sub(r).wrapping_add(m) >= 0 {
                return r;
            }
            u = self.next(31);
        }
    }

    /// `RandomSupport.boundedNextLong(rng, origin, bound)`.
    fn next_long_in(&mut self, origin: i64, bound: i64) -> i64 {
        let mut r = self.next_long();
        let n = bound.wrapping_sub(origin);
        let m = n.wrapping_sub(1);
        if n & m == 0 {
            return (r & m).wrapping_add(origin);
        }
        if n > 0 {
            let mut u = ((r as u64) >> 1) as i64;
            loop {
                r = u % n;
                if u.wrapping_add(m).wrapping_sub(r) >= 0 {
                    return r + origin;
                }
                u = ((self.next_long() as u64) >> 1) as i64;
            }
        }
        while r < origin || r >= bound {
            r = self.next_long();
        }
        r
    }
}

#[test]
fn the_random_port_draws_what_the_jvm_draws() {
    // new Random(20260905L): nextLong(1, 1_000_000) twice, nextLong over
    // ±10^11, nextInt(6), nextLong over ±10^18, as printed by a JVM.
    let mut r = JavaRandom::new(20260905);
    assert_eq!(r.next_long_in(1, 1_000_000), 329747);
    assert_eq!(r.next_long_in(1, 1_000_000), 51822);
    assert_eq!(r.next_long_in(-100_000_000_000, 100_000_000_000), -70782120038);
    assert_eq!(r.next_int(6), 0);
    assert_eq!(r.next_long_in(-1_000_000_000_000_000, 1_000_000_000_000_000), -298132328991886);
}

/// `value` rounded to 16 significant digits, half to even, for a positive
/// `num / den` with no carry past 10^38: an independent reference.
fn round16(num: u128, den: u128) -> (u128, i32) {
    // The shift that puts the quotient in [10^15, 10^16).
    let mut shift: i32 = 0;
    let (mut n, mut d) = (num, den);
    while n / d >= 10u128.pow(16) {
        d *= 10;
        shift -= 1;
    }
    while n / d < 10u128.pow(15) {
        n *= 10;
        shift += 1;
    }
    let (q, r) = (n / d, n % d);
    let q = match (2 * r).cmp(&d) {
        std::cmp::Ordering::Greater => q + 1,
        std::cmp::Ordering::Equal if q % 2 == 1 => q + 1,
        _ => q,
    };
    (q, shift)
}

fn reference(num: u128, den: u128, negative: bool, scale: i32) -> Decimal {
    let (q, shift) = round16(num, den);
    let sign = if negative { "-" } else { "" };
    unscaled(&format!("{sign}{q}"), scale + shift)
}

#[test]
fn generated_cross_rates() {
    let mut random = JavaRandom::new(20260905);
    for i in 0..500 {
        let usd = random.next_long_in(1, 1_000_000);
        let jpy = random.next_long_in(1, 1_000_000);
        let amount = random.next_long_in(-100_000_000_000, 100_000_000_000);
        let (usd_d, jpy_d) = (unscaled(&usd.to_string(), 3), unscaled(&jpy.to_string(), 2));
        let r = load(&format!("Date,USD,JPY\n2026-07-14,{usd_d},{jpy_d}\n"));
        // JPY per USD is (jpy / 100) / (usd / 1000).
        let expected_rate = reference(jpy as u128 * 10, usd as u128, false, 0);
        assert!(same(&expected_rate, &rate(&r, "USD", "JPY", DATE).unwrap()), "rate case {i}");
        let source = unscaled(&amount.to_string(), 6);
        let expected = if amount == 0 {
            dec("0")
        } else {
            reference(u128::from(amount.unsigned_abs()) * jpy as u128 * 10, usd as u128, amount < 0, 6)
        };
        let converted = convert(&r, &source, "USD", "JPY", day(DATE)).unwrap();
        assert!(same(&expected, &converted), "amount case {i}: {expected} {converted}");
    }
}

#[test]
fn conversion_rounding() {
    let r = load("Date,USD\n2026-07-14,1\n");
    for (amount, expected) in [
        ("12345678901234565", "12345678901234560"),
        ("12345678901234575", "12345678901234580"),
        ("-12345678901234565", "-12345678901234560"),
        ("-12345678901234575", "-12345678901234580"),
    ] {
        assert!(same(&dec(expected), &convert(&r, &dec(amount), "EUR", "USD", day(DATE)).unwrap()), "{amount}");
    }
}

#[test]
fn conversion_rounds_once() {
    let r = load("Date,USD,JPY\n2026-07-14,3,1\n");
    assert_rate("0.3333333333333333", rate(&r, "USD", "JPY", DATE));
    assert!(same(&dec("1"), &convert(&r, &dec("3"), "USD", "JPY", day(DATE)).unwrap()));
}

#[test]
fn concurrent_lookups() {
    let r = load("Date,USD,JPY\n2026-07-14,1.25,150\n");
    std::thread::scope(|s| {
        for t in 0..4 {
            let r = &r;
            s.spawn(move || {
                for value in (1..=200).filter(|v| v % 4 == t) {
                    let date = day(DATE).plus_days(value % 9).unwrap();
                    let result = convert(r, &dec(&value.to_string()), "USD", "JPY", date);
                    if value % 9 == 8 {
                        assert!(result.is_none());
                        assert!(r.rate(cur("USD"), cur("JPY"), date).unwrap().is_none());
                    } else {
                        assert_rate("120", r.rate(cur("USD"), cur("JPY"), date).unwrap());
                        assert!(same(&dec(&(value * 120).to_string()), &result.unwrap()));
                    }
                }
            });
        }
    });
}

// EcbFxRatesScaleTest

fn table(from: &str, to: &str) -> EcbRates {
    load(&format!("Date,USD,JPY\n2026-07-14,{from},{to}\n"))
}

#[test]
fn representable_conversion() {
    let (max, min) = (i32::MAX, i32::MIN);
    let q = "1.2345";
    let cases: [(&str, &str, &str, i32, &str, i32); 17] = [
        (q, q, "1", max, "1", max),
        (q, q, "-1", max, "-1", max),
        (q, q, "1", max - 7, "1", max - 7),
        (q, q, "1", min, "1", min),
        ("3", "1", "3", max, "1", max),
        ("1", "10", "1", min, "10", min),
        ("1", "1E+1", "1", min, "10", min),
        ("1", "1E+15", "1", min, "1000000000000000", min),
        ("1E+1", "1", "10", max, "1", max),
        ("1E-2147483647", "1", "1", max, "1", 0),
        ("1", "1E-2147483647", "1", min, "1", -1),
        ("1E-2147483647", "2E-2147483647", "1", max, "2", max),
        ("1E+2147483647", "2E+2147483647", "1", min, "2", min),
        (q, q, "12345678901234565", max, "1234567890123456", max - 1),
        (q, q, "12345678901234575", max, "1234567890123458", max - 1),
        (q, q, "-12345678901234565", max, "-1234567890123456", max - 1),
        (q, q, "-12345678901234575", max, "-1234567890123458", max - 1),
    ];
    for (from, to, coefficient, scale, want, want_scale) in cases {
        let r = table(from, to);
        let source = unscaled(coefficient, scale);
        let converted = try_convert(&r, &source, "USD", "JPY", day(DATE)).unwrap().unwrap();
        let case = format!("{from} {to} {coefficient}E{scale}");
        assert!(same(&unscaled(want, want_scale), &converted), "{case}: {converted:?}");
        assert!(converted.precision() <= 16, "{case}");
    }
}

#[test]
fn representable_rate() {
    for (from, to, expected) in [
        ("1E-2147483647", "2E-2147483647", "2"),
        ("1E+2147483647", "2E+2147483647", "2"),
        ("1", "1E-2147483647", "1E-2147483647"),
        ("1E-2147483647", "1", "1E+2147483647"),
        ("3E-2147483647", "1E-2147483647", "0.3333333333333333"),
    ] {
        assert_rate(expected, table(from, to).rate(cur("USD"), cur("JPY"), day(DATE)).unwrap());
    }
}

#[test]
fn zero_conversion() {
    let r = table("1E+2147483647", "1E-2147483647");
    for scale in [i32::MIN, 0, i32::MAX] {
        let converted = try_convert(&r, &Decimal::zero(scale), "USD", "JPY", day(DATE)).unwrap().unwrap();
        assert!(converted.is_zero(), "{scale}");
    }
}

#[test]
fn unrepresentable_rate() {
    for (from, to) in [("1E+2147483647", "1E-2147483647"), ("1E-2147483647", "1E+2147483647")] {
        let r = table(from, to);
        assert_eq!(r.rate(cur("USD"), cur("JPY"), day(DATE)), Err(ScaleRangeError));
        assert_eq!(try_convert(&r, &dec("1"), "USD", "JPY", day(DATE)), Err(ScaleRangeError));
    }
}

#[test]
fn unrepresentable_conversion() {
    for (scale, quote) in [(i32::MIN, "1E+16"), (i32::MAX, "0.1")] {
        let r = table("1", quote);
        assert_eq!(try_convert(&r, &unscaled("1", scale), "USD", "JPY", day(DATE)), Err(ScaleRangeError));
    }
}

#[test]
fn generated_scale_equivalence() {
    let shifts = [i32::MIN + 64, 0, i32::MAX - 64];
    for amount_scale in shifts {
        for quote_scale in shifts {
            let mut random = JavaRandom::new(20260905);
            for i in 0..100 {
                let from_c = random.next_long_in(1, 1_000_000);
                let from_s = random.next_int(6);
                let to_c = random.next_long_in(1, 1_000_000);
                let to_s = random.next_int(6);
                let amount = random.next_long_in(-1_000_000_000_000_000, 1_000_000_000_000_000);
                // The unshifted values, divided as BigDecimal.divide does.
                let plain = table(
                    &unscaled(&from_c.to_string(), from_s).to_string(),
                    &unscaled(&to_c.to_string(), to_s).to_string(),
                );
                let expected_rate = plain.rate(cur("USD"), cur("JPY"), day(DATE)).unwrap().unwrap();
                let expected_amount =
                    convert(&plain, &unscaled(&amount.to_string(), 0), "USD", "JPY", day(DATE)).unwrap();
                let shifted = table(
                    &decimal_string(&unscaled(&from_c.to_string(), from_s + quote_scale)),
                    &decimal_string(&unscaled(&to_c.to_string(), to_s + quote_scale)),
                );
                let got_rate = shifted.rate(cur("USD"), cur("JPY"), day(DATE)).unwrap().unwrap();
                assert_eq!(got_rate, expected_rate, "rate case {i}");
                let source = unscaled(&amount.to_string(), amount_scale);
                let converted = try_convert(&shifted, &source, "USD", "JPY", day(DATE)).unwrap().unwrap();
                let want = Decimal { scale: expected_amount.scale + amount_scale, ..expected_amount };
                assert_eq!(converted, want, "amount case {i}");
            }
        }
    }
}

/// A decimal as `BigDecimal.toString` would write it for a scale far
/// from zero: the digits with an exponent, which the loader reads back.
fn decimal_string(d: &Decimal) -> String {
    format!("{}{}E{}", if d.negative { "-" } else { "" }, d.digits, -i64::from(d.scale))
}

#[path = "money_rates/validation.rs"]
mod validation;
