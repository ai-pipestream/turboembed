//! Division rounded to 16 significant digits, half to even: Java's
//! `BigDecimal.divide(divisor, MathContext.DECIMAL64)`, including the
//! scale it gives the result and the cases where it throws because a
//! scale leaves the `int` range, and the fallback `EcbFxRates` takes then.

use std::cmp::Ordering;

use super::ScaleRangeError;
use super::natural::Natural;

/// The digits `DECIMAL64` keeps.
const PRECISION: i64 = 16;
/// 10¹⁶, one digit more than a rounded quotient may have.
const TEN_POW_PRECISION: u64 = 10_000_000_000_000_000;

/// A quotient: at most 16 digits and a `BigDecimal` scale.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Quotient {
    pub coefficient: u64,
    pub scale: i32,
}

/// `x / y` rounded to 16 digits, half to even, where `x` and `y` are the
/// unscaled values of decimals at the scales `x_scale` and `y_scale` (each
/// within the `int` range), `x` is not zero and `y` is positive.
///
/// The preferred scale is `x_scale - y_scale`. The quotient is computed at
/// the scale that gives it exactly 16 digits; an exact quotient then
/// drops trailing zeros down to the preferred scale, and a quotient that
/// rounding carried to 17 digits drops one. As in the JDK, the division
/// fails when the preferred scale or that 16-digit scale is outside the
/// `int` range, even when the result would fit after the zeros are gone.
pub(crate) fn divide64(x: &Natural, x_scale: i64, y: &Natural, y_scale: i64) -> Result<Quotient, ScaleRangeError> {
    debug_assert!(!x.is_zero() && !y.is_zero(), "a dividend and a divisor that are not zero");
    let preferred = x_scale - y_scale;
    let x_digits = x.digit_count() as i64;
    let mut y_digits = y.digit_count() as i64;
    // With the dividend's digits no larger than the divisor's, a shift by
    // 16 places gives a quotient of 16 digits; otherwise of 17, so the
    // divisor counts one digit more.
    if cmp_normalized(x, x_digits, y, y_digits) == Ordering::Greater {
        y_digits -= 1;
    }
    let shift = PRECISION + y_digits - x_digits;
    let mut scale = int(preferred + shift)?;
    int(preferred)?;
    let (mut q, exact) = rounded_quotient(x, y, shift);
    if exact {
        while q >= 10 && i64::from(scale) > preferred && q.is_multiple_of(10) {
            q /= 10;
            scale -= 1;
        }
    }
    if q >= TEN_POW_PRECISION {
        // Only 10^16 itself, which loses one zero.
        scale = int(i64::from(scale) - 1)?;
        q /= 10;
    }
    Ok(Quotient { coefficient: q, scale })
}

/// `EcbFxRates.decimalQuotient`: divides the unscaled values with no
/// scale of their own, then applies `scale`, the combined scale of the
/// operands. A result scale above the `int` range drops the quotient's
/// trailing zeros; one below it takes trailing zeros when 16 digits leave
/// room for them. Fails when the scale still does not fit.
pub(crate) fn decimal_quotient(
    numerator: &Natural,
    denominator: &Natural,
    scale: i64,
) -> Result<Quotient, ScaleRangeError> {
    let mut q = if numerator.is_zero() {
        Quotient { coefficient: 0, scale: 0 }
    } else {
        divide64(numerator, 0, denominator, 0)?
    };
    let mut result = scale + i64::from(q.scale);
    if result > i64::from(i32::MAX) {
        q = strip_trailing_zeros(q);
        result = scale + i64::from(q.scale);
    }
    if result < i64::from(i32::MIN) {
        let padding = i64::from(i32::MIN) - result;
        let room = PRECISION - digit_count(q.coefficient);
        if padding <= room {
            let coefficient = q.coefficient * 10u64.pow(padding as u32);
            return Ok(Quotient { coefficient, scale: i32::MIN });
        }
    }
    Ok(Quotient { coefficient: q.coefficient, scale: int(result)? })
}

/// `BigDecimal.stripTrailingZeros`: zero becomes zero at the scale 0.
fn strip_trailing_zeros(mut q: Quotient) -> Quotient {
    if q.coefficient == 0 {
        return Quotient { coefficient: 0, scale: 0 };
    }
    while q.coefficient.is_multiple_of(10) {
        q.coefficient /= 10;
        q.scale -= 1;
    }
    q
}

/// `BigDecimal.precision()` of a coefficient; 1 for zero.
fn digit_count(v: u64) -> i64 {
    i64::from(v.checked_ilog10().unwrap_or(0) + 1)
}

/// A scale as an `int`, or the error the JDK throws for it.
fn int(scale: i64) -> Result<i32, ScaleRangeError> {
    i32::try_from(scale).map_err(|_| ScaleRangeError)
}

/// `compareMagnitudeNormalized`: compares `x` and `y` as if both had their
/// leading digit at the same place.
fn cmp_normalized(x: &Natural, x_digits: i64, y: &Natural, y_digits: i64) -> Ordering {
    match x_digits.cmp(&y_digits) {
        Ordering::Less => x.mul_pow10((y_digits - x_digits) as u64).cmp(y),
        Ordering::Greater => x.cmp(&y.mul_pow10((x_digits - y_digits) as u64)),
        Ordering::Equal => x.cmp(y),
    }
}

/// `x · 10^shift / y` rounded half to even to an integer, which has at
/// most 17 digits, and whether the division was exact.
fn rounded_quotient(x: &Natural, y: &Natural, shift: i64) -> (u64, bool) {
    // The operands as the JDK scales them: the dividend up, or the divisor
    // up when the shift is negative.
    let (num_shift, den_shift) = if shift >= 0 { (shift as u64, 0) } else { (0, shift.unsigned_abs()) };
    let num_digits = x.digit_count() + num_shift;
    let den_digits = y.digit_count() + den_shift;
    if num_digits <= 38 && den_digits <= 38 {
        // Both fit a u128, and so does twice the remainder.
        let num = x.to_u128().expect("38 digits fit") * 10u128.pow(num_shift as u32);
        let den = y.to_u128().expect("38 digits fit") * 10u128.pow(den_shift as u32);
        let (q, r) = ((num / den) as u64, num % den);
        return (round_half_even(q, (2 * r).cmp(&den)), r == 0);
    }
    let num = x.mul_pow10(num_shift);
    let den = y.mul_pow10(den_shift);
    let (q, mut r) = num.div_rem_small_quotient(&den);
    let exact = r.is_zero();
    r.mul_small(2);
    (round_half_even(q, r.cmp(&den)), exact)
}

/// Rounds a truncated quotient up when twice the remainder is more than
/// the divisor, or equal to it and the quotient is odd.
fn round_half_even(q: u64, twice_remainder: Ordering) -> u64 {
    match twice_remainder {
        Ordering::Greater => q + 1,
        Ordering::Equal if q % 2 == 1 => q + 1,
        _ => q,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(s: &str) -> Natural {
        Natural::from_digits(s)
    }

    fn div(x: &str, xs: i64, y: &str, ys: i64) -> Result<(u64, i32), ScaleRangeError> {
        divide64(&n(x), xs, &n(y), ys).map(|q| (q.coefficient, q.scale))
    }

    #[test]
    fn rounds_to_sixteen_digits_half_even() {
        // 1/3 and 2/3.
        assert_eq!(div("1", 0, "3", 0), Ok((3333333333333333, 16)));
        assert_eq!(div("2", 0, "3", 0), Ok((6666666666666667, 16)));
        // 160.00 / 1.2000: the preferred scale is -2.
        assert_eq!(div("16000", 2, "12000", 4), Ok((1333333333333333, 13)));
        // Ties go to the even neighbour.
        assert_eq!(div("12345678901234565", 0, "1", 0), Ok((1234567890123456, -1)));
        assert_eq!(div("12345678901234575", 0, "1", 0), Ok((1234567890123458, -1)));
    }

    #[test]
    fn exact_quotients_keep_the_preferred_scale() {
        // 12.000 / 1.2 is 10 at the preferred scale 2: 10.00.
        assert_eq!(div("12000", 3, "12", 1), Ok((1000, 2)));
        assert_eq!(div("1", 0, "4", 0), Ok((25, 2)));
        assert_eq!(div("100", 0, "1", 2), Ok((100, -2)));
        // A power of ten with the same leading digits: 10^16 at first.
        assert_eq!(div("1", 0, "1", 0), Ok((1, 0)));
        assert_eq!(div("1", 20, "1", 0), Ok((1, 20)));
        // 10^20 needs a scale of -5 to fit 16 digits.
        assert_eq!(div("100000000000000000000", 0, "1", 0), Ok((1000000000000000, -5)));
    }

    #[test]
    fn scales_outside_the_int_range_fail() {
        let max = i64::from(i32::MAX);
        let min = i64::from(i32::MIN);
        assert_eq!(div("1", max, "1", -1), Err(ScaleRangeError));
        assert_eq!(div("1", min, "1", 1), Err(ScaleRangeError));
        // The 16-digit scale is out of range though the exact result is 1.
        assert_eq!(div("1", max, "1", 0), Err(ScaleRangeError));
        assert_eq!(div("1", min, "1", 0), Ok((1, i32::MIN)));
        assert_eq!(div("3", max - 20, "1", 0), Ok((3, i32::MAX - 20)));
    }

    #[test]
    fn large_operands_take_the_long_division() {
        let x = "123456789012345678901234567890123456789012345678901234567890";
        assert_eq!(div(x, 0, "7", 0), Ok((1763668414462081, -43)));
        assert_eq!(div("7", 0, x, 0), Ok((5670000051030000, 74)));
    }

    #[test]
    fn the_fallback_applies_the_combined_scale() {
        let q = |a: &str, b: &str, s: i64| decimal_quotient(&n(a), &n(b), s).map(|q| (q.coefficient, q.scale));
        assert_eq!(q("1", "1", i64::from(i32::MAX)), Ok((1, i32::MAX)));
        // 1E+1 at the largest scale: the zero goes, the scale stays.
        assert_eq!(q("10", "1", i64::from(i32::MAX) + 1), Ok((1, i32::MAX)));
        assert_eq!(q("1", "1", i64::from(i32::MIN) - 15), Ok((1_000_000_000_000_000, i32::MIN)));
        assert_eq!(q("1", "1", i64::from(i32::MIN) - 16), Err(ScaleRangeError));
        assert_eq!(q("1", "10", i64::from(i32::MAX)), Err(ScaleRangeError));
        assert_eq!(q("0", "7", 5), Ok((0, 5)));
    }
}
