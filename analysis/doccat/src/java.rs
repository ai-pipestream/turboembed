//! `Double.compare`, `Math.min` and `Math.max` as the JVM computes them:
//! the categorizer's scaling and LIBSVM's probability clamp call them, and
//! they differ from Rust's on NaN and on the sign of zero.

use std::cmp::Ordering;

/// `Double.compare`: numeric order, with -0.0 below 0.0 and every NaN equal
/// to every other and above positive infinity.
pub fn double_compare(a: f64, b: f64) -> Ordering {
    if a < b {
        return Ordering::Less;
    }
    if a > b {
        return Ordering::Greater;
    }
    // `doubleToLongBits` makes every NaN the one canonical NaN.
    let bits = |x: f64| if x.is_nan() { 0x7ff8_0000_0000_0000_i64 } else { x.to_bits() as i64 };
    bits(a).cmp(&bits(b))
}

/// `Double.compare(x, 0.0) != 0`: true for -0.0 and NaN as well.
#[inline]
pub fn is_not_zero(x: f64) -> bool {
    x.to_bits() != 0
}

/// `Math.min`: NaN if either is NaN, and -0.0 below 0.0.
#[inline]
pub fn min(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if a == 0.0 && b == 0.0 && b.to_bits() == (-0.0f64).to_bits() {
        return b;
    }
    if a <= b { a } else { b }
}

/// `Math.max`: NaN if either is NaN, and 0.0 above -0.0.
#[inline]
pub fn max(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if a == 0.0 && b == 0.0 && a.to_bits() == (-0.0f64).to_bits() {
        return b;
    }
    if a >= b { a } else { b }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_orders_signed_zeros_and_nan_as_java_does() {
        assert_eq!(double_compare(0.1, 0.2), Ordering::Less);
        assert_eq!(double_compare(-0.0, 0.0), Ordering::Less);
        assert_eq!(double_compare(0.0, 0.0), Ordering::Equal);
        assert_eq!(double_compare(f64::NAN, f64::INFINITY), Ordering::Greater);
        assert_eq!(double_compare(f64::NAN, -f64::NAN), Ordering::Equal);
        assert!(is_not_zero(-0.0));
        assert!(is_not_zero(f64::NAN));
        assert!(!is_not_zero(0.0));
    }

    #[test]
    fn min_and_max_keep_nan_and_the_sign_of_zero() {
        assert!(min(f64::NAN, 1.0).is_nan());
        assert!(min(1.0, f64::NAN).is_nan());
        assert!(max(f64::NAN, 1.0).is_nan());
        assert!(max(1.0, f64::NAN).is_nan());
        assert_eq!(min(0.0, -0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(min(-0.0, 0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(max(-0.0, 0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(max(0.0, -0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(min(2.0, 3.0), 2.0);
        assert_eq!(max(2.0, 3.0), 3.0);
    }
}
