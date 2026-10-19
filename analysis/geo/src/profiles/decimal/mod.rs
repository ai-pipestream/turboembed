//! The decimal arithmetic `PlaceProfiles` standardizes with: Java's
//! `BigDecimal` rounded to `MathContext.DECIMAL128`, 34 significant digits
//! and round half even.
//!
//! OpenNLP centers and squares the exact values of the doubles, which needs
//! integers of a few thousand bits, and rounds only the quotients and the
//! square root. [`Natural`] is the unsigned integer for the exact part and
//! [`Decimal`] a value rounded to 34 digits. Only values matter here, not
//! `BigDecimal` scales: every later step (division, comparison, `signum`,

mod natural;

pub(crate) use natural::{Integer, Natural};

use std::cmp::Ordering;
use std::fmt::Write;

const MIN_COEFFICIENT: u128 = 10u128.pow(33);
/// 10^34, one more than the largest coefficient.
const COEFFICIENT_LIMIT: u128 = 10u128.pow(34);

/// A value rounded to 34 significant digits: `coefficient * 10^exponent`,
/// with the coefficient zero or in `[10^33, 10^34)`, so that two nonzero
/// values compare by exponent first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Decimal {
    pub(crate) negative: bool,
    pub(crate) coefficient: u128,
    pub(crate) exponent: i64,
}

impl Decimal {
    pub(crate) const ZERO: Decimal = Decimal { negative: false, coefficient: 0, exponent: 0 };

    pub(crate) fn is_zero(self) -> bool {
        self.coefficient == 0
    }

    pub(crate) fn abs(self) -> Decimal {
        Decimal { negative: false, ..self }
    }

    /// The order of the absolute values.
    pub(crate) fn cmp_abs(self, other: Decimal) -> Ordering {
        match (self.is_zero(), other.is_zero()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => self.exponent.cmp(&other.exponent).then_with(|| self.coefficient.cmp(&other.coefficient)),
        }
    }

    /// `numerator / denominator * 10^power`, both positive, rounded to 34
    /// digits half even: `BigDecimal.divide(divisor, DECIMAL128)` on the
    /// exact quotient.
    pub(crate) fn quotient(negative: bool, numerator: &Natural, denominator: &Natural, power: i64) -> Decimal {
        if numerator.is_zero() {
            return Decimal::ZERO;
        }
        // log10 of the quotient is within one of this estimate.
        let log2 = numerator.bits() as i64 - denominator.bits() as i64;
        let mut exponent = (log2 as f64 * std::f64::consts::LOG10_2).floor() as i64 + power - 33;
        loop {
            let shift = power - exponent;
            let (n, d) = if shift >= 0 {
                (numerator.mul_pow10(shift as u64), denominator.clone())
            } else {
                (numerator.clone(), denominator.mul_pow10(shift.unsigned_abs()))
            };
            let (q, r) = n.div_rem(&d);
            let q = q.to_u128().unwrap_or(u128::MAX);
            if q >= COEFFICIENT_LIMIT {
                exponent += 1;
                continue;
            }
            if q < MIN_COEFFICIENT {
                exponent -= 1;
                continue;
            }
            let up = match r.shl(1).cmp(&d) {
                Ordering::Less => false,
                Ordering::Greater => true,
                Ordering::Equal => q % 2 == 1,
            };
            return Decimal::normalized(negative, q + u128::from(up), exponent);
        }
    }

    /// The coefficient brought back into `[10^33, 10^34)` after rounding up.
    fn normalized(negative: bool, coefficient: u128, exponent: i64) -> Decimal {
        if coefficient == COEFFICIENT_LIMIT {
            Decimal { negative, coefficient: MIN_COEFFICIENT, exponent: exponent + 1 }
        } else {
            Decimal { negative, coefficient, exponent }
        }
    }

    /// The square root of a positive value, rounded to 34 digits half even:
    /// `BigDecimal.sqrt(DECIMAL128)`, which JDK 21 rounds correctly.
    pub(crate) fn sqrt(self) -> Decimal {
        debug_assert!(!self.negative && !self.is_zero());
        // sqrt(c * 10^t) = sqrt(c * 10^u) * 10^((t - u) / 2), with u making
        // t - u even and c * 10^u at least 10^69, so the root has 35 or 36
        // digits before rounding.
        let u: i64 = if self.exponent.rem_euclid(2) == 0 { 36 } else { 37 };
        let n = Natural::from_u128(self.coefficient).mul_pow10(u as u64);
        // The root is below 10^35.5 < 2^118.
        let mut root = 0u128;
        for bit in (0..=118).rev() {
            let candidate = root | (1u128 << bit);
            let c = Natural::from_u128(candidate);
            if c.mul(&c).cmp(&n) != Ordering::Greater {
                root = candidate;
            }
        }
        let exact = {
            let r = Natural::from_u128(root);
            r.mul(&r) == n
        };
        let digits = root.ilog10() + 1;
        let dropped = digits - 34;
        let scale = 10u128.pow(dropped);
        let (q, rem) = (root / scale, root % scale);
        let half = scale / 2;
        // Below the root lies a fraction when the root is not exact, so a
        // remainder of exactly one half is then more than half.
        let up = match rem.cmp(&half) {
            Ordering::Less => false,
            Ordering::Greater => true,
            Ordering::Equal => !exact || q % 2 == 1,
        };
        Decimal::normalized(false, q + u128::from(up), (self.exponent - u) / 2 + i64::from(dropped))
    }

    /// `self / divisor` rounded to 34 digits; `divisor` must not be zero.
    pub(crate) fn div(self, divisor: Decimal) -> Decimal {
        Decimal::quotient(
            self.negative != divisor.negative,
            &Natural::from_u128(self.coefficient),
            &Natural::from_u128(divisor.coefficient),
            self.exponent - divisor.exponent,
        )
    }

    /// `BigDecimal.doubleValue()`, correctly rounded; `buffer` is scratch
    /// space for the decimal text Rust's parser reads.
    pub(crate) fn to_f64(self, buffer: &mut String) -> f64 {
        if self.is_zero() {
            return 0.0;
        }
        buffer.clear();
        if self.negative {
            buffer.push('-');
        }
        write!(buffer, "{}e{}", self.coefficient, self.exponent).expect("writing to a String");
        buffer.parse().expect("a decimal in exponent form")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(v: u128) -> Natural {
        Natural::from_u128(v)
    }

    fn d(coefficient: u128, exponent: i64) -> Decimal {
        Decimal { negative: false, coefficient, exponent }
    }

    #[test]
    fn long_division_matches_u128_division() {
        let values = [
            1u128,
            7,
            u32::MAX as u128,
            1 << 32,
            (1 << 64) + 5,
            u64::MAX as u128 * 3,
            12_345_678_901_234_567_890_123_456_789,
            u128::MAX,
            u128::MAX / 3,
            (1 << 96) - 1,
        ];
        for &a in &values {
            for &b in &values {
                let (q, r) = n(a).div_rem(&n(b));
                assert_eq!(q.to_u128(), Some(a / b), "{a} / {b}");
                assert_eq!(r.to_u128(), Some(a % b), "{a} % {b}");
            }
        }
    }

    #[test]
    fn long_division_of_wide_numbers_recombines() {
        let a = n(u128::MAX).mul(&n(u128::MAX - 12345)).mul(&n(987_654_321_987_654_321)).add(&n(42));
        let b = n((1 << 70) + 3).mul(&n(u64::MAX as u128));
        let (q, r) = a.div_rem(&b);
        assert_eq!(q.mul(&b).add(&r), a);
        assert_eq!(r.cmp(&b), Ordering::Less);
        // A divisor whose top limb has its high bit set needs no shift.
        let b = n(u128::MAX - 1);
        let (q, r) = a.div_rem(&b);
        assert_eq!(q.mul(&b).add(&r), a);
        assert_eq!(r.cmp(&b), Ordering::Less);
    }

    #[test]
    fn quotients_round_half_even_to_34_digits() {
        // 1/3 = 0.3333...: 34 threes.
        let third = Decimal::quotient(false, &n(1), &n(3), 0);
        assert_eq!(third, d(3_333_333_333_333_333_333_333_333_333_333_333, -34));
        // 2/3 rounds up.
        let two_thirds = Decimal::quotient(false, &n(2), &n(3), 0);
        assert_eq!(two_thirds, d(6_666_666_666_666_666_666_666_666_666_666_667, -34));
        // A tie: 35 digits ending in 5 go to the even neighbour.
        let tie = Decimal::quotient(false, &n(10u128.pow(34) + 5), &n(10), 0);
        assert_eq!(tie, d(10u128.pow(33), 0));
        let odd = Decimal::quotient(false, &n(10u128.pow(34) + 15), &n(10), 0);
        assert_eq!(odd, d(1_000_000_000_000_000_000_000_000_000_000_002, 0));
        // Rounding up to 10^34 carries into the exponent.
        let carry = Decimal::quotient(false, &n(10u128.pow(35) - 1), &n(1), 0);
        assert_eq!(carry, d(10u128.pow(33), 2));
        assert_eq!(Decimal::quotient(false, &n(5), &n(1), -400), d(5 * 10u128.pow(33), -433));
    }

    #[test]
    fn square_roots_are_correctly_rounded() {
        assert_eq!(d(4 * 10u128.pow(33), -33).sqrt(), d(2 * 10u128.pow(33), -33));
        // sqrt(2) = 1.41421356237309504880168872420969807856967...
        assert_eq!(d(2 * 10u128.pow(33), -33).sqrt(), d(1_414_213_562_373_095_048_801_688_724_209_698, -33));
        // sqrt(20) = 4.47213595499957939281834733746255247088123...
        assert_eq!(d(2 * 10u128.pow(33), -32).sqrt(), d(4_472_135_954_999_579_392_818_347_337_462_552, -33));
        // sqrt(10^-400) = 10^-200.
        assert_eq!(d(10u128.pow(33), -433).sqrt(), d(10u128.pow(33), -233));
    }

    #[test]
    fn doubles_are_read_correctly_rounded() {
        let mut buf = String::new();
        assert_eq!(d(10u128.pow(33), -33).to_f64(&mut buf), 1.0);
        assert_eq!(Decimal { negative: true, ..d(5 * 10u128.pow(33), -34) }.to_f64(&mut buf), -0.5);
        assert_eq!(Decimal::ZERO.to_f64(&mut buf), 0.0);
        assert_eq!(d(10u128.pow(33), -360).to_f64(&mut buf), 1e-327);
    }

    #[test]
    fn signed_addition_cancels() {
        let a = Integer::new(false, n(10));
        let b = Integer::new(true, n(10));
        assert_eq!(a.add(&b), Integer::default());
        assert_eq!(a.add(&b.negate().negate()).magnitude, Natural::default());
        assert_eq!(Integer::new(true, n(3)).add(&Integer::new(false, n(10))), Integer::new(false, n(7)));
        assert_eq!(Integer::new(true, n(13)).add(&Integer::new(false, n(10))), Integer::new(true, n(3)));
    }
}
