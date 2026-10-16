//! The decimal arithmetic `PlaceProfiles` standardizes with: Java's
//! `BigDecimal` rounded to `MathContext.DECIMAL128`, 34 significant digits
//! and round half even.
//!
//! OpenNLP centers and squares the exact values of the doubles, which needs
//! integers of a few thousand bits, and rounds only the quotients and the
//! square root. [`Natural`] is the unsigned integer for the exact part and
//! [`Decimal`] a value rounded to 34 digits. Only values matter here, not
//! `BigDecimal` scales: every later step (division, comparison, `signum`,
//! `doubleValue`) depends on the value alone.

use std::cmp::Ordering;
use std::fmt::Write;

/// 10^33, the smallest coefficient of a rounded nonzero value.
const MIN_COEFFICIENT: u128 = 10u128.pow(33);
/// 10^34, one more than the largest coefficient.
const COEFFICIENT_LIMIT: u128 = 10u128.pow(34);

/// An unsigned integer of any size: 32-bit limbs, least significant first,
/// with no zero limb at the top (zero has no limbs).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Natural(Vec<u32>);

impl Natural {
    pub(crate) fn from_u64(v: u64) -> Natural {
        Natural::from_u128(u128::from(v))
    }

    pub(crate) fn from_u128(mut v: u128) -> Natural {
        let mut limbs = Vec::with_capacity(4);
        while v != 0 {
            limbs.push(v as u32);
            v >>= 32;
        }
        Natural(limbs)
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    /// The number of significant bits.
    fn bits(&self) -> u64 {
        match self.0.last() {
            None => 0,
            Some(top) => 32 * self.0.len() as u64 - u64::from(top.leading_zeros()),
        }
    }

    fn trim(mut self) -> Natural {
        while self.0.last() == Some(&0) {
            self.0.pop();
        }
        self
    }

    /// The value, when it fits a `u128`.
    fn to_u128(&self) -> Option<u128> {
        if self.0.len() > 4 {
            return None;
        }
        Some(self.0.iter().rev().fold(0u128, |acc, &l| (acc << 32) | u128::from(l)))
    }

    pub(crate) fn add(&self, other: &Natural) -> Natural {
        let (long, short) = if self.0.len() >= other.0.len() { (self, other) } else { (other, self) };
        let mut out = Vec::with_capacity(long.0.len() + 1);
        let mut carry = 0u64;
        for (i, &l) in long.0.iter().enumerate() {
            let s = u64::from(l) + u64::from(short.0.get(i).copied().unwrap_or(0)) + carry;
            out.push(s as u32);
            carry = s >> 32;
        }
        if carry != 0 {
            out.push(carry as u32);
        }
        Natural(out)
    }

    /// `self - other`; `other` must not be greater.
    pub(crate) fn sub(&self, other: &Natural) -> Natural {
        debug_assert!(self.cmp(other) != Ordering::Less);
        let mut out = Vec::with_capacity(self.0.len());
        let mut borrow = 0i64;
        for (i, &l) in self.0.iter().enumerate() {
            let mut d = i64::from(l) - i64::from(other.0.get(i).copied().unwrap_or(0)) - borrow;
            borrow = 0;
            if d < 0 {
                d += 1 << 32;
                borrow = 1;
            }
            out.push(d as u32);
        }
        debug_assert_eq!(borrow, 0);
        Natural(out).trim()
    }

    pub(crate) fn mul(&self, other: &Natural) -> Natural {
        if self.is_zero() || other.is_zero() {
            return Natural::default();
        }
        let mut out = vec![0u32; self.0.len() + other.0.len()];
        for (i, &a) in self.0.iter().enumerate() {
            let mut carry = 0u64;
            for (j, &b) in other.0.iter().enumerate() {
                let t = u64::from(a) * u64::from(b) + u64::from(out[i + j]) + carry;
                out[i + j] = t as u32;
                carry = t >> 32;
            }
            out[i + other.0.len()] = carry as u32;
        }
        Natural(out).trim()
    }

    pub(crate) fn mul_small(&self, m: u32) -> Natural {
        let mut out = Vec::with_capacity(self.0.len() + 1);
        let mut carry = 0u64;
        for &l in &self.0 {
            let t = u64::from(l) * u64::from(m) + carry;
            out.push(t as u32);
            carry = t >> 32;
        }
        if carry != 0 {
            out.push(carry as u32);
        }
        Natural(out).trim()
    }

    pub(crate) fn shl(&self, bits: u64) -> Natural {
        if self.is_zero() {
            return Natural::default();
        }
        let limbs = (bits / 32) as usize;
        let shift = (bits % 32) as u32;
        let mut out = vec![0u32; limbs];
        out.reserve(self.0.len() + 1);
        if shift == 0 {
            out.extend_from_slice(&self.0);
        } else {
            let mut carry = 0u32;
            for &l in &self.0 {
                out.push((l << shift) | carry);
                carry = l >> (32 - shift);
            }
            if carry != 0 {
                out.push(carry);
            }
        }
        Natural(out)
    }

    /// `self * 10^k`.
    fn mul_pow10(&self, k: u64) -> Natural {
        const TEN_TO_NINE: u32 = 1_000_000_000;
        let mut out = self.clone();
        let mut k = k;
        while k >= 9 {
            out = out.mul_small(TEN_TO_NINE);
            k -= 9;
        }
        if k > 0 {
            out = out.mul_small(10u32.pow(k as u32));
        }
        out
    }

    pub(crate) fn cmp(&self, other: &Natural) -> Ordering {
        self.0.len().cmp(&other.0.len()).then_with(|| self.0.iter().rev().cmp(other.0.iter().rev()))
    }

    /// The quotient and remainder of `self / divisor`; `divisor` must not
    /// be zero. Knuth's algorithm D (TAOCP 4.3.1) as Hacker's Delight
    /// writes it.
    fn div_rem(&self, divisor: &Natural) -> (Natural, Natural) {
        assert!(!divisor.is_zero(), "division by zero");
        if self.cmp(divisor) == Ordering::Less {
            return (Natural::default(), self.clone());
        }
        let v = &divisor.0;
        if v.len() == 1 {
            let d = u64::from(v[0]);
            let mut q = vec![0u32; self.0.len()];
            let mut r = 0u64;
            for (i, &l) in self.0.iter().enumerate().rev() {
                let cur = (r << 32) | u64::from(l);
                q[i] = (cur / d) as u32;
                r = cur % d;
            }
            return (Natural(q).trim(), Natural::from_u64(r));
        }
        let n = v.len();
        let m = self.0.len() - n;
        let s = v[n - 1].leading_zeros();
        let vn = divisor.shl(u64::from(s)).0;
        let mut un = self.shl(u64::from(s)).0;
        un.resize(self.0.len() + 1, 0);
        let mut q = vec![0u32; m + 1];
        const BASE: u64 = 1 << 32;
        for j in (0..=m).rev() {
            let top = (u64::from(un[j + n]) << 32) | u64::from(un[j + n - 1]);
            let mut qhat = top / u64::from(vn[n - 1]);
            let mut rhat = top % u64::from(vn[n - 1]);
            while qhat >= BASE || qhat * u64::from(vn[n - 2]) > ((rhat << 32) | u64::from(un[j + n - 2])) {
                qhat -= 1;
                rhat += u64::from(vn[n - 1]);
                if rhat >= BASE {
                    break;
                }
            }
            let mut k = 0i64;
            for i in 0..n {
                let p = qhat * u64::from(vn[i]);
                let t = i64::from(un[i + j]) - k - (p & 0xFFFF_FFFF) as i64;
                un[i + j] = t as u32;
                k = (p >> 32) as i64 - (t >> 32);
            }
            let t = i64::from(un[j + n]) - k;
            un[j + n] = t as u32;
            q[j] = qhat as u32;
            if t < 0 {
                q[j] = q[j].wrapping_sub(1);
                let mut carry = 0u64;
                for i in 0..n {
                    let t = u64::from(un[i + j]) + u64::from(vn[i]) + carry;
                    un[i + j] = t as u32;
                    carry = t >> 32;
                }
                un[j + n] = un[j + n].wrapping_add(carry as u32);
            }
        }
        un.truncate(n);
        let r = Natural(un).trim();
        let r = if s == 0 { r } else { r.shr(s) };
        (Natural(q).trim(), r)
    }

    /// `self >> bits` for a shift below 32.
    fn shr(&self, bits: u32) -> Natural {
        debug_assert!(bits > 0 && bits < 32);
        let out = (0..self.0.len())
            .map(|i| (self.0[i] >> bits) | (self.0.get(i + 1).copied().unwrap_or(0) << (32 - bits)))
            .collect();
        Natural(out).trim()
    }
}

/// An integer with a sign, for the centered values.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Integer {
    pub(crate) negative: bool,
    pub(crate) magnitude: Natural,
}

impl Integer {
    pub(crate) fn new(negative: bool, magnitude: Natural) -> Integer {
        let negative = negative && !magnitude.is_zero();
        Integer { negative, magnitude }
    }

    pub(crate) fn add(&self, other: &Integer) -> Integer {
        if self.negative == other.negative {
            return Integer::new(self.negative, self.magnitude.add(&other.magnitude));
        }
        match self.magnitude.cmp(&other.magnitude) {
            Ordering::Less => Integer::new(other.negative, other.magnitude.sub(&self.magnitude)),
            _ => Integer::new(self.negative, self.magnitude.sub(&other.magnitude)),
        }
    }

    pub(crate) fn negate(&self) -> Integer {
        Integer::new(!self.negative, self.magnitude.clone())
    }
}

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
