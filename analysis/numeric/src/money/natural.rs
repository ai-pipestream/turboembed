//! Unbounded natural numbers in base 10⁹, for the coefficients of exact
//! decimal arithmetic. Only what the conversion needs is here: reading
//! and writing decimal digits, products, powers of ten, comparison,
//! subtraction and a division whose quotient fits in a `u64`.

use std::cmp::Ordering;

const BASE: u32 = 1_000_000_000;
const BASE_DIGITS: usize = 9;

/// Powers of ten that fit a limb.
const POW10: [u32; 10] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000];

/// A natural number: limbs of nine decimal digits, least significant
/// first, with no zero limb at the top. Zero has no limbs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Natural {
    limbs: Vec<u32>,
}

impl Natural {
    /// The number an ASCII digit string writes. Leading zeros are allowed.
    pub fn from_digits(digits: &str) -> Natural {
        let b = digits.as_bytes();
        debug_assert!(b.iter().all(u8::is_ascii_digit), "ASCII digits");
        let mut limbs = Vec::with_capacity(b.len() / BASE_DIGITS + 1);
        let mut end = b.len();
        while end > 0 {
            let start = end.saturating_sub(BASE_DIGITS);
            limbs.push(b[start..end].iter().fold(0u32, |v, &d| v * 10 + u32::from(d - b'0')));
            end = start;
        }
        let mut n = Natural { limbs };
        n.trim();
        n
    }

    #[cfg(test)]
    pub fn from_u128(mut v: u128) -> Natural {
        let mut limbs = Vec::new();
        while v > 0 {
            limbs.push((v % u128::from(BASE)) as u32);
            v /= u128::from(BASE);
        }
        Natural { limbs }
    }

    /// The value, when it fits a `u128`.
    pub fn to_u128(&self) -> Option<u128> {
        self.limbs.iter().rev().try_fold(0u128, |v, &l| v.checked_mul(u128::from(BASE))?.checked_add(u128::from(l)))
    }

    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// How many decimal digits the number has; 1 for zero, as
    /// `BigDecimal.precision()` counts.
    pub fn digit_count(&self) -> u64 {
        match self.limbs.last() {
            None => 1,
            Some(&top) => (self.limbs.len() as u64 - 1) * BASE_DIGITS as u64 + u64::from(top.ilog10() + 1),
        }
    }

    /// Appends the decimal digits, without leading zeros; `0` for zero.
    pub fn write_digits(&self, out: &mut String) {
        let Some((&top, rest)) = self.limbs.split_last() else {
            out.push('0');
            return;
        };
        out.push_str(&top.to_string());
        for &l in rest.iter().rev() {
            let s = l.to_string();
            out.extend(std::iter::repeat_n('0', BASE_DIGITS - s.len()));
            out.push_str(&s);
        }
    }

    pub fn mul(&self, other: &Natural) -> Natural {
        if self.is_zero() || other.is_zero() {
            return Natural::default();
        }
        let mut acc = vec![0u64; self.limbs.len() + other.limbs.len()];
        for (i, &a) in self.limbs.iter().enumerate() {
            let mut carry = 0u64;
            for (j, &b) in other.limbs.iter().enumerate() {
                // At most (10⁹ - 1)² + 2 · (10⁹ - 1), which fits a u64.
                let t = acc[i + j] + u64::from(a) * u64::from(b) + carry;
                acc[i + j] = t % u64::from(BASE);
                carry = t / u64::from(BASE);
            }
            let mut k = i + other.limbs.len();
            while carry > 0 {
                let t = acc[k] + carry;
                acc[k] = t % u64::from(BASE);
                carry = t / u64::from(BASE);
                k += 1;
            }
        }
        let mut n = Natural { limbs: acc.into_iter().map(|l| l as u32).collect() };
        n.trim();
        n
    }

    /// The number times 10^`k`.
    pub fn mul_pow10(&self, k: u64) -> Natural {
        if self.is_zero() {
            return Natural::default();
        }
        let whole = usize::try_from(k / BASE_DIGITS as u64).expect("a power of ten that fits in memory");
        let mut limbs = vec![0u32; whole];
        limbs.extend_from_slice(&self.limbs);
        let mut n = Natural { limbs };
        n.mul_small(POW10[(k % BASE_DIGITS as u64) as usize]);
        n
    }

    /// Multiplies in place by a factor below 2³².
    pub fn mul_small(&mut self, m: u32) {
        let mut carry = 0u64;
        for l in &mut self.limbs {
            let t = u64::from(*l) * u64::from(m) + carry;
            *l = (t % u64::from(BASE)) as u32;
            carry = t / u64::from(BASE);
        }
        while carry > 0 {
            self.limbs.push((carry % u64::from(BASE)) as u32);
            carry /= u64::from(BASE);
        }
        self.trim();
    }

    /// Subtracts `other`, which must not be larger.
    fn sub_assign(&mut self, other: &Natural) {
        debug_assert!(*self >= *other, "a subtrahend no larger than the minuend");
        let mut borrow = 0u32;
        for (i, l) in self.limbs.iter_mut().enumerate() {
            let s = other.limbs.get(i).copied().unwrap_or(0) + borrow;
            if s == 0 && i >= other.limbs.len() {
                break;
            }
            if *l >= s {
                *l -= s;
                borrow = 0;
            } else {
                *l += BASE - s;
                borrow = 1;
            }
        }
        self.trim();
    }

    /// The quotient and remainder of `self / divisor`, for a divisor that
    /// is not zero and a quotient below 10¹⁹. Long division in base ten:
    /// each decimal digit of the quotient takes at most nine subtractions.
    pub fn div_rem_small_quotient(&self, divisor: &Natural) -> (u64, Natural) {
        assert!(!divisor.is_zero(), "a divisor that is not zero");
        let mut rem = self.clone();
        if rem < *divisor {
            return (0, rem);
        }
        // The highest power of ten that the divisor times it can fit.
        let top = self.digit_count().saturating_sub(divisor.digit_count());
        assert!(top < 19, "a quotient below 10^19");
        let mut q = 0u64;
        for j in (0..=top).rev() {
            let step = divisor.mul_pow10(j);
            let mut digit = 0;
            while rem >= step {
                rem.sub_assign(&step);
                digit += 1;
            }
            q = q * 10 + digit;
        }
        (q, rem)
    }

    fn trim(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }
}

impl PartialOrd for Natural {
    fn partial_cmp(&self, other: &Natural) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Natural {
    fn cmp(&self, other: &Natural) -> Ordering {
        self.limbs.len().cmp(&other.limbs.len()).then_with(|| self.limbs.iter().rev().cmp(other.limbs.iter().rev()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(s: &str) -> Natural {
        Natural::from_digits(s)
    }

    fn digits(x: &Natural) -> String {
        let mut s = String::new();
        x.write_digits(&mut s);
        s
    }

    #[test]
    fn digits_round_trip() {
        for s in ["0", "1", "999999999", "1000000000", "123456789012345678901234567890", "1000000000000000000"] {
            assert_eq!(digits(&n(s)), s);
        }
        assert_eq!(digits(&n("000123")), "123");
        assert_eq!(digits(&n("")), "0");
        assert_eq!(n("0").digit_count(), 1);
        assert_eq!(n("999999999").digit_count(), 9);
        assert_eq!(n("1000000000").digit_count(), 10);
        assert_eq!(n("12345678901234567890").digit_count(), 20);
    }

    #[test]
    fn products_and_powers() {
        let a = n("123456789012345678901234567890");
        let b = n("987654321098765432109876543210");
        assert_eq!(digits(&a.mul(&b)), "121932631137021795226185032733622923332237463801111263526900");
        assert_eq!(digits(&a.mul(&n("0"))), "0");
        assert_eq!(digits(&n("7").mul_pow10(20)), "700000000000000000000");
        assert_eq!(digits(&n("123").mul_pow10(0)), "123");
        assert_eq!(digits(&n("0").mul_pow10(5)), "0");
        let mut x = n("999999999999");
        x.mul_small(1_000_000_000);
        assert_eq!(digits(&x), "999999999999000000000");
        assert_eq!(Natural::from_u128(u128::MAX).to_u128(), Some(u128::MAX));
        assert_eq!(digits(&Natural::from_u128(u128::MAX)), u128::MAX.to_string());
        assert_eq!(n("340282366920938463463374607431768211456").to_u128(), None);
    }

    #[test]
    fn division_with_a_small_quotient() {
        let (q, r) = n("1000000000000000000000").div_rem_small_quotient(&n("1000000"));
        assert_eq!((q, digits(&r)), (1_000_000_000_000_000, "0".into()));
        let (q, r) = n("12345678901234567890123").div_rem_small_quotient(&n("1234567"));
        assert_eq!(q, 10000007210005263);
        assert_eq!(digits(&r), "364002");
        let (q, r) = n("5").div_rem_small_quotient(&n("7"));
        assert_eq!((q, digits(&r)), (0, "5".into()));
        // Checked against u128 arithmetic.
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for _ in 0..2000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let d = u128::from(seed % 1_000_000_007 + 1) * u128::from(seed >> 40);
            let d = d.max(1);
            let q = u128::from(seed % 10_000_000_000_000_000);
            let num = d * q + u128::from(seed) % d;
            let (got_q, got_r) = Natural::from_u128(num).div_rem_small_quotient(&Natural::from_u128(d));
            assert_eq!(u128::from(got_q), num / d);
            assert_eq!(got_r.to_u128(), Some(num % d));
        }
    }

    #[test]
    fn order() {
        assert!(n("10") > n("9"));
        assert!(n("1000000000") > n("999999999"));
        assert!(n("0") < n("1"));
        assert_eq!(n("42").cmp(&n("042")), Ordering::Equal);
    }
}
