use std::cmp::Ordering;

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
    pub(super) fn bits(&self) -> u64 {
        match self.0.last() {
            None => 0,
            Some(top) => 32 * self.0.len() as u64 - u64::from(top.leading_zeros()),
        }
    }

    pub(super) fn trim(mut self) -> Natural {
        while self.0.last() == Some(&0) {
            self.0.pop();
        }
        self
    }

    /// The value, when it fits a `u128`.
    pub(super) fn to_u128(&self) -> Option<u128> {
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
    pub(super) fn mul_pow10(&self, k: u64) -> Natural {
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
    pub(super) fn div_rem(&self, divisor: &Natural) -> (Natural, Natural) {
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
    pub(super) fn shr(&self, bits: u32) -> Natural {
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
