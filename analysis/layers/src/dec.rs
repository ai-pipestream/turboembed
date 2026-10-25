//! Exact decimals for the values a grammar or unit normalizer computes.
//!
//! A [`Dec`] is an `i128` mantissa times a power of ten. Sums, products and
//! integer powers are exact while the mantissa stays within
//! [`MAX_DIGITS`] digits and are rounded half away from zero past that; a
//! quotient is cut at [`PRECISION`] significant digits, so `1/3` holds 34
//! threes. That is more than an `f64` holds, which is what the
//! Recognizers-Text specifications print.

use std::cmp::Ordering;
use std::fmt;

/// Significant digits a quotient keeps.
pub const PRECISION: u32 = 34;

/// Significant digits a mantissa keeps after any operation. Two mantissas
/// of this many digits add without overflowing an `i128`.
pub const MAX_DIGITS: u32 = 36;

/// `mantissa` times ten to the power of minus `scale`, with no trailing
/// zeros in the mantissa, so two equal values have equal fields and
/// `==` compares values. A negative scale stands for trailing zeros:
/// `1e10` is the mantissa 1 at the scale -10.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Dec {
    mantissa: i128,
    scale: i16,
}

/// `10^n` for `n` up to 38.
pub(crate) fn pow10(n: u32) -> i128 {
    10i128.pow(n)
}

/// How many decimal digits `m` has; 0 for zero.
pub(crate) fn digits(m: i128) -> u32 {
    let m = m.unsigned_abs();
    if m == 0 { 0 } else { m.ilog10() + 1 }
}

/// `m` divided by `10^drop`, rounded half away from zero.
fn shed(m: i128, drop: u32) -> i128 {
    if drop == 0 {
        return m;
    }
    if drop > 38 {
        return 0;
    }
    let d = pow10(drop);
    let (q, r) = (m / d, m % d);
    if r.unsigned_abs() * 2 >= d.unsigned_abs() { q + m.signum() } else { q }
}

impl Dec {
    pub const ZERO: Dec = Dec { mantissa: 0, scale: 0 };
    pub const ONE: Dec = Dec { mantissa: 1, scale: 0 };

    /// `mantissa` times ten to the minus `scale`, rounded to
    /// [`MAX_DIGITS`] digits; `None` when the scale leaves the `i16` range.
    pub fn new(mantissa: i128, scale: i32) -> Option<Dec> {
        let mut m = mantissa;
        let mut s = scale;
        let d = digits(m);
        if d > MAX_DIGITS {
            m = shed(m, d - MAX_DIGITS);
            s = s.checked_sub((d - MAX_DIGITS) as i32)?;
        }
        if m == 0 {
            return Some(Dec::ZERO);
        }
        while m % 10 == 0 {
            m /= 10;
            s = s.checked_sub(1)?;
        }
        let scale = i16::try_from(s).ok()?;
        Some(Dec { mantissa: m, scale })
    }

    pub fn from_i64(v: i64) -> Dec {
        Dec::new(i128::from(v), 0).expect("an i64 is in range")
    }

    /// Builds a [`Dec`] from an unscaled ASCII digit string (`coefficient`),
    /// sign (`negative`), and `scale` as stored in `Numbers`, `Quantities`,
    /// or `Money`.
    pub fn from_unscaled(negative: bool, coefficient: &str, scale: i32) -> Option<Dec> {
        if coefficient.is_empty() || !coefficient.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let mut m: i128 = 0;
        let mut kept = 0u32;
        let mut s = scale;
        let mut next: Option<u8> = None;
        for b in coefficient.bytes() {
            let d = b - b'0';
            if kept < MAX_DIGITS {
                if m != 0 || d != 0 {
                    m = m * 10 + i128::from(d);
                    kept += u32::from(m != 0);
                }
            } else {
                next.get_or_insert(d);
                s = s.checked_sub(1)?;
            }
        }
        if next.is_some_and(|d| d >= 5) {
            m += 1;
        }
        if negative {
            m = -m;
        }
        Dec::new(m, s)
    }

    /// The value of a written numeral: the ASCII digits of the integer
    /// part and of the fraction, each possibly empty. Digits past
    /// [`MAX_DIGITS`] significant ones are rounded off.
    pub fn from_digits(int: &[u8], frac: &[u8]) -> Dec {
        let mut m: i128 = 0;
        let mut kept = 0u32;
        let mut scale: i32 = 0;
        let mut next: Option<u8> = None;
        for (i, &b) in int.iter().chain(frac).enumerate() {
            let in_frac = i >= int.len();
            let d = b - b'0';
            if kept < MAX_DIGITS {
                if m != 0 || d != 0 {
                    m = m * 10 + i128::from(d);
                    kept += u32::from(m != 0);
                }
                if in_frac {
                    scale += 1;
                }
            } else {
                next.get_or_insert(d);
                if !in_frac {
                    scale -= 1;
                }
            }
        }
        if next.is_some_and(|d| d >= 5) {
            m += 1;
        }
        Dec::new(m, scale).unwrap_or(Dec::ZERO)
    }

    pub fn mantissa(self) -> i128 {
        self.mantissa
    }

    pub fn scale(self) -> i16 {
        self.scale
    }

    pub fn is_zero(self) -> bool {
        self.mantissa == 0
    }

    pub fn is_negative(self) -> bool {
        self.mantissa < 0
    }

    /// Whether the value has no fraction.
    pub fn is_integer(self) -> bool {
        self.scale <= 0
    }

    pub fn negated(self) -> Dec {
        Dec { mantissa: -self.mantissa, scale: self.scale }
    }

    pub fn abs(self) -> Dec {
        Dec { mantissa: self.mantissa.abs(), scale: self.scale }
    }

    /// The value as an `i64`, when it is an integer in range.
    pub fn to_i64(self) -> Option<i64> {
        if self.scale > 0 {
            return None;
        }
        let zeros = u32::try_from(-i32::from(self.scale)).ok()?;
        if zeros > 38 {
            return if self.mantissa == 0 { Some(0) } else { None };
        }
        i64::try_from(self.mantissa.checked_mul(pow10(zeros))?).ok()
    }

    /// The value times ten to the power of `e`.
    pub fn shift(self, e: i32) -> Option<Dec> {
        if self.is_zero() {
            return Some(self);
        }
        Dec::new(self.mantissa, i32::from(self.scale).checked_sub(e)?)
    }

    /// The order of magnitude: digits before the point, minus the scale.
    fn exponent(self) -> i32 {
        digits(self.mantissa) as i32 - i32::from(self.scale)
    }

    pub fn checked_add(self, other: Dec) -> Option<Dec> {
        if self.is_zero() {
            return Some(other);
        }
        if other.is_zero() {
            return Some(self);
        }
        // The common scale: the finer of the two, unless raising a mantissa
        // to it would pass 37 digits, in which case the finer one is
        // rounded to what fits.
        let fits = |d: Dec| i32::from(d.scale) + 37 - digits(d.mantissa) as i32;
        let target = i32::from(self.scale.max(other.scale)).min(fits(self)).min(fits(other));
        let at = |d: Dec| {
            let s = i32::from(d.scale);
            if target >= s { d.mantissa * pow10((target - s) as u32) } else { shed(d.mantissa, (s - target) as u32) }
        };
        Dec::new(at(self) + at(other), target)
    }

    pub fn checked_sub(self, other: Dec) -> Option<Dec> {
        self.checked_add(other.negated())
    }

    pub fn checked_mul(self, other: Dec) -> Option<Dec> {
        let (mut a, mut b) = (self, other);
        // Round the longer factor, one digit at a time, until the product
        // fits.
        while digits(a.mantissa) + digits(b.mantissa) > 37 {
            if digits(a.mantissa) >= digits(b.mantissa) {
                a = Dec::new(shed(a.mantissa, 1), i32::from(a.scale) - 1)?;
            } else {
                b = Dec::new(shed(b.mantissa, 1), i32::from(b.scale) - 1)?;
            }
        }
        Dec::new(a.mantissa * b.mantissa, i32::from(a.scale) + i32::from(b.scale))
    }

    /// The quotient to [`PRECISION`] significant digits; `None` for a zero
    /// divisor.
    pub fn checked_div(self, other: Dec) -> Option<Dec> {
        if other.is_zero() {
            return None;
        }
        if self.is_zero() {
            return Some(Dec::ZERO);
        }
        // Mantissas have at most MAX_DIGITS (36) digits, so ten times a
        // remainder is below 10^37 and stays well inside i128::MAX (~1.7e38).
        // A quotient of more than PRECISION digits needs a_m >= 10^34 * b_m
        // with a_m < 10^36, so b_m < 100 there and the single-step rounding
        // below cannot overflow either.
        let negative = self.is_negative() != other.is_negative();
        let (a_m, b_m) = (self.mantissa.abs(), other.mantissa.abs());
        let mut q = a_m / b_m;
        let mut r = a_m % b_m;
        let mut scale = i32::from(self.scale) - i32::from(other.scale);
        let dq = digits(q);
        if dq > PRECISION {
            let drop = dq - PRECISION;
            let d = pow10(drop);
            let rem = (q % d) * b_m + r;
            let denom = d * b_m;
            q /= d;
            if rem * 2 >= denom {
                q += 1;
            }
            scale -= drop as i32;
        } else {
            while r != 0 && digits(q) < PRECISION {
                q = q * 10 + (r * 10) / b_m;
                r = (r * 10) % b_m;
                scale += 1;
            }
            if r * 2 >= b_m {
                q += 1;
            }
        }
        Dec::new(if negative { -q } else { q }, scale)
    }

    /// The value to the power of `e`; a negative power divides one by it.
    /// `None` when `|e|` is above 4096 or the result leaves the range.
    pub fn pow(self, e: i32) -> Option<Dec> {
        if e.unsigned_abs() > 4096 {
            return None;
        }
        let mut result = Dec::ONE;
        let mut base = self;
        let mut n = e.unsigned_abs();
        while n > 0 {
            if n & 1 == 1 {
                result = result.checked_mul(base)?;
            }
            n >>= 1;
            if n > 0 {
                base = base.checked_mul(base)?;
            }
        }
        if e < 0 { Dec::ONE.checked_div(result) } else { Some(result) }
    }

    /// The nearest `f64`, rounded once from the decimal.
    pub fn to_f64(self) -> f64 {
        format!("{}e{}", self.mantissa, -i32::from(self.scale)).parse().expect("a mantissa with an exponent parses")
    }

    /// Appends the absolute value's digits without a decimal point, with
    /// the zeros a negative scale stands for, and returns how many of them
    /// are the fraction: the coefficient and scale of `turbo:numbers`.
    pub fn write_coefficient(self, out: &mut String) -> u32 {
        use std::fmt::Write;
        write!(out, "{}", self.mantissa.unsigned_abs()).expect("writing to a String");
        if self.scale < 0 {
            out.extend(std::iter::repeat_n('0', usize::from(self.scale.unsigned_abs())));
            0
        } else {
            u32::from(self.scale.unsigned_abs())
        }
    }
}

impl Ord for Dec {
    fn cmp(&self, other: &Dec) -> Ordering {
        let (a, b) = (*self, *other);
        let sign = |d: Dec| d.mantissa.signum();
        if sign(a) != sign(b) || a.is_zero() {
            return sign(a).cmp(&sign(b));
        }
        let magnitude = match a.exponent().cmp(&b.exponent()) {
            Ordering::Equal => {
                // Equal exponents: aligning the coarser mantissa to the finer
                // scale gives it the finer one's digit count, which fits.
                let s = a.scale.max(b.scale);
                let at = |d: Dec| d.mantissa.abs() * pow10((s - d.scale) as u32);
                at(a).cmp(&at(b))
            }
            o => o,
        };
        if a.is_negative() { magnitude.reverse() } else { magnitude }
    }
}

impl PartialOrd for Dec {
    fn partial_cmp(&self, other: &Dec) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The value written out without an exponent: `-1500`, `0.05`, `1000000`.
impl fmt::Display for Dec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        if self.is_negative() {
            s.push('-');
        }
        let mut digits = String::new();
        let scale = self.write_coefficient(&mut digits) as usize;
        if scale == 0 {
            s.push_str(&digits);
        } else if digits.len() > scale {
            s.push_str(&digits[..digits.len() - scale]);
            s.push('.');
            s.push_str(&digits[digits.len() - scale..]);
        } else {
            s.push_str("0.");
            s.extend(std::iter::repeat_n('0', scale - digits.len()));
            s.push_str(&digits);
        }
        f.write_str(&s)
    }
}

#[cfg(test)]
#[path = "dec_tests.rs"]
mod tests;
