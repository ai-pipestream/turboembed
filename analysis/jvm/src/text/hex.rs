/// A hexadecimal floating point literal taken apart, as
/// `FloatingDecimal.parseHexString` reads it: the value is the integer the
/// significant hex digits form times a power of two.
pub(super) struct Hex {
    /// The hex digits of the significand without leading zeros, so the
    /// first is not zero; empty when the significand is zero.
    digits: Vec<u8>,
    /// The power of two, when the exponent fit an `int`.
    exponent: Exponent,
}

/// The exponent written after `p`, minus four per fraction digit.
enum Exponent {
    Fits(i64),
    /// The digits after `p` exceed an `int`, which `Integer.parseInt`
    /// refuses: Java then returns infinity for a positive exponent and
    /// zero for a negative one (a zero significand is still zero).
    Overflow {
        positive: bool,
    },
}

impl Hex {
    /// The text after `0x`: `([-+])?0[xX](((\p{XDigit}+)\.?)|((\p{XDigit}*)\.(\p{XDigit}+)))[pP]([-+])?(\p{Digit}+)[fFdD]?`
    /// is Java's pattern, so hex digits with an optional point among them
    /// (at least one digit), then `p`, an optional sign, decimal digits
    /// and an optional type suffix. `None` when the text is not that.
    pub(super) fn parse(text: &str) -> Option<Hex> {
        let b = text.as_bytes();
        let mut i = 0;
        let mut digits = Vec::new();
        while i < b.len() && b[i].is_ascii_hexdigit() {
            digits.push((b[i] as char).to_digit(16).unwrap() as u8);
            i += 1;
        }
        let mut fraction_digits = 0usize;
        if i < b.len() && b[i] == b'.' {
            i += 1;
            while i < b.len() && b[i].is_ascii_hexdigit() {
                digits.push((b[i] as char).to_digit(16).unwrap() as u8);
                fraction_digits += 1;
                i += 1;
            }
        }
        if digits.is_empty() || i >= b.len() || (b[i] != b'p' && b[i] != b'P') {
            return None;
        }
        i += 1;
        let mut positive = true;
        if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
            positive = b[i] == b'+';
            i += 1;
        }
        let start = i;
        let mut magnitude: u64 = 0;
        let mut overflow = false;
        while i < b.len() && b[i].is_ascii_digit() {
            magnitude = magnitude * 10 + u64::from(b[i] - b'0');
            if magnitude > i32::MAX as u64 {
                overflow = true;
                magnitude = 0;
            }
            i += 1;
        }
        if i == start {
            return None;
        }
        if i < b.len() && matches!(b[i], b'f' | b'F' | b'd' | b'D') {
            i += 1;
        }
        if i != b.len() {
            return None;
        }
        let leading_zeros = digits.iter().take_while(|&&d| d == 0).count();
        digits.drain(..leading_zeros);
        let exponent = if overflow {
            Exponent::Overflow { positive }
        } else {
            let e = if positive { magnitude as i64 } else { -(magnitude as i64) };
            Exponent::Fits(e - 4 * fraction_digits as i64)
        };
        Some(Hex { digits, exponent })
    }

    pub(super) fn to_f64(&self) -> f64 {
        match self.round(53, -1022, 1023) {
            Binary::Infinity => f64::INFINITY,
            Binary::Finite { exponent, fraction } => f64::from_bits((exponent << 52) | fraction),
        }
    }

    pub(super) fn to_f32(&self) -> f32 {
        match self.round(24, -126, 127) {
            Binary::Infinity => f32::INFINITY,
            Binary::Finite { exponent, fraction } => f32::from_bits(((exponent << 23) | fraction) as u32),
        }
    }

    /// The value rounded to the nearest, ties to even, in a binary format
    /// of `precision` significant bits whose normal values have the
    /// exponents `emin..=emax` (53, -1022 and 1023 for a double). The
    /// digits beyond the first seventeen only say whether anything
    /// follows, which is all the rounding needs of them.
    fn round(&self, precision: u32, emin: i64, emax: i64) -> Binary {
        let zero = Binary::Finite { exponent: 0, fraction: 0 };
        if self.digits.is_empty() {
            return zero;
        }
        let exp2 = match self.exponent {
            Exponent::Fits(e) => e,
            Exponent::Overflow { positive: true } => return Binary::Infinity,
            Exponent::Overflow { positive: false } => return zero,
        };
        let p = i64::from(precision);
        let taken = self.digits.len().min(17);
        let top = self.digits[..taken].iter().fold(0u128, |acc, &d| (acc << 4) | u128::from(d));
        let top_len = i64::from(128 - top.leading_zeros());
        let sticky_rest = self.digits[taken..].iter().any(|&d| d != 0);
        // The integer of all the digits has `len` bits; its leading bit
        // has the exponent `e`, so the value is in [2^e, 2^(e+1)).
        let len = top_len + 4 * (self.digits.len() - taken) as i64;
        let mut e = exp2 + len - 1;
        if e > emax {
            return Binary::Infinity;
        }
        // The bits kept: the precision, fewer below the normal range.
        let mut keep = if e >= emin { p } else { p - (emin - e) };
        if keep < 0 {
            return zero;
        }
        let dropped = len - keep;
        let mut m: u64;
        if dropped <= 0 {
            m = (top << (-dropped)) as u64;
        } else {
            // Every dropped bit beyond the first seventeen digits is in
            // `sticky_rest`; the rest come out of `top`, which keeps more
            // bits than any format here, so the round bit is one of them.
            let from_top = (dropped - 4 * (self.digits.len() - taken) as i64) as u32;
            let round;
            let sticky;
            if i64::from(from_top) == top_len {
                m = 0;
                round = true;
                sticky = (top & !(1u128 << (top_len - 1))) != 0 || sticky_rest;
            } else {
                m = (top >> from_top) as u64;
                round = (top >> (from_top - 1)) & 1 == 1;
                sticky = (top & ((1u128 << (from_top - 1)) - 1)) != 0 || sticky_rest;
            }
            if round && (sticky || m & 1 == 1) {
                m += 1;
            }
            if m == 1u64 << keep {
                // Rounding carried into a new leading bit.
                e += 1;
                if e >= emin {
                    m = 1u64 << (p - 1);
                    keep = p;
                } else {
                    keep += 1;
                }
                if e > emax {
                    return Binary::Infinity;
                }
            }
        }
        debug_assert!(m < 1u64 << keep);
        if e >= emin {
            Binary::Finite { exponent: (e - emin + 1) as u64, fraction: m - (1u64 << (p - 1)) }
        } else {
            Binary::Finite { exponent: 0, fraction: m }
        }
    }
}

/// A rounded hexadecimal literal's fields, without its sign.
enum Binary {
    Infinity,
    /// The biased exponent field (zero for a subnormal or zero) and the
    /// fraction field.
    Finite {
        exponent: u64,
        fraction: u64,
    },
}
