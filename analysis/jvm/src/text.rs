//! Java's conversions between numbers and text: `Double.toString`,
//! `Float.toString`, `Double.parseDouble`, `Float.parseFloat`,
//! `Integer.parseInt` and `Boolean.parseBoolean`.
//!
//! OpenNLP writes numbers into event files, model manifests and training
//! reports with `toString`, and reads training parameters and real valued
//! features with the parse methods. The grammar Java accepts and the text
//! it writes differ from Rust's in small ways (`Infinity` against `inf`, a
//! `1.0E7` exponent form, type suffixes such as `1.5f`), so a port that
//! must read what the JVM wrote, or write what it would have written,
//! uses these.
//!
//! `toString` follows the JDK 19 and later specification (JDK-4511638):
//! the decimal written is the shortest that rounds to the value, and
//! among those of that length the closest; when the shortest has one
//! digit, decimals of two digits are considered as well and the closest
//! of them all is written. Rust's shortest formatting gives the same
//! digits except for that last rule, which is applied here on top of it.
//! The test vectors (`tests/vectors/text.tsv`) hold the JVM's output for
//! values on both sides of each rule.

use std::fmt;

/// Why Java would throw `NumberFormatException`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// The text is not a number in Java's grammar.
    NotANumber(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::NotANumber(s) => write!(f, "{s:?} is not a number"),
        }
    }
}

impl std::error::Error for ParseError {}

/// `Double.toString(d)`.
pub fn double_to_string(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let (digits, exponent) =
        shortest(format!("{:e}", x), |s| s.parse::<f64>().ok() == Some(x), || format!("{:.800e}", x));
    layout(x.is_sign_negative(), &digits, exponent, (1e-3..1e7).contains(&x.abs()))
}

/// `Float.toString(f)`.
pub fn float_to_string(x: f32) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let (digits, exponent) =
        shortest(format!("{:e}", x), |s| s.parse::<f32>().ok() == Some(x), || format!("{:.200e}", x));
    layout(x.is_sign_negative(), &digits, exponent, (1e-3..1e7).contains(&x.abs()))
}

/// The digits and the decimal exponent of the value Java writes. `short`
/// is Rust's `{:e}` form, the shortest decimal that rounds back to the
/// value; its length `m` is the length Java looks for, two when `m` is one.
/// Of the decimals of that length, Java writes the one closest to the
/// value, and on a tie the one whose last digit is even, so the value's
/// exact expansion (`exact`, in `{:e}` form) is rounded to that length
/// half to even. The closest decimal is at least as close as the one Rust
/// chose, so it rounds back to the value too; `rounds_back` checks that.
///
/// Rust's choice differs from Java's on a tie, which happens when the
/// value is exactly a decimal of `m + 1` digits ending in 5: for the float
/// 5392.15625, Rust writes 5392.1563 and Java 5392.1562.
fn shortest(short: String, rounds_back: impl Fn(&str) -> bool, exact: impl Fn() -> String) -> (Vec<u8>, i32) {
    let (digits, exponent) = split_e(&short);
    let n = digits.len().max(2);
    let (all, mut e) = split_e(&exact());
    let mut kept: Vec<u8> = all[..n.min(all.len())].to_vec();
    kept.resize(n, 0);
    let rest = &all[n.min(all.len())..];
    let up = match rest.first() {
        None => false,
        Some(&d) if d > 5 => true,
        Some(&d) if d < 5 => false,
        Some(_) => rest[1..].iter().any(|&d| d != 0) || kept[n - 1] % 2 == 1,
    };
    if up {
        let mut i = n;
        loop {
            if i == 0 {
                kept.insert(0, 1);
                kept.pop();
                e += 1;
                break;
            }
            i -= 1;
            if kept[i] == 9 {
                kept[i] = 0;
            } else {
                kept[i] += 1;
                break;
            }
        }
    }
    while kept.len() > 1 && kept.last() == Some(&0) {
        kept.pop();
    }
    let mut text = String::new();
    if short.starts_with('-') {
        text.push('-');
    }
    text.push((b'0' + kept[0]) as char);
    text.push('.');
    if kept.len() > 1 {
        kept[1..].iter().for_each(|&d| text.push((b'0' + d) as char));
    } else {
        text.push('0');
    }
    text.push_str(&format!("e{e}"));
    if !rounds_back(&text) {
        debug_assert!(false, "{text} does not round back; the shortest form was {short}");
        return (digits, exponent);
    }
    (kept, e)
}

/// The significant digits and the exponent of a `{:e}` formatted number,
/// ignoring its sign.
fn split_e(text: &str) -> (Vec<u8>, i32) {
    let text = text.strip_prefix('-').unwrap_or(text);
    let (mantissa, exponent) = text.split_once('e').expect("{:e} output has an exponent");
    let digits = mantissa.bytes().filter(u8::is_ascii_digit).map(|b| b - b'0').collect();
    (digits, exponent.parse().expect("{:e} output has an integer exponent"))
}

/// Lays out `digits` x 10^`exponent` (the first digit before the point)
/// as `Double.toString` does: plain when the magnitude is in
/// `[1e-3, 1e7)`, computerized scientific notation otherwise.
fn layout(negative: bool, digits: &[u8], exponent: i32, plain: bool) -> String {
    let mut out = String::with_capacity(digits.len() + 8);
    if negative {
        out.push('-');
    }
    let digit = |d: u8| (b'0' + d) as char;
    if plain {
        if exponent >= 0 {
            let int_len = exponent as usize + 1;
            for i in 0..int_len {
                out.push(digit(digits.get(i).copied().unwrap_or(0)));
            }
            out.push('.');
            if digits.len() > int_len {
                digits[int_len..].iter().for_each(|&d| out.push(digit(d)));
            } else {
                out.push('0');
            }
        } else {
            out.push_str("0.");
            for _ in 0..(-exponent - 1) {
                out.push('0');
            }
            digits.iter().for_each(|&d| out.push(digit(d)));
        }
    } else {
        out.push(digit(digits[0]));
        out.push('.');
        if digits.len() > 1 {
            digits[1..].iter().for_each(|&d| out.push(digit(d)));
        } else {
            out.push('0');
        }
        out.push('E');
        out.push_str(&exponent.to_string());
    }
    out
}

/// `Double.parseDouble(s)`: Java's grammar, with the type suffixes `f`,
/// `F`, `d` and `D`, `NaN` and `Infinity`, hexadecimal literals such as
/// `0x1.8p1`, and whitespace (characters up to U+0020) trimmed at both
/// ends.
pub fn parse_double(s: &str) -> Result<f64, ParseError> {
    let (negative, body) = body(s)?;
    let value: f64 = match body {
        Number::NaN => f64::NAN,
        Number::Infinity => f64::INFINITY,
        Number::Decimal(text) => text.parse().map_err(|_| ParseError::NotANumber(s.into()))?,
        Number::Hex(h) => h.to_f64(),
    };
    Ok(if negative { -value } else { value })
}

/// `Float.parseFloat(s)`, the grammar of [`parse_double`] read into an
/// `f32` with one correctly rounded conversion, as Java does it: a decimal
/// or hexadecimal literal is rounded to the float directly, not through
/// a double.
pub fn parse_float(s: &str) -> Result<f32, ParseError> {
    let (negative, body) = body(s)?;
    let value: f32 = match body {
        Number::NaN => f32::NAN,
        Number::Infinity => f32::INFINITY,
        Number::Decimal(text) => text.parse().map_err(|_| ParseError::NotANumber(s.into()))?,
        Number::Hex(h) => h.to_f32(),
    };
    Ok(if negative { -value } else { value })
}

enum Number<'a> {
    NaN,
    Infinity,
    Decimal(&'a str),
    Hex(Hex),
}

/// Trims, takes the sign, recognizes `NaN` and `Infinity`, takes a
/// hexadecimal literal apart and checks a decimal literal against Java's
/// grammar, returning it without its type suffix. The decimal that
/// remains is one Rust's parser reads, and reads to the same value.
fn body(s: &str) -> Result<(bool, Number<'_>), ParseError> {
    let error = || ParseError::NotANumber(s.into());
    let trimmed = s.trim_matches(|c: char| c <= ' ');
    if trimmed.is_empty() {
        return Err(error());
    }
    let (negative, rest) = match trimmed.as_bytes()[0] {
        b'-' => (true, &trimmed[1..]),
        b'+' => (false, &trimmed[1..]),
        _ => (false, trimmed),
    };
    match rest {
        "NaN" => return Ok((negative, Number::NaN)),
        "Infinity" => return Ok((negative, Number::Infinity)),
        _ => {}
    }
    if rest.starts_with("0x") || rest.starts_with("0X") {
        return Hex::parse(&rest[2..]).map(|h| (negative, Number::Hex(h))).ok_or_else(error);
    }
    let bytes = rest.as_bytes();
    let body_end = match bytes.last() {
        Some(b'f' | b'F' | b'd' | b'D') => bytes.len() - 1,
        _ => bytes.len(),
    };
    let body = &rest[..body_end];
    // digits [. digits] [(e|E) [sign] digits], with a digit somewhere in
    // the mantissa and at least one in the exponent.
    let b = body.as_bytes();
    let mut i = 0;
    let mut mantissa_digits = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
        mantissa_digits += 1;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
            mantissa_digits += 1;
        }
    }
    if mantissa_digits == 0 {
        return Err(error());
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        let start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == start {
            return Err(error());
        }
    }
    if i != b.len() {
        return Err(error());
    }
    Ok((negative, Number::Decimal(body)))
}

/// A hexadecimal floating point literal taken apart, as
/// `FloatingDecimal.parseHexString` reads it: the value is the integer the
/// significant hex digits form times a power of two.
struct Hex {
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
    fn parse(text: &str) -> Option<Hex> {
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

    fn to_f64(&self) -> f64 {
        match self.round(53, -1022, 1023) {
            Binary::Infinity => f64::INFINITY,
            Binary::Finite { exponent, fraction } => f64::from_bits((exponent << 52) | fraction),
        }
    }

    fn to_f32(&self) -> f32 {
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

/// `Integer.parseInt(s)`: an optional sign and ASCII decimal digits in
/// the range of an `i32`. Java also accepts the decimal digits of other
/// scripts (`Character.digit`); OpenNLP's parameters never hold them, and
/// they are refused here.
pub fn parse_int(s: &str) -> Result<i32, ParseError> {
    let error = || ParseError::NotANumber(s.into());
    let (negative, digits) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        Some(_) => (false, s),
        None => return Err(error()),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(error());
    }
    let mut value: i64 = 0;
    for b in digits.bytes() {
        value = value * 10 + i64::from(b - b'0');
        if value > i64::from(i32::MAX) + 1 {
            return Err(error());
        }
    }
    i32::try_from(if negative { -value } else { value }).map_err(|_| error())
}

/// `Boolean.parseBoolean(s)`: `true` when `s` is `true` ignoring case,
/// otherwise `false`.
pub fn parse_boolean(s: &str) -> bool {
    s.eq_ignore_ascii_case("true")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_and_scientific_doubles() {
        assert_eq!(double_to_string(1.0), "1.0");
        assert_eq!(double_to_string(-1.5), "-1.5");
        assert_eq!(double_to_string(0.001), "0.001");
        assert_eq!(double_to_string(0.0009), "9.0E-4");
        assert_eq!(double_to_string(1234567.0), "1234567.0");
        assert_eq!(double_to_string(1.0e7), "1.0E7");
        assert_eq!(double_to_string(123456.789), "123456.789");
        assert_eq!(double_to_string(1.0e23), "1.0E23");
        assert_eq!(double_to_string(1.0e-10), "1.0E-10");
        assert_eq!(double_to_string(12345.0e-20), "1.2345E-16");
        assert_eq!(double_to_string(100.0), "100.0");
        assert_eq!(double_to_string(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(double_to_string(f64::MAX), "1.7976931348623157E308");
        assert_eq!(double_to_string(f64::MIN_POSITIVE), "2.2250738585072014E-308");
    }

    #[test]
    fn a_one_digit_shortest_gives_way_to_a_closer_two_digit_decimal() {
        // Double.MIN_VALUE is about 4.94e-324: 5E-324 is the shortest,
        // 4.9E-324 is closer and also rounds to it.
        assert_eq!(double_to_string(f64::from_bits(1)), "4.9E-324");
        assert_eq!(double_to_string(2.0 * f64::from_bits(1)), "9.9E-324");
        assert_eq!(float_to_string(f32::from_bits(1)), "1.4E-45");
        // 2.0E-3 stays: it is exact.
        assert_eq!(double_to_string(0.002), "0.002");
        assert_eq!(double_to_string(2.0e-3), "0.002");
    }

    #[test]
    fn special_values() {
        assert_eq!(double_to_string(f64::NAN), "NaN");
        assert_eq!(double_to_string(f64::INFINITY), "Infinity");
        assert_eq!(double_to_string(f64::NEG_INFINITY), "-Infinity");
        assert_eq!(double_to_string(0.0), "0.0");
        assert_eq!(double_to_string(-0.0), "-0.0");
        assert_eq!(float_to_string(f32::NAN), "NaN");
        assert_eq!(float_to_string(-0.0), "-0.0");
    }

    #[test]
    fn floats() {
        assert_eq!(float_to_string(1.0), "1.0");
        assert_eq!(float_to_string(0.1), "0.1");
        assert_eq!(float_to_string(1.0e7), "1.0E7");
        assert_eq!(float_to_string(3.4028235e38), "3.4028235E38");
        assert_eq!(float_to_string(16777216.0), "1.6777216E7");
        assert_eq!(float_to_string(0.33333334), "0.33333334");
    }

    #[test]
    fn parse_double_follows_javas_grammar() {
        assert_eq!(parse_double("1.5"), Ok(1.5));
        assert_eq!(parse_double("  1.5  "), Ok(1.5));
        assert_eq!(parse_double("1.5f"), Ok(1.5));
        assert_eq!(parse_double("1.5D"), Ok(1.5));
        assert_eq!(parse_double("-.5"), Ok(-0.5));
        assert_eq!(parse_double("5."), Ok(5.0));
        assert_eq!(parse_double("+1e3"), Ok(1000.0));
        assert_eq!(parse_double("1E-3"), Ok(0.001));
        assert_eq!(parse_double("Infinity"), Ok(f64::INFINITY));
        assert_eq!(parse_double("-Infinity"), Ok(f64::NEG_INFINITY));
        assert!(parse_double("NaN").unwrap().is_nan());
        assert_eq!(parse_double("1e400"), Ok(f64::INFINITY));
        assert_eq!(parse_double("1e-400"), Ok(0.0));
        for bad in ["", " ", "inf", "nan", "1e", "1e+", ".", "1.5x", "1,5", "--1", "f", "1.5 f"] {
            assert_eq!(parse_double(bad), Err(ParseError::NotANumber(bad.into())), "{bad:?}");
        }
        assert_eq!(parse_float("0.1"), Ok(0.1f32));
        assert_eq!(parse_float("3.4028236e38"), Ok(f32::INFINITY));
        assert_eq!(parse_float("1.00000001"), Ok(1.0f32));
    }

    #[test]
    fn hexadecimal_literals_follow_javas_grammar() {
        assert_eq!(parse_double("0x1p3"), Ok(8.0));
        assert_eq!(parse_double("0X1.8P1"), Ok(3.0));
        assert_eq!(parse_double("-0x.8p1"), Ok(-1.0));
        assert_eq!(parse_double("+0x1.p0"), Ok(1.0));
        assert_eq!(parse_double(" 0x0001.0000p-0 "), Ok(1.0));
        assert_eq!(parse_double("0x.0001p16"), Ok(1.0));
        assert_eq!(parse_double("0xABCDEFp0"), Ok(11259375.0));
        assert_eq!(parse_double("0xabcdefp0f"), Ok(11259375.0));
        assert_eq!(parse_double("0x1p3D"), Ok(8.0));
        assert_eq!(parse_double("0x1p00003"), Ok(8.0));
        assert_eq!(parse_float("0x1p3"), Ok(8.0f32));
        // Too many digits for the type: rounded to the nearest, ties to even.
        assert_eq!(parse_double("0x1.00000000000008p0"), Ok(1.0));
        assert_eq!(parse_double("0x1.00000000000008000000000000000001p0"), Ok(1.0 + f64::EPSILON));
        assert_eq!(parse_double("0x1.00000000000018p0"), Ok(1.0 + 2.0 * f64::EPSILON));
        assert_eq!(parse_double("0x1.1234567890abcdef1234567890p0"), Ok(f64::from_bits(0x3ff1234567890abd)));
        // The float is rounded from the digits, not from the double: the
        // double of the second is above 1 + 2^-24 and would round up.
        assert_eq!(parse_float("0x1.000001p0"), Ok(1.0f32));
        assert_eq!(parse_double("0x1.000001p0"), Ok(1.0 + 2f64.powi(-24)));
        assert_eq!(parse_float("0x1.000003p0"), Ok(1.0f32 + 2.0 * f32::EPSILON));
        assert_eq!(parse_float("0x1.0000010000001p0"), Ok(1.0f32 + f32::EPSILON));
        // The ends of the ranges: the largest values, the smallest subnormals,
        // and the ties and carries at both.
        assert_eq!(parse_double("0x1.fffffffffffffp1023"), Ok(f64::MAX));
        assert_eq!(parse_double("0x1.fffffffffffff8p1023"), Ok(f64::INFINITY));
        assert_eq!(parse_double("0x1p1024"), Ok(f64::INFINITY));
        assert_eq!(parse_double("-0x1p1024"), Ok(f64::NEG_INFINITY));
        assert_eq!(parse_double("0x1p-1022"), Ok(f64::MIN_POSITIVE));
        assert_eq!(parse_double("0x1.ffffffffffffep-1023"), Ok(f64::MIN_POSITIVE - f64::from_bits(1)));
        // Half an ulp under the smallest normal: a tie, to the even mantissa.
        assert_eq!(parse_double("0x1.fffffffffffffp-1023"), Ok(f64::MIN_POSITIVE));
        assert_eq!(parse_double("0x1.fffffffffffff8p-1023"), Ok(f64::MIN_POSITIVE));
        assert_eq!(parse_double("0x1p-1074"), Ok(f64::from_bits(1)));
        assert_eq!(parse_double("0x1p-1075"), Ok(0.0));
        assert_eq!(parse_double("0x1.8p-1075"), Ok(f64::from_bits(1)));
        assert_eq!(parse_double("0x1.0000000000000000000000000001p-1075"), Ok(f64::from_bits(1)));
        assert_eq!(parse_double("0x1p-1076"), Ok(0.0));
        assert!(parse_double("-0x1p-1076").unwrap().is_sign_negative());
        assert_eq!(parse_float("0x1.fffffep127"), Ok(f32::MAX));
        assert_eq!(parse_float("0x1.ffffffp127"), Ok(f32::INFINITY));
        assert_eq!(parse_double("0x1.ffffffp127"), Ok(3.4028235677973366e38));
        assert_eq!(parse_float("0x1p-126"), Ok(f32::MIN_POSITIVE));
        assert_eq!(parse_float("0x1.ffffffp-127"), Ok(f32::MIN_POSITIVE));
        assert_eq!(parse_float("0x1p-149"), Ok(f32::from_bits(1)));
        assert_eq!(parse_float("0x1p-150"), Ok(0.0f32));
        assert_eq!(parse_float("0x1.8p-150"), Ok(f32::from_bits(1)));
        assert_eq!(parse_float("0x1p-151"), Ok(0.0f32));
        // A zero significand is zero whatever the exponent, and an exponent
        // over an int is infinity or zero by its sign.
        assert_eq!(parse_double("0x0p99999999999"), Ok(0.0));
        assert!(parse_double("-0x0.000p-99999999999").unwrap().is_sign_negative());
        assert_eq!(parse_double("0x1p2147483647"), Ok(f64::INFINITY));
        assert_eq!(parse_double("0x1p2147483648"), Ok(f64::INFINITY));
        assert_eq!(parse_double("0x1p-2147483648"), Ok(0.0));
        assert_eq!(parse_double("0x1p-2147483647"), Ok(0.0));
        assert_eq!(parse_float("0x1p99999999999"), Ok(f32::INFINITY));
        for bad in [
            "0x",
            "0x1",
            "0x1.8",
            "0xp1",
            "0x.p1",
            "0x1p",
            "0x1p+",
            "0x1p3.5",
            "0x1e5",
            "0x1p0x3",
            "0x+1p3",
            "0x1p3x",
            "0x1p3 f",
            "0x1pg",
            "0x1p\u{663}",
            "0xg1p0",
            "0x1,8p1",
        ] {
            assert_eq!(parse_double(bad), Err(ParseError::NotANumber(bad.into())), "{bad:?}");
            assert_eq!(parse_float(bad), Err(ParseError::NotANumber(bad.into())), "{bad:?}");
        }
    }

    #[test]
    fn parse_int_and_boolean() {
        assert_eq!(parse_int("5"), Ok(5));
        assert_eq!(parse_int("-5"), Ok(-5));
        assert_eq!(parse_int("+5"), Ok(5));
        assert_eq!(parse_int("2147483647"), Ok(i32::MAX));
        assert_eq!(parse_int("-2147483648"), Ok(i32::MIN));
        for bad in ["", "-", "2147483648", " 5", "5 ", "1.0", "0x10"] {
            assert_eq!(parse_int(bad), Err(ParseError::NotANumber(bad.into())), "{bad:?}");
        }
        assert!(parse_boolean("true"));
        assert!(parse_boolean("TRUE"));
        assert!(parse_boolean("tRuE"));
        assert!(!parse_boolean("yes"));
        assert!(!parse_boolean(" true"));
        assert!(!parse_boolean(""));
    }
}
