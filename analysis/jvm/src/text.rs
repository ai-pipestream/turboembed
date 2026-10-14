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

/// Why Java would throw `NumberFormatException`, or what this port does
/// not read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// The text is not a number in Java's grammar; Java throws
    /// `NumberFormatException` with this message.
    NotANumber(String),
    /// The text is a hexadecimal floating point literal such as `0x1p3`,
    /// which Java reads and this port does not.
    HexFloat(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::NotANumber(s) => write!(f, "{s:?} is not a number"),
            ParseError::HexFloat(s) => {
                write!(f, "{s:?} is a hexadecimal floating point literal, which is not read here")
            }
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
/// `F`, `d` and `D`, `NaN` and `Infinity`, and whitespace (characters up to
/// U+0020) trimmed at both ends. Hexadecimal literals are refused with
/// [`ParseError::HexFloat`].
pub fn parse_double(s: &str) -> Result<f64, ParseError> {
    let (negative, body) = decimal_body(s)?;
    let value: f64 = match body {
        Special::NaN => f64::NAN,
        Special::Infinity => f64::INFINITY,
        Special::Decimal(text) => text.parse().map_err(|_| ParseError::NotANumber(s.into()))?,
    };
    Ok(if negative { -value } else { value })
}

/// `Float.parseFloat(s)`, the grammar of [`parse_double`] read into an
/// `f32` with one correctly rounded conversion, as Java does it.
pub fn parse_float(s: &str) -> Result<f32, ParseError> {
    let (negative, body) = decimal_body(s)?;
    let value: f32 = match body {
        Special::NaN => f32::NAN,
        Special::Infinity => f32::INFINITY,
        Special::Decimal(text) => text.parse().map_err(|_| ParseError::NotANumber(s.into()))?,
    };
    Ok(if negative { -value } else { value })
}

enum Special<'a> {
    NaN,
    Infinity,
    Decimal(&'a str),
}

/// Trims, takes the sign, recognizes `NaN` and `Infinity`, refuses a
/// hexadecimal literal and checks a decimal literal against Java's
/// grammar, returning it without its type suffix. The decimal that
/// remains is one Rust's parser reads, and reads to the same value.
fn decimal_body(s: &str) -> Result<(bool, Special<'_>), ParseError> {
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
        "NaN" => return Ok((negative, Special::NaN)),
        "Infinity" => return Ok((negative, Special::Infinity)),
        _ => {}
    }
    if rest.starts_with("0x") || rest.starts_with("0X") {
        return Err(ParseError::HexFloat(s.into()));
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
    Ok((negative, Special::Decimal(body)))
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
        assert_eq!(parse_double("0x1p3"), Err(ParseError::HexFloat("0x1p3".into())));
        assert_eq!(parse_float("0.1"), Ok(0.1f32));
        assert_eq!(parse_float("3.4028236e38"), Ok(f32::INFINITY));
        assert_eq!(parse_float("1.00000001"), Ok(1.0f32));
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
