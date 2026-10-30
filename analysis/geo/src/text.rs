//! The Java string handling the table loaders depend on: line splitting as
//! `BufferedReader.readLine` does it, Unicode `White_Space` trimming and
//! blank tests (`StringUtil.trimUnicodeWhitespace`, `isUnicodeBlank`),
//! `String.split` with a single separator that keeps empty fields,
//! `Long.parseLong`, and the decimal numbers of the coordinate and metric
//! columns.

use turbo_unicode::java::digit;

/// The byte order mark, which OpenNLP removes from the start of a table.
pub(crate) const BYTE_ORDER_MARK: char = '\u{FEFF}';

/// The lines of `text` as `BufferedReader.readLine` returns them: a line
/// ends at a line feed, a carriage return, or a carriage return followed by
/// a line feed, and the text after the last line end is a line when it is
/// not empty. Each item is the line's one-based number, its byte offset in
/// `text`, and the line without its end.
pub(crate) struct Lines<'a> {
    text: &'a str,
    at: usize,
    number: usize,
}

impl<'a> Lines<'a> {
    pub(crate) fn new(text: &'a str) -> Lines<'a> {
        Lines { text, at: 0, number: 0 }
    }
}

impl<'a> Iterator for Lines<'a> {
    type Item = (usize, usize, &'a str);

    fn next(&mut self) -> Option<(usize, usize, &'a str)> {
        let rest = &self.text.as_bytes()[self.at..];
        if rest.is_empty() {
            return None;
        }
        let start = self.at;
        let (len, end_len) = match rest.iter().position(|&b| b == b'\n' || b == b'\r') {
            None => (rest.len(), 0),
            Some(i) if rest[i] == b'\r' && rest.get(i + 1) == Some(&b'\n') => (i, 2),
            Some(i) => (i, 1),
        };
        self.at = start + len + end_len;
        self.number += 1;
        Some((self.number, start, &self.text[start..start + len]))
    }
}

pub(crate) use turbo_unicode::whitespace::{is_blank, trim};

/// The text [`parse_long`] refused, where Java throws a
/// `NumberFormatException`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NotALong;

/// `Long.parseLong(s)`: an optional `+` or `-` and one or more decimal
/// digits in the range of an `i64`. Like Java it takes the decimal digits
/// of every script (`Character.digit`), but only those in the Basic
/// Multilingual Plane, since Java reads the text one UTF-16 unit at a time
/// and the half of a surrogate pair is no digit. Nothing is trimmed.
pub(crate) fn parse_long(s: &str) -> Result<i64, NotALong> {
    let (negative, digits) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        Some(_) => (false, s),
        None => return Err(NotALong),
    };
    if digits.is_empty() {
        return Err(NotALong);
    }
    // Accumulate negatively, as Java does, so i64::MIN parses.
    let mut value: i64 = 0;
    for c in digits.chars() {
        let d = if (c as u32) <= 0xFFFF { digit(c) } else { None }.ok_or(NotALong)?;
        value = value.checked_mul(10).and_then(|v| v.checked_sub(i64::from(d))).ok_or(NotALong)?;
    }
    if negative { Ok(value) } else { value.checked_neg().ok_or(NotALong) }
}

/// A decimal number as the tables write one: an optional `+` or `-`, ASCII
/// digits with at most one `.` and at least one digit, and an optional
/// exponent (`e` or `E`, an optional sign, one or more digits). Nothing is
/// trimmed, and `NaN`, `Infinity`, hexadecimal literals and type suffixes
/// are refused. The value is the correctly rounded `f64`; a magnitude past
/// its range reads as an infinity, which the caller refuses with its own
/// error.
pub(crate) fn parse_decimal(s: &str) -> Option<f64> {
    let bytes = s.as_bytes();
    let mut i = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    let mut digits = 0;
    let mut point = false;
    while let Some(&b) = bytes.get(i) {
        match b {
            b'0'..=b'9' => digits += 1,
            b'.' if !point => point = true,
            _ => break,
        }
        i += 1;
    }
    if digits == 0 {
        return None;
    }
    if let Some(b'e' | b'E') = bytes.get(i) {
        i += 1;
        if let Some(b'+' | b'-') = bytes.get(i) {
            i += 1;
        }
        let start = i;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        if i == start {
            return None;
        }
    }
    if i != bytes.len() {
        return None;
    }
    s.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(text: &str) -> Vec<&str> {
        Lines::new(text).map(|(_, _, l)| l).collect()
    }

    #[test]
    fn lines_end_at_lf_cr_and_crlf() {
        assert_eq!(lines("a\nb\r\nc\rd"), ["a", "b", "c", "d"]);
        assert_eq!(lines("a\n\nb\n"), ["a", "", "b"]);
        assert_eq!(lines("\r\r\n"), ["", ""]);
        assert!(lines("").is_empty());
        let numbered: Vec<(usize, usize)> = Lines::new("x\r\ny\nz").map(|(n, at, _)| (n, at)).collect();
        assert_eq!(numbered, [(1, 0), (2, 3), (3, 5)]);
    }

    #[test]
    fn trimming_and_blank_use_unicode_white_space() {
        assert_eq!(trim("\u{A0}\u{3000} Paris\t\u{2029}"), "Paris");
        // U+200B is not White_Space.
        assert_eq!(trim("\u{200B}x"), "\u{200B}x");
        assert!(is_blank(" \u{85}\u{A0}"));
        assert!(!is_blank("\u{FEFF}"));
        assert!(is_blank(""));
    }

    #[test]
    fn longs_parse_as_java_parses_them() {
        assert_eq!(parse_long("42"), Ok(42));
        assert_eq!(parse_long("+42"), Ok(42));
        assert_eq!(parse_long("-0"), Ok(0));
        assert_eq!(parse_long("9223372036854775807"), Ok(i64::MAX));
        assert_eq!(parse_long("-9223372036854775808"), Ok(i64::MIN));
        assert_eq!(parse_long("9223372036854775808"), Err(NotALong));
        assert_eq!(parse_long("\u{0661}\u{0662}"), Ok(12));
        assert_eq!(parse_long(" 1"), Err(NotALong));
        assert_eq!(parse_long("1.0"), Err(NotALong));
        assert_eq!(parse_long("-"), Err(NotALong));
        assert_eq!(parse_long(""), Err(NotALong));
        // MATHEMATICAL BOLD DIGIT ONE is outside the BMP.
        assert_eq!(parse_long("\u{1D7CF}"), Err(NotALong));
    }

    #[test]
    fn decimals_take_digits_a_point_and_an_exponent_only() {
        assert_eq!(parse_decimal("48.8566"), Some(48.8566));
        assert_eq!(parse_decimal("-90"), Some(-90.0));
        assert_eq!(parse_decimal("+1.5e3"), Some(1500.0));
        assert_eq!(parse_decimal("1."), Some(1.0));
        assert_eq!(parse_decimal(".5"), Some(0.5));
        assert_eq!(parse_decimal("2E-2"), Some(0.02));
        assert_eq!(parse_decimal("-0"), Some(-0.0));
        assert_eq!(parse_decimal("1e400"), Some(f64::INFINITY));
        assert_eq!(parse_decimal(""), None);
        assert_eq!(parse_decimal(" 1"), None);
        assert_eq!(parse_decimal("1 "), None);
        let refused = "+ - . 1.2.3 1e 1e+ 1d 1f 0x1p3 NaN Infinity inf -infinity nan 1_000 \u{0661} \u{FF11} e5";
        for text in refused.split(' ') {
            assert_eq!(parse_decimal(text), None, "{text:?}");
        }
    }
}
