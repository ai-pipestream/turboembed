//! The Java string handling the table loaders depend on: line splitting as
//! `BufferedReader.readLine` does it, Unicode `White_Space` trimming and
//! blank tests (`StringUtil.trimUnicodeWhitespace`, `isUnicodeBlank`),
//! `String.split` with a single separator that keeps empty fields, and
//! `Long.parseLong`.

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

/// `StringUtil.isUnicodeBlank`: empty, or Unicode `White_Space` only.
pub(crate) fn is_blank(s: &str) -> bool {
    s.chars().all(char::is_whitespace)
}

/// `StringUtil.trimUnicodeWhitespace`: `s` without leading and trailing
/// Unicode `White_Space`. Rust's `char::is_whitespace` is that property.
pub(crate) fn trim(s: &str) -> &str {
    s.trim_matches(char::is_whitespace)
}

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
}
