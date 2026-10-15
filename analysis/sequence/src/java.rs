//! The few `java.lang.String` and `java.lang.Double` operations the
//! feature generators and the beam search need, on Rust text.
//!
//! OpenNLP builds its features with `substring`, `length` and
//! `StringUtil.toLowerCase`, all of which count UTF-16 code units, so a
//! prefix of a token can end between the two halves of a surrogate pair.
//! The functions here write their result as Java modified UTF-8, the form
//! a model keys its predicates by, where such a half is three bytes like
//! any other unit; nothing is lost.

use std::cmp::Ordering;

use turbo_jvm::chars::Utf16Units;
use turbo_unicode::java::code_point::to_lower_case;
use turbo_unicode::modified_utf8::push_modified_utf8;

/// Appends `s.substring(start, end)`, with `start` and `end` counted in
/// UTF-16 code units, as modified UTF-8.
///
/// # Panics
///
/// If `end` is beyond the last unit of `s` or `start > end`, as Java
/// throws `StringIndexOutOfBoundsException`.
pub(crate) fn push_units(out: &mut Vec<u8>, s: &str, start: usize, end: usize) {
    assert!(start <= end, "substring({start}, {end})");
    if s.is_ascii() {
        assert!(end <= s.len(), "substring({start}, {end}) of a string of fewer units");
        let slice = &s.as_bytes()[start..end];
        if !slice.contains(&0) {
            out.extend_from_slice(slice);
            return;
        }
    }
    let mut n = 0;
    for unit in Utf16Units::new(s).skip(start).take(end - start) {
        push_modified_utf8(out, u32::from(unit.unit));
        n += 1;
    }
    assert_eq!(n, end - start, "substring({start}, {end}) of a string of fewer units");
}

/// Appends `StringUtil.toLowerCase(s)`: each code point mapped by
/// `Character.toLowerCase(int)`, as modified UTF-8.
pub(crate) fn push_lower(out: &mut Vec<u8>, s: &str) {
    for c in s.chars() {
        push_modified_utf8(out, to_lower_case(c as u32));
    }
}

/// `StringUtil.toLowerCase(s)` into a Rust string. The input holds no
/// lone surrogates, so neither does the result.
pub(crate) fn lowercase_into(out: &mut String, s: &str) {
    for c in s.chars() {
        out.push(char::from_u32(to_lower_case(c as u32)).expect("a lowercase mapping is a character"));
    }
}

/// Appends `StringUtil.toLowerCase` of a string given as UTF-16 code
/// units, which may hold lone surrogates; a lone surrogate is its own
/// code point and maps to itself, as `String.codePoints()` reports it.
pub(crate) fn push_lower_units(out: &mut Vec<u8>, units: &[u16]) {
    for cp in code_points(units) {
        push_modified_utf8(out, to_lower_case(cp));
    }
}

/// The code points of UTF-16 units as `String.codePoints()` reports them:
/// a high surrogate followed by a low one is one code point, any other
/// surrogate is a code point of its own.
pub(crate) fn code_points(units: &[u16]) -> impl Iterator<Item = u32> + '_ {
    let mut i = 0;
    std::iter::from_fn(move || {
        let u = *units.get(i)?;
        i += 1;
        if is_high(u)
            && let Some(&low) = units.get(i)
            && is_low(low)
        {
            i += 1;
            return Some(0x1_0000 + ((u32::from(u) - 0xD800) << 10) + (u32::from(low) - 0xDC00));
        }
        Some(u32::from(u))
    })
}

#[inline]
pub(crate) fn is_high(unit: u16) -> bool {
    (0xD800..0xDC00).contains(&unit)
}

#[inline]
pub(crate) fn is_low(unit: u16) -> bool {
    (0xDC00..0xE000).contains(&unit)
}

/// `Double.compare(a, b)`: numeric order, except that -0.0 is below 0.0
/// and every NaN is equal to every other NaN and above positive infinity.
/// This is the order `Arrays.sort(double[])` sorts by.
pub fn compare_doubles(a: f64, b: f64) -> Ordering {
    if a < b {
        Ordering::Less
    } else if a > b {
        Ordering::Greater
    } else {
        // Equal or unordered: compare doubleToLongBits, which maps every
        // NaN to the one canonical NaN.
        let bits = |x: f64| if x.is_nan() { 0x7FF8_0000_0000_0000i64 } else { x.to_bits() as i64 };
        bits(a).cmp(&bits(b))
    }
}

/// `Integer.parseInt(s)`: an optional `+` or `-`, then one or more
/// decimal digits of any script (`Character.digit(c, 10)`), within the
/// `int` range. `None` where Java throws `NumberFormatException`.
pub(crate) fn parse_int(s: &str) -> Option<i32> {
    let i = parse_long(s)?;
    i32::try_from(i).ok()
}

/// `Long.parseLong(s)`, as [`parse_int`] for the `long` range.
pub(crate) fn parse_long(s: &str) -> Option<i64> {
    let (negative, digits) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() {
        return None;
    }
    // Accumulate negatively, as Java does, so that the most negative
    // value parses.
    let mut value: i64 = 0;
    for c in digits.chars() {
        let d = turbo_unicode::java::digit(c)?;
        value = value.checked_mul(10)?.checked_sub(i64::from(d))?;
    }
    if negative { Some(value) } else { value.checked_neg() }
}

/// The lines of a text as `BufferedReader.readLine` returns them: split at
/// `\n`, `\r` and `\r\n`, with no empty line after a final line end.
pub(crate) fn lines(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        match rest.find(['\n', '\r']) {
            Some(end) => {
                let line = &rest[..end];
                let skip = if rest[end..].starts_with("\r\n") { 2 } else { 1 };
                rest = &rest[end + skip..];
                Some(line)
            }
            None => {
                let line = rest;
                rest = "";
                Some(line)
            }
        }
    })
}

/// `StringUtil.split(input, separator)`: the fields between separators,
/// with trailing empty fields removed; an empty input is one empty field.
pub(crate) fn split_trimmed(input: &str, separator: char) -> Vec<&str> {
    if input.is_empty() {
        return vec![""];
    }
    let mut fields: Vec<&str> = input.split(separator).collect();
    while fields.last().is_some_and(|f| f.is_empty()) {
        fields.pop();
    }
    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    fn units(s: &str, start: usize, end: usize) -> Vec<u8> {
        let mut out = Vec::new();
        push_units(&mut out, s, start, end);
        out
    }

    #[test]
    fn substrings_count_utf16_units() {
        assert_eq!(units("abc", 1, 3), b"bc");
        // ASCII containing a NUL byte encodes NUL as two bytes (C0 80) in
        // modified UTF-8 rather than taking the raw memcpy path.
        assert_eq!(units("a\0b", 0, 3), b"a\xC0\x80b");
        // U+1F600 is the two units D83D DE00; the first alone is three
        // bytes of modified UTF-8.
        assert_eq!(units("a\u{1F600}", 0, 2), b"a\xED\xA0\xBD");
        assert_eq!(units("a\u{1F600}", 0, 3), b"a\xED\xA0\xBD\xED\xB8\x80");
        assert_eq!(units("\u{E9}", 0, 1), "\u{E9}".as_bytes());
        assert_eq!(units("", 0, 0), b"");
    }

    #[test]
    #[should_panic(expected = "fewer units")]
    fn a_substring_past_the_end_panics() {
        units("ab", 0, 3);
    }

    #[test]
    fn lowercasing_maps_each_code_point() {
        let mut out = Vec::new();
        push_lower(&mut out, "A\u{130}\u{10400}");
        let mut want = Vec::new();
        for cp in ['a' as u32, 'i' as u32, 0x10428] {
            push_modified_utf8(&mut want, cp);
        }
        assert_eq!(out, want);
        let mut lone = Vec::new();
        push_lower_units(&mut lone, &[0xD801, 0x41, 0xD801, 0xDC00]);
        let mut want = Vec::new();
        for cp in [0xD801, 'a' as u32, 0x10428] {
            push_modified_utf8(&mut want, cp);
        }
        assert_eq!(lone, want);
    }

    #[test]
    fn integers_parse_as_java_does() {
        assert_eq!(parse_int("42"), Some(42));
        assert_eq!(parse_int("+7"), Some(7));
        assert_eq!(parse_int("-2147483648"), Some(i32::MIN));
        assert_eq!(parse_int("2147483648"), None);
        assert_eq!(parse_int(" 1"), None);
        assert_eq!(parse_int("-"), None);
        assert_eq!(parse_int(""), None);
        // Arabic-Indic digits are decimal digits to Java.
        assert_eq!(parse_int("\u{663}\u{664}"), Some(34));
        assert_eq!(parse_long("-9223372036854775808"), Some(i64::MIN));
    }

    #[test]
    fn lines_split_as_read_line_does() {
        assert_eq!(lines("a\nb\r\nc\rd").collect::<Vec<_>>(), ["a", "b", "c", "d"]);
        assert_eq!(lines("a\n\n").collect::<Vec<_>>(), ["a", ""]);
        assert_eq!(lines("").count(), 0);
        assert_eq!(split_trimmed("a  b ", ' '), ["a", "", "b"]);
        assert_eq!(split_trimmed("", ' '), [""]);
        assert_eq!(split_trimmed("  ", ' '), Vec::<&str>::new());
    }

    #[test]
    fn doubles_compare_as_java_does() {
        assert_eq!(compare_doubles(-0.0, 0.0), Ordering::Less);
        assert_eq!(compare_doubles(f64::NAN, f64::INFINITY), Ordering::Greater);
        assert_eq!(compare_doubles(f64::NAN, -f64::NAN), Ordering::Equal);
        assert_eq!(compare_doubles(1.0, 2.0), Ordering::Less);
        assert_eq!(compare_doubles(2.0, 2.0), Ordering::Equal);
    }
}
