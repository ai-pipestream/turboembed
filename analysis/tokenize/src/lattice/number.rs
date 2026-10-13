//! The number parsing of the dictionary readers: `Integer.parseInt` in
//! radix 10 and 16, after `StringUtil.trimUnicodeWhitespace` where the
//! Java trims.

use turbo_unicode::java;

/// `Integer.parseInt(s, radix)` for radix 10 or 16: an optional `+` or
/// `-` (not alone), then one or more characters `Character.digit(c,
/// radix)` accepts, within the range of a Java `int`.
///
/// `Character.digit` sees UTF-16 code units, so a digit outside the Basic
/// Multilingual Plane is two units that are not digits. It accepts the
/// decimal digits of every script (general category Nd), and in radix 16
/// also the Latin letters `a` to `f` in either case, ASCII or fullwidth.
pub(crate) fn parse_java_int(s: &str, radix: u32) -> Option<i32> {
    debug_assert!(radix == 10 || radix == 16);
    let (negative, digits) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() {
        return None;
    }
    // Accumulate negatively, as the Java does, so that the most negative
    // value is reachable.
    let mut value: i32 = 0;
    for c in digits.chars() {
        let d = java_digit(c, radix)?;
        value = value.checked_mul(radix as i32)?.checked_sub(d as i32)?;
    }
    if negative { Some(value) } else { value.checked_neg() }
}

/// `Character.digit(c, radix)` for radix 10 or 16, on a character of the
/// Basic Multilingual Plane; `None` for a character outside it.
fn java_digit(c: char, radix: u32) -> Option<u32> {
    let cp = c as u32;
    if cp > 0xFFFF {
        return None;
    }
    if let Some(d) = java::digit(c) {
        return Some(u32::from(d));
    }
    if radix != 16 {
        return None;
    }
    let letter = match cp {
        0x41..=0x46 => cp - 0x41,
        0x61..=0x66 => cp - 0x61,
        0xFF21..=0xFF26 => cp - 0xFF21,
        0xFF41..=0xFF46 => cp - 0xFF41,
        _ => return None,
    };
    Some(10 + letter)
}

/// `s.substring(2)` of a Java string: what is left after the first two
/// UTF-16 code units, or `None` when `s` has fewer than two units or the
/// second unit is the first half of a surrogate pair (the rest would then
/// start with a lone low surrogate, which no number holds).
pub(crate) fn after_two_units(s: &str) -> Option<&str> {
    let mut units = 0;
    for (i, c) in s.char_indices() {
        if units == 2 {
            return Some(&s[i..]);
        }
        units += c.len_utf16();
        if units > 2 {
            return None;
        }
    }
    (units == 2).then_some("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal() {
        assert_eq!(parse_java_int("42", 10), Some(42));
        assert_eq!(parse_java_int("+42", 10), Some(42));
        assert_eq!(parse_java_int("-42", 10), Some(-42));
        assert_eq!(parse_java_int("007", 10), Some(7));
        assert_eq!(parse_java_int("2147483647", 10), Some(i32::MAX));
        assert_eq!(parse_java_int("-2147483648", 10), Some(i32::MIN));
        assert_eq!(parse_java_int("2147483648", 10), None);
        assert_eq!(parse_java_int("-2147483649", 10), None);
        for bad in ["", "+", "-", "1a", "a", "1.0", "++1", " 1", "1 ", "0x10"] {
            assert_eq!(parse_java_int(bad, 10), None, "{bad:?}");
        }
        // Digits of other scripts, but not outside the BMP.
        assert_eq!(parse_java_int("\u{ff11}\u{ff12}", 10), Some(12));
        assert_eq!(parse_java_int("\u{663}\u{660}", 10), Some(30));
        assert_eq!(parse_java_int("\u{1d7cf}", 10), None);
        // Letters are not decimal digits, fullwidth or not.
        assert_eq!(parse_java_int("a", 10), None);
        assert_eq!(parse_java_int("\u{ff21}", 10), None);
    }

    #[test]
    fn hexadecimal() {
        assert_eq!(parse_java_int("3041", 16), Some(0x3041));
        assert_eq!(parse_java_int("fFaA", 16), Some(0xFFAA));
        assert_eq!(parse_java_int("+10", 16), Some(16));
        assert_eq!(parse_java_int("-1", 16), Some(-1));
        assert_eq!(parse_java_int("7fffffff", 16), Some(i32::MAX));
        assert_eq!(parse_java_int("80000000", 16), None);
        assert_eq!(parse_java_int("\u{ff21}\u{ff46}", 16), Some(0xAF));
        assert_eq!(parse_java_int("\u{ff10}\u{ff11}", 16), Some(1));
        assert_eq!(parse_java_int("g", 16), None);
        assert_eq!(parse_java_int("\u{ff27}", 16), None);
        assert_eq!(parse_java_int("", 16), None);
    }

    #[test]
    fn substring_two() {
        assert_eq!(after_two_units("0x41"), Some("41"));
        assert_eq!(after_two_units("0x"), Some(""));
        assert_eq!(after_two_units("0"), None);
        assert_eq!(after_two_units(""), None);
        // A supplementary character is the two units.
        assert_eq!(after_two_units("\u{1f600}41"), Some("41"));
        // The cut falls inside a pair.
        assert_eq!(after_two_units("a\u{1f600}41"), None);
        assert_eq!(after_two_units("ab\u{1f600}"), Some("\u{1f600}"));
    }
}
