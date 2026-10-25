//! The `java.lang` and `StringUtil` string operations the Java reader and
//! stemmer use, on `&str`.
//!
//! Java strings are UTF-16. The Java code compares lengths in UTF-16 units
//! in a few places ([`utf16_len`]); everywhere else it moves by whole code
//! points or compares whole strings, which UTF-8 does the same way.

use turbo_unicode::category::{GeneralCategory, general_category};
use turbo_unicode::{java, props};

/// `String.length()`: the number of UTF-16 units.
pub(crate) use turbo_unicode::utf16::utf16_len;

/// `StringUtil.isWhitespace(char)` in its default mode: the Unicode
/// `White_Space` property. No supplementary character has it, so a
/// surrogate half is never whitespace and testing whole characters gives
/// the same answer.
pub(crate) fn is_whitespace(c: char) -> bool {
    props::white_space(c)
}

/// The reader's `trim`: removes [`is_whitespace`] characters at both ends.
pub(crate) fn trim(s: &str) -> &str {
    s.trim_matches(is_whitespace)
}

/// The reader's `split`: the runs of characters between [`is_whitespace`]
/// characters, without empty ones.
pub(crate) fn split_fields(line: &str) -> Vec<&str> {
    line.split(is_whitespace).filter(|f| !f.is_empty()).collect()
}

/// The reader's `splitLines`: a line ends at a line feed, a carriage return
/// or a carriage return followed by a line feed. Text ending in a line
/// terminator has an empty last line, as in Java.
pub(crate) fn split_lines(content: &str) -> Vec<&str> {
    let bytes = content.as_bytes();
    let mut lines = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i <= bytes.len() {
        if i == bytes.len() || bytes[i] == b'\n' || bytes[i] == b'\r' {
            lines.push(&content[start..i]);
            if i + 1 < bytes.len() && bytes[i] == b'\r' && bytes[i + 1] == b'\n' {
                i += 1;
            }
            start = i + 1;
        }
        i += 1;
    }
    lines
}

/// `Integer.parseInt(s)`: an optional `+` or `-` and at least one decimal
/// digit of any script (`Character.digit(c, 10)`), within the range of an
/// `int`.
pub(crate) fn parse_int(s: &str) -> Option<i32> {
    let (negative, digits) = match s.as_bytes().first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() {
        return None;
    }
    let mut value: i64 = 0;
    for c in digits.chars() {
        // Character.digit(char, 10) reads one UTF-16 unit, so a digit
        // outside the Basic Multilingual Plane is two units, neither a digit.
        if java::is_supplementary(c) {
            return None;
        }
        value = value * 10 + i64::from(java::digit(c)?);
        if value > i64::from(i32::MAX) + 1 {
            return None;
        }
    }
    let value = if negative { -value } else { value };
    i32::try_from(value).ok()
}

/// A non-empty run of ASCII digits: the reader's `isCount`.
pub(crate) fn is_count(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// `Character.isLowerCase(int)`: the `Lowercase` property.
pub(crate) fn is_lower(c: char) -> bool {
    props::lowercase(c)
}

/// `Character.isUpperCase(int)`: the `Uppercase` property.
pub(crate) fn is_upper(c: char) -> bool {
    props::uppercase(c)
}

/// `Character.isTitleCase(int)`: general category Lt.
pub(crate) fn is_title(c: char) -> bool {
    general_category(c) == GeneralCategory::Lt
}

/// `Character.toLowerCase(int)`.
pub(crate) fn lower_char(c: char) -> char {
    java::to_lower_char(c)
}

/// `Character.toUpperCase(int)`.
pub(crate) fn upper_char(c: char) -> char {
    java::to_upper_char(c)
}

/// `StringUtil.toLowerCase`: `Character.toLowerCase(int)` on each code
/// point, appended to `out`.
pub(crate) use turbo_unicode::java::to_lowercase_code_points as push_lowercase;

/// `StringUtil.toUpperCase` on one character: `Character.toUpperCase(char)`
/// on each UTF-16 unit, so a character outside the Basic Multilingual
/// Plane, two surrogate units, is left as it is.
pub(crate) fn upper_unit(c: char) -> char {
    if java::is_supplementary(c) {
        return c;
    }
    let u = upper_char(c);
    // Character.toUpperCase(char) cannot return a character outside the
    // BMP; where the code point mapping would, the char is kept.
    if java::is_supplementary(u) { c } else { u }
}

/// The reader's `removeIgnored`: `text` without the characters of
/// `ignored` (an `IGNORE` declaration), borrowed when none occurs.
pub(crate) fn remove_ignored<'a>(text: &'a str, ignored: &str) -> std::borrow::Cow<'a, str> {
    if ignored.is_empty() || !text.chars().any(|c| ignored.contains(c)) {
        return text.into();
    }
    text.chars().filter(|&c| !ignored.contains(c)).collect::<String>().into()
}

/// `String.hashCode()`: the polynomial hash of the UTF-16 units.
pub(crate) fn java_hash(s: &str) -> i32 {
    let mut h: i32 = 0;
    let mut buf = [0u16; 2];
    for c in s.chars() {
        for &u in c.encode_utf16(&mut buf).iter() {
            h = h.wrapping_mul(31).wrapping_add(i32::from(u));
        }
    }
    h
}

/// The bucket of `hash` in a `java.util.HashMap` table of `capacity`
/// buckets: the hash with its high half folded into the low half.
pub(crate) fn hash_map_bucket(hash: i32, capacity: usize) -> usize {
    let h = hash as u32;
    ((h ^ (h >> 16)) as usize) & (capacity - 1)
}

/// The table size of a `java.util.HashMap` made with `new HashMap<>()`
/// after `keys` distinct keys were put into it: 16, doubled whenever the
/// size passes three quarters of it.
pub(crate) fn hash_map_capacity(keys: usize) -> usize {
    let mut capacity = 16;
    while keys > capacity / 4 * 3 {
        capacity *= 2;
    }
    capacity
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_parse_as_java_parses_them() {
        assert_eq!(parse_int("12"), Some(12));
        assert_eq!(parse_int("+7"), Some(7));
        assert_eq!(parse_int("-0"), Some(0));
        assert_eq!(parse_int("2147483647"), Some(i32::MAX));
        assert_eq!(parse_int("-2147483648"), Some(i32::MIN));
        assert_eq!(parse_int("2147483648"), None);
        assert_eq!(parse_int("\u{663}"), Some(3));
        assert_eq!(parse_int(""), None);
        assert_eq!(parse_int("-"), None);
        assert_eq!(parse_int("1a"), None);
        assert_eq!(parse_int("99999999999999999999"), None);
    }

    #[test]
    fn lines_split_as_java_splits_them() {
        assert_eq!(split_lines("a\r\nb\rc\n"), ["a", "b", "c", ""]);
        assert_eq!(split_lines(""), [""]);
        assert_eq!(split_fields(" a\u{a0}b\t c "), ["a", "b", "c"]);
        assert_eq!(trim("\u{2003} x \u{85}"), "x");
    }

    #[test]
    fn hashes_match_java() {
        assert_eq!(java_hash(""), 0);
        assert_eq!(java_hash("a"), 97);
        assert_eq!(java_hash("hello"), 99162322);
        assert_eq!(java_hash("\u{1F600}"), 0xD83D * 31 + 0xDE00);
        assert_eq!(hash_map_capacity(0), 16);
        assert_eq!(hash_map_capacity(12), 16);
        assert_eq!(hash_map_capacity(13), 32);
        assert_eq!(hash_map_bucket(99162322, 16), ((99162322u32 ^ (99162322u32 >> 16)) & 15) as usize);
    }

    #[test]
    fn upper_case_by_unit() {
        assert_eq!(upper_unit('a'), 'A');
        assert_eq!(upper_unit('\u{10428}'), '\u{10428}');
        assert_eq!(upper_char('\u{10428}'), '\u{10400}');
    }
}
