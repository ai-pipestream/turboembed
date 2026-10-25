//! Boundary checks shared by the scanners, OpenNLP's `Boundaries`, over
//! UTF-8 byte offsets.
//!
//! The Java walks UTF-16 units and asks ASCII questions of single units,
//! and `Character.isLetterOrDigit` of whole code points. A byte offset
//! inside a multibyte character holds a continuation byte, which is no
//! ASCII character, as a unit inside a surrogate pair is none; so asking
//! the same ASCII questions of bytes gives the same answers. Every offset
//! a check is asked about follows or precedes an ASCII character, so it is
//! a character boundary.

use turbo_unicode::java::code_point;

/// `Character.isLetterOrDigit(int)`.
#[inline]
pub fn letter_or_digit(c: char) -> bool {
    code_point::is_letter_or_digit(c as u32)
}

/// `Character.isWhitespace(int)`.
#[inline]
pub fn whitespace(c: char) -> bool {
    code_point::is_whitespace(c as u32)
}

/// The character that ends at byte `at`, when `at` is a character boundary.
#[inline]
pub fn before(text: &str, at: usize) -> Option<char> {
    text.get(..at)?.chars().next_back()
}

/// The character that starts at byte `at`, when `at` is a character boundary.
#[inline]
pub fn after(text: &str, at: usize) -> Option<char> {
    text.get(at..)?.chars().next()
}

/// `Boundaries.onWordStart`: `start` is a character boundary with no
/// letter or digit before it.
#[inline]
pub fn on_word_start(text: &str, start: usize) -> bool {
    text.is_char_boundary(start) && !before(text, start).is_some_and(letter_or_digit)
}

/// `Boundaries.onNumberStart`: `start` is a character boundary with no
/// letter or digit before it, and no `.` or `,` directly after an ASCII
/// digit, so a number is not read from inside a decimal or a grouped
/// number.
#[inline]
pub fn on_number_start(text: &str, start: usize) -> bool {
    if !text.is_char_boundary(start) {
        return false;
    }
    let Some(previous) = before(text, start) else { return true };
    if letter_or_digit(previous) {
        return false;
    }
    (previous != '.' && previous != ',') || start < 2 || !text.as_bytes()[start - 2].is_ascii_digit()
}

/// `Boundaries.onEnd`: `end` is a character boundary with no letter or
/// digit at `end`.
#[inline]
pub fn on_end(text: &str, end: usize) -> bool {
    text.is_char_boundary(end) && !after(text, end).is_some_and(letter_or_digit)
}

/// `Boundaries.onEndBefore`: [`on_end`], and a run of `separator` at
/// `end` is not followed by a letter or digit.
pub fn on_end_before(text: &str, end: usize, separator: u8) -> bool {
    if !on_end(text, end) {
        return false;
    }
    let b = text.as_bytes();
    let mut p = end;
    while p < b.len() && b[p] == separator {
        p += 1;
    }
    p == end || on_end(text, p)
}

/// Whether `text` holds `literal` at byte `at`.
#[inline]
pub(crate) fn starts_with(text: &[u8], at: usize, literal: &[u8]) -> bool {
    text.get(at..at + literal.len()) == Some(literal)
}

/// `Ascii.hexValue` for a byte known to be a hexadecimal digit.
#[inline]
pub(crate) fn hex_value(c: u8) -> u32 {
    match c {
        b'0'..=b'9' => u32::from(c - b'0'),
        b'a'..=b'f' => u32::from(c - b'a' + 10),
        _ => u32::from(c - b'A' + 10),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_starts_look_at_whole_characters() {
        assert!(on_word_start("x", 0));
        assert!(on_word_start(" x", 1));
        assert!(!on_word_start("ax", 1));
        assert!(!on_word_start("éx", 2));
        // U+1D400, a supplementary letter, as codePointBefore reads it.
        assert!(!on_word_start("\u{1D400}x", 4));
        assert!(on_word_start("\u{1F600}x", 4));
    }

    #[test]
    fn number_starts_refuse_decimals_and_groups() {
        assert!(on_number_start(" 5", 1));
        assert!(!on_number_start("1.5", 2));
        assert!(!on_number_start("1,5", 2));
        assert!(on_number_start("a.5", 2));
        assert!(on_number_start(".5", 1));
        assert!(!on_number_start("x5", 1));
        assert!(!on_number_start("٣5", 2));
    }

    #[test]
    fn ends_and_separator_runs() {
        assert!(on_end("12", 2));
        assert!(!on_end("12a", 2));
        assert!(!on_end("12é", 2));
        assert!(on_end_before("1.2.", 3, b'.'));
        assert!(!on_end_before("1.2..3", 3, b'.'));
        assert!(on_end_before("1.2.. x", 3, b'.'));
        assert!(!on_end_before("1.2x", 3, b'.'));
    }

    #[test]
    fn hex_values() {
        assert_eq!(hex_value(b'0'), 0);
        assert_eq!(hex_value(b'a'), 10);
        assert_eq!(hex_value(b'F'), 15);
        assert!(starts_with(b"abc", 1, b"bc"));
        assert!(!starts_with(b"abc", 2, b"cd"));
    }

    #[test]
    fn boundaries_cases() {
        for (text, at, want) in [
            ("abc", 0, true),
            ("a bc", 2, true),
            ("a-bc", 2, true),
            ("(abc", 1, true),
            ("abc", 1, false),
            ("1abc", 1, false),
            ("a.bc", 2, true),
        ] {
            assert_eq!(on_word_start(text, at), want, "{text}");
        }
        for (text, at, want) in [
            ("123", 0, true),
            ("a 123", 2, true),
            ("a123", 1, false),
            ("1.23", 2, false),
            ("1,23", 2, false),
            (". 23", 2, true),
            (".23", 1, true),
            ("a. 23", 3, true),
        ] {
            assert_eq!(on_number_start(text, at), want, "{text}");
        }
        for (text, at, want) in
            [("abc", 3, true), ("abc.", 3, true), ("abc ", 3, true), ("abcd", 3, false), ("abc9", 3, false)]
        {
            assert_eq!(on_end(text, at), want, "{text}");
        }
        for (text, want) in [("1.2.3.4", true), ("1.2.3.4.5", false), ("1.2.3.4.", true), ("1.2.3.4 and", true)] {
            assert_eq!(on_end_before(text, 7, b'.'), want, "{text}");
        }
        assert!(!on_word_start("\u{10400}abc", 4));
        assert!(!on_number_start("\u{10400}123", 4));
        assert!(!on_end("abc\u{10400}", 3));
        assert!(on_end("abc\u{1F600}", 3));
        assert!(on_end("", 0) && on_word_start("", 0) && on_number_start("", 0));
    }
}
