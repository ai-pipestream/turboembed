//! Reads one number at a position: a port of OpenNLP's `NumberScan`
//! (`opennlp.tools.extraction`, numeric add-on).
//!
//! The Java walks a `CharSequence` in UTF-16 units and reads a space
//! where an index falls outside the text, so a scan loop needs no bounds
//! checks. Here every offset is a UTF-8 byte offset into a `&str`, on a
//! character boundary, and a read outside the text gives the same space.
//! Each step the Java takes over one unit is a step over one character
//! here. The characters the scan compares against (digits, separators,
//! `e`, signs, the scale letters) are all in the Basic Multilingual Plane,
//! so where the Java would read half of a surrogate pair and match
//! nothing, this reads the whole character and matches nothing too, and
//! both scans stop at the same place.

use turbo_analysis_layers::{Multiplier, NumberNotation, ScaleMarker};
use turbo_unicode::GeneralCategory::{Ll, Lm, Lo, Lt, Lu, Mc, Me, Mn, Nd, Nl, No, Pc, Sc};
use turbo_unicode::WhitespaceMode;
use turbo_unicode::category::general_category;
use turbo_unicode::java;

/// The longest scale word, `thousand` and `trillion`, in UTF-16 units.
const MAX_SCALE_WORD_UNITS: usize = 8;

/// The scale words and what each multiplies by.
const SCALE_WORDS: [(&str, Multiplier); 4] = [
    ("thousand", Multiplier::Thousand),
    ("million", Multiplier::Million),
    ("billion", Multiplier::Billion),
    ("trillion", Multiplier::Trillion),
];

/// One number read by [`parse`]. Its value's digits are in the buffer
/// the caller passed in.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ScanResult {
    /// The byte after the number, its scale marker included.
    pub end: usize,
    /// The byte after the numeral: the digits, group separators and
    /// fraction, before any scale marker.
    pub numeral_end: usize,
    /// How many of the value's digits are the fraction: `BigDecimal.scale()`.
    pub scale: u32,
    pub multiplier: Multiplier,
    pub marker: ScaleMarker,
}

/// Reads a number whose first digit is at byte `start`, in `notation`.
///
/// `coefficient` is cleared and receives the value's digits without
/// leading zeros, `0` for zero, with the multiplier applied: the unscaled
/// value of the `BigDecimal` the Java returns. A caller scanning a whole
/// text passes the same buffer for every number.
///
/// Grouping is strict: a group separator may follow at most three leading
/// digits, and every group after it has exactly three. With `apply_scale`
/// an immediate `k`, `m`, `b` or `bn` (either case), or one `White_Space`
/// character and then `thousand`, `million`, `billion` or `trillion` (any
/// case), multiplies the value; an immediate letter that is none of those
/// fails the scan. Without it the scan stops after the fraction.
///
/// Returns `None` when no ASCII digit is at `start`; when the scan stops
/// at a group or decimal separator, a no-break, narrow no-break or thin
/// space, an apostrophe, a right single quotation mark or an Arabic
/// decimal or thousands separator directly followed by a digit, since the
/// text goes on with a number this scan cannot read (`1,00,000`, `1.2.3`);
/// when an exponent follows (`1e-3`, `1.e5`); and when a scale suffix is
/// not followed by a word boundary.
pub fn parse(
    text: &str,
    start: usize,
    apply_scale: bool,
    notation: NumberNotation,
    coefficient: &mut String,
) -> Option<ScanResult> {
    let b = text.as_bytes();
    let group = notation.group_separator();
    let decimal = notation.decimal_separator();
    let (group_byte, decimal_byte) = (group as u8, decimal as u8);
    coefficient.clear();
    let mut i = start;
    while byte_at(b, i).is_ascii_digit() {
        push_digit(coefficient, b[i]);
        i += 1;
    }
    if i == start {
        return None;
    }
    if byte_at(b, i) == group_byte && i - start <= 3 {
        while byte_at(b, i) == group_byte && group_of_three(b, i + 1) {
            for &d in &b[i + 1..i + 4] {
                push_digit(coefficient, d);
            }
            i += 4;
        }
    }
    let mut scale = 0u32;
    if byte_at(b, i) == decimal_byte && byte_at(b, i + 1).is_ascii_digit() {
        i += 1;
        while byte_at(b, i).is_ascii_digit() {
            push_digit(coefficient, b[i]);
            i += 1;
            scale += 1;
        }
    }
    let next = char_or_space(text, i);
    if (next == group || next == decimal || unsupported_separator(next))
        && byte_at(b, i + next.len_utf8()).is_ascii_digit()
    {
        // The text goes on with a number this scan cannot read; the prefix
        // read so far would be a wrong value.
        return None;
    }
    if exponent_at(b, i) || ((next == group || next == decimal) && exponent_at(b, i + 1)) {
        return None;
    }
    if coefficient.is_empty() {
        coefficient.push('0');
    }
    let unscaled = ScanResult { end: i, numeral_end: i, scale, multiplier: Multiplier::One, marker: ScaleMarker::None };
    if !apply_scale {
        return Some(unscaled);
    }
    let (end, multiplier, marker) = match parse_scale(text, i) {
        Scale::Rejected => return None,
        Scale::Absent => return Some(unscaled),
        Scale::Marked { end, multiplier, marker } => (end, multiplier, marker),
    };
    if coefficient != "0" {
        coefficient.extend(std::iter::repeat_n('0', multiplier.zeros() as usize));
    }
    Some(ScanResult { end, numeral_end: i, scale, multiplier, marker })
}

/// Appends a digit unless it would be a leading zero.
#[inline]
fn push_digit(coefficient: &mut String, d: u8) {
    if d != b'0' || !coefficient.is_empty() {
        coefficient.push(d as char);
    }
}

/// What follows a numeral, when scale markers are read.
enum Scale {
    /// No scale marker: the number ends with its numeral.
    Absent,
    /// A letter directly after the digits that is no scale suffix, or a
    /// suffix with no word boundary after it.
    Rejected,
    Marked {
        end: usize,
        multiplier: Multiplier,
        marker: ScaleMarker,
    },
}

/// `NumberScan.parseScale`: an immediate suffix, or a scale word after
/// one whitespace character, at byte `end`.
fn parse_scale(text: &str, end: usize) -> Scale {
    let next = char_at(text, end);
    if let Some(suffix) = next.filter(|&c| java::is_letter(c)) {
        let multiplier = match suffix {
            'k' | 'K' => Multiplier::Thousand,
            'm' | 'M' => Multiplier::Million,
            'b' | 'B' => Multiplier::Billion,
            _ => return Scale::Rejected,
        };
        let bn = multiplier == Multiplier::Billion && matches!(byte_at(text.as_bytes(), end + 1), b'n' | b'N');
        let suffix_end = end + if bn { 2 } else { 1 };
        if !boundary_after(text, suffix_end) {
            return Scale::Rejected;
        }
        return Scale::Marked { end: suffix_end, multiplier, marker: ScaleMarker::Suffix };
    }
    if whitespace_at(text, end) {
        // Past the end of the text the Java reads a space, and the word
        // scan after it finds nothing.
        let word_start = end + next.map_or(1, char::len_utf8);
        if let Some(word_end) = word_end(text, word_start, MAX_SCALE_WORD_UNITS)
            && let Some(multiplier) = scale_word(&text[word_start..word_end])
        {
            return Scale::Marked { end: word_end, multiplier, marker: ScaleMarker::Word };
        }
    }
    Scale::Absent
}

/// The multiplier a scale word names. The Java lowercases the word code
/// point by code point with `Character.toLowerCase` and compares, so a
/// letter whose simple lowercase mapping is an ASCII letter matches it:
/// `mİllion`, with U+0130, is `million`.
fn scale_word(word: &str) -> Option<Multiplier> {
    SCALE_WORDS.iter().find(|(w, _)| word.chars().map(java::to_lower_char).eq(w.chars())).map(|&(_, m)| m)
}

/// `NumberScan.continuesNumber`: whether a digit at byte `index` goes on
/// with a number before it, so a scan must not start there. That is the
/// case after a separator that follows a digit (the `000` of `1,00,000`)
/// and after an exponent marker, signed or not, that follows a mantissa
/// (the `3` of `1e-3`).
pub fn continues_number(text: &str, index: usize, notation: NumberNotation) -> bool {
    let b = text.as_bytes();
    let (group, decimal) = (notation.group_separator(), notation.decimal_separator());
    let Some((before_at, before)) = char_before(text, index) else {
        return false;
    };
    if (before == group || before == decimal || unsupported_separator(before)) && digit_before(b, before_at) {
        return true;
    }
    let (marker_at, marker) = if before == '+' || before == '-' {
        match char_before(text, before_at) {
            Some(c) => c,
            None => return false,
        }
    } else {
        (before_at, before)
    };
    if marker != 'e' && marker != 'E' {
        return false;
    }
    let Some((mantissa_at, mantissa_end)) = char_before(text, marker_at) else {
        return false;
    };
    mantissa_end.is_ascii_digit()
        || ((mantissa_end == group || mantissa_end == decimal) && digit_before(b, mantissa_at))
}

/// `NumberScan.exponentAt`: an ASCII `e` or `E`, an optional sign, then a
/// digit.
fn exponent_at(b: &[u8], start: usize) -> bool {
    if !matches!(byte_at(b, start), b'e' | b'E') {
        return false;
    }
    let sign = matches!(byte_at(b, start + 1), b'+' | b'-');
    byte_at(b, start + if sign { 2 } else { 1 }).is_ascii_digit()
}

/// `NumberScan.unsupportedSeparator`: a character that groups digits or
/// marks a fraction in some convention, but in neither notation read
/// here.
fn unsupported_separator(c: char) -> bool {
    matches!(c, '\u{A0}' | '\u{2009}' | '\u{202F}' | '\'' | '\u{2019}' | '\u{66B}' | '\u{66C}')
}

/// `NumberScan.groupOfThree`: three digits at `start` and no fourth.
fn group_of_three(b: &[u8], start: usize) -> bool {
    byte_at(b, start).is_ascii_digit()
        && byte_at(b, start + 1).is_ascii_digit()
        && byte_at(b, start + 2).is_ascii_digit()
        && !byte_at(b, start + 3).is_ascii_digit()
}

/// `NumberScan.wordEnd`: the end of the run of letters (`Character.isLetter`)
/// at byte `start`, when it is not empty, at most `max_units` UTF-16 units
/// long and followed by a word boundary. A letter outside the Basic
/// Multilingual Plane counts two units, as in the Java.
pub fn word_end(text: &str, start: usize, max_units: usize) -> Option<usize> {
    let mut i = start;
    let mut units = 0;
    while units < max_units {
        match char_at(text, i) {
            Some(c) if java::is_letter(c) => {
                i += c.len_utf8();
                units += c.len_utf16();
            }
            _ => break,
        }
    }
    (i != start && units <= max_units && boundary_after(text, i)).then_some(i)
}

/// `NumberScan.boundaryBefore`: whether a mention may start at byte
/// `index`, because the character before it, combining marks skipped, is
/// not part of a word or number.
pub fn boundary_before(text: &str, index: usize) -> bool {
    !base_char_before(text, index).is_some_and(word_continuation)
}

/// `NumberScan.signBoundaryBefore`: whether a minus at byte `index` may
/// open a negative number. Besides a letter, digit or mark, a currency
/// symbol or a percent sign before it makes the minus a range dash, so
/// `5%-10%` is two positive numbers.
pub fn sign_boundary_before(text: &str, index: usize) -> bool {
    match base_char_before(text, index) {
        None => true,
        Some(c) => !word_continuation(c) && c != '%' && general_category(c) != Sc,
    }
}

/// `NumberScan.boundaryAfter`: whether a mention may end at byte `index`:
/// the end of the text, or a character that is not part of a word or
/// number.
pub fn boundary_after(text: &str, index: usize) -> bool {
    !char_at(text, index).is_some_and(word_continuation)
}

/// `NumberScan.whitespaceAt`: whether the character at byte `index` has
/// the Unicode `White_Space` property. Outside the text this is true, as
/// the Java's space sentinel is; the part of a mention expected after it
/// then finds nothing.
pub fn whitespace_at(text: &str, index: usize) -> bool {
    WhitespaceMode::Unicode.is_whitespace(char_or_space(text, index) as u32)
}

/// `NumberScan.baseCodePointBefore`: the character before byte `index`,
/// skipping combining marks, or `None` at the start of the text.
pub fn base_char_before(text: &str, index: usize) -> Option<char> {
    text.get(..index)?.chars().rev().find(|&c| !combining_mark(c))
}

/// `NumberScan.wordContinuation`: a letter, a number of any kind, a
/// combining mark or a connector such as `_`.
fn word_continuation(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphanumeric() || c == '_';
    }
    matches!(general_category(c), Lu | Ll | Lt | Lm | Lo | Nd | Nl | No | Mn | Mc | Me | Pc)
}

/// `NumberScan.combiningMark`: a nonspacing, spacing or enclosing mark.
fn combining_mark(c: char) -> bool {
    !c.is_ascii() && matches!(general_category(c), Mn | Mc | Me)
}

/// The byte at `i`, or a space past the end: the Java's `charAt` sentinel
/// for the ASCII comparisons.
#[inline]
fn byte_at(b: &[u8], i: usize) -> u8 {
    b.get(i).copied().unwrap_or(b' ')
}

/// The character starting at byte `i`, or `None` past the end.
#[inline]
fn char_at(text: &str, i: usize) -> Option<char> {
    let &first = text.as_bytes().get(i)?;
    if first.is_ascii() {
        return Some(first as char);
    }
    text.get(i..)?.chars().next()
}

/// The character at byte `i`, or a space past the end, as the Java's
/// `charAt` gives.
#[inline]
fn char_or_space(text: &str, i: usize) -> char {
    char_at(text, i).unwrap_or(' ')
}

/// The character before byte `i` and where it starts, or `None` at the
/// start of the text.
#[inline]
fn char_before(text: &str, i: usize) -> Option<(usize, char)> {
    let c = text.get(..i)?.chars().next_back()?;
    Some((i - c.len_utf8(), c))
}

/// Whether the byte before `i` is an ASCII digit. A digit byte is never
/// part of a longer UTF-8 sequence, so no decoding is needed.
#[inline]
fn digit_before(b: &[u8], i: usize) -> bool {
    i > 0 && b[i - 1].is_ascii_digit()
}

#[cfg(test)]
mod tests {
    use super::*;
    use NumberNotation::{LatinEu, LatinUs};

    /// The value as `BigDecimal.toPlainString` writes it, and the end.
    fn scan(text: &str, apply_scale: bool, notation: NumberNotation) -> Option<(String, usize)> {
        let mut digits = String::new();
        let r = parse(text, 0, apply_scale, notation, &mut digits)?;
        let value = turbo_analysis_layers::DecimalRef { negative: false, digits: &digits, scale: r.scale };
        Some((value.to_plain_string(), r.end))
    }

    #[test]
    fn digits_without_leading_zeros() {
        let mut digits = String::from("left over");
        let r = parse("007", 0, false, LatinUs, &mut digits).unwrap();
        assert_eq!((digits.as_str(), r.scale, r.end), ("7", 0, 3));
        let r = parse("0.050", 0, false, LatinUs, &mut digits).unwrap();
        assert_eq!((digits.as_str(), r.scale), ("50", 3));
        let r = parse("0.00", 0, false, LatinUs, &mut digits).unwrap();
        assert_eq!((digits.as_str(), r.scale), ("0", 2));
        let r = parse("0k", 0, true, LatinUs, &mut digits).unwrap();
        assert_eq!((digits.as_str(), r.scale, r.multiplier), ("0", 0, Multiplier::Thousand));
    }

    #[test]
    fn a_scan_starts_at_a_digit() {
        let mut digits = String::new();
        for text in ["", "x", "-5", ".5", " 5"] {
            assert_eq!(parse(text, 0, true, LatinUs, &mut digits), None, "{text:?}");
        }
        // Past the end of the text.
        assert_eq!(parse("5", 1, true, LatinUs, &mut digits), None);
        assert_eq!(parse("5", 7, true, LatinUs, &mut digits), None);
    }

    #[test]
    fn the_scale_multiplies_and_keeps_the_fraction() {
        assert_eq!(scan("1.5k", true, LatinUs), Some(("1500.0".into(), 4)));
        assert_eq!(scan("1,5k", true, LatinEu), Some(("1500.0".into(), 4)));
        assert_eq!(scan("12k", true, LatinUs), Some(("12000".into(), 3)));
        assert_eq!(scan("12K", true, LatinUs), Some(("12000".into(), 3)));
        assert_eq!(scan("2m", true, LatinUs), Some(("2000000".into(), 2)));
        assert_eq!(scan("5b", true, LatinUs), Some(("5000000000".into(), 2)));
        assert_eq!(scan("5bn", true, LatinUs), Some(("5000000000".into(), 3)));
        assert_eq!(scan("5BN", true, LatinUs), Some(("5000000000".into(), 3)));
        assert_eq!(scan("3 million", true, LatinUs), Some(("3000000".into(), 9)));
        assert_eq!(scan("3 TRILLION", true, LatinUs), Some(("3000000000000".into(), 10)));
        assert_eq!(scan("0.25 Thousand", true, LatinUs), Some(("250.00".into(), 13)));
        // Without scaling the number ends with its digits.
        assert_eq!(scan("12k", false, LatinUs), Some(("12".into(), 2)));
        assert_eq!(scan("3 million", false, LatinUs), Some(("3".into(), 1)));
    }

    #[test]
    fn markers_are_recorded_with_the_numeral_end() {
        let mut digits = String::new();
        let r = parse("1,234 billion.", 0, true, LatinUs, &mut digits).unwrap();
        assert_eq!(
            r,
            ScanResult {
                end: 13,
                numeral_end: 5,
                scale: 0,
                multiplier: Multiplier::Billion,
                marker: ScaleMarker::Word
            }
        );
        let r = parse("7bn)", 0, true, LatinUs, &mut digits).unwrap();
        assert_eq!((r.end, r.numeral_end, r.marker), (3, 1, ScaleMarker::Suffix));
        let r = parse("7 apples", 0, true, LatinUs, &mut digits).unwrap();
        assert_eq!((r.end, r.numeral_end, r.marker, r.multiplier), (1, 1, ScaleMarker::None, Multiplier::One));
    }

    #[test]
    fn a_letter_after_the_digits_must_be_a_scale_suffix() {
        assert_eq!(scan("5x", true, LatinUs), None);
        assert_eq!(scan("5kg", true, LatinUs), None);
        assert_eq!(scan("5bnx", true, LatinUs), None);
        assert_eq!(scan("5k2", true, LatinUs), None);
        assert_eq!(scan("5m_", true, LatinUs), None);
        assert_eq!(scan("5é", true, LatinUs), None);
        // A digit that is not a letter ends the number before it.
        assert_eq!(scan("5\u{0661}m", true, LatinUs), Some(("5".into(), 1)));
        // A combining mark after a suffix is no boundary.
        assert_eq!(scan("5k\u{0301}", true, LatinUs), None);
    }

    #[test]
    fn scale_words_take_one_whitespace_character() {
        assert_eq!(scan("3\u{00A0}million", true, LatinUs), Some(("3000000".into(), 10)));
        assert_eq!(scan("3\u{3000}million", true, LatinUs), Some(("3000000".into(), 11)));
        assert_eq!(scan("3\tthousand", true, LatinUs), Some(("3000".into(), 10)));
        assert_eq!(scan("3  million", true, LatinUs), Some(("3".into(), 1)));
        assert_eq!(scan("3 millions", true, LatinUs), Some(("3".into(), 1)));
        assert_eq!(scan("3 millionaire", true, LatinUs), Some(("3".into(), 1)));
        assert_eq!(scan("3 mil", true, LatinUs), Some(("3".into(), 1)));
        assert_eq!(scan("3 ", true, LatinUs), Some(("3".into(), 1)));
        // U+0085 is White_Space; U+200B is not.
        assert_eq!(scan("3\u{85}million", true, LatinUs), Some(("3000000".into(), 10)));
        assert_eq!(scan("3\u{200B}million", true, LatinUs), Some(("3".into(), 1)));
    }

    #[test]
    fn scale_words_lowercase_by_code_point() {
        // U+0130's simple lowercase mapping is `i`, so Character.toLowerCase
        // turns the word into `million`.
        assert_eq!(scan("3 m\u{0130}LL\u{0130}ON", true, LatinUs), Some(("3000000".into(), 11)));
        // The Kelvin sign lowercases to `k`, which no scale word holds.
        assert_eq!(scan("3 \u{212A}", true, LatinUs), Some(("3".into(), 1)));
    }

    #[test]
    fn grouping_is_strict() {
        assert_eq!(scan("1,234,567.89", false, LatinUs), Some(("1234567.89".into(), 12)));
        assert_eq!(scan("1.234.567,89", false, LatinEu), Some(("1234567.89".into(), 12)));
        assert_eq!(scan("1.234,5", false, LatinEu), Some(("1234.5".into(), 7)));
        // Read in the other notation, 1.234 is a fraction and the comma then
        // goes on with a digit.
        assert_eq!(scan("1.234,5", false, LatinUs), None);
        assert_eq!(scan("1,00,000", false, LatinUs), None);
        assert_eq!(scan("1234,567", false, LatinUs), None);
        assert_eq!(scan("1,2345", false, LatinUs), None);
        assert_eq!(scan("1,23", false, LatinUs), None);
        assert_eq!(scan("1,", false, LatinUs), Some(("1".into(), 1)));
        assert_eq!(scan("1,23x", false, LatinUs), None);
        assert_eq!(scan("1,x", false, LatinUs), Some(("1".into(), 1)));
        // A group that ends at the end of the text, and one cut short.
        assert_eq!(scan("1,234", false, LatinUs), Some(("1234".into(), 5)));
        assert_eq!(scan("1,23", true, LatinUs), None);
        assert_eq!(scan("12,3", true, LatinUs), None);
        assert_eq!(scan("1.", false, LatinUs), Some(("1".into(), 1)));
    }

    #[test]
    fn other_separators_before_a_digit_reject_the_scan() {
        for sep in ['\u{A0}', '\u{2009}', '\u{202F}', '\'', '\u{2019}', '\u{66B}', '\u{66C}'] {
            for notation in [LatinUs, LatinEu] {
                let text = format!("1{sep}234");
                assert_eq!(scan(&text, false, notation), None, "{text:?}");
                assert!(continues_number(&text, 1 + sep.len_utf8(), notation), "{text:?}");
                // Not followed by a digit, the separator ends the number.
                let text = format!("1{sep}x");
                assert_eq!(scan(&text, false, notation), Some(("1".into(), 1)), "{text:?}");
            }
        }
    }

    #[test]
    fn exponents_reject_the_scan() {
        for text in ["1e3", "1E3", "1e-3", "1e+3", "1.5e3", "1.e3", "1,e3"] {
            assert_eq!(scan(text, true, LatinUs), None, "{text:?}");
        }
        assert_eq!(scan("1,e3", true, LatinEu), None);
        assert_eq!(scan("1e", true, LatinUs), None, "a letter that is no suffix");
        assert_eq!(scan("1e", false, LatinUs), Some(("1".into(), 1)));
        assert_eq!(scan("1e-", false, LatinUs), Some(("1".into(), 1)));
        assert_eq!(scan("1e-x", false, LatinUs), Some(("1".into(), 1)));
    }

    #[test]
    fn continuation_after_separators_and_exponents() {
        for notation in [LatinUs, LatinEu] {
            assert!(continues_number("1,00,000", 5, notation));
            assert!(continues_number("1.00.000", 5, notation));
            assert!(continues_number("1.2.3", 4, notation));
            assert!(!continues_number("1 000", 2, notation));
            assert!(!continues_number(",000", 1, notation));
            assert!(!continues_number("id,000", 3, notation));
            assert!(continues_number("1e-3", 3, notation));
            assert!(continues_number("1e3", 2, notation));
            assert!(continues_number("1.e3", 3, notation));
            assert!(continues_number("1,e3", 3, notation));
            assert!(!continues_number("e3", 1, notation));
            assert!(!continues_number("-3", 1, notation));
            assert!(!continues_number("3", 0, notation));
            assert!(!continues_number("a.e3", 3, notation));
        }
    }

    #[test]
    fn bounded_words() {
        // NumericBoundaryTest.testBoundedWord, with UTF-16 offsets turned
        // into bytes: U+1D48E is four bytes and two units.
        assert_eq!(word_end("cat!", 0, 3), Some(3));
        assert_eq!(word_end("cat3", 0, 3), None);
        assert_eq!(word_end("catastrophe", 0, 3), None);
        assert_eq!(word_end("xcat!", 1, 3), Some(4));
        assert_eq!(word_end("\u{1D48E}!", 0, 2), Some(4));
        assert_eq!(word_end("\u{1D48E}!", 0, 1), None);
        assert_eq!(word_end("\u{1D48E}x!", 0, 3), Some(5));
        assert_eq!(word_end("\u{1D48E}x!", 0, 2), None);
        assert_eq!(word_end("cat", 3, 3), None);
        assert_eq!(word_end("cat", 9, 3), None);
        assert_eq!(word_end("\u{3B1}\u{3B2}", 0, 2), Some(4));
    }

    #[test]
    fn boundaries_skip_combining_marks() {
        assert!(boundary_before("", 0));
        assert!(boundary_before(" 5", 1));
        assert!(!boundary_before("a5", 1));
        assert!(!boundary_before("_5", 1));
        assert!(!boundary_before("\u{00B2}5", 2), "an Other_Number before");
        assert!(!boundary_before("\u{2160}5", 3), "a Letter_Number before");
        // The base character before the marks decides.
        assert!(boundary_before(" \u{0301}\u{0302}5", 5));
        assert!(!boundary_before("e\u{0301}5", 3));
        // Marks alone back to the start count as the start.
        assert!(boundary_before("\u{0301}5", 2));
        assert!(boundary_after("5", 1));
        assert!(boundary_after("5 ", 1));
        assert!(!boundary_after("5\u{0301}", 1));
        assert!(!boundary_after("5\u{1D7D9}", 1), "a mathematical digit");
        assert!(boundary_after("5\u{1F4E6}", 1), "an emoji");
    }

    #[test]
    fn sign_boundaries() {
        assert!(sign_boundary_before("-5", 0));
        assert!(sign_boundary_before(" -5", 1));
        assert!(sign_boundary_before("(-5", 1));
        assert!(!sign_boundary_before("a-5", 1));
        assert!(!sign_boundary_before("5-5", 1));
        assert!(!sign_boundary_before("5%-10%", 2));
        assert!(!sign_boundary_before("50\u{20AC}-60", 5));
        assert!(!sign_boundary_before("$-5", 1));
        assert!(!sign_boundary_before("%\u{0301}-5", 3));
    }

    #[test]
    fn whitespace_past_the_end() {
        assert!(whitespace_at("5", 1));
        assert!(whitespace_at("5", 9));
        assert!(whitespace_at("5\u{2009}", 1));
        assert!(!whitespace_at("5x", 1));
    }
}
