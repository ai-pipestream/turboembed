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

mod extended;

pub(crate) use extended::{EXTENDED_START_BYTES, has_non_ascii_nd, is_nd_at, next_candidate_pos, utf8_char_width};
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
/// By default (`opennlp_parity == false`), supports standard `LatinUs` and
/// `LatinEu` grouping, Indian lakh/crore grouping (`1,00,000`), thin and
/// no-break space 3-digit grouping (`U+00A0`, `U+2009`, `U+202F`), Swiss
/// apostrophe 3-digit grouping (`'` and `U+2019`), Arabic thousands (`U+066C`)
/// and decimal (`U+066B`) separators, scientific notation exponents (`1e-3`,
/// `2.5e3`), and non-ASCII Unicode `Nd` decimal digits (such as Arabic-Indic,
/// Devanagari, and Fullwidth digits). Use [`parse_with_parity`] with
/// `opennlp_parity = true` for strict OpenNLP JVM parity.
#[inline]
pub fn parse(
    text: &str,
    start: usize,
    apply_scale: bool,
    notation: NumberNotation,
    coefficient: &mut String,
) -> Option<ScanResult> {
    parse_with_parity(text, start, apply_scale, notation, false, coefficient)
}

/// Reads a number whose first digit is at byte `start`, in `notation`,
/// optionally enforcing strict OpenNLP JVM parity (`opennlp_parity = true`).
#[inline]
pub fn parse_with_parity(
    text: &str,
    start: usize,
    apply_scale: bool,
    notation: NumberNotation,
    opennlp_parity: bool,
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
    let mut scale = 0u32;
    if i == start {
        if opennlp_parity {
            return None;
        }
        let (non_ascii_end, non_ascii_scale) = extended::parse_non_ascii_numeral(text, start, notation, coefficient)?;
        i = non_ascii_end;
        scale = non_ascii_scale;
    } else if opennlp_parity {
        if byte_at(b, i) == group_byte && i - start <= 3 {
            while byte_at(b, i) == group_byte && group_of_three(b, i + 1) {
                for &d in &b[i + 1..i + 4] {
                    push_digit(coefficient, d);
                }
                i += 4;
            }
        }
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
            return None;
        }
        if exponent_at(b, i) || ((next == group || next == decimal) && exponent_at(b, i + 1)) {
            return None;
        }
    } else {
        let lead_len = i - start;
        let mut used_alt_group = false;
        if lead_len <= 3 {
            let at_i = byte_at(b, i);
            if at_i == group_byte {
                while byte_at(b, i) == group_byte && group_of_three(b, i + 1) {
                    for &d in &b[i + 1..i + 4] {
                        push_digit(coefficient, d);
                    }
                    i += 4;
                }
                if i == start + lead_len
                    && group_byte == b','
                    && lead_len <= 2
                    && let Some(end) = extended::try_indian_ascii_groups(b, i, coefficient)
                {
                    i = end;
                }
            } else if matches!(at_i, b'\'' | 0xC2 | 0xD9 | 0xE2) {
                let (end, alt) = extended::consume_alt_ascii_groups(text, i, coefficient);
                i = end;
                used_alt_group = alt;
            }
        }
        let dec_at = byte_at(b, i);
        if (dec_at == decimal_byte || (used_alt_group && dec_at == b',')) && byte_at(b, i + 1).is_ascii_digit() {
            i += 1;
            while byte_at(b, i).is_ascii_digit() {
                push_digit(coefficient, b[i]);
                i += 1;
                scale += 1;
            }
        } else if dec_at == 0xD9 && char_or_space(text, i) == '\u{66B}' && byte_at(b, i + 2).is_ascii_digit() {
            i += 2;
            while byte_at(b, i).is_ascii_digit() {
                push_digit(coefficient, b[i]);
                i += 1;
                scale += 1;
            }
        }
        let next_b = byte_at(b, i);
        if next_b == b'e' || next_b == b'E' {
            if exponent_at(b, i) {
                let exp_end = extended::parse_exponent(text, i, apply_scale, notation, coefficient, &mut scale)?;
                return Some(ScanResult {
                    end: exp_end,
                    numeral_end: exp_end,
                    scale,
                    multiplier: Multiplier::One,
                    marker: ScaleMarker::None,
                });
            }
        } else if next_b == group_byte || next_b == decimal_byte || matches!(next_b, b'\'' | 0xC2 | 0xD9 | 0xE2) {
            let next = char_or_space(text, i);
            if (next == group || next == decimal || unsupported_separator(next))
                && (byte_at(b, i + next.len_utf8()).is_ascii_digit()
                    || exponent_at(b, i + next.len_utf8())
                    || is_nd_at(text, i + next.len_utf8()))
            {
                return None;
            }
        }
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
pub(crate) fn push_digit(coefficient: &mut String, d: u8) {
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
/// with a number before it, so a scan must not start there.
#[inline]
pub fn continues_number(text: &str, index: usize, notation: NumberNotation) -> bool {
    continues_number_with_parity(text, index, notation, false)
}

/// `NumberScan.continuesNumber`, optionally restricted to ASCII digits when
/// `opennlp_parity` is `true`.
#[inline]
pub fn continues_number_with_parity(text: &str, index: usize, notation: NumberNotation, opennlp_parity: bool) -> bool {
    let b = text.as_bytes();
    let (group, decimal) = (notation.group_separator(), notation.decimal_separator());
    let Some((before_at, before)) = char_before(text, index) else {
        return false;
    };
    let has_non_ascii_digit_before = |at: usize| {
        !opennlp_parity && at >= 2 && b[at - 1] >= 0x80 && char_before(text, at).is_some_and(|(_, c)| java::is_digit(c))
    };
    let has_digit_before = |at: usize| digit_before(b, at) || has_non_ascii_digit_before(at);
    let non_ascii_at_index = b.get(index).is_some_and(|&x| x >= 0x80);
    if ((before == group || before == decimal || unsupported_separator(before)) && has_digit_before(before_at))
        || (non_ascii_at_index && matches!(before, '\u{FF0C}' | '\u{FF0E}') && has_non_ascii_digit_before(before_at))
    {
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
    let is_mantissa_digit = if opennlp_parity { mantissa_end.is_ascii_digit() } else { java::is_digit(mantissa_end) };
    is_mantissa_digit || ((mantissa_end == group || mantissa_end == decimal) && has_digit_before(mantissa_at))
}

/// `NumberScan.exponentAt`: an ASCII `e` or `E`, an optional sign, then a
/// digit.
pub(crate) fn exponent_at(b: &[u8], start: usize) -> bool {
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
pub(crate) fn group_of_three(b: &[u8], start: usize) -> bool {
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
pub(crate) fn byte_at(b: &[u8], i: usize) -> u8 {
    b.get(i).copied().unwrap_or(b' ')
}

/// The character starting at byte `i`, or `None` past the end.
#[inline]
pub(crate) fn char_at(text: &str, i: usize) -> Option<char> {
    let &first = text.as_bytes().get(i)?;
    if first.is_ascii() {
        return Some(first as char);
    }
    text.get(i..)?.chars().next()
}

/// The character at byte `i`, or a space past the end, as the Java's
/// `charAt` gives.
#[inline]
pub(crate) fn char_or_space(text: &str, i: usize) -> char {
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
mod tests;
