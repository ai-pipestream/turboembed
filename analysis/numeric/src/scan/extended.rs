//! Extended numeral scanning beyond OpenNLP's strict ASCII Latin US/EU rules:
//!
//! - Indian lakh/crore comma grouping (`1,00,000`, `12,34,567.89`).
//! - Thin and no-break space 3-digit grouping (`U+00A0`, `U+2009`, `U+202F`).
//! - Swiss apostrophe 3-digit grouping (`'` and `U+2019`).
//! - Arabic thousands (`U+066C`) and decimal (`U+066B`) separators.
//! - Scientific notation exponents (`1e3`, `1E-03`, `2.5e-3`, `2,500e+3`).
//! - Non-ASCII Unicode decimal digit scripts (`Nd`, such as Arabic-Indic
//!   `U+0660..=U+0669`, Extended Arabic-Indic `U+06F0..=U+06F9`, Devanagari
//!   `U+0966..=U+096F`, and Fullwidth `U+FF10..=U+FF19`), requiring all digits
//!   in a numeral to belong to the same 10-digit block.

use turbo_analysis_layers::NumberNotation;
use turbo_unicode::java;

use super::{boundary_after, byte_at, char_at, char_or_space, exponent_at, group_of_three, push_digit};

/// Upper bound on scientific notation exponent magnitude (`10^10000`) so a
/// pathological input cannot trigger unbounded allocation.
const MAX_EXPONENT: i32 = 10_000;

/// Lookup table for bytes that can open a candidate mention when extended
/// scanning is active: ASCII `-`, ASCII `'0'..='9'`, and the UTF-8 lead bytes
/// of every Unicode `Nd` decimal digit block (`DECIMAL_ZEROS`).
pub(crate) static EXTENDED_START_BYTES: [bool; 256] = {
    let mut table = [false; 256];
    table[b'-' as usize] = true;
    let mut d = b'0';
    while d <= b'9' {
        table[d as usize] = true;
        d += 1;
    }
    // Every non-ASCII `Nd` character in Unicode has one of these 8 UTF-8 lead
    // bytes (`U+0660` -> 0xD9, `U+06F0` -> 0xDB, `U+07C0` -> 0xDF,
    // `U+0966..=U+1C59` -> 0xE0/0xE1, `U+A620..=U+ABF9` -> 0xEA,
    // `U+FF10..=U+FF19` -> 0xEF, supplementary planes -> 0xF0).
    let leads: [u8; 8] = [0xD9, 0xDB, 0xDF, 0xE0, 0xE1, 0xEA, 0xEF, 0xF0];
    let mut k = 0;
    while k < leads.len() {
        table[leads[k] as usize] = true;
        k += 1;
    }
    table
};

/// Whether byte `x` is one of the 8 UTF-8 lead bytes that can start a
/// non-ASCII Unicode `Nd` decimal digit (`0xD9, 0xDB, 0xDF, 0xE0, 0xE1, 0xEA, 0xEF, 0xF0`).
#[inline]
pub(crate) fn is_nd_lead_byte(x: u8) -> bool {
    matches!(x, 0xD9 | 0xDB | 0xDF | 0xE0 | 0xE1 | 0xEA | 0xEF | 0xF0)
}

/// Whether `c` is an alternative 3-digit group separator (thin/no-break space,
/// Swiss apostrophe, or Arabic thousands separator).
#[inline]
pub(crate) fn is_alt_group_sep(c: char) -> bool {
    matches!(c, '\u{A0}' | '\u{2009}' | '\u{202F}' | '\'' | '\u{2019}' | '\u{66C}')
}

/// Two ASCII digits at `start` followed by `,`.
#[inline]
fn group_of_two_comma(b: &[u8], start: usize) -> bool {
    byte_at(b, start).is_ascii_digit() && byte_at(b, start + 1).is_ascii_digit() && byte_at(b, start + 2) == b','
}

/// Tries to read Indian lakh/crore comma grouping (`1,00,000`, `12,34,567`)
/// starting at the first comma `i` after 1 or 2 leading ASCII digits.
///
/// Requires one or more `,dd` (2-digit) groups followed by a final `,ddd`
/// (3-digit) group not followed by another comma-and-digit.
pub(crate) fn try_indian_ascii_groups(b: &[u8], i: usize, coefficient: &mut String) -> Option<usize> {
    if byte_at(b, i) != b',' || !group_of_two_comma(b, i + 1) {
        return None;
    }
    let mut pos = i;
    while byte_at(b, pos) == b',' && group_of_two_comma(b, pos + 1) {
        pos += 3;
    }
    if byte_at(b, pos) != b',' || !group_of_three(b, pos + 1) {
        return None;
    }
    let end = pos + 4;
    if byte_at(b, end) == b',' && byte_at(b, end + 1).is_ascii_digit() {
        return None;
    }
    let mut cur = i;
    while cur < pos {
        push_digit(coefficient, b[cur + 1]);
        push_digit(coefficient, b[cur + 2]);
        cur += 3;
    }
    for &d in &b[pos + 1..pos + 4] {
        push_digit(coefficient, d);
    }
    Some(end)
}

/// Consumes consecutive 3-digit ASCII groups separated by `sep` (`\u{00A0}`,
/// `\u{2009}`, `\u{202F}`, `'`, `\u{2019}`, or `\u{066C}`) starting at `i`.
pub(crate) fn consume_alt_ascii_groups(text: &str, mut i: usize, coefficient: &mut String) -> (usize, bool) {
    let b = text.as_bytes();
    let sep = char_or_space(text, i);
    if !is_alt_group_sep(sep) {
        return (i, false);
    }
    let sep_len = sep.len_utf8();
    if !group_of_three(b, i + sep_len) {
        return (i, false);
    }
    while char_or_space(text, i) == sep && group_of_three(b, i + sep_len) {
        for &d in &b[i + sep_len..i + sep_len + 3] {
            push_digit(coefficient, d);
        }
        i += sep_len + 3;
    }
    (i, true)
}

/// Parses an ASCII scientific notation exponent (`e`/`E`, optional `+`/`-`,
/// and ASCII digits) at `i`, adjusting `coefficient` and `scale`.
pub(crate) fn parse_exponent(
    text: &str,
    i: usize,
    apply_scale: bool,
    notation: NumberNotation,
    coefficient: &mut String,
    scale: &mut u32,
) -> Option<usize> {
    let b = text.as_bytes();
    let sign_byte = byte_at(b, i + 1);
    let neg_exp = sign_byte == b'-';
    let has_sign = neg_exp || sign_byte == b'+';
    let exp_start = i + 1 + usize::from(has_sign);
    let mut exp_end = exp_start;
    let mut exp_val: i32 = 0;
    while byte_at(b, exp_end).is_ascii_digit() {
        let d = i32::from(b[exp_end] - b'0');
        exp_val = exp_val.checked_mul(10)?.checked_add(d)?;
        if exp_val > MAX_EXPONENT {
            return None;
        }
        exp_end += 1;
    }
    if exp_end == exp_start {
        return None;
    }
    let next = char_or_space(text, exp_end);
    let (group, decimal) = (notation.group_separator(), notation.decimal_separator());
    if exponent_at(b, exp_end)
        || ((next == group || next == decimal || is_alt_group_sep(next) || next == '\u{66B}')
            && (char_at(text, exp_end + next.len_utf8()).is_some_and(java::is_digit)
                || exponent_at(b, exp_end + next.len_utf8())))
    {
        return None;
    }
    if apply_scale && !boundary_after(text, exp_end) {
        return None;
    }
    let signed_exp = if neg_exp { -exp_val } else { exp_val };
    if coefficient.is_empty() {
        *scale = if signed_exp >= 0 {
            scale.saturating_sub(signed_exp as u32)
        } else {
            scale.checked_add(signed_exp.unsigned_abs())?
        };
        coefficient.push('0');
    } else {
        let adj = i64::from(*scale) - i64::from(signed_exp);
        if adj >= 0 {
            *scale = u32::try_from(adj).ok()?;
        } else {
            let zeros = usize::try_from(-adj).ok()?;
            coefficient.extend(std::iter::repeat_n('0', zeros));
            *scale = 0;
        }
    }
    Some(exp_end)
}

/// Checks if the character at `pos` is a digit in the 10-digit Unicode block
/// starting at `zero_base`, returning its canonical ASCII digit (`b'0'..=b'9'`)
/// and UTF-8 byte length.
#[inline]
fn digit_in_script(text: &str, pos: usize, zero_base: u32) -> Option<(u8, usize)> {
    let c = char_at(text, pos)?;
    let d = (c as u32).checked_sub(zero_base)?;
    (d < 10).then(|| (b'0' + d as u8, c.len_utf8()))
}

/// Three digits in `zero_base` starting at `pos`, and no fourth digit in
/// `zero_base` after them.
fn script_group_of_three(text: &str, pos: usize, zero_base: u32) -> Option<([u8; 3], usize)> {
    let (d0, l0) = digit_in_script(text, pos, zero_base)?;
    let (d1, l1) = digit_in_script(text, pos + l0, zero_base)?;
    let (d2, l2) = digit_in_script(text, pos + l0 + l1, zero_base)?;
    let total = l0 + l1 + l2;
    if digit_in_script(text, pos + total, zero_base).is_some() {
        return None;
    }
    Some(([d0, d1, d2], total))
}

/// Two digits in `zero_base` starting at `pos`, followed by `,`.
fn script_group_of_two_comma(text: &str, pos: usize, zero_base: u32) -> Option<([u8; 2], usize)> {
    let (d0, l0) = digit_in_script(text, pos, zero_base)?;
    let (d1, l1) = digit_in_script(text, pos + l0, zero_base)?;
    let total = l0 + l1;
    (byte_at(text.as_bytes(), pos + total) == b',').then_some(([d0, d1], total))
}

/// Reads a numeral starting with a non-ASCII Unicode `Nd` digit at `start`,
/// converting its digits into canonical ASCII digits in `coefficient` and
/// returning `(numeral_end, scale)`.
pub(crate) fn parse_non_ascii_numeral(
    text: &str,
    start: usize,
    notation: NumberNotation,
    coefficient: &mut String,
) -> Option<(usize, u32)> {
    let first_char = char_at(text, start)?;
    if first_char.is_ascii() {
        return None;
    }
    let first_d = java::digit(first_char)?;
    let zero_base = (first_char as u32) - u32::from(first_d);
    let mut i = start;
    let mut lead_count = 0usize;
    while let Some((ascii_d, len)) = digit_in_script(text, i, zero_base) {
        push_digit(coefficient, ascii_d);
        i += len;
        lead_count += 1;
    }
    let (group, decimal) = (notation.group_separator(), notation.decimal_separator());
    let next_sep = char_or_space(text, i);
    let is_group = next_sep == group || is_alt_group_sep(next_sep) || (group == ',' && next_sep == '\u{FF0C}');
    if lead_count <= 3 && is_group {
        let sep_len = next_sep.len_utf8();
        if script_group_of_three(text, i + sep_len, zero_base).is_some() {
            while char_or_space(text, i) == next_sep
                && let Some((digits, group_len)) = script_group_of_three(text, i + sep_len, zero_base)
            {
                for d in digits {
                    push_digit(coefficient, d);
                }
                i += sep_len + group_len;
            }
        } else if notation == NumberNotation::LatinUs && lead_count <= 2 && next_sep == ',' {
            let mut pos = i;
            let mut pairs: Vec<[u8; 2]> = Vec::new();
            while byte_at(text.as_bytes(), pos) == b','
                && let Some((pair, len)) = script_group_of_two_comma(text, pos + 1, zero_base)
            {
                pairs.push(pair);
                pos += 1 + len;
            }
            if !pairs.is_empty()
                && byte_at(text.as_bytes(), pos) == b','
                && let Some((last_three, len3)) = script_group_of_three(text, pos + 1, zero_base)
            {
                let end = pos + 1 + len3;
                if !(byte_at(text.as_bytes(), end) == b',' && char_at(text, end + 1).is_some_and(java::is_digit)) {
                    for pair in pairs {
                        push_digit(coefficient, pair[0]);
                        push_digit(coefficient, pair[1]);
                    }
                    for d in last_three {
                        push_digit(coefficient, d);
                    }
                    i = end;
                }
            }
        }
    }
    let mut scale = 0u32;
    let dec_candidate = char_or_space(text, i);
    let is_dec =
        dec_candidate == decimal || dec_candidate == '\u{66B}' || (decimal == '.' && dec_candidate == '\u{FF0E}');
    if is_dec && digit_in_script(text, i + dec_candidate.len_utf8(), zero_base).is_some() {
        i += dec_candidate.len_utf8();
        while let Some((ascii_d, len)) = digit_in_script(text, i, zero_base) {
            push_digit(coefficient, ascii_d);
            i += len;
            scale += 1;
        }
    }
    let next = char_or_space(text, i);
    if (next == group
        || next == decimal
        || is_alt_group_sep(next)
        || matches!(next, '\u{66B}' | '\u{FF0C}' | '\u{FF0E}'))
        && char_at(text, i + next.len_utf8()).is_some_and(java::is_digit)
    {
        return None;
    }
    Some((i, scale))
}
