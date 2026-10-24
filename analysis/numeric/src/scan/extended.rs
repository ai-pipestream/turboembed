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

/// Pure-arithmetic check on 3 adjacent bytes `(x0, x1, x2)` that is `true` for
/// every non-ASCII Unicode `Nd` decimal digit and uses only branchless bitwise
/// operations so LLVM auto-vectorizes it across 32-byte SIMD lanes.
#[inline]
fn may_be_nd_triple(x0: u8, x1: u8, x2: u8) -> bool {
    let lead = (x0 >= 0xD9) & (x0.wrapping_sub(0xE2) > 7);
    let e0_ef = (x1 >= 0xA5) & (x1.max(x2) >= 0xA6) & (x1 != 0xB8) & (x2 >= 0x90) & (x2 != 0x9A);
    let other = (x0 != 0xE0) & (x0 != 0xEF) & ((x1 >= 0xA0) | (x0 != 0xD9)) & ((x1 != 0x9F) | (x2 >= 0xA0));
    lead & (e0_ef | other)
}

#[inline]
fn has_nd_lead_candidate(b: &[u8]) -> bool {
    let (chunks, rem) = b.as_chunks::<32>();
    for c in chunks {
        let mut any = false;
        for &x in c {
            any |= x >= 0xD9;
        }
        if any {
            return true;
        }
    }
    rem.iter().any(|&x| x >= 0xD9)
}

/// Fast SIMD-vectorized first pass that reports whether `text` contains any
/// non-ASCII Unicode `Nd` decimal digit, falling back to `main`'s ASCII start
/// scan whenever `text` has no non-ASCII `Nd` digits.
#[inline]
pub(crate) fn has_non_ascii_nd(text: &str) -> bool {
    let b = text.as_bytes();
    let len = b.len();
    if len < 2 || !has_nd_lead_candidate(b) {
        return false;
    }
    let mut i = 0;
    while i + 18 <= len {
        while i + 18 <= len {
            let w = &b[i..i + 18];
            let (s0, s1, s2) = (&w[0..16], &w[1..17], &w[2..18]);
            let mut any = false;
            for k in 0..16 {
                any |= may_be_nd_triple(s0[k], s1[k], s2[k]);
            }
            if any {
                break;
            }
            i += 16;
        }
        if i + 18 > len {
            break;
        }
        for k in 0..16 {
            if is_nd_at(text, i + k) {
                return true;
            }
        }
        i += 16;
    }
    while i + 2 < len {
        if may_be_nd_triple(b[i], b[i + 1], b[i + 2]) && is_nd_at(text, i) {
            return true;
        }
        i += 1;
    }
    is_nd_at(text, len - 2)
}

/// Finds the next byte offset `>= start` that can open a number candidate:
/// ASCII `'-'`, ASCII `'0'..='9'`, or (when `!ascii_only`) a non-ASCII
/// Unicode `Nd` decimal digit.
#[inline(always)]
pub(crate) fn next_candidate_pos(text: &str, start: usize, ascii_only: bool) -> Option<usize> {
    if ascii_only {
        let b = text.as_bytes();
        return b[start..].iter().position(|&x| x == b'-' || x.is_ascii_digit()).map(|s| start + s);
    }
    next_candidate_pos_extended(text, start)
}

#[inline(always)]
fn next_candidate_pos_extended(text: &str, mut i: usize) -> Option<usize> {
    let b = text.as_bytes();
    while let Some(rel) = b[i..].iter().position(|&x| EXTENDED_START_BYTES[x as usize]) {
        let pos = i + rel;
        if b[pos] == b'-' || b[pos].is_ascii_digit() || is_nd_at(text, pos) {
            return Some(pos);
        }
        i = pos + utf8_char_width(b[pos]);
    }
    None
}

/// Branchless UTF-8 byte length of the character starting with lead byte `b0`.
#[inline]
pub(crate) fn utf8_char_width(b0: u8) -> usize {
    1 + usize::from(b0 >= 0x80) + usize::from(b0 >= 0xE0) + usize::from(b0 >= 0xF0)
}

/// Whether `text` has a non-ASCII Unicode `Nd` decimal digit at byte `pos`.
///
/// Checks the UTF-8 lead and continuation bytes before touching the Unicode
/// category trie so non-digit Devanagari, Thai, CJK punctuation, and emoji
/// sequences starting with `0xE0`, `0xE1`, `0xEA`, `0xEF`, or `0xF0` are
/// rejected in 1-2 byte comparisons.
#[inline]
pub(crate) fn is_nd_at(text: &str, pos: usize) -> bool {
    let b = text.as_bytes();
    let Some(&b0) = b.get(pos) else { return false };
    match b0 {
        0xD9 => b.get(pos + 1).is_some_and(|&b1| (0xA0..=0xA9).contains(&b1)),
        0xDB => b.get(pos + 1).is_some_and(|&b1| (0xB0..=0xB9).contains(&b1)),
        0xDF => b.get(pos + 1).is_some_and(|&b1| (0x80..=0x89).contains(&b1)),
        0xEF => b.get(pos + 1..pos + 3).is_some_and(|s| s[0] == 0xBC && (0x90..=0x99).contains(&s[1])),
        0xE0 => b.get(pos + 1..pos + 3).is_some_and(|s| {
            let (b1, b2) = (s[0], s[1]);
            ((0xA5..=0xB7).contains(&b1) && (b1 & 1 == 1) && (0xA6..=0xAF).contains(&b2))
                || (matches!(b1, 0xB9 | 0xBB) && (0x90..=0x99).contains(&b2))
                || (b1 == 0xBC && (0xA0..=0xA9).contains(&b2))
        }),
        0xE1 => {
            b.get(pos + 1)
                .is_some_and(|&b1| matches!(b1, 0x81 | 0x82 | 0x9F | 0xA0 | 0xA5 | 0xA7 | 0xAA | 0xAD | 0xAE | 0xB1))
                && char_at(text, pos).is_some_and(java::is_digit)
        }
        0xEA => {
            b.get(pos + 1).is_some_and(|&b1| matches!(b1, 0x98 | 0xA3 | 0xA4 | 0xA7 | 0xA9 | 0xAF))
                && char_at(text, pos).is_some_and(java::is_digit)
        }
        0xF0 => {
            b.get(pos + 1..pos + 3).is_some_and(|s| {
                matches!(s[0], 0x90 | 0x91 | 0x96 | 0x9C | 0x9D | 0x9E) || (s[0] == 0x9F && s[1] == 0xAF)
            }) && char_at(text, pos).is_some_and(java::is_digit)
        }
        _ => false,
    }
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
