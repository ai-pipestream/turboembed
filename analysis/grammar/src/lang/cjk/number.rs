//! Decimal arithmetic directly over source text, without a normalized copy.
use crate::dec::Dec;
use turbo_unicode::whitespace::WhitespaceMode;

// Nonnegative digits, powers of ten, and syntax markers. The same inventory
// drives the lexicon and this reader so segmentation and arithmetic agree.
pub(super) const WORDS: &[(&str, i64)] = &[
    ("零", 0),
    ("〇", 0),
    ("一", 1),
    ("二", 2),
    ("两", 2),
    ("兩", 2),
    ("三", 3),
    ("四", 4),
    ("五", 5),
    ("六", 6),
    ("七", 7),
    ("八", 8),
    ("九", 9),
    ("十", 10),
    ("百", 100),
    ("千", 1000),
    ("万", 10000),
    ("萬", 10000),
    ("億", 100000000),
    ("亿", 100000000),
    ("兆", 1000000000000),
    ("れい", 0),
    ("ぜろ", 0),
    ("いち", 1),
    ("に", 2),
    ("さん", 3),
    ("よん", 4),
    ("し", 4),
    ("ご", 5),
    ("ろく", 6),
    ("なな", 7),
    ("しち", 7),
    ("はち", 8),
    ("きゅう", 9),
    ("く", 9),
    ("じゅう", 10),
    ("ひゃく", 100),
    ("びゃく", 100),
    ("ぴゃく", 100),
    ("せん", 1000),
    ("ぜん", 1000),
    ("まん", 10000),
    ("おく", 100000000),
    ("ちょう", 1000000000000),
    ("０", 0),
    ("１", 1),
    ("２", 2),
    ("３", 3),
    ("４", 4),
    ("５", 5),
    ("６", 6),
    ("７", 7),
    ("８", 8),
    ("９", 9),
    ("点", -1),
    ("點", -1),
    ("．", -1),
    (".", -1),
    ("，", -2),
    (",", -2),
    ("负", -3),
    ("負", -3),
    ("マイナス", -3),
    ("-", -3),
    ("－", -3),
    ("/", -4),
    ("／", -4),
];

fn white(c: char) -> bool {
    WhitespaceMode::Unicode.is_whitespace(c as u32)
}
fn token(text: &str) -> Option<(usize, i64)> {
    let c = text.chars().next()?;
    if c.is_ascii_digit() {
        return Some((1, i64::from(c as u8 - b'0')));
    }
    WORDS
        .iter()
        .filter(|(word, _)| text.starts_with(word))
        .max_by_key(|(word, _)| word.len())
        .map(|(word, value)| (word.len(), *value))
}

/// Read an exact nonempty CJK numeral, including mixed digit/scaled values.
/// Overflow, repeated signs, malformed grouping and multiple decimals fail.
pub fn parse_number(text: &str) -> Option<Dec> {
    let text = text.trim_matches(white);
    let (negative, text) = match token(text) {
        Some((n, -3)) => (true, text[n..].trim_start_matches(white)),
        _ => (false, text),
    };
    let result = if let Some(at) = text.find(['/', '／']) {
        let width = text[at..].chars().next()?.len_utf8();
        unsigned(&text[..at])?.checked_div(unsigned(&text[at + width..])?)?
    } else {
        unsigned(text)?
    };
    Some(if negative { result.negated() } else { result })
}

fn unsigned(mut text: &str) -> Option<Dec> {
    let mut total = Dec::ZERO;
    let mut section = Dec::ZERO;
    let mut coefficient = 0i128;
    let mut digits = 0u32;
    let mut kana_digits = false;
    let mut fraction = None;
    let mut group = None;
    let mut last_small = 10000;
    let mut last_large = 0;
    let mut seen = false;
    let mut last_digit = false;
    while !text.is_empty() {
        let trimmed = text.trim_start_matches(white);
        if trimmed.len() != text.len() && last_digit && token(trimmed).is_some_and(|(_, v)| (0..10).contains(&v)) {
            return None;
        }
        text = trimmed;
        if text.is_empty() {
            break;
        }
        let (width, value) = token(text)?;
        let kana =
            matches!(turbo_unicode::script::script(text.chars().next()?), turbo_unicode::script::Script::Hiragana);
        text = &text[width..];
        match value {
            0..=9 => {
                if digits > 0 && kana != kana_digits {
                    return None;
                }
                kana_digits = kana;
                coefficient = coefficient.checked_mul(10)?.checked_add(i128::from(value))?;
                digits += 1;
                if let Some(ref mut places) = fraction {
                    *places += 1;
                }
                if let Some(ref mut count) = group {
                    *count += 1;
                }
                last_digit = true;
            }
            -1 if digits > 0 && fraction.is_none() && group.is_none_or(|n| n == 3) => {
                fraction = Some(0);
                group = None;
                last_digit = false;
            }
            -2 if last_digit && fraction.is_none() && group.map_or(digits <= 3, |n| n == 3) => {
                group = Some(0);
                last_digit = false;
            }
            10.. => {
                if fraction == Some(0) || group.is_some_and(|n| n != 3) {
                    return None;
                }
                let current = if digits == 0 { Dec::ONE } else { Dec::new(coefficient, fraction.unwrap_or(0))? };
                let scale = Dec::from_i64(value);
                if value < 10000 {
                    if value >= last_small {
                        return None;
                    }
                    section = section.checked_add(current.checked_mul(scale)?)?;
                    last_small = value;
                } else {
                    let chunk =
                        section.checked_add(if digits == 0 && !section.is_zero() { Dec::ZERO } else { current })?;
                    if value == last_large {
                        return None;
                    }
                    if last_large > 0 && value > last_large {
                        total = total
                            .checked_add(if digits == 0 && section.is_zero() { Dec::ZERO } else { chunk })?
                            .checked_mul(scale)?;
                    } else {
                        total = total.checked_add(chunk.checked_mul(scale)?)?;
                    }
                    section = Dec::ZERO;
                    last_large = value;
                    last_small = 10000;
                }
                coefficient = 0;
                digits = 0;
                fraction = None;
                group = None;
                last_digit = false;
            }
            _ => return None,
        }
        seen = true;
    }
    if !seen || fraction == Some(0) || group.is_some_and(|n| n != 3) {
        return None;
    }
    total.checked_add(section)?.checked_add(Dec::new(coefficient, fraction.unwrap_or(0))?)
}
