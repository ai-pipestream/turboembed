//! The word type of a segment, as OpenNLP's `WordType.of` decides it.
//!
//! An emoji (any `Extended_Pictographic` character or regional indicator)
//! makes the segment `Emoji` at once. Otherwise the first character of a
//! script with its own type (Han, Hiragana, Katakana, Hangul, or one of
//! the Southeast Asian scripts) decides, then a letter makes it
//! `Alphanumeric` and a digit `Numeric`. A segment with none of these
//! (spaces, punctuation, symbols) is not a word.

use turbo_tokenize_layers::WordType;
use turbo_unicode::Script;
use turbo_unicode::java;
use turbo_unicode::word_break::{is_extended_pictographic, is_regional_indicator};

use crate::utf8::decode;

/// No character below this has a script with its own type (the lowest is
/// Thai, U+0E00), so the script lookup is skipped for it.
const LOWEST_SCRIPT_CODE_POINT: u32 = 0x0E00;

/// The type a script gives a word, or `None` for a script with no type of
/// its own (Latin, Greek, Common, ...).
#[inline]
fn script_type(cp: u32) -> Option<WordType> {
    match java::script(cp) {
        Script::Han => Some(WordType::Ideographic),
        Script::Hiragana => Some(WordType::Hiragana),
        Script::Katakana => Some(WordType::Katakana),
        Script::Hangul => Some(WordType::Hangul),
        Script::Thai
        | Script::Lao
        | Script::Myanmar
        | Script::Khmer
        | Script::TaiLe
        | Script::NewTaiLue
        | Script::TaiViet => Some(WordType::SoutheastAsian),
        Script::Other => None,
    }
}

/// The word type of `segment`, or `None` when it is not a word.
pub fn word_type(segment: &str) -> Option<WordType> {
    let bytes = segment.as_bytes();
    let mut letter = false;
    let mut digit = false;
    let mut script = None;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b < 0x80 {
            letter |= b.is_ascii_alphabetic();
            digit |= b.is_ascii_digit();
            i += 1;
            continue;
        }
        let (cp, len) = decode(bytes, i);
        i += len;
        if is_extended_pictographic(cp) || is_regional_indicator(cp) {
            return Some(WordType::Emoji);
        }
        if script.is_none() && cp >= LOWEST_SCRIPT_CODE_POINT {
            script = script_type(cp);
        }
        if java::is_letter(cp) {
            letter = true;
        } else if java::is_digit(cp) {
            digit = true;
        }
    }
    if script.is_some() {
        script
    } else if letter {
        Some(WordType::Alphanumeric)
    } else if digit {
        Some(WordType::Numeric)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letters_and_digits() {
        assert_eq!(word_type("abc"), Some(WordType::Alphanumeric));
        assert_eq!(word_type("a1"), Some(WordType::Alphanumeric));
        assert_eq!(word_type("3.14"), Some(WordType::Numeric));
        assert_eq!(word_type("\u{663}\u{664}"), Some(WordType::Numeric));
        assert_eq!(word_type("\u{3b1}\u{3b2}"), Some(WordType::Alphanumeric));
        assert_eq!(word_type("_"), None);
        assert_eq!(word_type(" "), None);
        assert_eq!(word_type("..."), None);
        assert_eq!(word_type(""), None);
        // Superscript two is No, not a Java digit; the Roman numeral eight is Nl, not a letter.
        assert_eq!(word_type("\u{b2}"), None);
        assert_eq!(word_type("\u{2167}"), None);
    }

    #[test]
    fn scripts() {
        assert_eq!(word_type("\u{4e2d}"), Some(WordType::Ideographic));
        assert_eq!(word_type("\u{3072}"), Some(WordType::Hiragana));
        assert_eq!(word_type("\u{30ab}\u{30fc}"), Some(WordType::Katakana));
        assert_eq!(word_type("\u{d55c}\u{ad6d}"), Some(WordType::Hangul));
        assert_eq!(word_type("\u{e44}\u{e17}\u{e22}"), Some(WordType::SoutheastAsian));
        assert_eq!(word_type("\u{1780}"), Some(WordType::SoutheastAsian));
        // The first script with a type decides, even after Latin letters.
        assert_eq!(word_type("ab\u{30ab}\u{4e2d}"), Some(WordType::Katakana));
        // The prolonged sound mark alone is Common: a letter, so alphanumeric.
        assert_eq!(word_type("\u{30fc}"), Some(WordType::Alphanumeric));
    }

    #[test]
    fn emoji_win() {
        assert_eq!(word_type("\u{1f600}"), Some(WordType::Emoji));
        assert_eq!(word_type("\u{4e2d}\u{1f600}"), Some(WordType::Emoji));
        assert_eq!(word_type("\u{1f1fa}\u{1f1f8}"), Some(WordType::Emoji));
        // Copyright is Extended_Pictographic, so it is kept as an emoji.
        assert_eq!(word_type("\u{a9}"), Some(WordType::Emoji));
        // A keycap's digit is ASCII, and U+20E3 is not pictographic: numeric.
        assert_eq!(word_type("1\u{fe0f}\u{20e3}"), Some(WordType::Numeric));
    }
}
