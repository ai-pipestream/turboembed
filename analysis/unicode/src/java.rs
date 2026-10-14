//! `java.lang.Character` and `String.toLowerCase(Locale.ROOT)` semantics
//! on this crate's Unicode data.
//!
//! The OpenNLP normalizers are written against these JDK methods. The
//! functions here follow the JDK's definitions (which categories count as
//! letters, which spaces count as whitespace) but read the Unicode 18.0
//! tables of this crate, so for characters assigned after the JDK's own
//! Unicode version the answers can differ from a given JVM.
//!
//! The functions here take a `char`. [`code_point`] has the classes the
//! segmentation ports need on a code point given as `u32`, which may be a
//! lone surrogate code unit, with `getType`'s numbering and
//! `UnicodeScript.of`.

pub mod code_point;

use crate::ascii::push_lowercase_run;
use crate::category::{GeneralCategory, general_category};
use crate::props::bits;
use crate::tables::case::{
    LOWER_KEYS, LOWER_VALUES, SPECIAL_LOWER_KEYS, SPECIAL_LOWER_VALUES, UPPER_KEYS, UPPER_VALUES,
};
use crate::tables::category::DECIMAL_ZEROS;
use crate::tables::props::{HAS_LOWER, HAS_SPECIAL_LOWER, HAS_UPPER};
use crate::trie::find;

const CAPITAL_SIGMA: char = '\u{3A3}';
const FINAL_SIGMA: char = '\u{3C2}';
const SMALL_SIGMA: char = '\u{3C3}';

/// `Character.isLetter`: general category Lu, Ll, Lt, Lm or Lo.
#[inline]
pub fn is_letter(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    general_category(c).is_letter()
}

/// `Character.isLetterOrDigit`: a letter or a decimal digit (Nd).
#[inline]
pub fn is_letter_or_digit(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphanumeric();
    }
    let category = general_category(c);
    category.is_letter() || category == GeneralCategory::Nd
}

/// `Character.isDigit`: general category Nd.
#[inline]
pub fn is_digit(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    general_category(c) == GeneralCategory::Nd
}

/// `Character.digit(c, 10)` for a decimal digit: the digit value of an Nd
/// character, `None` for every other character. Unlike the JDK method
/// this does not accept the Latin letters of larger radixes.
pub fn digit(c: char) -> Option<u8> {
    if c.is_ascii() {
        return c.is_ascii_digit().then(|| c as u8 - b'0');
    }
    if general_category(c) != GeneralCategory::Nd {
        return None;
    }
    // Every Nd character is in a run of ten that starts at a zero, so its
    // value is its distance from the closest zero at or below it.
    let cp = c as u32;
    let zero = DECIMAL_ZEROS[DECIMAL_ZEROS.partition_point(|&z| z <= cp) - 1];
    Some((cp - zero) as u8)
}

/// `Character.isWhitespace`: a space separator, line separator or
/// paragraph separator other than the no-break spaces U+00A0, U+2007 and
/// U+202F, or one of U+0009 to U+000D and U+001C to U+001F. U+0085 NEXT
/// LINE is not whitespace by this definition.
#[inline]
pub fn is_whitespace(c: char) -> bool {
    match c {
        ' ' | '\t'..='\r' | '\u{1C}'..='\u{1F}' => true,
        '\u{A0}' | '\u{2007}' | '\u{202F}' => false,
        _ if c.is_ascii() => false,
        _ => general_category(c).is_separator(),
    }
}

/// `Character.isSpaceChar`: general category Zs, Zl or Zp.
#[inline]
pub fn is_space_char(c: char) -> bool {
    general_category(c).is_separator()
}

/// `Character.isSupplementaryCodePoint`: `c` is above U+FFFF and takes
/// two UTF-16 code units.
#[inline]
pub fn is_supplementary(c: char) -> bool {
    c as u32 > 0xFFFF
}

/// `Character.toLowerCase(int)`: the simple lowercase mapping, `c` itself
/// when it has none.
#[inline]
pub fn to_lower_char(c: char) -> char {
    if c.is_ascii() {
        return c.to_ascii_lowercase();
    }
    if bits(c) & HAS_LOWER == 0 {
        return c;
    }
    find(&LOWER_KEYS, &LOWER_VALUES, c as u32).unwrap_or(c)
}

/// `Character.toUpperCase(int)`: the simple uppercase mapping, `c` itself
/// when it has none.
#[inline]
pub fn to_upper_char(c: char) -> char {
    if c.is_ascii() {
        return c.to_ascii_uppercase();
    }
    if bits(c) & HAS_UPPER == 0 {
        return c;
    }
    find(&UPPER_KEYS, &UPPER_VALUES, c as u32).unwrap_or(c)
}

/// `String.toLowerCase(Locale.ROOT)`, appended to `out`: the simple
/// lowercase mapping of each character, except that the unconditional
/// lowercase mappings of SpecialCasing.txt apply (U+0130 becomes
/// `i\u{307}`) and U+03A3 becomes final sigma U+03C2 where the JDK's final
/// sigma condition holds ([`is_final_sigma`] says how that differs from
/// Unicode's Final_Sigma). The language-specific mappings of SpecialCasing.txt
/// (Turkish, Azeri, Lithuanian) are not applied.
pub fn to_lowercase(text: &str, out: &mut String) {
    out.reserve(text.len());
    let mut i = 0;
    while i < text.len() {
        let ascii_end = push_lowercase_run(text, i, out);
        if ascii_end > i {
            i = ascii_end;
            continue;
        }
        let c = text[i..].chars().next().expect("i is at a character boundary");
        push_lower(text, i, c, out);
        i += c.len_utf8();
    }
}

/// What `String.toLowerCase(Locale.ROOT)` writes for one character.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lowercase {
    /// One character, which is the input character itself when it has no
    /// lowercase mapping.
    Char(char),
    /// More than one character (U+0130 becomes `i\u{307}`).
    Str(&'static str),
}

/// The lowercase mapping [`to_lowercase`] applies to the character `c`
/// that starts at byte `at` of `text`. The text is needed only for U+03A3
/// GREEK CAPITAL LETTER SIGMA, whose mapping depends on the characters
/// around it (the JDK's final sigma condition); every other character maps the
/// same wherever it stands. Calling this for each character of a text and
/// writing the results gives the same string as [`to_lowercase`], so a
/// caller can lowercase character by character and see which ones change.
pub fn lowercase_at(text: &str, at: usize, c: char) -> Lowercase {
    debug_assert!(text[at..].starts_with(c), "c is not the character at byte {at}");
    if c == CAPITAL_SIGMA {
        return Lowercase::Char(if is_final_sigma(text, at) { FINAL_SIGMA } else { SMALL_SIGMA });
    }
    if bits(c) & HAS_SPECIAL_LOWER != 0
        && let Some(lower) = find(&SPECIAL_LOWER_KEYS, &SPECIAL_LOWER_VALUES, c as u32)
    {
        return Lowercase::Str(lower);
    }
    Lowercase::Char(to_lower_char(c))
}

fn push_lower(text: &str, at: usize, c: char, out: &mut String) {
    match lowercase_at(text, at, c) {
        Lowercase::Char(lower) => out.push(lower),
        Lowercase::Str(lower) => out.push_str(lower),
    }
}

/// Whether `String.toLowerCase(Locale.ROOT)` writes the sigma at byte
/// `at` as final sigma. The JDK's condition is not Unicode's Final_Sigma:
/// it walks the word the sigma is in, as the root locale's word
/// `BreakIterator` delimits it, backwards until a cased character (every
/// other character is passed over, not only the case-ignorable ones), and
/// then forwards, where any cased character before the word's end makes
/// the sigma non-final. So "ΑΣ1" and "Α1Σ" end in final sigma, "ΑΣ1a" and
/// "ΑΣ.a" do not, and a space, a comma or an ideograph after the sigma
/// ends the word where Unicode would look on.
fn is_final_sigma(text: &str, at: usize) -> bool {
    let after = at + CAPITAL_SIGMA.len_utf8();
    cased_before_in_word(&text[..at]) && !cased_after_in_word(&text[after..])
}

/// `isCased` of the JDK's conditional casing: the three cased letter
/// categories and a fixed list of other characters (modifier letters,
/// U+0345, the Roman numerals, the circled letters), which is an older
/// Unicode's Other_Uppercase and Other_Lowercase, not the current ones.
fn java_cased(c: char) -> bool {
    matches!(general_category(c), GeneralCategory::Lu | GeneralCategory::Ll | GeneralCategory::Lt)
        || matches!(
            c as u32,
            0x02B0..=0x02B8 | 0x02C0..=0x02C1 | 0x02E0..=0x02E4 | 0x0345 | 0x037A | 0x1D2C..=0x1D61 | 0x2160..=0x217F | 0x24B6..=0x24E9
        )
}

/// What a character is to the root locale's word `BreakIterator`, as far
/// as the sigma condition needs: the iterator's rules predate UAX #29.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WordClass {
    /// Nonspacing and enclosing marks and format characters attach to the
    /// character before them.
    Ignorable,
    /// Letters and spacing marks, except the kanji, hiragana and katakana
    /// ranges, which make words of their own.
    Letter,
    /// Every numeric category.
    Digit,
    /// Everything else ends a word, unless it is one of the characters
    /// allowed between two letters (`.`, `'`, `"`, the dashes, the
    /// connectors, the soft hyphen and U+2027) or between two digits
    /// (`.`, `'`, `"`, `,` and U+066B), which [`mid_word`] and [`mid_num`]
    /// tell apart.
    Other,
}

fn word_class(c: char) -> WordClass {
    if matches!(c as u32, 0x3005 | 0x4E00..=0x9FA5 | 0xF900..=0xFA2D | 0x3099..=0x309C | 0x30A1..=0x30FE | 0x3041..=0x309E)
    {
        return WordClass::Other;
    }
    use GeneralCategory::*;
    match general_category(c) {
        Mn | Me | Cf => WordClass::Ignorable,
        Lu | Ll | Lt | Lm | Lo | Mc => WordClass::Letter,
        Nd | Nl | No => WordClass::Digit,
        _ => WordClass::Other,
    }
}

fn mid_word(c: char) -> bool {
    matches!(c, '.' | '\'' | '"' | '\u{AD}' | '\u{2027}')
        || matches!(general_category(c), GeneralCategory::Pd | GeneralCategory::Pc)
}

fn mid_num(c: char) -> bool {
    matches!(c, '.' | '\'' | '"' | ',' | '\u{66B}')
}

/// Walks `before` backwards from the sigma: true when a cased character
/// comes before the start of the sigma's word. A cased character that is
/// a mark or format character counts only when a letter or digit stands
/// before it in the word, as the iterator makes a lone mark a word of its
/// own.
fn cased_before_in_word(before: &str) -> bool {
    let mut chars = before.chars().rev().peekable();
    let mut last = WordClass::Letter;
    let mut pending = false;
    while let Some(c) = chars.next() {
        let class = word_class(c);
        match class {
            WordClass::Ignorable => {
                pending |= java_cased(c);
                continue;
            }
            WordClass::Letter | WordClass::Digit => {
                if pending || java_cased(c) {
                    return true;
                }
            }
            WordClass::Other => {
                // Allowed between two characters of the class on this side.
                let beyond = chars.peek().map(|&b| word_class(b));
                let joins = (last == WordClass::Letter && mid_word(c) && beyond == Some(WordClass::Letter))
                    || (last == WordClass::Digit && mid_num(c) && beyond == Some(WordClass::Digit));
                if !joins {
                    return false;
                }
            }
        }
        last = class;
    }
    false
}

/// Walks `after` forwards from the sigma: true when a cased character
/// comes before the end of the sigma's word.
fn cased_after_in_word(after: &str) -> bool {
    let mut chars = after.chars().peekable();
    let mut last = WordClass::Letter;
    while let Some(c) = chars.next() {
        let class = word_class(c);
        match class {
            WordClass::Ignorable | WordClass::Letter | WordClass::Digit => {
                if java_cased(c) {
                    return true;
                }
                if class == WordClass::Ignorable {
                    continue;
                }
            }
            // A word boundary comes before the cased check: a circled
            // letter is on the JDK's cased list and ends the word.
            WordClass::Other => {
                // The character after the joiner must follow it directly:
                // a mark between them ends the word.
                let beyond = chars.peek().map(|&b| word_class(b));
                let joins = (last == WordClass::Letter && mid_word(c) && beyond == Some(WordClass::Letter))
                    || (last == WordClass::Digit && mid_num(c) && beyond == Some(WordClass::Digit));
                if !joins {
                    return false;
                }
            }
        }
        last = class;
    }
    false
}
