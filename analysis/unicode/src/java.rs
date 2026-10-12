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
use crate::tables::props::{CASE_IGNORABLE, CASED, HAS_LOWER, HAS_SPECIAL_LOWER, HAS_UPPER};
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
/// `i\u{307}`) and U+03A3 becomes final sigma U+03C2 where the Final_Sigma
/// condition holds. The language-specific mappings of SpecialCasing.txt
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
/// around it (the Final_Sigma condition); every other character maps the
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

/// The Final_Sigma condition of Unicode section 3.13 for the sigma at
/// byte `at`: a cased character comes before it with only case-ignorable
/// characters between, and no cased character follows it with only
/// case-ignorable characters between. A character that is both cased and
/// case-ignorable counts as cased, as the definition's regular expressions
/// allow.
fn is_final_sigma(text: &str, at: usize) -> bool {
    let after = at + CAPITAL_SIGMA.len_utf8();
    cased_beyond_ignorables(text[..at].chars().rev()) && !cased_beyond_ignorables(text[after..].chars())
}

/// The first character of `chars` that is not case-ignorable is cased.
fn cased_beyond_ignorables(chars: impl Iterator<Item = char>) -> bool {
    for c in chars {
        let bits = bits(c);
        if bits & CASED != 0 {
            return true;
        }
        if bits & CASE_IGNORABLE == 0 {
            return false;
        }
    }
    false
}
