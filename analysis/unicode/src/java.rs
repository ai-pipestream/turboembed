//! `java.lang.Character` and `String.toLowerCase(Locale.ROOT)` semantics on this crate's Unicode data.
//!
//! The OpenNLP normalizers use these JDK methods. The functions follow the JDK's definitions
//! of letters and whitespace, using this crate's Unicode 18.0 tables. Characters assigned after
//! a JVM's Unicode version can therefore have different results.
//!
//! The functions here take a `char`. [`code_point`] has the segmentation classes on `u32`,
//! including lone surrogate code units, with `getType`'s numbering and `UnicodeScript.of`.

pub mod code_point;
mod codepoint_strings;
mod ignore_case;
pub mod locale;
mod sigma;
mod strings;
mod word_classes;

use crate::ascii::push_lowercase_run;
use crate::category::{GeneralCategory, general_category};
use crate::props::bits;
use crate::tables::case::{
    LOWER_KEYS, LOWER_VALUES, SPECIAL_LOWER_KEYS, SPECIAL_LOWER_VALUES, UPPER_KEYS, UPPER_VALUES,
};
use crate::tables::category::DECIMAL_ZEROS;
use crate::tables::props::{HAS_LOWER, HAS_UPPER};
use crate::trie::find;
pub use codepoint_strings::{
    eq_lowercase_code_points, lowercase_code_points, lowercase_code_points_changes, to_lowercase_code_points,
};
pub use ignore_case::equals_ignore_case;
pub use sigma::{SigmaCursor, WordClass, word_class};
pub use strings::is_blank;

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
    code_point::is_whitespace(c as u32)
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
/// sigma condition holds ([`SigmaCursor`] says how that differs from
/// Unicode's Final_Sigma). The language-specific mappings of SpecialCasing.txt
/// (Turkish, Azeri, Lithuanian) are not applied.
pub fn to_lowercase(text: &str, out: &mut String) {
    out.reserve(text.len());
    let mut sigma = SigmaCursor::new(text);
    let mut i = 0;
    while i < text.len() {
        let ascii_end = push_lowercase_run(text, i, out);
        if ascii_end > i {
            i = ascii_end;
            continue;
        }
        let c = text[i..].chars().next().expect("i is at a character boundary");
        match sigma.lowercase_at(i, c) {
            Lowercase::Char(lower) => out.push(lower),
            Lowercase::Str(lower) => out.push_str(lower),
        }
        i += c.len_utf8();
    }
}

mod borrowed_lowercase;
pub use borrowed_lowercase::lowercase_cow;

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
    SigmaCursor::new(text).lowercase_at(at, c)
}
