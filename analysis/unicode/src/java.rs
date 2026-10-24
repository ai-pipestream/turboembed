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
mod strings;
mod word_classes;
pub use strings::is_blank;

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
        assert!(text.get(at..).is_some_and(|s| s.starts_with(c)), "c is not the character at byte {at}");
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
/// ends the word where Unicode would look on. [`word_class`] and
/// [`boundary_before`] model the iterator, from probing JDK 21.
///
/// Only the text around the sigma is read, without allocating: backwards
/// to the nearest cased character and on to the nearest character whose
/// class fixes what the word ends in whatever came before it (a letter, a
/// digit, a currency sign or a character that is a word of its own), from
/// where the rules are replayed; and forwards to the word's end. A
/// character that is a word of its own before any cased one ends the
/// search at once, as a boundary lies between it and the sigma or no cased
/// character does.
fn is_final_sigma(text: &str, at: usize) -> bool {
    let fixes_tail = |class: WordClass| {
        matches!(
            class,
            WordClass::Letter | WordClass::Digit | WordClass::Other | WordClass::Control | WordClass::PreNum
        )
    };
    let mut start = 0;
    let mut cased_seen = false;
    for (i, c) in text[..at].char_indices().rev() {
        cased_seen |= java_cased(c);
        let class = word_class(c);
        if cased_seen && fixes_tail(class) {
            start = i;
            break;
        }
        if matches!(class, WordClass::Other | WordClass::Control | WordClass::PreNum) {
            return false;
        }
    }
    // Replaying the rules from `start`: the boundary before the character
    // at `start` is not read (the backward walk stops at the cased
    // character at or after it), only the tail it leaves.
    let mut last = Tail::Closed;
    // Where the word holding the sigma starts, and the nearest cased
    // character before the sigma, as byte indexes.
    let mut word_start = start;
    let mut last_cased = None;
    let mut previous: Option<(usize, char)> = None;
    for (offset, c) in text[start..].char_indices() {
        let i = start + offset;
        let class = word_class(c);
        let after = i + c.len_utf8();
        let next = || text[after..].chars().map(word_class).find(|&k| k != WordClass::Transparent);
        let rules = boundary_before(class, c == '\u{AD}', next, &mut last);
        // The resynchronization quirk: `isBoundary` right after a
        // supplementary character answers true unless it starts the text.
        let quirk = previous.is_some_and(|(p, pc)| p > 0 && pc as u32 > 0xFFFF);
        let boundary = i == start || rules || quirk;
        if i < at {
            if boundary {
                word_start = i;
            }
            if java_cased(c) {
                last_cased = Some(i);
            }
        } else if i == at {
            // Backwards: the nearest cased character must be in the
            // sigma's word.
            if boundary || !last_cased.is_some_and(|cased| cased >= word_start) {
                return false;
            }
        } else {
            // Forwards: a cased character before the word's end makes it
            // non-final.
            if boundary {
                return true;
            }
            if java_cased(c) {
                return false;
            }
        }
        previous = Some((i, c));
    }
    true
}

/// `isCased` of the JDK's conditional casing: the three cased letter
/// categories and a fixed list of other characters (modifier letters,
/// U+0345, the Roman numerals, the circled letters), which is an older
/// Unicode's Other_Uppercase and Other_Lowercase, not the current ones.
/// U+0295 is Ll in JDK 21 (Unicode 15.0) and Lo since Unicode 16.0, so
/// it is listed.
fn java_cased(c: char) -> bool {
    matches!(general_category(c), GeneralCategory::Lu | GeneralCategory::Ll | GeneralCategory::Lt)
        || matches!(
            c as u32,
            0x0295
                | 0x02B0..=0x02B8
                | 0x02C0..=0x02C1
                | 0x02E0..=0x02E4
                | 0x0345
                | 0x037A
                | 0x1D2C..=0x1D61
                | 0x2160..=0x217F
                | 0x24B6..=0x24E9
        )
}

/// What the root locale's word `BreakIterator` of the JDK makes of a
/// character. Its rules predate UAX #29: a word is letters and digits in
/// any order, with a few joiners allowed between two letters or two
/// digits, a currency sign before a number, a percent sign after one and
/// a danda after letters. The class of each code point comes from probing
/// JDK 21 (`word_classes.rs`); supplementary characters keep the classes
/// of their categories.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordClass {
    /// Ends a word on both sides: punctuation, symbols, separators,
    /// the tab, private use, unassigned, and the kana and the common
    /// kanji, which the iterator segments on their own. A mark after one
    /// attaches to it.
    Other,
    /// Letters and spacing marks, in every plane.
    Letter,
    /// The numeric categories, in every plane.
    Digit,
    /// Nonspacing and enclosing marks: attached to the character before
    /// them, except after a word the rules have closed (a danda or percent
    /// sign that joined a word, or at the start of the text), where a mark
    /// is a word of its own.
    Mark,
    /// Format characters, invisible to the rules (the soft hyphen is a
    /// joiner instead; a few format characters are `Other`).
    Transparent,
    /// Dashes, connectors, the soft hyphen and U+2027: joins two letters.
    MidWord,
    /// `,` and U+066B: joins two digits.
    MidNum,
    /// `'` and `"`: joins two letters or two digits.
    MidBoth,
    /// `.`: joins two letters or two digits, and starts a number.
    Dot,
    /// Currency signs and `#`: attached to the number after them.
    PreNum,
    /// `%`, `&`, U+00A2, U+066A, U+2030 and U+2031: attached to the number
    /// before them.
    PostNum,
    /// The dandas: attached to the letters before them, and digits may
    /// follow; a second danda is a word of its own.
    Danda,
    /// The controls other than the tab, the line and paragraph separators
    /// and seven format characters (U+110BD, U+110CD, U+1343F, U+1BCA3,
    /// U+1D17A, U+E0001, U+E007F): a word of its own that even a mark does
    /// not join. No sigma decision depends on the difference from `Other`,
    /// as a mark's word never holds a letter; the class table keeps it.
    Control,
}

/// The class of `c`, by the table's class number.
fn word_class_number(c: char) -> u8 {
    let cp = c as u32;
    let table = word_classes::JDK_WORD_CLASSES;
    let i = table.partition_point(|&(_, last, _)| last < cp);
    match table.get(i) {
        Some(&(first, _, class)) if first <= cp => class,
        _ => 0,
    }
}

/// What the JDK's root locale word `BreakIterator` makes of `c`, from
/// the probed table; `analysis/unicode/tests/jdk_sigma.rs` checks it
/// against a running JDK.
pub fn word_class(c: char) -> WordClass {
    match word_class_number(c) {
        1 => WordClass::Letter,
        2 => WordClass::Digit,
        3 => WordClass::Mark,
        4 => WordClass::Transparent,
        5 => WordClass::MidWord,
        6 => WordClass::MidNum,
        7 => WordClass::MidBoth,
        8 => WordClass::Dot,
        9 => WordClass::PreNum,
        10 => WordClass::PostNum,
        11 => WordClass::Danda,
        12 => WordClass::Control,
        _ => WordClass::Other,
    }
}

/// What the word so far ends in, for the rules of [`boundary_before`]
/// that look back. Marks and format characters do not change it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tail {
    /// A character that is a word of its own, or a joiner that did not
    /// join: a mark attaches to it, everything else starts a word.
    Other,
    /// A letter, or a spacing mark.
    Letter,
    Digit,
    /// A danda that followed letters: digits may follow it, another danda
    /// may not.
    Danda,
    /// A currency sign, `#` or a lone dot before a digit: the digit attaches.
    PreNum,
    /// A word the rules have closed: the start of the text, a control, a
    /// soft hyphen that did not join, a danda or a percent sign that joined
    /// a word, or a mark that attached to nothing. Everything starts a word
    /// here, and a mark is a word of its own.
    Closed,
}

/// Whether the JDK's root locale word `BreakIterator` puts a boundary
/// before a character of class `class`, given what the word so far ends
/// in (`last`, which the character then updates) and `next`, the class of
/// the first character after it that is not a format character. The rules,
/// as probed: letters and digits chain into one word; a mark attaches to
/// the character before it, but is a word of its own after a danda or
/// percent sign that joined a word, after a control or a soft hyphen that
/// did not join, after such a mark, and at the start of the text; a format
/// character is invisible; a joiner stays in the word when the character
/// before it (marks attached) and the character right after it are both
/// letters, or both digits for the numeric joiners; `.` and a currency
/// sign start a number; a percent sign ends one; a danda follows letters
/// and may be followed by digits. Everything else is a word of its own.
/// The caller lays one quirk over the answers: `isBoundary` right after a
/// supplementary character answers true unless that character starts the
/// text, as the iterator resynchronizes from inside the surrogate pair,
/// while the word goes on as the rules say ("a\u{20000}\u{345}-a" is one
/// word with a boundary before the mark).
fn boundary_before(
    class: WordClass,
    soft_hyphen: bool,
    next: impl FnOnce() -> Option<WordClass>,
    last: &mut Tail,
) -> bool {
    match class {
        WordClass::Transparent => false,
        WordClass::Mark => match *last {
            Tail::Closed => true,
            Tail::Danda => {
                *last = Tail::Closed;
                true
            }
            Tail::PreNum => {
                *last = Tail::Other;
                false
            }
            Tail::Other | Tail::Letter | Tail::Digit => false,
        },
        WordClass::Letter => {
            let joins = matches!(*last, Tail::Letter | Tail::Digit);
            *last = Tail::Letter;
            !joins
        }
        WordClass::Digit => {
            let joins = matches!(*last, Tail::Letter | Tail::Digit | Tail::Danda | Tail::PreNum);
            *last = Tail::Digit;
            !joins
        }
        WordClass::MidWord | WordClass::MidNum | WordClass::MidBoth | WordClass::Dot => {
            let next = next();
            let letters = matches!(class, WordClass::MidWord | WordClass::MidBoth | WordClass::Dot);
            let digits = matches!(class, WordClass::MidNum | WordClass::MidBoth | WordClass::Dot);
            let joins = (letters && *last == Tail::Letter && next == Some(WordClass::Letter))
                || (digits && *last == Tail::Digit && next == Some(WordClass::Digit));
            if joins {
                false
            } else {
                // A lone dot before digits starts the number; a lone soft
                // hyphen is a control.
                *last = if class == WordClass::Dot && next == Some(WordClass::Digit) {
                    Tail::PreNum
                } else if soft_hyphen {
                    Tail::Closed
                } else {
                    Tail::Other
                };
                true
            }
        }
        WordClass::PreNum => {
            *last = Tail::PreNum;
            true
        }
        WordClass::PostNum => {
            let joins = *last == Tail::Digit;
            *last = if joins { Tail::Closed } else { Tail::Other };
            !joins
        }
        WordClass::Danda => {
            let joins = *last == Tail::Letter;
            *last = if joins { Tail::Danda } else { Tail::Other };
            !joins
        }
        WordClass::Other => {
            *last = Tail::Other;
            true
        }
        WordClass::Control => {
            *last = Tail::Closed;
            true
        }
    }
}
