//! Forward word-boundary state machine for the JDK's `Locale.ROOT` final-sigma
//! condition (`ConditionalSpecialCasing.isFinalCased`).
//!
//! The JDK's condition is not Unicode's `Final_Sigma`: it walks the word the
//! sigma is in, as the root locale's word `BreakIterator` delimits it,
//! backwards until a cased character (every other character is passed over,
//! not only the case-ignorable ones), and then forwards, where any cased
//! character before the word's end makes the sigma non-final. So "ΑΣ1" and
//! "Α1Σ" end in final sigma, "ΑΣ1a" and "ΑΣ.a" do not, and a space, a comma
//! or an ideograph after the sigma ends the word where Unicode would look on.
//! [`word_class`] and [`boundary_before`] model the iterator, from probing
//! JDK 21.

use super::{
    CAPITAL_SIGMA, FINAL_SIGMA, Lowercase, SMALL_SIGMA, SPECIAL_LOWER_KEYS, SPECIAL_LOWER_VALUES, to_lower_char,
    word_classes,
};
use crate::category::{GeneralCategory, general_category};
use crate::props::bits;
use crate::tables::props::HAS_SPECIAL_LOWER;
use crate::trie::find;

/// Forward state machine that evaluates the JDK's `Locale.ROOT` final-sigma
/// condition across `text` in linear time per word.
///
/// The cursor advances monotonically through each word holding a capital sigma,
/// deciding the sigma's form as soon as the following cased character or word
/// boundary is reached and reusing the word-boundary state for any subsequent
/// sigmas in the same word.
pub struct SigmaCursor<'a> {
    text: &'a str,
    /// Byte offset in `text` up to which the word-boundary state has been advanced.
    cursor: usize,
    /// Word-boundary tail state at `cursor`.
    last: Tail,
    /// Character immediately before `cursor`, for the supplementary resynchronization quirk.
    previous: Option<(usize, char)>,
    /// Whether the current word at `cursor` already contains a `java_cased` character.
    cased_in_word: bool,
    /// When a previous sigma's lookahead stopped on another `CAPITAL_SIGMA` at byte `at`,
    /// stores `(at, has_preceding_cased)` so `at` does not re-step itself.
    pending_sigma: Option<(usize, bool)>,
    /// Start of the slice already inspected by `next_non_transparent`.
    nt_from: usize,
    /// Furthest byte offset reached by `next_non_transparent`.
    nt_scanned: usize,
    /// Cached first non-`Transparent` character `(byte_offset, WordClass)` at or after `nt_from`.
    nt_cached: Option<(usize, WordClass)>,
}

impl<'a> SigmaCursor<'a> {
    /// Creates a new forward cursor over `text`.
    #[inline]
    pub fn new(text: &'a str) -> SigmaCursor<'a> {
        SigmaCursor {
            text,
            cursor: 0,
            last: Tail::Closed,
            previous: None,
            cased_in_word: false,
            pending_sigma: None,
            nt_from: 0,
            nt_scanned: 0,
            nt_cached: None,
        }
    }

    /// Returns the `Locale.ROOT` lowercase mapping of the character `c` at byte
    /// `at` of `text`, advancing the word-boundary state machine when `c` is
    /// U+03A3 GREEK CAPITAL LETTER SIGMA.
    #[inline]
    pub fn lowercase_at(&mut self, at: usize, c: char) -> Lowercase {
        debug_assert!(self.text[at..].starts_with(c), "c is not the character at byte {at}");
        if c == CAPITAL_SIGMA {
            assert!(self.text.get(at..).is_some_and(|s| s.starts_with(c)), "c is not the character at byte {at}");
            return Lowercase::Char(if self.is_final(at) { FINAL_SIGMA } else { SMALL_SIGMA });
        }
        if bits(c) & HAS_SPECIAL_LOWER != 0
            && let Some(lower) = find(&SPECIAL_LOWER_KEYS, &SPECIAL_LOWER_VALUES, c as u32)
        {
            return Lowercase::Str(lower);
        }
        Lowercase::Char(to_lower_char(c))
    }

    /// Returns whether `String.toLowerCase(Locale.ROOT)` writes the capital
    /// sigma at byte `at` of `text` as final sigma U+03C2.
    pub fn is_final(&mut self, at: usize) -> bool {
        assert!(
            self.text.get(at..).is_some_and(|s| s.starts_with(CAPITAL_SIGMA)),
            "c is not the character at byte {at}"
        );
        let has_preceding_cased = if let Some((pending_at, has_cased)) = self.pending_sigma.take()
            && pending_at == at
        {
            has_cased
        } else {
            if at < self.cursor {
                self.cursor = 0;
                self.last = Tail::Closed;
                self.previous = None;
                self.cased_in_word = false;
            }
            self.advance_to_sigma(at)
        };
        if !has_preceding_cased {
            return false;
        }
        self.resolve_after_sigma()
    }

    /// Advances `self.cursor` through `(at, CAPITAL_SIGMA)` and returns whether
    /// a `java_cased` character preceded `at` in the same word.
    fn advance_to_sigma(&mut self, at: usize) -> bool {
        let mut jump_start = None;
        let mut cased_seen = false;
        for (offset, c) in self.text[self.cursor..at].char_indices().rev() {
            let i = self.cursor + offset;
            cased_seen |= java_cased(c);
            let class = word_class(c);
            if cased_seen && fixes_tail(class) {
                jump_start = Some(i);
                break;
            }
            if !cased_seen && matches!(class, WordClass::Other | WordClass::Control | WordClass::PreNum) {
                self.finish_sigma_at(at);
                return false;
            }
        }
        if let Some(start) = jump_start {
            self.cursor = start;
            self.last = Tail::Closed;
            self.previous = None;
            self.cased_in_word = false;
        } else if !cased_seen && !self.cased_in_word {
            self.finish_sigma_at(at);
            return false;
        }
        let start = self.cursor;
        while self.cursor < at {
            let i = self.cursor;
            let c = self.text[i..].chars().next().expect("cursor is at a character boundary");
            let boundary = self.step_char(i, c, jump_start == Some(start) && i == start);
            if boundary {
                self.cased_in_word = false;
            }
            if java_cased(c) {
                self.cased_in_word = true;
            }
        }
        let boundary = self.step_char(at, CAPITAL_SIGMA, false);
        if boundary {
            self.cased_in_word = false;
        }
        let has_preceding_cased = self.cased_in_word;
        self.cased_in_word = true;
        has_preceding_cased
    }

    /// Sets the state immediately after `(at, CAPITAL_SIGMA)` when no replay up
    /// to `at` is needed. Because `CAPITAL_SIGMA` is `WordClass::Letter` and
    /// `java_cased`, the state at `at + 2` is always fixed by `at` alone.
    #[inline]
    fn finish_sigma_at(&mut self, at: usize) {
        self.cursor = at + CAPITAL_SIGMA.len_utf8();
        self.last = Tail::Letter;
        self.previous = Some((at, CAPITAL_SIGMA));
        self.cased_in_word = true;
        self.pending_sigma = None;
    }

    /// Steps forward from `at + 2` until either a word boundary (returning
    /// `true` for final sigma) or a `java_cased` character in the same word
    /// (returning `false` for small sigma) is reached.
    fn resolve_after_sigma(&mut self) -> bool {
        while self.cursor < self.text.len() {
            let i = self.cursor;
            let c = self.text[i..].chars().next().expect("cursor is at a character boundary");
            let boundary = self.step_char(i, c, false);
            if boundary {
                self.cased_in_word = java_cased(c);
                if c == CAPITAL_SIGMA {
                    self.pending_sigma = Some((i, false));
                }
                return true;
            }
            if java_cased(c) {
                self.cased_in_word = true;
                if c == CAPITAL_SIGMA {
                    self.pending_sigma = Some((i, true));
                }
                return false;
            }
        }
        true
    }

    /// Steps the word-boundary state machine across character `c` at byte `i`
    /// and returns whether a word boundary precedes `c`.
    fn step_char(&mut self, i: usize, c: char, force_boundary: bool) -> bool {
        let class = word_class(c);
        let after = i + c.len_utf8();
        let text = self.text;
        let nt_from = &mut self.nt_from;
        let nt_scanned = &mut self.nt_scanned;
        let nt_cached = &mut self.nt_cached;
        let next = || next_non_transparent(text, after, nt_from, nt_scanned, nt_cached);
        let rules = boundary_before(class, c == '\u{AD}', next, &mut self.last);
        // The resynchronization quirk: `isBoundary` right after a
        // supplementary character answers true unless it starts the text.
        let quirk = self.previous.is_some_and(|(p, pc)| p > 0 && pc as u32 > 0xFFFF);
        let boundary = i == 0 || force_boundary || rules || quirk;
        self.previous = Some((i, c));
        self.cursor = after;
        boundary
    }
}

/// Returns the `WordClass` of the first character at or after `after` that is
/// not `WordClass::Transparent`, advancing `nt_scanned` monotonically so a run
/// of format characters is scanned at most once across the word.
fn next_non_transparent(
    text: &str,
    after: usize,
    nt_from: &mut usize,
    nt_scanned: &mut usize,
    nt_cached: &mut Option<(usize, WordClass)>,
) -> Option<WordClass> {
    if *nt_from <= after {
        if let Some((pos, class)) = *nt_cached
            && after <= pos
        {
            return Some(class);
        }
        if nt_cached.is_none() && *nt_scanned == text.len() {
            return None;
        }
    }
    let scan_start = if *nt_from <= after && *nt_scanned > after { *nt_scanned } else { after };
    *nt_from = after;
    for (offset, c) in text[scan_start..].char_indices() {
        let pos = scan_start + offset;
        let class = word_class(c);
        *nt_scanned = pos + c.len_utf8();
        if class != WordClass::Transparent {
            *nt_cached = Some((pos, class));
            return Some(class);
        }
    }
    *nt_scanned = text.len();
    *nt_cached = None;
    None
}

#[inline]
fn fixes_tail(class: WordClass) -> bool {
    matches!(class, WordClass::Letter | WordClass::Digit | WordClass::Other | WordClass::Control | WordClass::PreNum)
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
/// the first character after it that is not a format character.
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
