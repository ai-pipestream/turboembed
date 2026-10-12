//! UAX #29 word boundaries, rules WB1 to WB999, over the Unicode 17.0
//! `Word_Break` property. This is OpenNLP's `WordSegmenter`.
//!
//! The text is walked once, forward, decoding each character once. The
//! state is the class of the character before (for WB3 to WB4), the last
//! two significant classes (WB4 makes Extend, Format and ZWJ invisible to
//! the later rules) and the number of regional indicators in a row. The
//! next significant class is read ahead only at the few (last, current)
//! pairs where a rule needs it (WB6, WB7b, WB12) or where the regional
//! indicator parity decides (WB15, WB16).
//!
//! The decision for WB5 to WB999 comes from a table over (last, current)
//! pairs, derived at compile time from the rule cascade: a cell holds the
//! answer when it is the same for every other input, or says to run the
//! cascade. A second table folds in whether the current class can fire a
//! WB3 or WB4 rule, so the common case is one table read per character.

use turbo_unicode::WordBreak;
use turbo_unicode::word_break::{is_extended_pictographic, word_break};

use crate::utf8::decode;

/// The number of `Word_Break` classes.
const N: usize = WordBreak::ALL.len();

const OTHER: u8 = WordBreak::Other as u8;
const CR: u8 = WordBreak::Cr as u8;
const LF: u8 = WordBreak::Lf as u8;
const NEWLINE: u8 = WordBreak::Newline as u8;
const EXTEND: u8 = WordBreak::Extend as u8;
const ZWJ: u8 = WordBreak::Zwj as u8;
const REGIONAL_INDICATOR: u8 = WordBreak::RegionalIndicator as u8;
const FORMAT: u8 = WordBreak::Format as u8;
const KATAKANA: u8 = WordBreak::Katakana as u8;
const HEBREW_LETTER: u8 = WordBreak::HebrewLetter as u8;
const ALETTER: u8 = WordBreak::ALetter as u8;
const SINGLE_QUOTE: u8 = WordBreak::SingleQuote as u8;
const DOUBLE_QUOTE: u8 = WordBreak::DoubleQuote as u8;
const MID_NUM_LET: u8 = WordBreak::MidNumLet as u8;
const MID_LETTER: u8 = WordBreak::MidLetter as u8;
const MID_NUM: u8 = WordBreak::MidNum as u8;
const NUMERIC: u8 = WordBreak::Numeric as u8;
const EXTEND_NUM_LET: u8 = WordBreak::ExtendNumLet as u8;
const WSEG_SPACE: u8 = WordBreak::WSegSpace as u8;

/// Table cells. `CONSULT` means the cascade must run with the look-ahead
/// and the parity; `SPECIAL` (only in `FAST`) means the current class can
/// fire a WB3 or WB4 rule.
const NO_BREAK: u8 = 0;
const BREAK: u8 = 1;
const CONSULT: u8 = 2;
const SPECIAL: u8 = 3;

const fn ignorable(c: u8) -> bool {
    c == EXTEND || c == FORMAT || c == ZWJ
}

const fn ah_letter(c: u8) -> bool {
    c == ALETTER || c == HEBREW_LETTER
}

/// MidLetter, MidNumLet or Single_Quote: the middle of WB6 and WB7.
const fn mid_letter(c: u8) -> bool {
    c == MID_LETTER || c == MID_NUM_LET || c == SINGLE_QUOTE
}

/// MidNum, MidNumLet or Single_Quote: the middle of WB11 and WB12.
const fn mid_num(c: u8) -> bool {
    c == MID_NUM || c == MID_NUM_LET || c == SINGLE_QUOTE
}

/// Whether a class can fire a WB3 family or WB4 rule.
const fn special(c: u8) -> bool {
    c == CR || c == LF || c == NEWLINE || c == ZWJ || c == WSEG_SPACE || c == EXTEND || c == FORMAT
}

/// Rules WB5 to WB999: whether there is a break before a character of
/// class `current`, given the last two significant classes, the next
/// significant class and whether an odd number of regional indicators
/// precede. These rules never look at the character immediately before,
/// which is what lets a table hold them.
const fn after_prefix(current: u8, last: u8, second_last: u8, next: u8, odd_regional: bool) -> bool {
    if ah_letter(last) && ah_letter(current) {
        return false; // WB5
    }
    if ah_letter(last) && mid_letter(current) && ah_letter(next) {
        return false; // WB6
    }
    if ah_letter(second_last) && mid_letter(last) && ah_letter(current) {
        return false; // WB7
    }
    if last == HEBREW_LETTER && current == SINGLE_QUOTE {
        return false; // WB7a
    }
    if last == HEBREW_LETTER && current == DOUBLE_QUOTE && next == HEBREW_LETTER {
        return false; // WB7b
    }
    if second_last == HEBREW_LETTER && last == DOUBLE_QUOTE && current == HEBREW_LETTER {
        return false; // WB7c
    }
    if last == NUMERIC && current == NUMERIC {
        return false; // WB8
    }
    if ah_letter(last) && current == NUMERIC {
        return false; // WB9
    }
    if last == NUMERIC && ah_letter(current) {
        return false; // WB10
    }
    if second_last == NUMERIC && mid_num(last) && current == NUMERIC {
        return false; // WB11
    }
    if last == NUMERIC && mid_num(current) && next == NUMERIC {
        return false; // WB12
    }
    if last == KATAKANA && current == KATAKANA {
        return false; // WB13
    }
    if (ah_letter(last) || last == NUMERIC || last == KATAKANA || last == EXTEND_NUM_LET) && current == EXTEND_NUM_LET {
        return false; // WB13a
    }
    if last == EXTEND_NUM_LET && (ah_letter(current) || current == NUMERIC || current == KATAKANA) {
        return false; // WB13b
    }
    if current == REGIONAL_INDICATOR && last == REGIONAL_INDICATOR && odd_regional {
        return false; // WB15, WB16
    }
    true // WB999
}

/// The cell for (last, current): the answer of `after_prefix` when it is
/// the same for every second last class, next class and parity, else
/// `CONSULT`.
const fn derive(last: u8, current: u8) -> u8 {
    let first = after_prefix(current, last, OTHER, OTHER, false);
    let mut second_last = 0;
    while second_last < N as u8 {
        let mut next = 0;
        while next < N as u8 {
            if after_prefix(current, last, second_last, next, false) != first
                || after_prefix(current, last, second_last, next, true) != first
            {
                return CONSULT;
            }
            next += 1;
        }
        second_last += 1;
    }
    if first { BREAK } else { NO_BREAK }
}

const fn transition_table() -> [u8; N * N] {
    let mut table = [0; N * N];
    let mut last = 0;
    while last < N {
        let mut current = 0;
        while current < N {
            table[last * N + current] = derive(last as u8, current as u8);
            current += 1;
        }
        last += 1;
    }
    table
}

const fn fast_table() -> [u8; N * N] {
    let mut table = TRANSITION;
    let mut i = 0;
    while i < N * N {
        if special((i % N) as u8) {
            table[i] = SPECIAL;
        }
        i += 1;
    }
    table
}

/// `TRANSITION[last * N + current]`: the WB5 to WB999 decision.
static TRANSITION: [u8; N * N] = transition_table();
/// `FAST[last * N + current]`: the decision, or `SPECIAL` when the current
/// class can fire a WB3 or WB4 rule.
static FAST: [u8; N * N] = fast_table();

/// The `Word_Break` class of every ASCII character, so ASCII text skips
/// the two-stage table. A test checks it against the property table.
const fn ascii_classes() -> [u8; 128] {
    let mut table = [OTHER; 128];
    let mut b = 0;
    while b < 128 {
        let c = b as u8;
        table[b] = match c {
            b'\n' => LF,
            b'\r' => CR,
            0x0B | 0x0C => NEWLINE,
            b' ' => WSEG_SPACE,
            b'"' => DOUBLE_QUOTE,
            b'\'' => SINGLE_QUOTE,
            b'.' => MID_NUM_LET,
            b',' | b';' => MID_NUM,
            b':' => MID_LETTER,
            b'_' => EXTEND_NUM_LET,
            b'0'..=b'9' => NUMERIC,
            b'A'..=b'Z' | b'a'..=b'z' => ALETTER,
            _ => OTHER,
        };
        b += 1;
    }
    table
}

static ASCII_CLASSES: [u8; 128] = ascii_classes();

/// The class of a code point as a table index.
#[inline(always)]
fn class(cp: u32) -> u8 {
    if cp < 0x80 { ASCII_CLASSES[cp as usize] } else { word_break(cp) as u8 }
}

/// The class of the first significant character at or after byte `from`,
/// or `Other` at the end of the text.
#[inline]
fn next_significant(bytes: &[u8], mut from: usize) -> u8 {
    while from < bytes.len() {
        let (cp, len) = decode(bytes, from);
        let c = class(cp);
        if !ignorable(c) {
            return c;
        }
        from += len;
    }
    OTHER
}

/// The running state of the pass: the significant classes and the
/// regional indicator run.
struct Context {
    last: u8,
    second_last: u8,
    regional_run: u32,
}

impl Context {
    /// Runs the cascade for a `CONSULT` cell, reading ahead from `next_from`.
    #[inline(never)]
    fn consult(&self, bytes: &[u8], next_from: usize, current: u8) -> bool {
        let next = next_significant(bytes, next_from);
        after_prefix(current, self.last, self.second_last, next, self.regional_run & 1 == 1)
    }

    /// The decision when the class before or the current class is special:
    /// the WB3 family and WB4 first, then the table.
    #[inline(never)]
    fn at_special(&self, prev: u8, current: u8, cp: u32, bytes: &[u8], next_from: usize) -> bool {
        if prev == CR && current == LF {
            return false; // WB3
        }
        if prev == CR || prev == LF || prev == NEWLINE {
            return true; // WB3a
        }
        if current == CR || current == LF || current == NEWLINE {
            return true; // WB3b
        }
        if prev == ZWJ && is_extended_pictographic(cp) {
            return false; // WB3c
        }
        if prev == WSEG_SPACE && current == WSEG_SPACE {
            return false; // WB3d
        }
        if ignorable(current) {
            return false; // WB4
        }
        match TRANSITION[self.last as usize * N + current as usize] {
            CONSULT => self.consult(bytes, next_from, current),
            cell => cell == BREAK,
        }
    }
}

/// Calls `segment(start, end)` for each word segment of `text`, in order,
/// with byte offsets. The segments are contiguous and cover the text; an
/// empty text has none.
pub fn for_each_segment(text: &str, mut segment: impl FnMut(u32, u32)) {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return;
    }
    let (first, mut i) = decode(bytes, 0);
    let mut prev = class(first);
    let mut prev_special = special(prev);
    let mut cx = Context { last: OTHER, second_last: OTHER, regional_run: 0 };
    if !ignorable(prev) {
        cx.last = prev;
        cx.regional_run = u32::from(prev == REGIONAL_INDICATOR);
    }
    let mut start = 0u32;
    while i < bytes.len() {
        let (cp, len) = decode(bytes, i);
        let current = class(cp);
        let cell = FAST[cx.last as usize * N + current as usize];
        let current_special = cell == SPECIAL;
        let split = if prev_special || current_special {
            cx.at_special(prev, current, cp, bytes, i + len)
        } else if cell == CONSULT {
            cx.consult(bytes, i + len, current)
        } else {
            cell == BREAK
        };
        if split {
            segment(start, i as u32);
            start = i as u32;
        }
        if !ignorable(current) {
            cx.second_last = cx.last;
            cx.last = current;
            cx.regional_run = if current == REGIONAL_INDICATOR { cx.regional_run + 1 } else { 0 };
        }
        prev = current;
        prev_special = current_special;
        i += len;
    }
    segment(start, bytes.len() as u32);
}

/// Appends the word boundaries of `text` to `out`, as byte offsets in
/// ascending order: 0 (WB1), every boundary inside, and the text length
/// (WB2). An empty text has the one boundary 0.
pub fn word_boundaries(text: &str, out: &mut Vec<u32>) {
    out.push(0);
    for_each_segment(text, |_, end| out.push(end));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segments(text: &str) -> Vec<&str> {
        let mut out = Vec::new();
        for_each_segment(text, |s, e| out.push(&text[s as usize..e as usize]));
        out
    }

    #[test]
    fn ascii_classes_match_the_property() {
        for cp in 0..0x80 {
            assert_eq!(ASCII_CLASSES[cp as usize], word_break(cp) as u8, "U+{cp:04X}");
        }
    }

    #[test]
    fn table_cells() {
        let cell = |last: WordBreak, current: WordBreak| TRANSITION[last as usize * N + current as usize];
        assert_eq!(cell(WordBreak::ALetter, WordBreak::ALetter), NO_BREAK);
        assert_eq!(cell(WordBreak::Other, WordBreak::ALetter), BREAK);
        // WB6 needs the next class, WB7 the one before last.
        assert_eq!(cell(WordBreak::ALetter, WordBreak::MidLetter), CONSULT);
        assert_eq!(cell(WordBreak::MidLetter, WordBreak::ALetter), CONSULT);
        assert_eq!(cell(WordBreak::RegionalIndicator, WordBreak::RegionalIndicator), CONSULT);
        assert_eq!(cell(WordBreak::HebrewLetter, WordBreak::SingleQuote), NO_BREAK);
        assert_eq!(FAST[ALETTER as usize * N + EXTEND as usize], SPECIAL);
        assert_eq!(FAST[ALETTER as usize * N + ALETTER as usize], NO_BREAK);
    }

    #[test]
    fn words_numbers_and_punctuation() {
        assert_eq!(segments("can't stop"), ["can't", " ", "stop"]);
        assert_eq!(segments("3.14, 1,000.5"), ["3.14", ",", " ", "1,000.5"]);
        assert_eq!(segments("e.g. a."), ["e.g", ".", " ", "a", "."]);
        assert_eq!(segments("a  b"), ["a", "  ", "b"]);
        assert_eq!(segments("\r\n\n"), ["\r\n", "\n"]);
        assert_eq!(segments("foo_bar42"), ["foo_bar42"]);
        assert_eq!(segments(""), Vec::<&str>::new());
    }

    #[test]
    fn ignorables_flags_and_emoji() {
        // A combining mark joins the letter before it and is invisible to WB5.
        assert_eq!(segments("e\u{301}x"), ["e\u{301}x"]);
        // Four regional indicators make two flags.
        assert_eq!(segments("\u{1f1fa}\u{1f1f8}\u{1f1eb}\u{1f1f7}"), ["\u{1f1fa}\u{1f1f8}", "\u{1f1eb}\u{1f1f7}"]);
        // WB3c: a ZWJ sequence stays together.
        assert_eq!(segments("\u{1f468}\u{200d}\u{1f469}!"), ["\u{1f468}\u{200d}\u{1f469}", "!"]);
        // Han characters are Other: every one is a segment.
        assert_eq!(segments("\u{4e2d}\u{6587}"), ["\u{4e2d}", "\u{6587}"]);
        // A leading combining mark is a segment start like any character.
        assert_eq!(segments("\u{301}a"), ["\u{301}", "a"]);
    }

    #[test]
    fn hebrew_quotes() {
        // WB7b and WB7c: a gershayim written as a double quote inside a word.
        assert_eq!(segments("\u{5e6}\"\u{5d4}\u{5dc}"), ["\u{5e6}\"\u{5d4}\u{5dc}"]);
        assert_eq!(segments("\u{5e6}\" x"), ["\u{5e6}", "\"", " ", "x"]);
    }

    #[test]
    fn boundaries_include_both_ends() {
        let mut out = Vec::new();
        word_boundaries("ab cd", &mut out);
        assert_eq!(out, [0, 2, 3, 5]);
        out.clear();
        word_boundaries("", &mut out);
        assert_eq!(out, [0]);
    }
}
