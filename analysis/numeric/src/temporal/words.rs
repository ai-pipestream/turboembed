//! The words a temporal mention is made of: month names, the relative
//! keywords, unit words and ordinal suffixes.
//!
//! OpenNLP lowercases a word with `StringUtil.toLowerCase`, which maps each
//! code point with `Character.toLowerCase`, and looks the result up. Two
//! characters outside ASCII lowercase into ASCII that way: U+0130 (`İ`)
//! becomes `i` and U+212A (the Kelvin sign) becomes `k`, so `İn 2 days`
//! and `next wee\u{212A}` are read as `in 2 days` and `next week`. The
//! comparisons here map the same way and allocate nothing.

use turbo_analysis_layers::Granularity;
use turbo_unicode::java;

use crate::scan::word_end;

/// How many UTF-16 units of letters a word may have before the scan gives
/// up: one more than `september`, the longest word read.
const MAX_WORD_UNITS: usize = 10;

/// The month names; the first three letters of each are its abbreviation.
const MONTHS: [&str; 12] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

/// `CursorTemporalExtractor.word`: the end of the word at byte `start`,
/// a run of letters of at most ten UTF-16 units followed by a word
/// boundary.
pub(super) fn word(text: &str, start: usize) -> Option<usize> {
    word_end(text, start, MAX_WORD_UNITS)
}

/// Whether `word`, lowercased as OpenNLP lowercases it, is `lower`.
fn is(word: &str, lower: &str) -> bool {
    word.chars().map(java::to_lower_char).eq(lower.chars())
}

/// The month, 1 to 12, that `word` names in full or by its first three
/// letters.
pub(super) fn month(word: &str) -> Option<u8> {
    let i = MONTHS.iter().position(|m| is(word, m) || is(word, &m[..3]))?;
    Some(i as u8 + 1)
}

/// Whether `word` is an ordinal suffix: `st`, `nd`, `rd` or `th`.
pub(super) fn is_ordinal(word: &str) -> bool {
    ["st", "nd", "rd", "th"].iter().any(|s| is(word, s))
}

/// Whether `word` is `ago`.
pub(super) fn is_ago(word: &str) -> bool {
    is(word, "ago")
}

/// The words a relative expression starts with.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Keyword {
    Today,
    Yesterday,
    Tomorrow,
    Last,
    This,
    Next,
    In,
}

pub(super) fn keyword(word: &str) -> Option<Keyword> {
    const KEYWORDS: [(&str, Keyword); 7] = [
        ("today", Keyword::Today),
        ("yesterday", Keyword::Yesterday),
        ("tomorrow", Keyword::Tomorrow),
        ("last", Keyword::Last),
        ("this", Keyword::This),
        ("next", Keyword::Next),
        ("in", Keyword::In),
    ];
    KEYWORDS.iter().find(|(k, _)| is(word, k)).map(|&(_, k)| k)
}

const UNITS: [(&str, Granularity); 5] = [
    ("day", Granularity::Day),
    ("week", Granularity::Week),
    ("month", Granularity::Month),
    ("quarter", Granularity::Quarter),
    ("year", Granularity::Year),
];

/// The granularity a unit word names as written: after `last`, `this` and
/// `next` OpenNLP takes the singular only, so `next weeks` is no mention.
pub(super) fn unit(word: &str) -> Option<Granularity> {
    UNITS.iter().find(|(u, _)| is(word, u)).map(|&(_, g)| g)
}

/// The granularity a counted unit word names: `CursorTemporalExtractor.singular`
/// drops one final `s` of the lowercased word when it is longer than one
/// unit, so `3 days ago` and `in 1 weeks` both read, and `in 2 dayss`
/// does not.
pub(super) fn counted_unit(word: &str) -> Option<Granularity> {
    let (lowered_len, last) =
        word.chars().fold((0, None), |(n, _), c| (n + java::to_lower_char(c).len_utf16(), Some(c)));
    let last = last?;
    if java::to_lower_char(last) == 's' && lowered_len > 1 {
        return unit(&word[..word.len() - last.len_utf8()]);
    }
    unit(word)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn months_in_full_and_abbreviated_in_any_case() {
        assert_eq!(month("July"), Some(7));
        assert_eq!(month("SEP"), Some(9));
        assert_eq!(month("fEb"), Some(2));
        assert_eq!(month("May"), Some(5));
        assert_eq!(month("Sept"), None);
        assert_eq!(month("Jul."), None);
        assert_eq!(month("\u{130}"), None);
    }

    #[test]
    fn only_two_characters_outside_ascii_lowercase_into_it() {
        // The start table of the scan relies on this: no other character
        // can begin a word that lowercases to a keyword or a month.
        let into_ascii: Vec<u32> = (0x80..=0x10FFFFu32)
            .filter_map(char::from_u32)
            .filter(|&c| java::to_lower_char(c).is_ascii())
            .map(|c| c as u32)
            .collect();
        assert_eq!(into_ascii, [0x130, 0x212A]);
        assert_eq!(keyword("\u{130}n"), Some(Keyword::In));
        assert_eq!(unit("wee\u{212A}"), Some(Granularity::Week));
    }

    #[test]
    fn counted_units_drop_one_final_s() {
        assert_eq!(counted_unit("days"), Some(Granularity::Day));
        assert_eq!(counted_unit("WEEKS"), Some(Granularity::Week));
        assert_eq!(counted_unit("quarter"), Some(Granularity::Quarter));
        assert_eq!(counted_unit("dayss"), None);
        assert_eq!(counted_unit("s"), None);
        assert_eq!(unit("days"), None);
        assert_eq!(counted_unit("year\u{17F}"), None);
    }

    #[test]
    fn ordinals_and_keywords() {
        assert!(is_ordinal("TH") && is_ordinal("nd") && !is_ordinal("\u{17F}t") && !is_ordinal("x"));
        assert_eq!(keyword("Yesterday"), Some(Keyword::Yesterday));
        assert_eq!(keyword("lasts"), None);
        assert!(is_ago("AGO") && !is_ago("agos"));
    }
}
