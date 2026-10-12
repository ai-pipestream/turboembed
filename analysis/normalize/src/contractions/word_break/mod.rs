//! Word boundaries by the Unicode Text Segmentation algorithm (UAX #29),
//! rules WB1 to WB999, as OpenNLP's `opennlp.tools.tokenize.uax29.WordSegmenter`
//! applies them. The contraction normalizer expands a contraction only
//! when it is a complete word, so it asks whether its two ends are word
//! boundaries.
//!
//! OpenNLP finds boundaries with a forward pass over the whole text. The
//! rules only look at a bounded neighbourhood of a position: the character
//! before it, the character at it, the last two characters before it that
//! are not Extend, Format or ZWJ (rule WB4 skips those), the next such
//! character after it, and the length of the run of regional indicators
//! before it. [`is_boundary`] reads exactly those, so one position costs a
//! few character decodes instead of a pass from the start of the text,
//! and gives the same answer as the forward pass. The conformance test
//! runs Unicode's WordBreakTest.txt through it.

mod table;

use std::cmp::Ordering;

/// The Word_Break property values (UAX #29, table 3), `Other` for every
/// code point the property file does not list.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum WordBreak {
    Other,
    Cr,
    Lf,
    Newline,
    Extend,
    Zwj,
    RegionalIndicator,
    Format,
    Katakana,
    HebrewLetter,
    ALetter,
    SingleQuote,
    DoubleQuote,
    MidNumLet,
    MidLetter,
    MidNum,
    Numeric,
    ExtendNumLet,
    WSegSpace,
}

use WordBreak::*;

/// The Word_Break value of `c`.
pub(crate) fn word_break(c: char) -> WordBreak {
    let cp = u32::from(c);
    if cp < 0x80 {
        return table::ASCII[cp as usize];
    }
    let found = table::RANGES.binary_search_by(|&(first, last, _)| {
        if last < cp {
            Ordering::Less
        } else if first > cp {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
    found.map_or(Other, |k| table::RANGES[k].2)
}

/// Extend, Format and ZWJ: the classes WB4 attaches to the character
/// before them.
fn ignorable(w: WordBreak) -> bool {
    matches!(w, Extend | Format | Zwj)
}

/// AHLetter: ALetter or Hebrew_Letter.
fn ah_letter(w: WordBreak) -> bool {
    matches!(w, ALetter | HebrewLetter)
}

/// MidLetter, MidNumLet or Single_Quote (WB6, WB7).
fn mid_letter(w: WordBreak) -> bool {
    matches!(w, MidLetter | MidNumLet | SingleQuote)
}

/// MidNum, MidNumLet or Single_Quote (WB11, WB12).
fn mid_num(w: WordBreak) -> bool {
    matches!(w, MidNum | MidNumLet | SingleQuote)
}

/// What the rules after WB4 read around a position.
struct Context {
    /// The last character before the position that is not ignorable, and
    /// the one before that (`Other` at the start of the text).
    last: WordBreak,
    second_last: WordBreak,
    /// How many regional indicators end the sequence of non-ignorable
    /// characters before the position.
    regional_run: usize,
}

fn context_before(before: &str) -> Context {
    let mut significant = before.chars().rev().map(word_break).filter(|&w| !ignorable(w));
    let last = significant.next().unwrap_or(Other);
    let second_last = significant.next().unwrap_or(Other);
    let regional_run = match (last, second_last) {
        (RegionalIndicator, RegionalIndicator) => 2 + significant.take_while(|&w| w == RegionalIndicator).count(),
        (RegionalIndicator, _) => 1,
        _ => 0,
    };
    Context { last, second_last, regional_run }
}

/// The class of the first non-ignorable character of `after`, `Other` when
/// there is none.
fn next_significant(after: &str) -> WordBreak {
    after.chars().map(word_break).find(|&w| !ignorable(w)).unwrap_or(Other)
}

/// Rules WB3 to WB4, which read the characters right before and at the
/// position. `None` when none of them decides.
fn newline_and_attachment_rules(prev: WordBreak, current: WordBreak, c: char) -> Option<bool> {
    if prev == Cr && current == Lf {
        return Some(false); // WB3
    }
    if matches!(prev, Cr | Lf | Newline) {
        return Some(true); // WB3a
    }
    if matches!(current, Cr | Lf | Newline) {
        return Some(true); // WB3b
    }
    if prev == Zwj && turbo_unicode::props::extended_pictographic(c) {
        return Some(false); // WB3c
    }
    if prev == WSegSpace && current == WSegSpace {
        return Some(false); // WB3d
    }
    if ignorable(current) {
        return Some(false); // WB4
    }
    None
}

/// Rules WB5 to WB999. `next` is computed only for the rules that look
/// ahead.
fn letter_and_number_rules(current: WordBreak, ctx: &Context, next: impl Fn() -> WordBreak) -> bool {
    let Context { last, second_last, regional_run } = *ctx;
    let no_break = (ah_letter(last) && ah_letter(current)) // WB5
        || (ah_letter(last) && mid_letter(current) && ah_letter(next())) // WB6
        || (ah_letter(second_last) && mid_letter(last) && ah_letter(current)) // WB7
        || (last == HebrewLetter && current == SingleQuote) // WB7a
        || (last == HebrewLetter && current == DoubleQuote && next() == HebrewLetter) // WB7b
        || (second_last == HebrewLetter && last == DoubleQuote && current == HebrewLetter) // WB7c
        || (last == Numeric && current == Numeric) // WB8
        || (ah_letter(last) && current == Numeric) // WB9
        || (last == Numeric && ah_letter(current)) // WB10
        || (second_last == Numeric && mid_num(last) && current == Numeric) // WB11
        || (last == Numeric && mid_num(current) && next() == Numeric) // WB12
        || (last == Katakana && current == Katakana) // WB13
        || ((ah_letter(last) || matches!(last, Numeric | Katakana | ExtendNumLet)) && current == ExtendNumLet) // WB13a
        || (last == ExtendNumLet && (ah_letter(current) || matches!(current, Numeric | Katakana))) // WB13b
        || (last == RegionalIndicator && current == RegionalIndicator && regional_run % 2 == 1); // WB15, WB16
    !no_break // WB999
}

/// Whether byte `at` of `text`, a character boundary, is a word boundary.
/// The start and the end of the text are (WB1, WB2).
pub(crate) fn is_boundary(text: &str, at: usize) -> bool {
    if at == 0 || at >= text.len() {
        return true;
    }
    let (before, after) = text.split_at(at);
    let p = before.chars().next_back().expect("text before the position");
    let c = after.chars().next().expect("text after the position");
    let (prev, current) = (word_break(p), word_break(c));
    if let Some(decided) = newline_and_attachment_rules(prev, current, c) {
        return decided;
    }
    let ctx = context_before(before);
    letter_and_number_rules(current, &ctx, || next_significant(&after[c.len_utf8()..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROPERTY_FILE: &str = include_str!("../../../tests/data/WordBreakProperty-18.0.0.txt");
    const TEST_FILE: &str = include_str!("../../../tests/data/WordBreakTest-18.0.0.txt");

    fn value(name: &str) -> WordBreak {
        match name {
            "CR" => Cr,
            "LF" => Lf,
            "Newline" => Newline,
            "Extend" => Extend,
            "ZWJ" => Zwj,
            "Regional_Indicator" => RegionalIndicator,
            "Format" => Format,
            "Katakana" => Katakana,
            "Hebrew_Letter" => HebrewLetter,
            "ALetter" => ALetter,
            "Single_Quote" => SingleQuote,
            "Double_Quote" => DoubleQuote,
            "MidNumLet" => MidNumLet,
            "MidLetter" => MidLetter,
            "MidNum" => MidNum,
            "Numeric" => Numeric,
            "ExtendNumLet" => ExtendNumLet,
            "WSegSpace" => WSegSpace,
            other => panic!("unknown Word_Break value {other}"),
        }
    }

    #[test]
    fn table_matches_the_property_file() {
        let mut expected = vec![Other; 0x11_0000];
        for line in PROPERTY_FILE.lines() {
            let data = line.split('#').next().unwrap().trim();
            if data.is_empty() {
                continue;
            }
            let (range, name) = data.split_once(';').unwrap();
            let (first, last) = match range.trim().split_once("..") {
                Some((a, b)) => (u32::from_str_radix(a, 16).unwrap(), u32::from_str_radix(b, 16).unwrap()),
                None => {
                    let a = u32::from_str_radix(range.trim(), 16).unwrap();
                    (a, a)
                }
            };
            for cp in first..=last {
                expected[cp as usize] = value(name.trim());
            }
        }
        for (cp, &want) in expected.iter().enumerate() {
            if let Some(c) = char::from_u32(cp as u32) {
                assert_eq!(word_break(c), want, "U+{cp:04X}");
            }
        }
    }

    /// Every line of WordBreakTest.txt: the boundaries `is_boundary` finds
    /// at each code point boundary are the ones the line marks with ÷.
    /// Lines with surrogate code points cannot be written as a Rust string
    /// and are counted as skipped.
    #[test]
    fn conformance_with_word_break_test() {
        let (mut run, mut skipped) = (0, 0);
        for (n, line) in TEST_FILE.lines().enumerate() {
            let data = line.split('#').next().unwrap().trim();
            if data.is_empty() {
                continue;
            }
            let fields: Vec<&str> = data.split_whitespace().collect();
            let mut text = String::new();
            let mut expected = vec![0];
            let mut surrogate = false;
            for pair in fields[1..].chunks(2) {
                match char::from_u32(u32::from_str_radix(pair[0], 16).unwrap()) {
                    Some(c) => text.push(c),
                    None => surrogate = true,
                }
                if pair[1] == "÷" {
                    expected.push(text.len());
                }
            }
            if surrogate {
                skipped += 1;
                continue;
            }
            let found: Vec<usize> =
                (0..=text.len()).filter(|&i| text.is_char_boundary(i) && is_boundary(&text, i)).collect();
            assert_eq!(found, expected, "line {}: {data}", n + 1);
            run += 1;
        }
        assert!(run > 1900, "ran {run} lines");
        assert_eq!(skipped, 0, "lines with surrogates");
    }
}
