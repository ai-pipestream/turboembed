//! Document-level quality statistics in the manner of the Gopher quality
//! filters (Rae et al. 2021) and the C4 line filter (Raffel et al. 2020).
//! They are measurements only: no threshold is applied here, the caller
//! (the chunker) decides what a value means.

use std::collections::HashSet;

use turbo_unicode::props;

use super::token::core;
use crate::artifacts::lines::trimmed_lines;

/// Statistics of one text. Ratios and fractions are 0 when their
/// denominator is 0.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct NoiseStats {
    /// Code points in the text.
    pub chars: u32,
    /// Words: `White_Space`-delimited tokens that keep at least one
    /// character once surrounding punctuation is trimmed.
    pub words: u32,
    /// Lines that are not blank.
    pub lines: u32,
    /// `#` characters plus ellipses (U+2026, or a run of three or more
    /// full stops counted once).
    pub symbols: u32,
    /// `symbols / words`. Gopher drops documents above 0.1.
    pub symbol_to_word_ratio: f64,
    /// The mean length of the words in code points, surrounding
    /// punctuation trimmed. Gopher keeps documents between 3 and 10.
    pub mean_word_length: f64,
    /// The share of lines that end in terminal punctuation (after trailing
    /// whitespace). C4 keeps only such lines.
    pub terminal_punctuation_fraction: f64,
    /// The share of lines that repeat an earlier line exactly (after
    /// trimming `White_Space` at both ends).
    pub duplicate_line_fraction: f64,
    /// The code points of those repeating lines over all code points of
    /// the text. Gopher drops documents above 0.2.
    pub duplicate_line_char_fraction: f64,
}

/// Characters that end a sentence or a quoted sentence at the end of a
/// line: C4's `.`, `!`, `?` and `"`, the apostrophe, and their common
/// non-ASCII forms (curly quotes, guillemet, ellipsis, the CJK full stops
/// and marks, the Arabic question mark and full stop, the Devanagari
/// danda).
const TERMINAL: [char; 16] = [
    '.', '!', '?', '"', '\'', '\u{2019}', '\u{201D}', '\u{00BB}', '\u{2026}', '\u{3002}', '\u{FF01}', '\u{FF1F}',
    '\u{FF0E}', '\u{061F}', '\u{06D4}', '\u{0964}',
];

impl NoiseStats {
    /// Measures `text`.
    pub fn of(text: &str) -> NoiseStats {
        let chars = text.chars().count() as u32;
        let (words, word_chars) = words(text);
        let symbols = symbols(text);
        let mut lines = 0u32;
        let mut terminal = 0u32;
        let mut duplicates = 0u32;
        let mut duplicate_chars = 0u32;
        let mut seen: HashSet<&str> = HashSet::new();
        for range in trimmed_lines(text) {
            let line = &text[range];
            lines += 1;
            terminal += u32::from(line.chars().next_back().is_some_and(|c| TERMINAL.contains(&c)));
            if !seen.insert(line) {
                duplicates += 1;
                duplicate_chars += line.chars().count() as u32;
            }
        }
        NoiseStats {
            chars,
            words,
            lines,
            symbols,
            symbol_to_word_ratio: ratio(symbols, words),
            mean_word_length: ratio(word_chars, words),
            terminal_punctuation_fraction: ratio(terminal, lines),
            duplicate_line_fraction: ratio(duplicates, lines),
            duplicate_line_char_fraction: ratio(duplicate_chars, chars),
        }
    }
}

fn ratio(a: u32, b: u32) -> f64 {
    if b == 0 { 0.0 } else { f64::from(a) / f64::from(b) }
}

/// The number of words and the code points in them.
fn words(text: &str) -> (u32, u32) {
    let mut words = 0;
    let mut chars = 0;
    for token in text.split(props::white_space).filter(|t| !t.is_empty()) {
        let (from, to) = core(token);
        if from < to {
            words += 1;
            chars += token[from..to].chars().count() as u32;
        }
    }
    (words, chars)
}

/// `#` characters, U+2026 HORIZONTAL ELLIPSIS, and runs of three or more
/// full stops.
fn symbols(text: &str) -> u32 {
    let mut count = 0;
    let mut stops = 0;
    for c in text.chars() {
        if c == '.' {
            stops += 1;
            continue;
        }
        count += u32::from(stops >= 3);
        stops = 0;
        count += u32::from(c == '#' || c == '\u{2026}');
    }
    count + u32::from(stops >= 3)
}
