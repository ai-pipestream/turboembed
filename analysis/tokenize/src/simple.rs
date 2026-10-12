//! OpenNLP's `SimpleTokenizer`: a token is a run of one character class.
//!
//! The classes are whitespace (in the configured mode), alphabetic
//! (`Character.isAlphabetic`), digit (`Character.isDigit`) and other.
//! Whitespace separates tokens and is in none. A run of "other"
//! characters splits wherever the character changes, so `...` is one
//! token and `.,` is two.
//!
//! The Java classifies UTF-16 code units, so it makes each half of a
//! character outside the Basic Multilingual Plane an "other" token of its
//! own. Half a character is not a UTF-8 span, so here such a character is
//! one "other" token covering the whole character, never merged with its
//! neighbours, which is exactly where the Java puts the outer edges of
//! its two halves.
//!
//! With `keep_newlines`, every line feed and carriage return is also a
//! token of its own at its real position. The Java throws when a line
//! break comes before the first token and gives a stale span to a line
//! break that follows whitespace; both are fixed here.

use turbo_analysis::{Annotator, Error, Refs, Span, View};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_unicode::WhitespaceMode;
use turbo_unicode::java::code_point as java;

use crate::TokenizerOptions;
use crate::unit::tokenize_units;
use crate::utf8::decode;

/// The character classes of `SimpleTokenizer.CharacterEnum`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Class {
    Whitespace,
    Alphabetic,
    Numeric,
    Other,
}

/// The class of every ASCII character in one whitespace mode. ASCII
/// letters are the only alphabetic ASCII characters and ASCII digits the
/// only digits; the two modes differ on U+001C to U+001F.
const fn ascii_classes(legacy: bool) -> [Class; 128] {
    let mut table = [Class::Other; 128];
    let mut b = 0;
    while b < 128 {
        let c = b as u8;
        table[b] = if (c >= 0x09 && c <= 0x0D) || c == b' ' || (legacy && c >= 0x1C && c <= 0x1F) {
            Class::Whitespace
        } else if c.is_ascii_alphabetic() {
            Class::Alphabetic
        } else if c.is_ascii_digit() {
            Class::Numeric
        } else {
            Class::Other
        };
        b += 1;
    }
    table
}

static UNICODE_ASCII: [Class; 128] = ascii_classes(false);
static LEGACY_ASCII: [Class; 128] = ascii_classes(true);

/// The class of a character at or above U+0080. A character outside the
/// BMP is "other", as each of its surrogate halves is in Java.
#[inline]
fn classify(cp: u32, mode: WhitespaceMode) -> Class {
    if mode.is_whitespace(cp) {
        Class::Whitespace
    } else if cp >= 0x1_0000 {
        Class::Other
    } else if java::is_alphabetic(cp) {
        Class::Alphabetic
    } else if java::is_digit(cp) {
        Class::Numeric
    } else {
        Class::Other
    }
}

/// A previous character no character equals: what follows a character
/// outside the BMP, so nothing extends its token.
const NONE: u32 = u32::MAX;

/// Appends the simple tokens of `text` to `out`, as byte spans of `text`.
pub fn simple_spans(text: &str, options: TokenizerOptions, out: &mut Vec<Span>) {
    let mode = options.whitespace;
    let table = match mode {
        WhitespaceMode::Unicode => &UNICODE_ASCII,
        WhitespaceMode::Legacy => &LEGACY_ASCII,
    };
    let bytes = text.as_bytes();
    let mut state = Class::Whitespace;
    let mut start = 0u32;
    let mut prev = NONE;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        let (cp, len, class) = if b < 0x80 {
            (u32::from(b), 1, table[b as usize])
        } else {
            let (cp, len) = decode(bytes, i);
            (cp, len, classify(cp, mode))
        };
        let at = i as u32;
        if state == Class::Whitespace {
            if class != Class::Whitespace {
                start = at;
            }
        } else if class != state || (class == Class::Other && cp != prev) {
            out.push(Span { start, end: at });
            start = at;
        }
        if options.keep_newlines && (b == b'\n' || b == b'\r') {
            out.push(Span { start: at, end: at + 1 });
        }
        state = class;
        prev = if cp >= 0x1_0000 { NONE } else { cp };
        i += len;
    }
    if state != Class::Whitespace {
        out.push(Span { start, end: bytes.len() as u32 });
    }
}

/// The simple tokenizer as an annotator.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct SimpleTokenizer {
    pub options: TokenizerOptions,
}

impl SimpleTokenizer {
    pub fn new(options: TokenizerOptions) -> SimpleTokenizer {
        SimpleTokenizer { options }
    }
}

impl Annotator for SimpleTokenizer {
    const ID: &'static str = "turbo-tokenize:simple";
    type Reads = (View, Option<Sentences>);
    type Writes = (Tokens,);

    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<(Tokens,), Error> {
        let options = self.options;
        let tokens = tokenize_units(Self::ID, view, sentences, |text, _, out| simple_spans(text, options, out))?;
        Ok((tokens,))
    }

    fn config_hash(&self) -> u64 {
        self.options.config_hash()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(text: &str, options: TokenizerOptions) -> Vec<&str> {
        let mut out = Vec::new();
        simple_spans(text, options, &mut out);
        out.iter().map(|s| &text[s.range()]).collect()
    }

    fn plain(text: &str) -> Vec<&str> {
        tokens(text, TokenizerOptions::default())
    }

    const NEWLINES: TokenizerOptions = TokenizerOptions { whitespace: WhitespaceMode::Unicode, keep_newlines: true };

    #[test]
    fn ascii_table_matches_the_classifier() {
        for mode in [WhitespaceMode::Unicode, WhitespaceMode::Legacy] {
            let table = if mode == WhitespaceMode::Unicode { &UNICODE_ASCII } else { &LEGACY_ASCII };
            for cp in 0..0x80u32 {
                let expected = if mode.is_whitespace(cp) {
                    Class::Whitespace
                } else if java::is_alphabetic(cp) {
                    Class::Alphabetic
                } else if java::is_digit(cp) {
                    Class::Numeric
                } else {
                    Class::Other
                };
                assert_eq!(table[cp as usize], expected, "U+{cp:04X} {mode:?}");
            }
        }
    }

    #[test]
    fn classes_split_tokens() {
        // Worked through SimpleTokenizer.tokenizePos by hand.
        assert_eq!(plain("Mr. Smith's 3rd car..."), ["Mr", ".", "Smith", "'", "s", "3", "rd", "car", "..."]);
        assert_eq!(plain("a.,b"), ["a", ".", ",", "b"]);
        assert_eq!(plain("  x  "), ["x"]);
        assert_eq!(plain(""), Vec::<&str>::new());
        assert_eq!(plain(" \t\n "), Vec::<&str>::new());
        // U+00E9 is alphabetic, U+0663 an Arabic-Indic digit, U+00BD (No) neither.
        assert_eq!(plain("caf\u{e9}\u{663}\u{663}\u{bd}\u{bd}"), ["caf\u{e9}", "\u{663}\u{663}", "\u{bd}\u{bd}"]);
        // A Roman numeral is alphabetic in Java (Nl with Other_Alphabetic).
        assert_eq!(plain("x\u{2167}1"), ["x\u{2167}", "1"]);
    }

    #[test]
    fn whitespace_modes() {
        // U+0085 is whitespace in Unicode mode only, U+001F in legacy mode only.
        let unicode = TokenizerOptions::default();
        let legacy = TokenizerOptions { whitespace: WhitespaceMode::Legacy, keep_newlines: false };
        assert_eq!(tokens("a\u{85}b", unicode), ["a", "b"]);
        assert_eq!(tokens("a\u{85}b", legacy), ["a", "\u{85}", "b"]);
        assert_eq!(tokens("a\u{1f}b", unicode), ["a", "\u{1f}", "b"]);
        assert_eq!(tokens("a\u{1f}b", legacy), ["a", "b"]);
        // The no-break space is whitespace in both.
        assert_eq!(tokens("a\u{a0}b", legacy), ["a", "b"]);
    }

    #[test]
    fn characters_outside_the_bmp_are_one_token_each() {
        // Java gives each surrogate half a token; here each character is one.
        assert_eq!(plain("a\u{1f600}\u{1f600}b"), ["a", "\u{1f600}", "\u{1f600}", "b"]);
        // A letter and a digit outside the BMP are "other" too, as in Java.
        assert_eq!(plain("x\u{10400}y\u{1d7ce}"), ["x", "\u{10400}", "y", "\u{1d7ce}"]);
        // The character after one is never merged with it.
        assert_eq!(plain("\u{1f600}!!"), ["\u{1f600}", "!!"]);
        assert_eq!(plain("!\u{1f600}"), ["!", "\u{1f600}"]);
    }

    #[test]
    fn keep_newlines_at_real_positions() {
        assert_eq!(tokens("ab\ncd", NEWLINES), ["ab", "\n", "cd"]);
        assert_eq!(tokens("a\r\n\r\nb", NEWLINES), ["a", "\r", "\n", "\r", "\n", "b"]);
        // Java throws here: the line break comes before the first token.
        let mut out = Vec::new();
        simple_spans("\n x", NEWLINES, &mut out);
        assert_eq!(out, [Span::new(0, 1), Span::new(2, 3)]);
        // Java reports Span(2, 3) for the break after the space: it covers the space.
        out.clear();
        simple_spans("ab \nc", NEWLINES, &mut out);
        assert_eq!(out, [Span::new(0, 2), Span::new(3, 4), Span::new(4, 5)]);
        // U+2028 and U+0085 are whitespace but not kept.
        assert_eq!(tokens("a\u{2028}\u{85}b", NEWLINES), ["a", "b"]);
    }
}
