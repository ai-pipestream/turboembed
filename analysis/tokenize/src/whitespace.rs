//! OpenNLP's `WhitespaceTokenizer`: a token is a run of non-whitespace in
//! the configured mode. With `keep_newlines`, every line feed and carriage
//! return is also a token of its own.
//!
//! The scan works on bytes, through a 256-entry table built for the mode:
//! an ASCII byte is whitespace or not by the table alone, and only the
//! four lead bytes that can start a non-ASCII whitespace character make
//! the scan decode the character and ask the mode. Java tests UTF-16 code
//! units, but
//! no whitespace character is outside the BMP and a surrogate half is
//! never whitespace, so testing whole characters gives the same tokens.

use turbo_analysis::{Annotator, Error, Refs, Span, View};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_unicode::WhitespaceMode;

use crate::TokenizerOptions;
use crate::unit::tokenize_units;
use crate::utf8::decode;

/// What a byte says about whitespace, for a scan that steps one byte at
/// a time.
const TOKEN: u8 = 0;
/// An ASCII whitespace character.
const SPACE: u8 = 1;
/// The first byte of a character that may be whitespace: decode it and
/// ask the mode.
const DECODE: u8 = 2;

/// The byte table of one mode. ASCII whitespace is U+0009 to U+000D and
/// the space in both modes, plus U+001C to U+001F in legacy mode. Every
/// whitespace character at or above U+0080, in either mode, starts with
/// the byte 0xC2 (U+0085, U+00A0), 0xE1 (U+1680), 0xE2 (U+2000 to
/// U+205F) or 0xE3 (U+3000); a character starting with any other byte is
/// never whitespace, and neither is a continuation byte, so the scan steps
/// over them without decoding.
const fn byte_table(legacy: bool) -> [u8; 256] {
    let mut table = [TOKEN; 256];
    let mut b = 0;
    while b < 0x80 {
        if (b >= 0x09 && b <= 0x0D) || b == 0x20 || (legacy && b >= 0x1C && b <= 0x1F) {
            table[b] = SPACE;
        }
        b += 1;
    }
    table[0xC2] = DECODE;
    table[0xE1] = DECODE;
    table[0xE2] = DECODE;
    table[0xE3] = DECODE;
    table
}

static UNICODE_BYTES: [u8; 256] = byte_table(false);
static LEGACY_BYTES: [u8; 256] = byte_table(true);

/// Appends the whitespace tokens of `text` to `out`, as byte spans of
/// `text`.
pub fn whitespace_spans(text: &str, options: TokenizerOptions, out: &mut Vec<Span>) {
    let mode = options.whitespace;
    let table = match mode {
        WhitespaceMode::Unicode => &UNICODE_BYTES,
        WhitespaceMode::Legacy => &LEGACY_BYTES,
    };
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    loop {
        // Whitespace up to the next token.
        while i < n {
            let b = bytes[i];
            match table[b as usize] {
                TOKEN => break,
                SPACE => {
                    if options.keep_newlines && (b == b'\n' || b == b'\r') {
                        out.push(Span { start: i as u32, end: i as u32 + 1 });
                    }
                    i += 1;
                }
                _ => {
                    let (cp, len) = decode(bytes, i);
                    if !mode.is_whitespace(cp) {
                        break;
                    }
                    i += len;
                }
            }
        }
        if i == n {
            return;
        }
        // The token, up to the next whitespace or the end.
        let start = i as u32;
        i += 1;
        while i < n {
            match table[bytes[i] as usize] {
                TOKEN => i += 1,
                SPACE => break,
                _ => {
                    let (cp, len) = decode(bytes, i);
                    if mode.is_whitespace(cp) {
                        break;
                    }
                    i += len;
                }
            }
        }
        out.push(Span { start, end: i as u32 });
    }
}

/// The whitespace tokenizer as an annotator.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct WhitespaceTokenizer {
    pub options: TokenizerOptions,
}

impl WhitespaceTokenizer {
    pub fn new(options: TokenizerOptions) -> WhitespaceTokenizer {
        WhitespaceTokenizer { options }
    }
}

impl Annotator for WhitespaceTokenizer {
    const ID: &'static str = "turbo-tokenize:whitespace";
    type Reads = (View, Option<Sentences>);
    type Writes = (Tokens,);

    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<(Tokens,), Error> {
        let options = self.options;
        let tokens = tokenize_units(Self::ID, view, sentences, |text, _, out| whitespace_spans(text, options, out))?;
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
        whitespace_spans(text, options, &mut out);
        out.iter().map(|s| &text[s.range()]).collect()
    }

    #[test]
    fn byte_tables_match_the_modes() {
        for (mode, table) in [(WhitespaceMode::Unicode, &UNICODE_BYTES), (WhitespaceMode::Legacy, &LEGACY_BYTES)] {
            for b in 0..0x80u8 {
                assert_eq!(table[b as usize] == SPACE, mode.is_ascii_whitespace(b), "{b:#04x} {mode:?}");
                assert_ne!(table[b as usize], DECODE);
            }
            // Every whitespace character above ASCII starts with a byte marked for decoding.
            for c in (0x80..=0x10_FFFF).filter_map(char::from_u32) {
                let mut buf = [0; 4];
                let lead = c.encode_utf8(&mut buf).as_bytes()[0];
                if mode.is_whitespace(c as u32) {
                    assert_eq!(table[lead as usize], DECODE, "U+{:04X} {mode:?}", c as u32);
                }
            }
            assert!((0x80..0xC0).all(|b| table[b] == TOKEN));
        }
    }

    #[test]
    fn runs_of_non_whitespace() {
        let o = TokenizerOptions::default();
        assert_eq!(tokens("  Mr. Smith's\tcar...  ", o), ["Mr.", "Smith's", "car..."]);
        assert_eq!(tokens("", o), Vec::<&str>::new());
        assert_eq!(tokens(" \u{3000}\u{2029} ", o), Vec::<&str>::new());
        assert_eq!(tokens("a\u{1f600}b\u{3000}\u{e9}", o), ["a\u{1f600}b", "\u{e9}"]);
    }

    #[test]
    fn modes_differ() {
        let unicode = TokenizerOptions::default();
        let legacy = TokenizerOptions { whitespace: WhitespaceMode::Legacy, keep_newlines: false };
        assert_eq!(tokens("a\u{85}b\u{1c}c", unicode), ["a", "b\u{1c}c"]);
        assert_eq!(tokens("a\u{85}b\u{1c}c", legacy), ["a\u{85}b", "c"]);
    }

    #[test]
    fn keep_newlines() {
        let o = TokenizerOptions { whitespace: WhitespaceMode::Unicode, keep_newlines: true };
        let mut out = Vec::new();
        whitespace_spans("\nab \r\ncd\u{2028}e", o, &mut out);
        let expected = [(0, 1), (1, 3), (4, 5), (5, 6), (6, 8), (11, 12)];
        assert_eq!(out, expected.map(|(s, e)| Span::new(s, e)));
    }
}
