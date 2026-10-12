//! The whitespace units TokenizerME starts from: OpenNLP's
//! `WhitespaceTokenizer.tokenizePos` in the configured whitespace mode.
//!
//! A unit is a maximal run of characters that are not whitespace. With
//! `keep_newlines`, every line feed and carriage return is also a unit of
//! its own, at its own position.

use turbo_unicode::WhitespaceMode;

/// The whitespace units of a text, as byte ranges `(start, end)`.
pub struct WhitespaceUnits<'a> {
    text: &'a str,
    at: usize,
    mode: WhitespaceMode,
    keep_newlines: bool,
}

impl<'a> WhitespaceUnits<'a> {
    pub fn new(text: &'a str, mode: WhitespaceMode, keep_newlines: bool) -> WhitespaceUnits<'a> {
        WhitespaceUnits { text, at: 0, mode, keep_newlines }
    }

    /// Whether the character starting at byte `i` is whitespace, and its
    /// length in bytes.
    #[inline]
    fn whitespace_at(&self, i: usize) -> (bool, usize) {
        let b = self.text.as_bytes()[i];
        if b < 0x80 {
            return (self.mode.is_ascii_whitespace(b), 1);
        }
        let c = self.text[i..].chars().next().expect("a character boundary");
        (self.mode.is_whitespace(c as u32), c.len_utf8())
    }
}

impl Iterator for WhitespaceUnits<'_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        let bytes = self.text.as_bytes();
        // Skip whitespace, stopping at a kept line break.
        while self.at < bytes.len() {
            let (ws, len) = self.whitespace_at(self.at);
            if !ws {
                break;
            }
            if self.keep_newlines && matches!(bytes[self.at], b'\n' | b'\r') {
                self.at += 1;
                return Some((self.at - 1, self.at));
            }
            self.at += len;
        }
        if self.at >= bytes.len() {
            return None;
        }
        let start = self.at;
        while self.at < bytes.len() {
            let (ws, len) = self.whitespace_at(self.at);
            if ws {
                break;
            }
            self.at += len;
        }
        Some((start, self.at))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use WhitespaceMode::{Legacy, Unicode};

    fn units(text: &str, mode: WhitespaceMode, keep_newlines: bool) -> Vec<&str> {
        WhitespaceUnits::new(text, mode, keep_newlines).map(|(s, e)| &text[s..e]).collect()
    }

    #[test]
    fn runs_of_non_whitespace() {
        assert_eq!(units("  a bc\tdef  ", Unicode, false), ["a", "bc", "def"]);
        assert_eq!(units("", Unicode, false), Vec::<&str>::new());
        assert_eq!(units(" \n ", Unicode, false), Vec::<&str>::new());
        assert_eq!(units("caf\u{e9}\u{3000}\u{1f600}x", Unicode, false), ["caf\u{e9}", "\u{1f600}x"]);
    }

    #[test]
    fn the_modes_differ() {
        // U+0085 is whitespace only in the Unicode mode; U+001F only in legacy.
        assert_eq!(units("a\u{85}b", Unicode, false), ["a", "b"]);
        assert_eq!(units("a\u{85}b", Legacy, false), ["a\u{85}b"]);
        assert_eq!(units("a\u{1f}b", Unicode, false), ["a\u{1f}b"]);
        assert_eq!(units("a\u{1f}b", Legacy, false), ["a", "b"]);
        // The no-break space is whitespace in both.
        assert_eq!(units("a\u{a0}b", Legacy, false), ["a", "b"]);
    }

    #[test]
    fn kept_line_breaks() {
        assert_eq!(units("a\r\n b\n", Unicode, true), ["a", "\r", "\n", "b", "\n"]);
        assert_eq!(units("\nx", Unicode, true), ["\n", "x"]);
        // U+2028 is whitespace but not a kept line break.
        assert_eq!(units("a\u{2028}b", Unicode, true), ["a", "b"]);
    }
}
