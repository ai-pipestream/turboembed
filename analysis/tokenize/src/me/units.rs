//! The whitespace units TokenizerME starts from: OpenNLP's
//! `WhitespaceTokenizer.tokenizePos`, splitting at Unicode `White_Space`.
//!
//! A unit is a maximal run of characters that are not whitespace. With
//! `keep_newlines`, every line feed and carriage return is also a unit of
//! its own, at its own position.

use turbo_unicode::whitespace::{is_ascii_whitespace, is_whitespace};

/// The whitespace units of a text, as byte ranges `(start, end)`.
pub struct WhitespaceUnits<'a> {
    text: &'a str,
    at: usize,
    keep_newlines: bool,
}

impl<'a> WhitespaceUnits<'a> {
    pub fn new(text: &'a str, keep_newlines: bool) -> WhitespaceUnits<'a> {
        WhitespaceUnits { text, at: 0, keep_newlines }
    }

    /// Whether the character starting at byte `i` is whitespace, and its
    /// length in bytes.
    #[inline]
    fn whitespace_at(&self, i: usize) -> (bool, usize) {
        let b = self.text.as_bytes()[i];
        if b < 0x80 {
            return (is_ascii_whitespace(b), 1);
        }
        let c = self.text[i..].chars().next().expect("a character boundary");
        (is_whitespace(c as u32), c.len_utf8())
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

    fn units(text: &str, keep_newlines: bool) -> Vec<&str> {
        WhitespaceUnits::new(text, keep_newlines).map(|(s, e)| &text[s..e]).collect()
    }

    #[test]
    fn runs_of_non_whitespace() {
        assert_eq!(units("  a bc\tdef  ", false), ["a", "bc", "def"]);
        assert_eq!(units("", false), Vec::<&str>::new());
        assert_eq!(units(" \n ", false), Vec::<&str>::new());
        assert_eq!(units("caf\u{e9}\u{3000}\u{1f600}x", false), ["caf\u{e9}", "\u{1f600}x"]);
    }

    #[test]
    fn next_line_separates_and_information_separators_do_not() {
        assert_eq!(units("a\u{85}b", false), ["a", "b"]);
        assert_eq!(units("a\u{1f}b", false), ["a\u{1f}b"]);
        assert_eq!(units("a\u{a0}b", false), ["a", "b"]);
    }

    #[test]
    fn kept_line_breaks() {
        assert_eq!(units("a\r\n b\n", true), ["a", "\r", "\n", "b", "\n"]);
        assert_eq!(units("\nx", true), ["\n", "x"]);
        // U+2028 is whitespace but not a kept line break.
        assert_eq!(units("a\u{2028}b", true), ["a", "b"]);
    }
}
