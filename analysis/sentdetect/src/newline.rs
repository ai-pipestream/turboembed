//! One sentence per line: OpenNLP's `NewlineSentenceDetector`.
//!
//! A sentence is each run of text between `\n` or `\r` characters, trimmed
//! of whitespace in the configured mode, and dropped when nothing remains.
//! `U+2028`, `U+2029` and `U+0085` do not end a line here, as they do not
//! in the Java. `\r\n` is two separators with an empty piece between them,
//! which is dropped, so it ends one line as the reader expects.

use turbo_analysis::{Annotator, Error, Refs, Span, Spans, View};
use turbo_analysis_layers::Sentences;
use turbo_unicode::WhitespaceMode;

use crate::scan::{next_break, trim};
use crate::{check_len, mode_hash};

/// Writes one sentence per non-blank line of a view.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct NewlineDetector {
    /// What counts as whitespace when a line is trimmed.
    pub whitespace: WhitespaceMode,
}

impl NewlineDetector {
    pub fn new(whitespace: WhitespaceMode) -> NewlineDetector {
        NewlineDetector { whitespace }
    }
}

impl Annotator for NewlineDetector {
    const ID: &'static str = "turbo-sentdetect:newline";
    type Reads = (View,);
    type Writes = (Sentences,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Sentences,), Error> {
        let text = view.text();
        check_len(Self::ID, text)?;
        let mut spans = Spans::new(view.id());
        each_line(text, self.whitespace, |start, end| {
            spans.push(Span::new(start, end));
        });
        Ok((Sentences { spans },))
    }

    fn config_hash(&self) -> u64 {
        mode_hash(self.whitespace)
    }
}

/// Appends to `out` the span of every sentence in `text`, in text order,
/// as UTF-8 byte offsets.
///
/// # Panics
///
/// When `text` is longer than `u32::MAX` bytes, since offsets are `u32`.
pub fn newline_spans(text: &str, mode: WhitespaceMode, out: &mut Vec<Span>) {
    assert!(u32::try_from(text.len()).is_ok(), "a text of at most u32::MAX bytes");
    each_line(text, mode, |start, end| out.push(Span::new(start, end)));
}

/// Calls `emit` with the byte offsets of every sentence. This follows
/// `sentPosDetect` step for step: a separator closes the piece since the
/// last one only when that piece is not empty, so after a run of
/// separators the next piece begins with the last of them, which the trim
/// removes. Offsets are counted in bytes rather than UTF-16 units; since
/// the separators are single ASCII characters, every comparison the Java
/// makes on units comes out the same on bytes.
#[inline]
fn each_line(text: &str, mode: WhitespaceMode, mut emit: impl FnMut(u32, u32)) {
    let bytes = text.as_bytes();
    let mut start = 0;
    let mut i = next_break(bytes, 0);
    while i < bytes.len() {
        if i > start {
            if let Some((s, e)) = trim(text, start, i, mode) {
                emit(s as u32, e as u32);
            }
            start = i + 1;
        }
        i = next_break(bytes, i + 1);
    }
    if bytes.len() > start
        && let Some((s, e)) = trim(text, start, bytes.len(), mode)
    {
        emit(s as u32, e as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use WhitespaceMode::{Legacy, Unicode};
    use turbo_analysis::Document;
    use turbo_unicode::utf16::byte_to_utf16;

    fn spans(text: &str, mode: WhitespaceMode) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        newline_spans(text, mode, &mut out);
        out.iter().map(|s| (s.start, s.end)).collect()
    }

    fn utf16(text: &str, mode: WhitespaceMode) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        newline_spans(text, mode, &mut out);
        out.iter()
            .map(|s| (byte_to_utf16(text, s.start as usize).unwrap(), byte_to_utf16(text, s.end as usize).unwrap()))
            .collect()
    }

    #[test]
    fn empty_and_whitespace_only() {
        for mode in [Unicode, Legacy] {
            assert_eq!(spans("", mode), []);
            assert_eq!(spans(" \t\n \r\n\u{3000}\r", mode), []);
            assert_eq!(spans("\n\n\r\r", mode), []);
        }
    }

    #[test]
    fn crlf_runs() {
        // a=0 \r=1 \n=2 b=3 \r=4 \n=5 \r=6 \n=7 c=8
        assert_eq!(spans("a\r\nb\r\n\r\nc", Unicode), [(0, 1), (3, 4), (8, 9)]);
        // A lone CR ends a line too, which str::lines would not do.
        assert_eq!(spans("one\rtwo", Unicode), [(0, 3), (4, 7)]);
    }

    #[test]
    fn last_line_without_a_newline() {
        assert_eq!(spans("x\n  last  ", Unicode), [(0, 1), (4, 8)]);
        assert_eq!(spans("only", Legacy), [(0, 4)]);
    }

    #[test]
    fn leading_separators() {
        // The Java keeps the first separator in the next piece; the trim
        // removes it, so the sentence starts at the text.
        assert_eq!(spans("\n\nab", Unicode), [(2, 4)]);
        assert_eq!(spans("\r\n  ab\n", Unicode), [(4, 6)]);
    }

    #[test]
    fn unicode_spaces_by_mode() {
        // U+00A0 at 0..2, a at 2, U+2028 at 3..6, b at 6, U+0085 at 7..9.
        let text = "\u{A0}a\u{2028}b\u{85}";
        // U+2028 does not end a line in either mode.
        assert_eq!(spans(text, Unicode), [(2, 7)]);
        assert_eq!(utf16(text, Unicode), [(1, 4)]);
        // U+0085 is not legacy whitespace, so it stays in the sentence.
        assert_eq!(spans(text, Legacy), [(2, 9)]);
        assert_eq!(utf16(text, Legacy), [(1, 5)]);
        // U+001C is legacy whitespace only.
        assert_eq!(spans("a\u{1C}\nb", Unicode), [(0, 2), (3, 4)]);
        assert_eq!(spans("a\u{1C}\nb", Legacy), [(0, 1), (3, 4)]);
    }

    #[test]
    fn outside_the_bmp() {
        // U+1F600 at 0..4 (units 0..2), space, x at 5, LF at 6, U+10400 at 7..11.
        let text = "\u{1F600} x\n\u{10400}";
        assert_eq!(spans(text, Unicode), [(0, 6), (7, 11)]);
        assert_eq!(utf16(text, Unicode), [(0, 4), (5, 7)]);
    }

    #[test]
    fn annotator_writes_sentences() {
        let doc = Document::new("first\r\n\r\n second \n");
        let view = doc.get::<View>().unwrap();
        let (sentences,) = NewlineDetector::default().annotate((view,)).unwrap();
        assert_eq!(sentences.spans.view(), view.id());
        assert_eq!(sentences.spans.columns(), (&[0, 10][..], &[5, 16][..]));
        assert_ne!(NewlineDetector::new(Unicode).config_hash(), NewlineDetector::new(Legacy).config_hash());
    }
}
