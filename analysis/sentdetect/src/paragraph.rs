//! Paragraphs: maximal runs of non-blank lines.
//!
//! A line ends at `\n`, at `\r\n` or at a lone `\r`. These three are the
//! line breaks `NewlineDetector` sees: it splits at every `\n` and `\r`,
//! and the empty piece inside `\r\n` is dropped, so `\r\n` ends one line,
//! not two. Counting it as one here keeps the two detectors agreeing on
//! where lines are; counting it as two would put a blank line inside
//! every CR LF break and make each line of a Windows file a paragraph.
//! `U+2028`, `U+2029` and `U+0085` do not end a line, again as in the
//! newline detector; they are whitespace inside a line. That holds for
//! `U+2029 PARAGRAPH SEPARATOR` too, despite its name: OpenNLP's
//! `NewlineSentenceDetector` splits only at `\n` and `\r`, so ending a
//! paragraph at `U+2029` would put a paragraph boundary inside what
//! `NewlineDetector` reports as one line.
//!
//! A line is blank when every character on it is whitespace in the
//! configured mode (an empty line is blank). A paragraph's span starts at
//! the first non-whitespace character of its first line and ends after the
//! last non-whitespace character of its last line, so it holds its inner
//! line breaks but no leading or trailing whitespace. This is the
//! definition docs/segmentation.md gives for `turbo:structure`; OpenNLP's
//! own paragraph readers each differ slightly from it and from each other.

use turbo_analysis::{Annotator, Error, Refs, Span, Spans, View};
use turbo_analysis_layers::{Block, Structure};
use turbo_unicode::WhitespaceMode;

use crate::scan::{next_break, skip_whitespace, skip_whitespace_back};
use crate::{check_len, mode_hash};

/// Writes every paragraph of a view as a top-level `Block::Paragraph`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ParagraphDetector {
    /// What counts as whitespace on a blank line and at a paragraph's ends.
    pub whitespace: WhitespaceMode,
}

impl ParagraphDetector {
    pub fn new(whitespace: WhitespaceMode) -> ParagraphDetector {
        ParagraphDetector { whitespace }
    }
}

impl Annotator for ParagraphDetector {
    const ID: &'static str = "turbo-sentdetect:paragraph";
    type Reads = (View,);
    type Writes = (Structure,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Structure,), Error> {
        let text = view.text();
        check_len(Self::ID, text)?;
        let mut spans = Spans::new(view.id());
        each_paragraph(text, self.whitespace, |start, end| {
            spans.push(Span::new(start, end));
        });
        let n = spans.len();
        Ok((Structure { spans, kind: vec![Block::Paragraph; n], parent: vec![None; n], depth: vec![0; n] },))
    }

    fn config_hash(&self) -> u64 {
        mode_hash(self.whitespace)
    }
}

/// Appends to `out` the span of every paragraph in `text`, in text order,
/// as UTF-8 byte offsets.
///
/// # Panics
///
/// When `text` is longer than `u32::MAX` bytes, since offsets are `u32`.
pub fn paragraph_spans(text: &str, mode: WhitespaceMode, out: &mut Vec<Span>) {
    assert!(u32::try_from(text.len()).is_ok(), "a text of at most u32::MAX bytes");
    each_paragraph(text, mode, |start, end| out.push(Span::new(start, end)));
}

/// Calls `emit` with the byte offsets of every paragraph. One pass over
/// the lines: a non-blank line opens a paragraph or extends the open one
/// to its last non-whitespace character, and a blank line or the end of
/// the text closes it.
#[inline]
fn each_paragraph(text: &str, mode: WhitespaceMode, mut emit: impl FnMut(u32, u32)) {
    let bytes = text.as_bytes();
    let mut open: Option<(usize, usize)> = None;
    let mut line = 0;
    loop {
        let brk = next_break(bytes, line);
        let first = skip_whitespace(text, line, brk, mode);
        if first == brk {
            if let Some((s, e)) = open.take() {
                emit(s as u32, e as u32);
            }
        } else {
            let last = skip_whitespace_back(text, first, brk, mode);
            match &mut open {
                Some((_, end)) => *end = last,
                None => open = Some((first, last)),
            }
        }
        if brk == bytes.len() {
            break;
        }
        line = brk + 1 + usize::from(bytes[brk] == b'\r' && bytes.get(brk + 1) == Some(&b'\n'));
    }
    if let Some((s, e)) = open {
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
        paragraph_spans(text, mode, &mut out);
        out.iter().map(|s| (s.start, s.end)).collect()
    }

    fn utf16(text: &str, mode: WhitespaceMode) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        paragraph_spans(text, mode, &mut out);
        out.iter()
            .map(|s| (byte_to_utf16(text, s.start as usize).unwrap(), byte_to_utf16(text, s.end as usize).unwrap()))
            .collect()
    }

    #[test]
    fn empty_and_whitespace_only() {
        for mode in [Unicode, Legacy] {
            assert_eq!(spans("", mode), []);
            assert_eq!(spans(" \n\t\r\n\u{3000}\r\u{2029}", mode), []);
            assert_eq!(spans("\r\n\r\n", mode), []);
        }
    }

    #[test]
    fn crlf_is_one_break() {
        // a=0 \r=1 \n=2 b=3 \r=4 \n=5 \r=6 \n=7 c=8: CR LF CR LF is one blank line.
        assert_eq!(spans("a\r\nb\r\n\r\nc", Unicode), [(0, 4), (8, 9)]);
        // A line of one space between two CR LF breaks is blank.
        assert_eq!(spans("a\r\n \r\n\tb", Unicode), [(0, 1), (7, 8)]);
        // A lone CR is a break; LF then CR is two breaks, so a blank line.
        assert_eq!(spans("a\rb", Unicode), [(0, 3)]);
        assert_eq!(spans("a\n\rb", Unicode), [(0, 1), (3, 4)]);
    }

    #[test]
    fn trimmed_and_last_without_newline() {
        let text = "  p1 line1\n p1 line2  \n\n\n p2";
        assert_eq!(spans(text, Unicode), [(2, 20), (26, 28)]);
        assert_eq!(&text[2..20], "p1 line1\n p1 line2");
        assert_eq!(spans("one paragraph", Legacy), [(0, 13)]);
    }

    #[test]
    fn unicode_spaces_by_mode() {
        // a=0 LF=1 U+0085=2..4 LF=4 b=5. U+0085 is White_Space but not
        // legacy whitespace, so its line is blank in one mode only.
        let text = "a\n\u{85}\nb";
        assert_eq!(spans(text, Unicode), [(0, 1), (5, 6)]);
        assert_eq!(utf16(text, Unicode), [(0, 1), (4, 5)]);
        assert_eq!(spans(text, Legacy), [(0, 6)]);
        assert_eq!(utf16(text, Legacy), [(0, 5)]);
        // U+00A0 and U+2028 are whitespace in both modes; U+2028 is not a break.
        for mode in [Unicode, Legacy] {
            assert_eq!(spans("a\n\u{A0}\nb", mode), [(0, 1), (5, 6)]);
            assert_eq!(spans("a\n\u{2028}\nb", mode), [(0, 1), (6, 7)]);
            assert_eq!(spans("a\u{2028}\u{2028}b", mode), [(0, 8)]);
        }
        // U+001C is legacy whitespace only.
        assert_eq!(spans("a\n\u{1C}\nb", Unicode), [(0, 5)]);
        assert_eq!(spans("a\n\u{1C}\nb", Legacy), [(0, 1), (4, 5)]);
    }

    #[test]
    fn outside_the_bmp() {
        // U+1F600=0..4 LF=4 LF=5 U+10400=6..10 space=10 U+1F600=11..15.
        let text = "\u{1F600}\n\n\u{10400} \u{1F600}";
        assert_eq!(spans(text, Unicode), [(0, 4), (6, 15)]);
        assert_eq!(utf16(text, Unicode), [(0, 2), (4, 9)]);
    }

    #[test]
    fn annotator_writes_top_level_paragraphs() {
        let doc = Document::new("one\ntwo\n\n three \n");
        let view = doc.get::<View>().unwrap();
        let (s,) = ParagraphDetector::default().annotate((view,)).unwrap();
        assert_eq!(s.spans.columns(), (&[0, 10][..], &[7, 15][..]));
        assert_eq!(s.kind, [Block::Paragraph, Block::Paragraph]);
        assert_eq!(s.parent, [None, None]);
        assert_eq!(s.depth, [0, 0]);
        assert_ne!(ParagraphDetector::new(Unicode).config_hash(), ParagraphDetector::new(Legacy).config_hash());
    }
}
