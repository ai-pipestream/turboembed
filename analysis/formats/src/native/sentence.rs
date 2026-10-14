//! The sentence format: `SentenceSampleStream` over
//! `EmptyLinePreprocessorStream`.
//!
//! One sentence per line, a document (paragraph) ending at an empty
//! line. `<LF>` and `<CR>` in a sentence stand for a line feed and a
//! carriage return inside it.
//!
//! A sample's document is its sentences, each trimmed with `String.trim`
//! (characters up to U+0020) and with the escapes replaced, each followed
//! by one space. The trim comes before the replacement, so an escape at
//! the end of a sentence stays in it as a line break. A line is empty
//! when it is empty after `String.trim`; empty lines at the start and
//! runs of empty lines count as one, so no sample is empty.

use std::borrow::Cow;

use turbo_analysis::Span;

use crate::error::FormatError;
use crate::sample::sentence::SentenceSample;
use crate::source::{Line, Lines, java_trim};

/// Reads sentence samples from the sentence format.
#[derive(Clone, Debug)]
pub struct SentenceReader<'a> {
    lines: Lines<'a>,
    last_was_empty: bool,
}

impl<'a> SentenceReader<'a> {
    pub fn new(source: &'a str) -> Result<SentenceReader<'a>, FormatError> {
        Ok(SentenceReader { lines: Lines::new(source)?, last_was_empty: true })
    }

    /// `EmptyLinePreprocessorStream.read`: the next line, `Err(())` for an
    /// empty line (after skipping the empty lines that follow an empty
    /// line or the start).
    fn next_line(&mut self) -> Option<Result<Line<'a>, ()>> {
        let blank = |l: &Line<'_>| java_trim(l.piece).text.is_empty();
        let mut line = self.lines.next();
        if self.last_was_empty {
            self.last_was_empty = false;
            while line.as_ref().is_some_and(blank) {
                line = self.lines.next();
            }
        }
        let line = line?;
        if blank(&line) {
            self.last_was_empty = true;
            return Some(Err(()));
        }
        Some(Ok(line))
    }
}

impl<'a> Iterator for SentenceReader<'a> {
    type Item = Result<SentenceSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut document = String::new();
        let mut sentences = Vec::new();
        let mut sources = Vec::new();
        let mut source: Option<Span> = None;
        // A trimmed line and its space take no more bytes than the line and
        // its terminator (one more for a last line without one), and the
        // escapes shrink, so no sentence ends past the source's length and
        // the `u32` offsets below cannot overflow.
        while let Some(Ok(line)) = self.next_line() {
            let trimmed = java_trim(line.piece);
            let start = document.len() as u32;
            push_unescaped(&mut document, trimmed.text);
            sentences.push(Span::new(start, document.len() as u32));
            sources.push(trimmed.span);
            source = Some(source.map_or(line.span(), |s| Span::new(s.start, line.span().end)));
            document.push(' ');
        }
        let source = source?;
        Some(SentenceSample::new(Cow::Owned(document), sentences, sources, source))
    }
}

/// `SentenceSampleStream.replaceNewLineEscapeTags`: `<LF>` to a line
/// feed, then `<CR>` to a carriage return.
fn push_unescaped(out: &mut String, text: &str) {
    let mut rest = text;
    while let Some(at) = rest.find("<LF>") {
        push_cr(out, &rest[..at]);
        out.push('\n');
        rest = &rest[at + 4..];
    }
    push_cr(out, rest);
}

fn push_cr(out: &mut String, text: &str) {
    let mut rest = text;
    while let Some(at) = rest.find("<CR>") {
        out.push_str(&rest[..at]);
        out.push('\r');
        rest = &rest[at + 4..];
    }
    out.push_str(rest);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paragraphs_and_escapes() {
        let src = "\n \n  One.  \nTwo<LF>\n\n\n\t\nThree<CR<LF>>.\n";
        let samples: Vec<SentenceSample<'_>> = SentenceReader::new(src).unwrap().map(Result::unwrap).collect();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].document(), "One. Two\n ");
        assert_eq!(samples[0].sentence(1), "Two\n");
        let sources: Vec<&str> = samples[0].sources().iter().map(|s| &src[s.range()]).collect();
        assert_eq!(sources, ["One.", "Two<LF>"]);
        assert_eq!(samples[0].to_string(), "One.\nTwo<LF>\n");
        // A line feed escape inside a carriage return escape leaves the
        // outer one broken.
        assert_eq!(samples[1].document(), "Three<CR\n>. ");
    }

    #[test]
    fn no_samples_from_blank_text() {
        assert_eq!(SentenceReader::new(" \n\n\t").unwrap().count(), 0);
    }
}
