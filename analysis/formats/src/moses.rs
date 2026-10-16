//! Moses sentence sample stream and empty-line preprocessor
//! (`opennlp.tools.formats.moses`).
//!
//! Port of `MosesSentenceSampleStream`, `MosesSentenceSampleStreamFactory`, and
//! `EmptyLinePreprocessorStream` from OpenNLP format utilities.

use std::borrow::Cow;

use turbo_analysis::Span;

use crate::error::FormatError;
use crate::sample::SentenceSample;
use crate::source::{Line, Lines, Piece, java_trim};

/// Number of sentences grouped into each [`SentenceSample`] by [`MosesSentenceReader`].
pub const SENTENCES_PER_SAMPLE: usize = 25;

/// Stream preprocessor that skips leading empty lines and normalizes consecutive
/// empty lines into a single empty string (`""`).
///
/// Mirrors `opennlp.tools.formats.moses.EmptyLinePreprocessorStream`.
#[derive(Debug, Clone)]
pub struct EmptyLinePreprocessor<'a> {
    lines: Lines<'a>,
    last_line_was_empty: bool,
}

impl<'a> EmptyLinePreprocessor<'a> {
    /// Wraps a source text in an empty-line preprocessor stream.
    pub fn new(source: &'a str) -> Result<Self, FormatError> {
        Ok(Self { lines: Lines::new(source)?, last_line_was_empty: true })
    }

    #[inline]
    fn is_line_empty(piece: Piece<'_>) -> bool {
        java_trim(piece).text.is_empty()
    }

    /// Reads the next preprocessed line piece.
    pub fn next_piece(&mut self) -> Option<Piece<'a>> {
        let mut line: Line<'a> = self.lines.next()?;
        if self.last_line_was_empty {
            self.last_line_was_empty = false;
            while Self::is_line_empty(line.piece) {
                line = self.lines.next()?;
            }
        }
        if Self::is_line_empty(line.piece) {
            self.last_line_was_empty = true;
            let start = line.span().start;
            return Some(Piece { text: "", span: Span::new(start, start) });
        }
        Some(line.piece)
    }
}

impl<'a> Iterator for EmptyLinePreprocessor<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.next_piece().map(|p| p.text)
    }
}

/// Reader for Moses parallel-corpus sentence files (one sentence per line).
///
/// Groups up to 25 lines from [`EmptyLinePreprocessor`] into each [`SentenceSample`],
/// trimming each line with Java's `String.trim()` semantics and appending a trailing
/// ASCII space `' '` after every sentence (matching `MosesSentenceSampleStream.read()`).
#[derive(Debug, Clone)]
pub struct MosesSentenceReader<'a> {
    lines: EmptyLinePreprocessor<'a>,
}

impl<'a> MosesSentenceReader<'a> {
    /// Creates a new Moses sentence reader over `source`.
    pub fn new(source: &'a str) -> Result<Self, FormatError> {
        Ok(Self { lines: EmptyLinePreprocessor::new(source)? })
    }

    /// Reads the next [`SentenceSample`], or `None` at end of stream.
    pub fn next_sample(&mut self) -> Option<Result<SentenceSample<'a>, FormatError>> {
        let mut document = String::new();
        let mut sentences = Vec::new();
        let mut sources = Vec::new();

        for _ in 0..SENTENCES_PER_SAMPLE {
            let Some(piece) = self.lines.next_piece() else {
                break;
            };
            let trimmed = java_trim(piece);
            let begin = document.len() as u32;
            document.push_str(trimmed.text);
            let end = document.len() as u32;
            document.push(' ');

            sentences.push(Span::new(begin, end));
            sources.push(trimmed.span);
        }

        if sentences.is_empty() {
            None
        } else {
            let source = Span::new(sources.first().map_or(0, |s| s.start), sources.last().map_or(0, |s| s.end));
            Some(SentenceSample::new(Cow::Owned(document), sentences, sources, source))
        }
    }
}

impl<'a> Iterator for MosesSentenceReader<'a> {
    type Item = Result<SentenceSample<'a>, FormatError>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.next_sample()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MOSES_TINY: &str = "\
je|PRO vous|PRO achète|VB un|ART aardvark|NN
je|PRO vous|PRO achète|VB un|ART chat|NN
je|PRO vous|PRO achète|VB un|ART grand|ADJ chat|NN blanc|ADJ
";

    #[test]
    fn empty_line_preprocessor_collapses_blank_lines() {
        let input = "\n  \nfirst line\nsecond line\n\n  \nthird line\n\n";
        let lines: Vec<&str> = EmptyLinePreprocessor::new(input).unwrap().collect();
        assert_eq!(lines, vec!["first line", "second line", "", "third line", ""]);
    }

    #[test]
    fn moses_tiny_sample_matches_opennlp() {
        let mut reader = MosesSentenceReader::new(MOSES_TINY).unwrap();
        let sample = reader.next().expect("one sample").unwrap();
        assert!(reader.next().is_none());
        assert_eq!(sample.sentences().len(), 3);
        assert_eq!(sample.sentence(0), "je|PRO vous|PRO achète|VB un|ART aardvark|NN");
        assert_eq!(sample.sentence(1), "je|PRO vous|PRO achète|VB un|ART chat|NN");
        assert_eq!(sample.sentence(2), "je|PRO vous|PRO achète|VB un|ART grand|ADJ chat|NN blanc|ADJ");
        // Trailing space is kept in sample.document() just like MosesSentenceSampleStream.
        assert!(sample.document().ends_with(' '));
        assert_eq!(&MOSES_TINY[sample.sources()[0].range()], "je|PRO vous|PRO achète|VB un|ART aardvark|NN");
    }

    #[test]
    fn moses_chunks_every_25_sentences() {
        let mut input = String::new();
        for i in 0..30 {
            input.push_str(&format!("Sentence number {i}.\n"));
        }
        let samples: Vec<SentenceSample<'_>> =
            MosesSentenceReader::new(&input).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].sentences().len(), 25);
        assert_eq!(samples[1].sentences().len(), 5);
    }
}
