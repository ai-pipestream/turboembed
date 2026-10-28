//! The token and sentence formats.
//!
//! Token: one sentence per line, written as the tokenizer should split
//! it: whitespace between tokens the text separates, `<SPLIT>` between
//! tokens it writes together. The sample's text is the line rebuilt, each
//! piece followed by one space.
//!
//! Sentence: one sentence per line, a document (paragraph) ending at an
//! empty line. `<LF>` and `<CR>` in a sentence stand for a line feed and
//! a carriage return inside it; each sentence is trimmed.

use std::borrow::Cow;

use turbo_analysis::Span;

use super::{java_trim, whitespace_tokens};
use turbo_formats::sample::{SPLIT_MARKER, SentenceSample, TokenSample};
use turbo_formats::{FormatError, Line, Lines};

/// Reads token samples from the token format.
#[derive(Clone, Debug)]
pub struct TokenReader<'a> {
    lines: Lines<'a>,
}

impl<'a> TokenReader<'a> {
    /// A reader of `source` with the `<SPLIT>` marker, splitting lines at
    /// Unicode whitespace.
    pub fn new(source: &'a str) -> Result<TokenReader<'a>, FormatError> {
        Ok(TokenReader { lines: Lines::new(source)? })
    }
}

impl<'a> Iterator for TokenReader<'a> {
    type Item = Result<TokenSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let line = self.lines.next()?;
        Some(parse_line(line))
    }
}

/// One line of the token format.
fn parse_line(line: Line<'_>) -> Result<TokenSample<'_>, FormatError> {
    // Each piece is followed by one space, so no token ends past the
    // text's length and the `u32` offsets below cannot overflow.
    let mut out = String::with_capacity(line.text().len() + 1);
    let mut tokens = Vec::new();
    let mut sources = Vec::new();
    for piece in whitespace_tokens(line.piece) {
        let mut at = 0;
        for part in piece.text.split(SPLIT_MARKER) {
            let start = out.len() as u32;
            out.push_str(part);
            tokens.push(Span::new(start, out.len() as u32));
            sources.push(piece.slice(at, at + part.len()).span);
            at += part.len() + SPLIT_MARKER.len();
        }
        out.push(' ');
    }
    TokenSample::new(Cow::Owned(out), tokens, sources, line.span())
}

/// Reads sentence samples from the sentence format.
#[derive(Clone, Debug)]
pub struct SentenceReader<'a> {
    lines: EmptyLinePreprocessor<'a>,
}

impl<'a> SentenceReader<'a> {
    /// A reader of `source`.
    pub fn new(source: &'a str) -> Result<SentenceReader<'a>, FormatError> {
        Ok(SentenceReader { lines: EmptyLinePreprocessor::new(source)? })
    }
}

/// The lines of a text, each run of empty lines (empty after a trim)
/// given as one empty line, and the empty lines at the start skipped.
#[derive(Clone, Debug)]
struct EmptyLinePreprocessor<'a> {
    lines: Lines<'a>,
    last_was_empty: bool,
}

impl<'a> EmptyLinePreprocessor<'a> {
    fn new(source: &'a str) -> Result<EmptyLinePreprocessor<'a>, FormatError> {
        Ok(EmptyLinePreprocessor { lines: Lines::new(source)?, last_was_empty: true })
    }

    /// The next line, `Err(line)` for an empty line (after skipping the
    /// empty lines that follow an empty line or the start).
    fn next_line(&mut self) -> Option<Result<Line<'a>, Line<'a>>> {
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
            return Some(Err(line));
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
        while let Some(Ok(line)) = self.lines.next_line() {
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

/// `<LF>` to a line feed, then `<CR>` to a carriage return.
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
