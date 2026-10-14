//! The token format: `TokenSampleStream` and `TokenSample.parse`.
//!
//! One sentence per line, written as the tokenizer should split it:
//! whitespace between tokens that the text separates, `<SPLIT>` between
//! tokens it writes together:
//!
//! ```text
//! He said<SPLIT>, "<SPLIT>Yes<SPLIT>."
//! ```
//!
//! The sample's text is the line rebuilt: each whitespace-separated piece
//! with its `<SPLIT>` markers removed, followed by one space, so the text
//! ends with a space and any run of whitespace in the line becomes one.
//! Every line is a sample; an empty line is a sample with an empty text.
//! Markers next to each other or at the ends of a piece make empty
//! tokens (`a<SPLIT><SPLIT>b` is `a`, an empty token and `b`).

use std::borrow::Cow;

use turbo_analysis::Span;
use turbo_unicode::WhitespaceMode;

use crate::error::FormatError;
use crate::sample::token::{SplitMarker, TokenSample};
use crate::source::{Line, Lines, Piece, whitespace_tokens};

/// Reads token samples from the token format.
#[derive(Clone, Debug)]
pub struct TokenReader<'a, 'm> {
    lines: Lines<'a>,
    mode: WhitespaceMode,
    marker: SplitMarker<'m>,
}

impl<'a> TokenReader<'a, 'static> {
    /// A reader of `source` with the `<SPLIT>` marker, splitting lines on
    /// whitespace in `mode`.
    pub fn new(source: &'a str, mode: WhitespaceMode) -> Result<TokenReader<'a, 'static>, FormatError> {
        TokenReader::with_marker(source, mode, SplitMarker::default())
    }
}

impl<'a, 'm> TokenReader<'a, 'm> {
    /// A reader of `source` with another split marker.
    pub fn with_marker(
        source: &'a str,
        mode: WhitespaceMode,
        marker: SplitMarker<'m>,
    ) -> Result<TokenReader<'a, 'm>, FormatError> {
        Ok(TokenReader { lines: Lines::new(source)?, mode, marker })
    }
}

impl<'a> Iterator for TokenReader<'a, '_> {
    type Item = Result<TokenSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let line = self.lines.next()?;
        Some(parse_line(line, self.mode, self.marker))
    }
}

/// `TokenSample.parse` of one line.
pub fn parse_line<'a>(
    line: Line<'a>,
    mode: WhitespaceMode,
    marker: SplitMarker<'_>,
) -> Result<TokenSample<'a>, FormatError> {
    parse_text(line.piece, mode, marker, |span| span, line.span())
}

/// `TokenSample.parse` of `text`, a piece of a source or a text made from
/// one. `to_source` maps the span of each token in `text` (offset by the
/// piece's span) to its bytes in the source.
pub(crate) fn parse_text<'a>(
    text: Piece<'_>,
    mode: WhitespaceMode,
    marker: SplitMarker<'_>,
    to_source: impl Fn(Span) -> Span,
    source: Span,
) -> Result<TokenSample<'a>, FormatError> {
    // Each piece is followed by one space where the text has at least one
    // whitespace character or its end, so no token ends past the text's
    // length and the `u32` offsets below cannot overflow.
    let mut out = String::with_capacity(text.text.len() + 1);
    let mut tokens = Vec::new();
    let mut sources = Vec::new();
    for piece in whitespace_tokens(text, mode) {
        let mut at = 0;
        for part in piece.text.split(marker.as_str()) {
            let start = out.len() as u32;
            out.push_str(part);
            tokens.push(Span::new(start, out.len() as u32));
            sources.push(to_source(piece.slice(at, at + part.len()).span));
            at += part.len() + marker.as_str().len();
        }
        out.push(' ');
    }
    TokenSample::new(Cow::Owned(out), tokens, sources, source)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(s: &str) -> Vec<TokenSample<'_>> {
        TokenReader::new(s, WhitespaceMode::Unicode).unwrap().map(Result::unwrap).collect()
    }

    #[test]
    fn rebuilds_the_text_with_a_space_after_each_piece() {
        let src = "He said<SPLIT>,  \"<SPLIT>Yes\n\n<SPLIT>a<SPLIT><SPLIT>b<SPLIT>";
        let samples = read(src);
        assert_eq!(samples.len(), 3);
        let s = &samples[0];
        assert_eq!(s.text(), "He said, \"Yes ");
        let tokens: Vec<&str> = (0..s.tokens().len()).map(|i| s.token(i)).collect();
        assert_eq!(tokens, ["He", "said", ",", "\"", "Yes"]);
        let sources: Vec<&str> = s.sources().iter().map(|sp| &src[sp.range()]).collect();
        assert_eq!(sources, tokens);
        assert_eq!(s.to_string(), "He said<SPLIT>, \"<SPLIT>Yes");
        assert_eq!(samples[1].text(), "");
        assert!(samples[1].tokens().is_empty());
        let s = &samples[2];
        let tokens: Vec<&str> = (0..s.tokens().len()).map(|i| s.token(i)).collect();
        assert_eq!(tokens, ["", "a", "", "b", ""]);
        assert_eq!(s.to_string(), "<SPLIT>a<SPLIT><SPLIT>b<SPLIT>");
    }

    #[test]
    fn another_marker() {
        let m = SplitMarker::new("|").unwrap();
        let samples: Vec<_> = TokenReader::with_marker("a|b c", WhitespaceMode::Unicode, m).unwrap().collect();
        let s = samples[0].as_ref().unwrap();
        assert_eq!(s.text(), "ab c ");
        let mut out = String::new();
        s.write_with(m, &mut out).unwrap();
        assert_eq!(out, "a|b c");
        assert!(SplitMarker::new("").is_none());
    }
}
