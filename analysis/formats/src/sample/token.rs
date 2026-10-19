//! Token samples: `opennlp.tools.tokenize.TokenSample`.

use std::borrow::Cow;
use std::fmt;

use turbo_analysis::Span;

use crate::error::FormatError;

/// The marker the token format writes between two tokens with no space
/// between them (`TokenSample.DEFAULT_SEPARATOR_CHARS`).
pub const SPLIT_MARKER: &str = "<SPLIT>";

/// A non-empty split marker. OpenNLP accepts an empty one and then loops
/// forever on the first token, so this port refuses it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SplitMarker<'m>(&'m str);

impl<'m> SplitMarker<'m> {
    /// `None` for the empty string.
    pub fn new(marker: &'m str) -> Option<SplitMarker<'m>> {
        (!marker.is_empty()).then_some(SplitMarker(marker))
    }

    pub fn as_str(&self) -> &'m str {
        self.0
    }
}

impl Default for SplitMarker<'static> {
    fn default() -> SplitMarker<'static> {
        SplitMarker(SPLIT_MARKER)
    }
}

/// A text and the tokens in it.
///
/// The text is the sample's own: the token format writes the tokens of
/// a line with a space after each run of joined tokens (so the text ends
/// with a space), and a converted sample rebuilds it with a detokenizer.
/// Each token has its span in that text and the span of the source it
/// came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenSample<'a> {
    text: Cow<'a, str>,
    tokens: Vec<Span>,
    sources: Vec<Span>,
    source: Span,
}

impl<'a> TokenSample<'a> {
    /// A sample of `text` with `tokens`, spans in `text`, and for each
    /// token the bytes of the source it came from; `source` is the bytes
    /// of the whole sample.
    ///
    /// # Errors
    ///
    /// [`FormatError::OutOfBounds`] for a token span outside the text or
    /// not on its character boundaries, and [`FormatError::Lengths`] when
    /// there is not one source span per token.
    pub fn new(
        text: Cow<'a, str>,
        tokens: Vec<Span>,
        sources: Vec<Span>,
        source: Span,
    ) -> Result<TokenSample<'a>, FormatError> {
        if sources.len() != tokens.len() {
            return Err(FormatError::Lengths { what: "token sources", expected: tokens.len(), found: sources.len() });
        }
        for &span in &tokens {
            if text.get(span.range()).is_none() {
                return Err(FormatError::OutOfBounds { span, len: text.len() });
            }
        }
        Ok(TokenSample { text, tokens, sources, source })
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// The tokens' spans in [`TokenSample::text`].
    pub fn tokens(&self) -> &[Span] {
        &self.tokens
    }

    /// Each token's span in the source it was read or converted from.
    pub fn sources(&self) -> &[Span] {
        &self.sources
    }

    /// The bytes of the source the sample came from.
    pub fn source(&self) -> Span {
        self.source
    }

    /// The text of token `i`.
    pub fn token(&self, i: usize) -> &str {
        &self.text[self.tokens[i].range()]
    }

    /// `TokenSample.toString` with another marker than `<SPLIT>`.
    pub fn write_with(&self, marker: SplitMarker<'_>, out: &mut impl fmt::Write) -> fmt::Result {
        let mut last_end: Option<u32> = None;
        for &span in &self.tokens {
            if let Some(end) = last_end {
                out.write_str(if end == span.start { marker.as_str() } else { " " })?;
            }
            out.write_str(&self.text[span.range()])?;
            last_end = Some(span.end);
        }
        Ok(())
    }
}

/// `TokenSample.toString`, the token format's line: the tokens in order,
/// separated by `<SPLIT>` where one ends where the next starts and by a
/// space otherwise, whatever the text holds between them.
impl fmt::Display for TokenSample<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_with(SplitMarker::default(), f)
    }
}
