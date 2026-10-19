//! Part-of-speech samples: `opennlp.tools.postag.POSSample`.

use std::fmt;

use turbo_analysis::Span;

use crate::error::FormatError;
use crate::source::Field;

/// A sentence with a part-of-speech tag per token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PosSample<'a> {
    tokens: Vec<Field<'a>>,
    tags: Vec<Field<'a>>,
    source: Span,
}

impl<'a> PosSample<'a> {
    /// A sample of `tokens` and their `tags`, read from `source`.
    ///
    /// # Errors
    ///
    /// [`FormatError::Lengths`] when there is not one tag per token.
    pub fn new(tokens: Vec<Field<'a>>, tags: Vec<Field<'a>>, source: Span) -> Result<PosSample<'a>, FormatError> {
        if tokens.len() != tags.len() {
            return Err(FormatError::Lengths { what: "tags", expected: tokens.len(), found: tags.len() });
        }
        Ok(PosSample { tokens, tags, source })
    }

    /// A sample of token and tag pairs, read from `source`.
    pub(crate) fn from_pairs(pairs: impl Iterator<Item = (Field<'a>, Field<'a>)>, source: Span) -> PosSample<'a> {
        let (tokens, tags) = pairs.unzip();
        PosSample { tokens, tags, source }
    }

    /// A sample with no tokens, at `source`: what `WordTagSampleStream`
    /// returns in place of a line it cannot read.
    pub fn empty(source: Span) -> PosSample<'a> {
        PosSample { tokens: Vec::new(), tags: Vec::new(), source }
    }

    pub fn tokens(&self) -> &[Field<'a>] {
        &self.tokens
    }

    /// The tags, one per token.
    pub fn tags(&self) -> &[Field<'a>] {
        &self.tags
    }

    /// The bytes of the source the sample was read from.
    pub fn source(&self) -> Span {
        self.source
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// `POSSample.toString`, the word and tag format's line: `token_tag`
/// pairs separated by single spaces.
impl fmt::Display for PosSample<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (token, tag)) in self.tokens.iter().zip(&self.tags).enumerate() {
            if i > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{}_{}", token.text, tag.text)?;
        }
        Ok(())
    }
}
