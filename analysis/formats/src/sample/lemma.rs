//! Lemma samples: `opennlp.tools.lemmatizer.LemmaSample`.

use turbo_analysis::Span;

use crate::error::FormatError;
use crate::source::Field;

/// A sentence with a part-of-speech tag and a lemma per token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LemmaSample<'a> {
    tokens: Vec<Field<'a>>,
    tags: Vec<Field<'a>>,
    lemmas: Vec<Field<'a>>,
    source: Span,
}

impl<'a> LemmaSample<'a> {
    /// A sample of `tokens` with their `tags` and `lemmas`, read from
    /// `source`.
    ///
    /// # Errors
    ///
    /// [`FormatError::Lengths`] when the three are not the same length.
    pub fn new(
        tokens: Vec<Field<'a>>,
        tags: Vec<Field<'a>>,
        lemmas: Vec<Field<'a>>,
        source: Span,
    ) -> Result<LemmaSample<'a>, FormatError> {
        if tags.len() != tokens.len() {
            return Err(FormatError::Lengths { what: "tags", expected: tokens.len(), found: tags.len() });
        }
        if lemmas.len() != tokens.len() {
            return Err(FormatError::Lengths { what: "lemmas", expected: tokens.len(), found: lemmas.len() });
        }
        Ok(LemmaSample { tokens, tags, lemmas, source })
    }

    pub fn tokens(&self) -> &[Field<'a>] {
        &self.tokens
    }

    pub fn tags(&self) -> &[Field<'a>] {
        &self.tags
    }

    pub fn lemmas(&self) -> &[Field<'a>] {
        &self.lemmas
    }

    /// The bytes of the source the sample was read from, from the start
    /// of its first line to the end of its last.
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
