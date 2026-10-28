//! Sentence samples: `opennlp.tools.sentdetect.SentenceSample`.

use std::borrow::Cow;

use turbo_analysis::Span;

use crate::error::FormatError;

/// A document and the sentences in it.
///
/// The document is the sample's own text. Each sentence has its span in
/// that text and the span of the source it came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SentenceSample<'a> {
    document: Cow<'a, str>,
    sentences: Vec<Span>,
    sources: Vec<Span>,
    source: Span,
}

impl<'a> SentenceSample<'a> {
    /// A sample of `document` with `sentences`, spans in it, and for each
    /// sentence the bytes of the source it came from; `source` is the
    /// bytes of the whole sample.
    ///
    /// # Errors
    ///
    /// [`FormatError::OutOfBounds`] for a sentence span outside the
    /// document or not on its character boundaries, and
    /// [`FormatError::Lengths`] when there is not one source span per
    /// sentence.
    pub fn new(
        document: Cow<'a, str>,
        sentences: Vec<Span>,
        sources: Vec<Span>,
        source: Span,
    ) -> Result<SentenceSample<'a>, FormatError> {
        if sources.len() != sentences.len() {
            return Err(FormatError::Lengths {
                what: "sentence sources",
                expected: sentences.len(),
                found: sources.len(),
            });
        }
        for &span in &sentences {
            if document.get(span.range()).is_none() {
                return Err(FormatError::OutOfBounds { span, len: document.len() });
            }
        }
        Ok(SentenceSample { document, sentences, sources, source })
    }

    pub fn document(&self) -> &str {
        &self.document
    }

    /// The sentences' spans in [`SentenceSample::document`].
    pub fn sentences(&self) -> &[Span] {
        &self.sentences
    }

    /// Each sentence's span in the source it was read or converted from.
    pub fn sources(&self) -> &[Span] {
        &self.sources
    }

    /// The bytes of the source the sample came from.
    pub fn source(&self) -> Span {
        self.source
    }

    /// The text of sentence `i`.
    pub fn sentence(&self, i: usize) -> &str {
        &self.document[self.sentences[i].range()]
    }
}
