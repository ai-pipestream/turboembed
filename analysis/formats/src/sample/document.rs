//! Document category and language samples:
//! `opennlp.tools.doccat.DocumentSample` and
//! `opennlp.tools.langdetect.LanguageSample`.

use turbo_analysis::Span;

use crate::source::Field;

/// A document's tokens and its category.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentSample<'a> {
    pub category: Field<'a>,
    pub tokens: Vec<Field<'a>>,
    /// The bytes of the source the sample was read from.
    pub source: Span,
}

/// A text and the language it is in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LanguageSample<'a> {
    /// The language code, as the sample gives it.
    pub language: Field<'a>,
    /// The text, everything after the first tab of the line.
    pub context: Field<'a>,
    /// The bytes of the source the sample was read from.
    pub source: Span,
}
