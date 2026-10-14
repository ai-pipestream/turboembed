//! Document category and language samples:
//! `opennlp.tools.doccat.DocumentSample` and
//! `opennlp.tools.langdetect.LanguageSample`.

use std::fmt;

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

/// `DocumentSample.toString`, the document category format's line: the
/// category, a tab and the tokens separated by single spaces; only the
/// category when there are no tokens.
impl fmt::Display for DocumentSample<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.category.text)?;
        for (i, token) in self.tokens.iter().enumerate() {
            f.write_str(if i == 0 { "\t" } else { " " })?;
            f.write_str(&token.text)?;
        }
        Ok(())
    }
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

/// `LanguageSample.toString`, the language format's line: the language,
/// a tab and the text.
impl fmt::Display for LanguageSample<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}\t{}", self.language.text, self.context.text)
    }
}
