//! The document category and language formats: `DocumentSampleStream`
//! and `LanguageDetectorSampleStream`.
//!
//! Document category: one document per line, its category and then its
//! tokens, all separated by whitespace. A line with fewer than two tokens,
//! an empty line included, is refused (the stream throws; the next read
//! goes on with the next line).
//!
//! Language: one text per line, its language code, a tab and the text,
//! which runs to the end of the line and may hold more tabs. A line
//! without a tab, or that starts with one, is passed over silently
//! ([`Warning::NoLanguage`]).

use turbo_unicode::WhitespaceMode;

use crate::error::{FormatError, Warning};
use crate::sample::document::{DocumentSample, LanguageSample};
use crate::source::{Field, Lines, whitespace_tokens};

/// Reads document category samples.
#[derive(Clone, Debug)]
pub struct DocumentReader<'a> {
    lines: Lines<'a>,
    mode: WhitespaceMode,
}

impl<'a> DocumentReader<'a> {
    /// A reader of `source`, splitting lines on whitespace in `mode`.
    pub fn new(source: &'a str, mode: WhitespaceMode) -> Result<DocumentReader<'a>, FormatError> {
        Ok(DocumentReader { lines: Lines::new(source)?, mode })
    }
}

impl<'a> Iterator for DocumentReader<'a> {
    type Item = Result<DocumentSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let line = self.lines.next()?;
        let mut tokens = whitespace_tokens(line.field, self.mode);
        let category = tokens.next();
        let tokens: Vec<Field<'a>> = tokens.collect();
        Some(match category {
            Some(category) if !tokens.is_empty() => Ok(DocumentSample { category, tokens, source: line.span() }),
            _ => Err(FormatError::NoDocumentText { line: line.number }),
        })
    }
}

/// Reads language samples.
#[derive(Clone, Debug)]
pub struct LanguageReader<'a> {
    lines: Lines<'a>,
    warnings: Vec<Warning>,
}

impl<'a> LanguageReader<'a> {
    pub fn new(source: &'a str) -> Result<LanguageReader<'a>, FormatError> {
        Ok(LanguageReader { lines: Lines::new(source)?, warnings: Vec::new() })
    }

    /// The lines read so far that were passed over.
    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }
}

impl<'a> Iterator for LanguageReader<'a> {
    type Item = Result<LanguageSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let line = self.lines.next()?;
            let text = line.text();
            match text.find('\t') {
                Some(tab) if tab > 0 => {
                    let field = line.field;
                    return Some(Ok(LanguageSample {
                        language: field.slice(0, tab),
                        context: field.slice(tab + 1, text.len()),
                        source: line.span(),
                    }));
                }
                _ => self.warnings.push(Warning::NoLanguage { line: line.number }),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categories_need_a_token() {
        let src = "spam buy now\n\nham\nham  see\tyou ";
        let r: Vec<_> = DocumentReader::new(src, WhitespaceMode::Unicode).unwrap().collect();
        assert_eq!(r.len(), 4);
        assert_eq!(r[0].as_ref().unwrap().to_string(), "spam\tbuy now");
        assert_eq!(r[1].as_ref().unwrap_err().line(), Some(2));
        assert_eq!(
            r[1].as_ref().unwrap_err().to_string(),
            "Empty lines, or lines with only a category string are not allowed!"
        );
        assert!(r[2].is_err());
        assert_eq!(r[3].as_ref().unwrap().to_string(), "ham\tsee you");
    }

    #[test]
    fn languages_take_everything_after_the_first_tab() {
        let src = "eng\ta\tb\n\tx\nnone\nde\t\n";
        let mut r = LanguageReader::new(src).unwrap();
        let samples: Vec<LanguageSample<'_>> = r.by_ref().map(Result::unwrap).collect();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].context.text, "a\tb");
        assert_eq!(samples[0].to_string(), "eng\ta\tb");
        assert_eq!(samples[1].context.text, "");
        assert_eq!(r.warnings(), [Warning::NoLanguage { line: 2 }, Warning::NoLanguage { line: 3 }]);
    }
}
