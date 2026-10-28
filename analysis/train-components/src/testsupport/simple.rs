//! The word-and-tag, document category and language formats.
//!
//! Word and tag: one sentence per line, tokens separated by whitespace,
//! each `word_tag` (the tag is everything after the token's last `_`). A
//! token without `_` makes the whole line an empty sample.
//!
//! Document category: one document per line, its category then its
//! tokens, all separated by whitespace. A line with fewer than two
//! tokens is refused.
//!
//! Language: one text per line, its language code, a tab and the text,
//! which runs to the end of the line. A line without a tab, or starting
//! with one, is passed over.

use super::whitespace_tokens;
use turbo_formats::sample::{DocumentSample, LanguageSample, PosSample};
use turbo_formats::{FormatError, Lines, Piece};

/// Reads part-of-speech samples from the word and tag format.
#[derive(Clone, Debug)]
pub struct WordTagReader<'a> {
    lines: Lines<'a>,
}

impl<'a> WordTagReader<'a> {
    /// A reader of `source`, splitting lines at Unicode whitespace.
    pub fn new(source: &'a str) -> Result<WordTagReader<'a>, FormatError> {
        Ok(WordTagReader { lines: Lines::new(source)? })
    }
}

impl<'a> Iterator for WordTagReader<'a> {
    type Item = Result<PosSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let line = self.lines.next()?;
        let mut tokens = Vec::new();
        let mut tags = Vec::new();
        for part in whitespace_tokens(line.piece) {
            let Some(split) = part.text.rfind('_') else {
                // A token without a tag separator: an empty sample, as the
                // stream logs the line and returns one.
                return Some(PosSample::new(Vec::new(), Vec::new(), line.span()));
            };
            tokens.push(part.slice(0, split).field());
            tags.push(part.slice(split + 1, part.text.len()).field());
        }
        Some(PosSample::new(tokens, tags, line.span()))
    }
}

/// Reads document category samples.
#[derive(Clone, Debug)]
pub struct DocumentReader<'a> {
    lines: Lines<'a>,
}

impl<'a> DocumentReader<'a> {
    /// A reader of `source`, splitting lines at Unicode whitespace.
    pub fn new(source: &'a str) -> Result<DocumentReader<'a>, FormatError> {
        Ok(DocumentReader { lines: Lines::new(source)? })
    }
}

impl<'a> Iterator for DocumentReader<'a> {
    type Item = Result<DocumentSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let line = self.lines.next()?;
        let mut tokens = whitespace_tokens(line.piece);
        let category = tokens.next().map(Piece::field);
        let tokens: Vec<_> = tokens.map(Piece::field).collect();
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
}

impl<'a> LanguageReader<'a> {
    /// A reader of `source`.
    pub fn new(source: &'a str) -> Result<LanguageReader<'a>, FormatError> {
        Ok(LanguageReader { lines: Lines::new(source)? })
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
                    let field = line.piece;
                    return Some(Ok(LanguageSample {
                        language: field.slice(0, tab).field(),
                        context: field.slice(tab + 1, text.len()).field(),
                        source: line.span(),
                    }));
                }
                _ => {}
            }
        }
    }
}
