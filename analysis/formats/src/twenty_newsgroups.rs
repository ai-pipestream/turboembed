//! Twenty Newsgroups document categorization corpus reader
//! (`opennlp.tools.formats.TwentyNewsgroupSampleStream`).
//!
//! Port of `TwentyNewsgroupSampleStream` and `TwentyNewsgroupSampleStreamFactory`
//! from OpenNLP format utilities.

use std::fs;
use std::path::{Path, PathBuf};

use turbo_analysis::Span;
use turbo_tokenize::{TokenizerOptions, WhitespaceMode, simple_spans, whitespace_spans};

use crate::error::FormatError;
use crate::sample::DocumentSample;
use crate::source::Field;

/// Rule-based tokenizer choice for [`TwentyNewsgroupReader`].
///
/// Mirrors the `-ruleBasedTokenizer` option in `TwentyNewsgroupSampleStreamFactory`
/// (`"whitespace"` by default, or `"simple"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwentyNewsgroupTokenizer {
    /// Tokenize with `WhitespaceTokenizer` in the given [`WhitespaceMode`].
    Whitespace(WhitespaceMode),
    /// Tokenize with `SimpleTokenizer` in the given [`WhitespaceMode`].
    Simple(WhitespaceMode),
}

impl Default for TwentyNewsgroupTokenizer {
    #[inline]
    fn default() -> Self {
        Self::Whitespace(WhitespaceMode::Legacy)
    }
}

impl TwentyNewsgroupTokenizer {
    /// Parses a rule-based tokenizer name (`"whitespace"` or `"simple"`), matching
    /// `TwentyNewsgroupSampleStreamFactory`.
    pub fn from_name(name: &str) -> Result<Self, FormatError> {
        match name {
            "whitespace" => Ok(Self::Whitespace(WhitespaceMode::Legacy)),
            "simple" => Ok(Self::Simple(WhitespaceMode::Legacy)),
            other => Err(FormatError::UnknownTokenizer { name: other.to_owned() }),
        }
    }

    /// Tokenizes `text` into [`Span`]s.
    pub fn tokenize_spans(self, text: &str, out: &mut Vec<Span>) {
        match self {
            Self::Whitespace(whitespace) => {
                whitespace_spans(text, TokenizerOptions { whitespace, keep_newlines: false }, out);
            }
            Self::Simple(whitespace) => {
                simple_spans(text, TokenizerOptions { whitespace, keep_newlines: false }, out);
            }
        }
    }
}

/// Reader for a Twenty Newsgroups directory tree (`data_dir/<category>/<file>`).
///
/// Mirrors `opennlp.tools.formats.TwentyNewsgroupSampleStream`.
#[derive(Debug, Clone)]
pub struct TwentyNewsgroupReader {
    tokenizer: TwentyNewsgroupTokenizer,
    entries: Vec<(PathBuf, String)>,
    index: usize,
}

impl TwentyNewsgroupReader {
    /// Opens a Twenty Newsgroups directory tree using the default whitespace tokenizer.
    pub fn new(data_dir: impl AsRef<Path>) -> Result<Self, FormatError> {
        Self::with_tokenizer(data_dir, TwentyNewsgroupTokenizer::default())
    }

    /// Opens a Twenty Newsgroups directory tree using `tokenizer`.
    pub fn with_tokenizer(
        data_dir: impl AsRef<Path>,
        tokenizer: TwentyNewsgroupTokenizer,
    ) -> Result<Self, FormatError> {
        let root = data_dir.as_ref();
        let dir_iter = fs::read_dir(root)
            .map_err(|err| FormatError::Io { path: root.display().to_string(), message: err.to_string() })?;

        let mut subdirs = Vec::new();
        for entry in dir_iter {
            let entry =
                entry.map_err(|err| FormatError::Io { path: root.display().to_string(), message: err.to_string() })?;
            let path = entry.path();
            if entry.metadata().is_ok_and(|m| m.is_dir()) {
                let Some(cat_name) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                subdirs.push((path.clone(), cat_name.to_owned()));
            }
        }
        subdirs.sort_by(|a, b| a.0.cmp(&b.0));

        let mut entries = Vec::new();
        for (subdir, category) in subdirs {
            let file_iter = fs::read_dir(&subdir)
                .map_err(|err| FormatError::Io { path: subdir.display().to_string(), message: err.to_string() })?;
            let mut files = Vec::new();
            for fentry in file_iter {
                let fentry = fentry
                    .map_err(|err| FormatError::Io { path: subdir.display().to_string(), message: err.to_string() })?;
                let fpath = fentry.path();
                if fentry.metadata().is_ok_and(|m| m.is_file()) {
                    files.push(fpath);
                }
            }
            files.sort();
            for fpath in files {
                entries.push((fpath, category.clone()));
            }
        }

        Ok(Self { tokenizer, entries, index: 0 })
    }

    /// Resets the stream to the first document in the corpus.
    #[inline]
    pub fn reset(&mut self) {
        self.index = 0;
    }

    /// Reads the next [`DocumentSample`], or `None` when all files have been read.
    pub fn next_sample(&mut self) -> Option<Result<DocumentSample<'static>, FormatError>> {
        let (path, category) = self.entries.get(self.index)?;
        self.index += 1;

        let bytes = match fs::read(path) {
            Ok(b) => b,
            Err(err) => {
                return Some(Err(FormatError::Io { path: path.display().to_string(), message: err.to_string() }));
            }
        };
        let text = String::from_utf8_lossy(&bytes);
        let mut spans = Vec::new();
        self.tokenizer.tokenize_spans(&text, &mut spans);

        let tokens: Vec<Field<'static>> =
            spans.into_iter().map(|span| Field::made(text[span.range()].to_owned(), span)).collect();
        let doc_len = text.len().min(u32::MAX as usize) as u32;

        Some(Ok(DocumentSample {
            category: Field::made(category.clone(), Span::new(0, 0)),
            tokens,
            source: Span::new(0, doc_len),
        }))
    }
}

impl Iterator for TwentyNewsgroupReader {
    type Item = Result<DocumentSample<'static>, FormatError>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.next_sample()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn twenty_newsgroups_reads_categories_and_tokenizes() {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("turbo_formats_20news_{}_{id}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let graphics = dir.join("comp.graphics");
        let space = dir.join("sci.space");
        fs::create_dir_all(&graphics).unwrap();
        fs::create_dir_all(&space).unwrap();

        fs::write(graphics.join("1001"), "Polygon rendering in 3D.").unwrap();
        fs::write(space.join("2001"), "Orbiting the Earth!").unwrap();

        let ws_samples: Vec<DocumentSample<'static>> =
            TwentyNewsgroupReader::new(&dir).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(ws_samples.len(), 2);
        assert_eq!(ws_samples[0].category.as_str(), "comp.graphics");
        let ws_toks: Vec<&str> = ws_samples[0].tokens.iter().map(|t| t.as_str()).collect();
        assert_eq!(ws_toks, vec!["Polygon", "rendering", "in", "3D."]);

        let simple_samples: Vec<DocumentSample<'static>> =
            TwentyNewsgroupReader::with_tokenizer(&dir, TwentyNewsgroupTokenizer::from_name("simple").unwrap())
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
        assert_eq!(simple_samples.len(), 2);
        let simple_toks: Vec<&str> = simple_samples[0].tokens.iter().map(|t| t.as_str()).collect();
        assert_eq!(simple_toks, vec!["Polygon", "rendering", "in", "3", "D", "."]);

        let _ = fs::remove_dir_all(&dir);
    }
}
