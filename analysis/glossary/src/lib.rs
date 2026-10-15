//! Glossary term matching (`AhoCorasickGlossaryMatcher`,
//! `TermAnalyzingGlossaryMatcher`, `CompositeGlossaryMatcher`), document
//! annotation (`GlossaryAnnotator`, `TokenGlossaryAnnotator`, `GlossaryTerms`),
//! and glossary format readers and writers (`CsvGlossaryReader`,
//! `TbxGlossaryReader`).

use std::fmt;
use std::io;

use turbo_analysis::Span;
use turbo_unicode::WhitespaceMode;

pub mod aho_corasick;
pub mod annotate;
pub mod composite;
pub mod csv;
pub mod selection;
pub mod tbx;
pub mod term_analyzer;

pub use aho_corasick::AhoCorasickGlossaryMatcher;
pub use annotate::{GlossaryAnnotator, GlossaryTerms, TokenGlossaryAnnotator, matches_to_layer};
pub use composite::CompositeGlossaryMatcher;
pub use csv::CsvGlossaryReader;
pub use tbx::TbxGlossaryReader;
pub use term_analyzer::{Dimension, Term, TermAnalyzer, TermAnalyzerBuilder, TermAnalyzingGlossaryMatcher};

/// Whether `s` is empty or consists solely of Unicode `White_Space` code
/// points (`StringUtil.isBlank` under OpenNLP's default Unicode whitespace
/// mode).
#[inline]
pub fn is_blank(s: &str) -> bool {
    s.chars().all(|c| WhitespaceMode::Unicode.is_whitespace(c as u32))
}

/// Errors returned when constructing or reading glossary entries and matchers.
#[derive(Debug)]
pub enum GlossaryError {
    /// A glossary entry's `id` is empty or blank.
    BlankId,
    /// A glossary entry's `term` is empty or blank.
    BlankTerm,
    /// No glossary entries were supplied to a matcher.
    EmptyEntries,
    /// No delegate matchers were supplied to [`CompositeGlossaryMatcher`].
    EmptyMatchers,
    /// A glossary entry's term normalized to an empty or blank string.
    BlankNormalizedTerm { entry_id: String, term: String },
    /// A glossary entry's term produced zero tokens when analyzed.
    ZeroTokensTerm { entry_id: String, term: String },
    /// A token inside a glossary entry's term normalized to an empty or blank
    /// string.
    BlankTokenInTerm { entry_id: String, term: String, token: String },
    /// Both `CaseFold` and `FullCaseFold` were enabled on a [`TermAnalyzer`].
    ConflictingCaseFolds,
    /// [`Dimension::Stem`] was enabled without configuring a stemmer.
    MissingStemmer,
    /// [`Dimension::Lemma`] was enabled on an operation over raw text without
    /// part-of-speech tags.
    LemmaDimensionUnsupported,
    /// `transform(...)` was called with a non-character dimension (`Original`,
    /// `Stem`, or `Lemma`).
    InvalidTransformDimension(Dimension),
    /// An invalid CSV/TSV delimiter (`'"'`, `'\n'`, or `'\r'`) was supplied.
    InvalidDelimiter(char),
    /// A blank target language tag was supplied to [`TbxGlossaryReader`].
    BlankLanguage,
    /// A format error occurred while parsing CSV, TSV, or TBX input.
    Format(String),
    /// An underlying I/O error occurred.
    Io(io::Error),
}

impl fmt::Display for GlossaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GlossaryError::BlankId => write!(f, "GlossaryEntry id must not be blank"),
            GlossaryError::BlankTerm => write!(f, "GlossaryEntry term must not be blank"),
            GlossaryError::EmptyEntries => write!(f, "Glossary entries must not be empty"),
            GlossaryError::EmptyMatchers => write!(f, "Glossary matchers must not be empty"),
            GlossaryError::BlankNormalizedTerm { entry_id, term } => {
                write!(f, "Glossary entry '{entry_id}' term '{term}' normalizes to blank")
            }
            GlossaryError::ZeroTokensTerm { entry_id, term } => {
                write!(f, "Glossary entry '{entry_id}' term '{term}' produces no tokens")
            }
            GlossaryError::BlankTokenInTerm { entry_id, term, token } => write!(
                f,
                "Glossary entry '{entry_id}' term '{term}' contains a token '{token}' that normalizes to blank"
            ),
            GlossaryError::ConflictingCaseFolds => {
                write!(f, "case_fold() and full_case_fold() must not both be configured")
            }
            GlossaryError::MissingStemmer => {
                write!(f, "Dimension::Stem requires a configured Stemmer")
            }
            GlossaryError::LemmaDimensionUnsupported => {
                write!(f, "Dimension::Lemma requires part-of-speech tags and cannot be used on raw text")
            }
            GlossaryError::InvalidTransformDimension(dim) => {
                write!(f, "transform(...) only applies to character-level dimensions, not {dim:?}")
            }
            GlossaryError::InvalidDelimiter(d) => {
                write!(f, "CSV delimiter must not be a quote or line break: {d:?}")
            }
            GlossaryError::BlankLanguage => write!(f, "TBX language tag must not be blank"),
            GlossaryError::Format(msg) => f.write_str(msg),
            GlossaryError::Io(err) => fmt::Display::fmt(err, f),
        }
    }
}

impl std::error::Error for GlossaryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GlossaryError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for GlossaryError {
    fn from(err: io::Error) -> Self {
        GlossaryError::Io(err)
    }
}

/// An immutable glossary entry pairing a concept or term identifier with a
/// surface form (`term`).
///
/// Multiple entries may share the same `id` when a concept has several surface
/// forms or synonyms. Neither `id` nor `term` is trimmed on construction; the
/// caller's exact strings are preserved once validated as non-blank.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GlossaryEntry {
    id: String,
    term: String,
}

impl GlossaryEntry {
    /// Creates a new [`GlossaryEntry`] after validating that neither `id` nor
    /// `term` is empty or blank.
    ///
    /// # Errors
    ///
    /// Returns [`GlossaryError::BlankId`] if `id` is blank, or
    /// [`GlossaryError::BlankTerm`] if `term` is blank.
    pub fn new(id: impl Into<String>, term: impl Into<String>) -> Result<Self, GlossaryError> {
        let id = id.into();
        if is_blank(&id) {
            return Err(GlossaryError::BlankId);
        }
        let term = term.into();
        if is_blank(&term) {
            return Err(GlossaryError::BlankTerm);
        }
        Ok(Self { id, term })
    }

    /// The concept or term identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The surface form to match in text.
    pub fn term(&self) -> &str {
        &self.term
    }
}

/// A matched glossary term at a UTF-8 byte [`Span`] in the input text.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GlossaryMatch {
    span: Span,
    id: String,
    term: String,
}

impl GlossaryMatch {
    /// Creates a new [`GlossaryMatch`] after validating that `id` and `term`
    /// are non-blank.
    ///
    /// # Errors
    ///
    /// Returns [`GlossaryError::BlankId`] if `id` is blank, or
    /// [`GlossaryError::BlankTerm`] if `term` is blank.
    pub fn new(span: Span, id: impl Into<String>, term: impl Into<String>) -> Result<Self, GlossaryError> {
        let id = id.into();
        if is_blank(&id) {
            return Err(GlossaryError::BlankId);
        }
        let term = term.into();
        if is_blank(&term) {
            return Err(GlossaryError::BlankTerm);
        }
        Ok(Self { span, id, term })
    }

    /// Creates a [`GlossaryMatch`] directly from a validated [`GlossaryEntry`].
    pub fn from_entry(span: Span, entry: &GlossaryEntry) -> Self {
        Self { span, id: entry.id.clone(), term: entry.term.clone() }
    }

    /// The UTF-8 byte span of the match in the original text.
    pub fn span(&self) -> Span {
        self.span
    }

    /// The matched glossary entry's concept/term identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The matched glossary entry's canonical surface form.
    pub fn term(&self) -> &str {
        &self.term
    }
}

/// Matches glossary terms in raw text and returns resolved [`GlossaryMatch`]
/// items with original byte spans.
pub trait GlossaryMatcher: Send + Sync {
    /// Finds all glossary term matches in `text`.
    fn match_text(&self, text: &str) -> Vec<GlossaryMatch>;
}
