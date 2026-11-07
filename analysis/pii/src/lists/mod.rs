//! Caller-configured allow lists (ignore rules) and deny lists (custom PII
//! dictionary matchers), loaded from caller terms, regex-free patterns, or
//! line-oriented files.
//!
//! - [`AllowList`] suppresses detected mentions whose covered surface text
//!   or normalized form matches an exact term or a [`Pattern`]. An allow
//!   list can be attached globally across all [`turbo_analysis_layers::PiiKind`]s
//!   or scoped to a single [`turbo_analysis_layers::PiiKind`].
//! - [`DenyList`] matches caller terms using [`turbo_glossary::TextMatcher`]
//!   (Aho-Corasick over Unicode scalars) with optional bounded Levenshtein
//!   fuzzy matching (`0`, `1`, or `2` edits over word tokens) and a
//!   `capitalised_only` guard, reporting hits under a caller-named
//!   [`turbo_analysis_layers::PiiKind::Custom`].
//! - [`Pattern`] provides two deterministic, regex-free pattern syntaxes:
//!   [`Pattern::shape`] (`d` ASCII digit, `a` ASCII letter, `w` ASCII
//!   alphanumeric, `?` any character, `*` zero or more characters, `\x`
//!   escaped literal, and any other character as itself) and
//!   [`Pattern::glob`] (`*` and `?` wildcards with `\x` escapes).

mod allow;
mod deny;
mod pattern;

use std::fmt;
use std::path::{Path, PathBuf};

pub use allow::AllowList;
pub use deny::DenyList;
pub use pattern::{Pattern, PatternKind};
pub use turbo_glossary::Case;

use turbo_analysis_layers::{KindError, PiiKind};

/// Why an [`AllowList`], [`DenyList`], or [`Pattern`] configuration was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ListError {
    /// A term is empty or contains only whitespace.
    EmptyTerm,
    /// A pattern string is empty.
    EmptyPattern,
    /// A pattern ends with an unescaped trailing backslash (`\`).
    TrailingEscape(String),
    /// A fuzzy edit distance exceeds the supported `0..=2` range.
    InvalidEditDistance(u8),
    /// A deny list targets a built-in kind; deny lists report custom kinds.
    BuiltinKind(PiiKind),
    /// A [`PiiKind::Custom`] outside the detector's custom-kind table.
    Kind(KindError),
    /// Reading a term or pattern file from disk failed.
    Io {
        /// Path that could not be read.
        path: PathBuf,
        /// Underlying OS error description.
        message: String,
    },
}

impl fmt::Display for ListError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ListError::EmptyTerm => f.write_str("list term must not be empty or whitespace"),
            ListError::EmptyPattern => f.write_str("pattern must not be empty"),
            ListError::TrailingEscape(p) => write!(f, "pattern {p:?} ends with an unescaped backslash"),
            ListError::InvalidEditDistance(d) => {
                write!(f, "fuzzy edit distance {d} is out of range (must be 0, 1, or 2)")
            }
            ListError::BuiltinKind(k) => write!(f, "deny list kind {k} must be a custom kind, not a built-in kind"),
            ListError::Kind(e) => e.fmt(f),
            ListError::Io { path, message } => write!(f, "failed to read {}: {message}", path.display()),
        }
    }
}

impl std::error::Error for ListError {}

impl From<KindError> for ListError {
    fn from(e: KindError) -> ListError {
        ListError::Kind(e)
    }
}

#[inline]
pub(crate) const fn case_tag(case: Case) -> u8 {
    match case {
        Case::Sensitive => 0,
        Case::Insensitive => 1,
    }
}

/// Parses non-empty, non-comment lines from a line-oriented list text.
/// Blank lines and lines whose first non-whitespace character is `#` are skipped.
pub(crate) fn parse_lines(content: &str) -> impl Iterator<Item = &str> {
    let content = content.strip_prefix('\u{feff}').unwrap_or(content);
    content.lines().map(str::trim).filter(|line| !line.is_empty() && !line.starts_with('#'))
}

/// Reads non-blank, non-`#`-comment lines from `path`.
pub(crate) fn read_file_lines(path: &Path) -> Result<Vec<String>, ListError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ListError::Io { path: path.to_path_buf(), message: e.to_string() })?;
    Ok(parse_lines(&text).map(str::to_owned).collect())
}
