//! What building a matcher, matching and reading a glossary can fail with.

use std::fmt;

use turbo_glossary_layers::{EntryError, EntryIx};

/// Why a matcher could not be built from a glossary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuildError {
    /// The glossary has no entries; OpenNLP's matchers refuse an empty one.
    Empty,
    /// The entry's term is blank once normalized for matching.
    BlankTerm { entry: EntryIx },
    /// The entry's term gave no token when analyzed.
    NoTokens { entry: EntryIx },
    /// One of the entry's tokens is blank once analyzed.
    BlankToken { entry: EntryIx, token: usize },
    /// A composite matcher with no matchers in it.
    NoMatchers,
    /// An entry number past the end of the glossary.
    NoEntry { entry: EntryIx },
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::Empty => f.write_str("glossary must not be empty"),
            BuildError::BlankTerm { entry } => write!(f, "the term of entry {} normalizes to blank", entry.0),
            BuildError::NoTokens { entry } => write!(f, "the term of entry {} yields no token after analysis", entry.0),
            BuildError::BlankToken { entry, token } => {
                write!(f, "token {token} of the term of entry {} normalizes to blank", entry.0)
            }
            BuildError::NoMatchers => f.write_str("a composite matcher needs at least one matcher"),
            BuildError::NoEntry { entry } => write!(f, "the glossary has no entry {}", entry.0),
        }
    }
}

impl std::error::Error for BuildError {}

/// Why a document could not be matched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatchError {
    /// A token span is outside the text or not on character boundaries.
    Token { index: usize, start: u32, end: u32 },
    /// The token values are not one per token.
    Misaligned { values: usize, tokens: usize },
    /// A token matcher reads a value per token and none was given.
    NoValues,
    /// The token starts before the one before it ends: tokens must be in
    /// text order without overlaps.
    Unordered { index: usize },
}

impl fmt::Display for MatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatchError::Token { index, start, end } => {
                write!(f, "token {index} ({start}..{end}) is outside the text or cuts a character")
            }
            MatchError::Misaligned { values, tokens } => write!(f, "{values} token values for {tokens} tokens"),
            MatchError::NoValues => f.write_str("the token matcher needs a value per token"),
            MatchError::Unordered { index } => write!(f, "token {index} starts before the token before it ends"),
        }
    }
}

impl std::error::Error for MatchError {}

/// What is wrong with a glossary file. The line numbers count from 1;
/// a line ends at a line feed, a carriage return or the pair of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadError {
    /// The bytes are not UTF-8; `offset` is the first byte that is not.
    Utf8 { offset: usize },
    /// A delimited file has a quote inside a field that does not start with one.
    QuoteInUnquotedField { line: u32 },
    /// A delimited file has something other than a delimiter or a line end
    /// after a closing quote.
    AfterClosingQuote { line: u32 },
    /// A delimited file ends inside a quoted field of the record that
    /// starts on `line`.
    UnclosedQuote { line: u32 },
    /// A record of a delimited file has fewer than two fields.
    MissingColumns { line: u32, found: usize },
    /// An entry's identifier or term is blank. `line` is the record's
    /// first line in a delimited file and the line of the term element or
    /// entry in a termbase.
    Entry { line: u32, error: EntryError },
    /// A termbase is not well-formed XML, or uses XML this reader does
    /// not read (see [`crate::TbxReader`]).
    Xml { line: u32, detail: String },
    /// The termbase's root is not a TBX `martif` or `tbx` element.
    NotTbx { line: u32, root: String },
    /// A termbase declares an external entity, which is never loaded.
    ExternalEntity { line: u32, name: String },
}

impl ReadError {
    /// The line the problem is on, when it has one.
    pub fn line(&self) -> Option<u32> {
        match self {
            ReadError::Utf8 { .. } => None,
            ReadError::QuoteInUnquotedField { line }
            | ReadError::AfterClosingQuote { line }
            | ReadError::UnclosedQuote { line }
            | ReadError::MissingColumns { line, .. }
            | ReadError::Entry { line, .. }
            | ReadError::Xml { line, .. }
            | ReadError::NotTbx { line, .. }
            | ReadError::ExternalEntity { line, .. } => Some(*line),
        }
    }
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadError::Utf8 { offset } => write!(f, "glossary content is not valid UTF-8 at byte {offset}"),
            ReadError::QuoteInUnquotedField { line } => write!(f, "quote in unquoted field on line {line}"),
            ReadError::AfterClosingQuote { line } => {
                write!(f, "unexpected character after closing quote on line {line}")
            }
            ReadError::UnclosedQuote { line } => write!(f, "unclosed quoted field in record starting on line {line}"),
            ReadError::MissingColumns { line, found } => {
                write!(f, "expected at least an id and a term column on line {line}, found {found} column(s)")
            }
            ReadError::Entry { line, error } => write!(f, "{error} on line {line}"),
            ReadError::Xml { line, detail } => write!(f, "malformed TBX content on line {line}: {detail}"),
            ReadError::NotTbx { line, root } => {
                write!(f, "expected a TBX martif or tbx root element on line {line}, found: {root}")
            }
            ReadError::ExternalEntity { line, name } => {
                write!(f, "external entity declarations are not supported: {name} (line {line})")
            }
        }
    }
}

impl std::error::Error for ReadError {}
