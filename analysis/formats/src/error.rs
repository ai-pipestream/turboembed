//! What a reader refuses, and what it skips.
//!
//! A [`FormatError`] is a sample OpenNLP's stream throws on. Its
//! `Display` is the message of the exception Java throws there, and
//! [`FormatError::java_class`] the exception's class, so a test compares
//! both with the JVM's. A [`Warning`] is a line OpenNLP's stream logs and
//! skips or replaces; the reader carries on as Java does and keeps the
//! warning for the caller.

use std::fmt;

use turbo_analysis::Span;

/// A sample a reader cannot read, or a sample value that cannot be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FormatError {
    /// A `<START>` tag inside a name: `NameSample.parse` throws an
    /// `IOException`. `context` is the tag with up to two parts of the
    /// line before and after it, the tag marked with `###`.
    NestedName { line: u32, context: String },
    /// `<START:>`, a start tag with an empty type.
    MissingNameType { line: u32, context: String },
    /// An `<END>` tag outside a name.
    UnexpectedEnd { line: u32, context: String },
    /// Two names of one sample overlap after `NameSample` sorts them: the
    /// `RuntimeException` of its constructor. The parser makes this only
    /// from an empty name followed at the same token by a longer one.
    OverlappingNames { line: u32, first: String, second: String, sentence: String },
    /// A document category line with no text after the category, or an
    /// empty line: `DocumentSampleStream` throws an `IOException`.
    NoDocumentText { line: u32 },
    /// A chunk tag that starts a phrase but has fewer than two UTF-16
    /// code units, so `substring(2)` throws a
    /// `StringIndexOutOfBoundsException`; `len` is the tag's length in
    /// code units.
    ShortChunkTag { token: usize, len: usize },
    /// A chunk tag whose first two UTF-16 code units end inside a
    /// surrogate pair. Java cuts the pair and keeps half of it in the
    /// phrase type; a Rust string cannot hold half a pair, so this port
    /// refuses the tag instead.
    SplitsSurrogatePair { token: usize },
    /// Columns of different lengths given to a sample's constructor.
    Lengths { what: &'static str, expected: usize, found: usize },
    /// A name built over tokens a sample does not have.
    NameOutsideTokens { name: String, tokens: usize },
    /// A span outside the text it indexes, or not on its character
    /// boundaries.
    OutOfBounds { span: Span, len: usize },
    /// A source text longer than `u32::MAX` bytes, which [`Span`] cannot
    /// index.
    TooLong { len: usize },
}

impl FormatError {
    /// The class of the exception OpenNLP throws in the same place, for
    /// the errors that have one.
    pub fn java_class(&self) -> Option<&'static str> {
        use FormatError::*;
        match self {
            NestedName { .. } | MissingNameType { .. } | UnexpectedEnd { .. } | NoDocumentText { .. } => {
                Some("java.io.IOException")
            }
            OverlappingNames { .. } => Some("java.lang.RuntimeException"),
            ShortChunkTag { .. } => Some("java.lang.StringIndexOutOfBoundsException"),
            Lengths { .. } | OutOfBounds { .. } => Some("java.lang.IllegalArgumentException"),
            SplitsSurrogatePair { .. } | NameOutsideTokens { .. } | TooLong { .. } => None,
        }
    }

    /// The source line the error is on, for the errors a reader makes.
    pub fn line(&self) -> Option<u32> {
        use FormatError::*;
        match *self {
            NestedName { line, .. }
            | MissingNameType { line, .. }
            | UnexpectedEnd { line, .. }
            | OverlappingNames { line, .. }
            | NoDocumentText { line } => Some(line),
            _ => None,
        }
    }
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use FormatError::*;
        match self {
            NestedName { context, .. } => {
                write!(f, "Found unexpected annotation while handling a name sequence: {context}")
            }
            MissingNameType { context, .. } => write!(f, "Missing a name type: {context}"),
            UnexpectedEnd { context, .. } => write!(f, "Found unexpected annotation: {context}"),
            OverlappingNames { first, second, sentence, .. } => {
                write!(f, "name spans {first} and {second} are overlapped in file: null, sentence: {sentence}")
            }
            NoDocumentText { .. } => f.write_str("Empty lines, or lines with only a category string are not allowed!"),
            ShortChunkTag { len, .. } => write!(f, "Range [2, {len}) out of bounds for length {len}"),
            SplitsSurrogatePair { token } => {
                write!(f, "the chunk tag of token {token} has a surrogate pair across its second code unit")
            }
            Lengths { what, expected, found } => {
                write!(f, "{what}: expected {expected} values, found {found}")
            }
            NameOutsideTokens { name, tokens } => write!(f, "the name {name} reaches past the {tokens} tokens"),
            OutOfBounds { span, len } => {
                write!(
                    f,
                    "the span {}..{} is outside a text of {len} bytes or not on its characters",
                    span.start, span.end
                )
            }
            TooLong { len } => write!(f, "a source of {len} bytes is longer than spans can index"),
        }
    }
}

impl std::error::Error for FormatError {}

/// A line a reader skipped or replaced, as OpenNLP's stream logs it and
/// carries on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Warning {
    /// A word and tag line with a token that has no `_`.
    /// `WordTagSampleStream` logs "Error during parsing, ignoring
    /// sentence" and returns a sample with no tokens in its place.
    NoTagSeparator { line: u32, token: Span },
    /// A lemma or chunk line without exactly three columns, skipped
    /// ("Skipping corrupt line").
    CorruptLine { line: u32, columns: usize },
    /// A language sample line without a tab after a non-empty language
    /// code, which `LanguageDetectorSampleStream` passes over silently.
    NoLanguage { line: u32 },
}
