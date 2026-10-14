use std::fmt;

use turbo_tokenize::me::PatternError;

/// Why a factory, an event stream or a reproduction step was refused.
#[derive(Debug)]
pub enum Error {
    /// The trainer, the indexer or a parameter.
    Train(turbo_train::Error),
    /// A sample could not be read.
    Format(turbo_formats::FormatError),
    /// Dictionary XML could not be read.
    DictionaryXml(turbo_maxent::Error),
    /// `new StringList()` with no tokens: `IllegalArgumentException("tokens
    /// must not be empty")`.
    EmptyEntry,
    /// A case-sensitive dictionary built with `put` was given two entries
    /// that differ only in case. The JVM keeps both and finds each by an
    /// exact lookup; the lookup dictionary this crate hands the context
    /// generators keeps one, so the pair is refused rather than looked up
    /// differently.
    CaseVariants { first: String, second: String },
    /// A `java.util.HashMap` bucket would hold nine entries in a table of
    /// 64 or more, where the JVM turns the bucket into a tree whose
    /// iteration order this crate does not simulate; the serialized
    /// dictionary's entry order would be a guess, so it is refused.
    TreeifiedBucket { entries: usize },
    /// A dictionary token holds a character XML 1.0 cannot carry (a control
    /// character other than tab, line feed and carriage return, or U+FFFE
    /// or U+FFFF): the JDK's XML serializer throws `SAXException` and
    /// `Dictionary.serialize` wraps it in an `IOException`, so such a
    /// dictionary cannot be written to a model file.
    InvalidXmlCharacter(u32),
    /// The tokenizer's alphanumeric pattern is one `turbo-tokenize` cannot
    /// evaluate.
    Pattern(PatternError),
    /// An end-of-sentence character given as a surrogate code unit.
    SurrogateEos(u16),
    /// The treebank could not be fetched or verified.
    Zoo(turbo_zoo::ZooError),
    /// A published model file could not be read.
    Model(turbo_maxent::Error),
    /// A published model's manifest lacks an entry the comparison needs.
    MissingManifestEntry(&'static str),
    /// A file could not be read.
    Io { path: std::path::PathBuf, message: String },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Train(e) => write!(f, "training: {e}"),
            Error::Format(e) => write!(f, "samples: {e}"),
            Error::DictionaryXml(e) => write!(f, "dictionary XML: {e}"),
            Error::EmptyEntry => f.write_str("tokens must not be empty"),
            Error::CaseVariants { first, second } => {
                write!(f, "a case-sensitive dictionary with {first:?} and {second:?}, which differ only in case")
            }
            Error::TreeifiedBucket { entries } => {
                write!(f, "a HashMap bucket of {entries} entries in a table of 64 or more would be a tree")
            }
            Error::InvalidXmlCharacter(c) => {
                write!(f, "An invalid XML character (Unicode: 0x{c:x}) was found in the node's character data content.")
            }
            Error::Pattern(e) => write!(f, "alphanumeric pattern: {e}"),
            Error::SurrogateEos(u) => write!(f, "end of sentence character U+{u:04X} is a surrogate"),
            Error::Zoo(e) => write!(f, "zoo: {e}"),
            Error::Model(e) => write!(f, "model file: {e}"),
            Error::MissingManifestEntry(k) => write!(f, "the published manifest has no {k}"),
            Error::Io { path, message } => write!(f, "{}: {message}", path.display()),
        }
    }
}

impl std::error::Error for Error {}

impl From<turbo_train::Error> for Error {
    fn from(e: turbo_train::Error) -> Error {
        Error::Train(e)
    }
}

impl From<turbo_formats::FormatError> for Error {
    fn from(e: turbo_formats::FormatError) -> Error {
        Error::Format(e)
    }
}

impl From<PatternError> for Error {
    fn from(e: PatternError) -> Error {
        Error::Pattern(e)
    }
}

impl From<turbo_zoo::ZooError> for Error {
    fn from(e: turbo_zoo::ZooError) -> Error {
        Error::Zoo(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
