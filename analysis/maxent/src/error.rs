use std::fmt;
use std::path::PathBuf;

/// Why a model file, or a part of one, could not be read. Every variant
/// names what was wrong and where: the byte offset, the zip entry, the
/// manifest line or the model type.
#[derive(Debug)]
pub enum Error {
    /// The file could not be read from disk.
    Io { path: PathBuf, error: std::io::Error },
    /// A deflate stream is malformed. The message says how; the zip
    /// entry it belongs to is added by [`Error::Entry`].
    Inflate(&'static str),
    /// The zip archive's structure is malformed or uses a feature this
    /// reader does not support (zip64, encryption, multiple disks, a
    /// compression method other than stored or deflated).
    Zip { offset: usize, message: &'static str },
    /// The archive has no entry with this name.
    MissingEntry(Box<str>),
    /// An entry's data does not match the CRC-32 its directory record
    /// states.
    Crc { entry: Box<str>, expected: u32, actual: u32 },
    /// Something went wrong inside the named entry.
    Entry { entry: Box<str>, error: Box<Error> },
    /// Binary data in Java `DataInput` form ended early or held malformed
    /// modified UTF-8.
    Data { offset: usize, message: &'static str },
    /// A properties file holds a malformed `\uXXXX` escape on the given
    /// line (1-based, counting physical lines).
    Properties { line: usize, message: &'static str },
    /// The manifest lacks a property every OpenNLP model must have, or
    /// holds one that cannot be used.
    Manifest { key: &'static str, message: &'static str },
    /// The model's type string names no model this crate evaluates.
    ModelType(Box<str>),
    /// A count in the model is negative, above the limit OpenNLP enforces
    /// (`ResourceLimits.MAX_ENTRIES`, ten million), or zero where at least
    /// one is needed (a model must have an outcome).
    Count { what: &'static str, value: i32, offset: usize },
    /// An outcome pattern is not a list of integers, or refers to an
    /// outcome the model does not have.
    OutcomePattern { pattern: u32, message: &'static str },
    /// The outcome patterns describe more predicates than the model lists.
    PatternPredicates { described: u64, listed: u32 },
    /// The dictionary XML is malformed at this byte offset.
    Xml { offset: usize, message: &'static str },
    /// The dictionary XML declares a character encoding this reader does
    /// not decode.
    XmlEncoding(Box<str>),
    /// The dictionary is well-formed XML but not a valid dictionary.
    Dictionary { offset: usize, message: &'static str },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, error } => write!(f, "{}: {error}", path.display()),
            Error::Inflate(message) => write!(f, "inflate: {message}"),
            Error::Zip { offset, message } => write!(f, "zip at byte {offset}: {message}"),
            Error::MissingEntry(name) => write!(f, "the archive has no entry {name:?}"),
            Error::Crc { entry, expected, actual } => {
                write!(f, "entry {entry:?}: CRC-32 is {actual:08x}, the directory says {expected:08x}")
            }
            Error::Entry { entry, error } => write!(f, "entry {entry:?}: {error}"),
            Error::Data { offset, message } => write!(f, "at byte {offset}: {message}"),
            Error::Properties { line, message } => write!(f, "properties line {line}: {message}"),
            Error::Manifest { key, message } => write!(f, "manifest property {key}: {message}"),
            Error::ModelType(found) => write!(f, "model type {found:?} is not GIS, QN or Perceptron"),
            Error::Count { what, value, offset } => {
                write!(f, "at byte {offset}: {what} count {value} is out of range")
            }
            Error::OutcomePattern { pattern, message } => write!(f, "outcome pattern {pattern}: {message}"),
            Error::PatternPredicates { described, listed } => {
                write!(f, "the outcome patterns describe {described} predicates, the model lists {listed}")
            }
            Error::Xml { offset, message } => write!(f, "XML at byte {offset}: {message}"),
            Error::XmlEncoding(name) => write!(f, "XML encoding {name:?} is not UTF-8, US-ASCII or ISO-8859-1"),
            Error::Dictionary { offset, message } => write!(f, "dictionary at byte {offset}: {message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { error, .. } => Some(error),
            Error::Entry { error, .. } => Some(error.as_ref()),
            _ => None,
        }
    }
}

/// The crate's result type.
pub type Result<T> = std::result::Result<T, Error>;
