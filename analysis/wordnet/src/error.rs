//! Error type for WordNet readers, lexicon validation, and algorithms.

use std::fmt;
use std::io;

/// An error returned while reading, validating, or querying a WordNet resource.
#[derive(Debug)]
pub enum WordNetError {
    /// An I/O error occurred while reading a file or stream.
    Io(io::Error),
    /// A structural, format, or validation error occurred while parsing a WordNet resource.
    InvalidData(String),
    /// A parse or schema validation error occurred in a WNDB or WN-LMF resource.
    Parse(String),
    /// An invalid argument was supplied to a constructor or method.
    InvalidInput(String),
}

impl WordNetError {
    /// Creates a validation error with the given message.
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidData(msg.into())
    }

    /// Creates a parse/format error with the given message.
    pub fn format(msg: impl Into<String>) -> Self {
        Self::Parse(msg.into())
    }
}

impl fmt::Display for WordNetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::InvalidData(msg) | Self::Parse(msg) | Self::InvalidInput(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for WordNetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::InvalidData(_) | Self::Parse(_) | Self::InvalidInput(_) => None,
        }
    }
}

impl From<io::Error> for WordNetError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}
