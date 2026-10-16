//! Error types for Morfologik automata, metadata, and dictionary readers.

use std::fmt;

/// An error raised when reading or building a Morfologik FSA automaton,
/// `.info` metadata file, or morphological dictionary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MorfologikError {
    /// An argument or metadata property is invalid (`IllegalArgumentException` in OpenNLP).
    InvalidArgument(String),
    /// An I/O, binary format, or character decoding error (`IOException` in OpenNLP).
    Format(String),
}

impl MorfologikError {
    pub fn invalid_argument(msg: impl Into<String>) -> Self {
        Self::InvalidArgument(msg.into())
    }

    pub fn format(msg: impl Into<String>) -> Self {
        Self::Format(msg.into())
    }

    /// The error message.
    pub fn message(&self) -> &str {
        match self {
            Self::InvalidArgument(msg) | Self::Format(msg) => msg,
        }
    }
}

impl fmt::Display for MorfologikError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for MorfologikError {}
