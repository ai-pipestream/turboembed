//! Errors produced by n-gram models, XML serialization, probability utilities,
//! and Stupid Backoff language models.

use std::fmt;

/// Result type returned by `turbo-ngram` operations.
pub type Result<T> = std::result::Result<T, Error>;

/// An error returned by `turbo-ngram` operations.
#[derive(Debug)]
pub enum Error {
    /// An n-gram token list was empty (`StringList` requires at least one token).
    EmptyNGram,
    /// `set_count` was called for an n-gram that is not present in the model.
    NoSuchNGram,
    /// `min_length` or `max_length` was less than 1, or `min_length > max_length`.
    InvalidLengthRange {
        /// Minimum n-gram length requested.
        min_length: usize,
        /// Maximum n-gram length requested.
        max_length: usize,
    },
    /// `NGramLanguageModel` was constructed with `n == 0`.
    InvalidOrder(usize),
    /// Linear interpolation weights did not sum to `1.0` within `0.0001`.
    InvalidInterpolationWeights {
        /// Weight for the trigram term.
        lambda1: f64,
        /// Weight for the bigram term.
        lambda2: f64,
        /// Weight for the unigram term.
        lambda3: f64,
    },
    /// Dictionary XML could not be parsed.
    DictionaryXml(turbo_maxent::Error),
    /// An `<entry>` in a serialized `NGramModel` XML lacked a `count` attribute.
    MissingCountAttribute {
        /// Zero-based index of the entry in the XML document.
        entry_index: usize,
    },
    /// An `<entry>` in a serialized `NGramModel` XML had a non-integer `count` attribute.
    InvalidCountAttribute {
        /// Zero-based index of the entry in the XML document.
        entry_index: usize,
        /// The raw attribute value.
        value: String,
    },
    /// A token contained a character that XML 1.0 cannot represent.
    InvalidXmlCharacter(u32),
}

impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Error::EmptyNGram, Error::EmptyNGram) | (Error::NoSuchNGram, Error::NoSuchNGram) => true,
            (
                Error::InvalidLengthRange { min_length: a_min, max_length: a_max },
                Error::InvalidLengthRange { min_length: b_min, max_length: b_max },
            ) => a_min == b_min && a_max == b_max,
            (Error::InvalidOrder(a), Error::InvalidOrder(b)) => a == b,
            (
                Error::InvalidInterpolationWeights { lambda1: a1, lambda2: a2, lambda3: a3 },
                Error::InvalidInterpolationWeights { lambda1: b1, lambda2: b2, lambda3: b3 },
            ) => a1.to_bits() == b1.to_bits() && a2.to_bits() == b2.to_bits() && a3.to_bits() == b3.to_bits(),
            (Error::DictionaryXml(a), Error::DictionaryXml(b)) => a.to_string() == b.to_string(),
            (Error::MissingCountAttribute { entry_index: a }, Error::MissingCountAttribute { entry_index: b }) => {
                a == b
            }
            (
                Error::InvalidCountAttribute { entry_index: ai, value: av },
                Error::InvalidCountAttribute { entry_index: bi, value: bv },
            ) => ai == bi && av == bv,
            (Error::InvalidXmlCharacter(a), Error::InvalidXmlCharacter(b)) => a == b,
            _ => false,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::EmptyNGram => f.write_str("tokens must not be empty"),
            Error::NoSuchNGram => f.write_str("n-gram is not present in the model"),
            Error::InvalidLengthRange { min_length, max_length } => write!(
                f,
                "minLength ({min_length}) and maxLength ({max_length}) must be at least 1, and minLength must not exceed maxLength"
            ),
            Error::InvalidOrder(n) => write!(f, "Parameter 'n' must be greater than 0 (got {n})."),
            Error::InvalidInterpolationWeights { lambda1, lambda2, lambda3 } => {
                write!(f, "lambdas sum should be equal to 1 (got {lambda1} + {lambda2} + {lambda3})")
            }
            Error::DictionaryXml(err) => write!(f, "dictionary XML error: {err}"),
            Error::MissingCountAttribute { entry_index } => {
                write!(f, "The count attribute must be set (entry {entry_index})!")
            }
            Error::InvalidCountAttribute { entry_index, value } => {
                write!(f, "The count attribute '{value}' must be a number (entry {entry_index})!")
            }
            Error::InvalidXmlCharacter(cp) => {
                write!(f, "token contains a character XML 1.0 cannot represent: U+{cp:04X}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::DictionaryXml(err) => Some(err),
            _ => None,
        }
    }
}
