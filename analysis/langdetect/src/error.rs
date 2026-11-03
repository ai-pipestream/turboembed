//! Why a language detector model could not be opened.

use std::fmt;

/// A native language model that could not be constructed or loaded.
#[derive(Debug)]
pub enum ModelError {
    /// A classifier outcome is the empty string, which `Language` layers refuse.
    EmptyOutcome,
    /// Two classifier outcomes have the same language code.
    DuplicateOutcome(String),
    /// The n-gram weight vector has the wrong length.
    Weights { expected: usize, found: usize },
    /// The n-gram bias vector has the wrong length.
    Bias { expected: usize, found: usize },
    /// The serialized n-gram model is malformed.
    Format(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::EmptyOutcome => {
                write!(f, "language detector model: an outcome is the empty string")
            }
            ModelError::DuplicateOutcome(c) => {
                write!(f, "language detector model: duplicate outcome {c:?}")
            }
            ModelError::Weights { expected, found } => {
                write!(f, "language detector model: expected {expected} weights, found {found}")
            }
            ModelError::Bias { expected, found } => {
                write!(f, "language detector model: expected {expected} biases, found {found}")
            }
            ModelError::Format(detail) => write!(f, "language detector model: {detail}"),
        }
    }
}

impl std::error::Error for ModelError {}
