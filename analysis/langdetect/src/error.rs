//! Why a language detector model could not be opened.

use std::fmt;

/// A model file that is not a usable language model. Each variant names
/// what was wrong.
#[derive(Debug)]
pub enum ModelError {
    /// The model's component name is not the expected one.
    Component(String),
    /// A classifier outcome is the empty string, which `Language` layers refuse.
    EmptyOutcome,
    /// Two classifier outcomes have the same language code.
    DuplicateOutcome(String),
    /// The weight vector has the wrong length.
    Weights { expected: usize, found: usize },
    /// The bias vector has the wrong length.
    Bias { expected: usize, found: usize },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Component(c) => {
                write!(f, "language detector model: component {c:?} is not the expected one")
            }
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
        }
    }
}

impl std::error::Error for ModelError {}
