//! Why a language detector model could not be opened.

use std::fmt;

/// A model file that is not a usable `LanguageDetectorME` model. Each
/// variant names what was wrong.
#[derive(Debug)]
pub enum ModelError {
    /// The file or its classifier entry could not be read.
    File(turbo_maxent::Error),
    /// The manifest's `Component-Name` is not `LanguageDetectorME`.
    Component(String),
    /// The manifest names a factory other than `LanguageDetectorFactory`.
    /// A subclass can change the context generator, which this port does
    /// not reproduce.
    Factory(String),
    /// A classifier outcome holds half of a surrogate pair, which no
    /// language code can hold.
    OutcomeNotText(turbo_maxent::JavaString),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::File(e) => write!(f, "language detector model: {e}"),
            ModelError::Component(c) => {
                write!(f, "language detector model: component {c:?} is not LanguageDetectorME")
            }
            ModelError::Factory(c) => write!(
                f,
                "language detector model: factory {c:?} is not opennlp.tools.langdetect.LanguageDetectorFactory"
            ),
            ModelError::OutcomeNotText(o) => {
                write!(f, "language detector model: the outcome {o:?} holds half of a surrogate pair")
            }
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelError::File(e) => Some(e),
            _ => None,
        }
    }
}

impl From<turbo_maxent::Error> for ModelError {
    fn from(e: turbo_maxent::Error) -> ModelError {
        ModelError::File(e)
    }
}
