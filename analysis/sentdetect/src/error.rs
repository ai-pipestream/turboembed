//! Why a sentence model could not be opened.

use std::fmt;

use crate::eos::SurrogateInSet;

/// A model file that is not a usable `SentenceDetectorME` model. Each
/// variant names what was wrong.
#[derive(Debug)]
pub enum ModelError {
    /// The file or one of its entries could not be read.
    File(turbo_maxent::Error),
    /// The manifest's `Component-Name` is not `SentenceDetectorME`.
    Component(String),
    /// The manifest names a factory other than `SentenceDetectorFactory`.
    /// A subclass can change the context generator and the scanner, which
    /// this port does not reproduce.
    Factory(String),
    /// The manifest has no `useTokenEnd`, which OpenNLP requires.
    MissingUseTokenEnd,
    /// The manifest's `eosCharacters` hold a character outside the BMP.
    EosCharacters(SurrogateInSet),
    /// The classifier's outcomes are not exactly `s` and `n`; these are
    /// the ones it has.
    Outcomes(Vec<String>),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::File(e) => write!(f, "sentence model: {e}"),
            ModelError::Component(c) => write!(f, "sentence model: component {c:?} is not SentenceDetectorME"),
            ModelError::Factory(c) => {
                write!(f, "sentence model: factory {c:?} is not opennlp.tools.sentdetect.SentenceDetectorFactory")
            }
            ModelError::MissingUseTokenEnd => write!(f, "sentence model: the manifest has no useTokenEnd property"),
            ModelError::EosCharacters(e) => write!(f, "sentence model: {e}"),
            ModelError::Outcomes(o) => write!(f, "sentence model: outcomes {o:?} are not s and n"),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelError::File(e) => Some(e),
            ModelError::EosCharacters(e) => Some(e),
            _ => None,
        }
    }
}

impl From<turbo_maxent::Error> for ModelError {
    fn from(e: turbo_maxent::Error) -> ModelError {
        ModelError::File(e)
    }
}
