//! Errors returned when opening, configuring, or training a document
//! categorizer model.

use std::fmt;

use turbo_maxent::JavaString;

/// Why a document categorizer model or feature generator could not be
/// built, opened, or trained.
#[derive(Debug)]
pub enum DoccatError {
    /// The archive's `Component-Name` is not `DocumentCategorizerME`.
    Component(String),
    /// The archive's `factory` is not `opennlp.tools.doccat.DoccatFactory`.
    Factory(String),
    /// The archive's `doccat.featureGenerators` names an unknown feature
    /// generator class.
    UnknownFeatureGenerator(String),
    /// `NGramFeatureGenerator` was configured with `min_gram == 0` or
    /// `min_gram > max_gram`.
    InvalidNGramRange { min_gram: usize, max_gram: usize },
    /// Reading `doccat.model` failed.
    Maxent(turbo_maxent::Error),
    /// An outcome in `doccat.model` contains lone UTF-16 surrogates and is
    /// not valid UTF-8 text.
    OutcomeNotText(JavaString),
    /// Training or packaging a model failed.
    Train(turbo_train_components::Error),
}

impl fmt::Display for DoccatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DoccatError::Component(c) => {
                write!(f, "expected Component-Name DocumentCategorizerME, got {c:?}")
            }
            DoccatError::Factory(fac) => {
                write!(f, "unsupported DoccatModel factory {fac:?}")
            }
            DoccatError::UnknownFeatureGenerator(cls) => {
                write!(f, "unsupported doccat feature generator class {cls:?}")
            }
            DoccatError::InvalidNGramRange { min_gram, max_gram } => {
                write!(
                    f,
                    "invalid NGramFeatureGenerator range: min_gram ({min_gram}) must be >= 1 and <= max_gram ({max_gram})"
                )
            }
            DoccatError::Maxent(e) => write!(f, "{e}"),
            DoccatError::OutcomeNotText(o) => {
                write!(f, "category outcome {o} is not valid UTF-8 text")
            }
            DoccatError::Train(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for DoccatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DoccatError::Maxent(e) => Some(e),
            DoccatError::Train(e) => Some(e),
            _ => None,
        }
    }
}

impl From<turbo_maxent::Error> for DoccatError {
    fn from(e: turbo_maxent::Error) -> Self {
        DoccatError::Maxent(e)
    }
}

impl From<turbo_train_components::Error> for DoccatError {
    fn from(e: turbo_train_components::Error) -> Self {
        DoccatError::Train(e)
    }
}

impl From<turbo_train::Error> for DoccatError {
    fn from(e: turbo_train::Error) -> Self {
        DoccatError::Train(turbo_train_components::Error::from(e))
    }
}
