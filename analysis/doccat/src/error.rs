//! Errors returned when configuring or parsing document categorizer feature
//! generators.

use std::fmt;

/// Why a document categorizer feature generator or configuration could not be
/// built or parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DoccatError {
    /// A feature generator specification names an unknown generator class or
    /// malformed parameter list.
    UnknownFeatureGenerator(String),
    /// `NGramFeatureGenerator` was configured with `min_gram == 0` or
    /// `min_gram > max_gram`.
    InvalidNGramRange { min_gram: usize, max_gram: usize },
}

impl fmt::Display for DoccatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DoccatError::UnknownFeatureGenerator(cls) => {
                write!(f, "unsupported doccat feature generator specification {cls:?}")
            }
            DoccatError::InvalidNGramRange { min_gram, max_gram } => {
                write!(
                    f,
                    "invalid NGramFeatureGenerator range: min_gram ({min_gram}) must be >= 1 and <= max_gram ({max_gram})"
                )
            }
        }
    }
}

impl std::error::Error for DoccatError {}
