//! Native hashed character n-gram language classification with script priors.

pub mod error;
pub mod ngram;

pub use error::ModelError;
pub use ngram::NgramModel;

/// Error message when a text contains no language evidence.
pub const NO_EVIDENCE: &str = "no language evidence";
