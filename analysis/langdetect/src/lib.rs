//! Native hashed character n-gram language classification with script priors.

pub mod error;
pub mod ngram;

pub use error::ModelError;
pub use ngram::NgramModel;
