//! Document language detection, including the native hashed character
//! n-gram classifier with script priors (#1525).

pub mod context;
pub mod detector;
pub mod error;
pub mod model;
pub mod ngram;
pub mod probing;

pub use context::ContextGenerator;
pub use detector::{LanguageDetector, NO_EVIDENCE, Prediction, Ranking, Scratch};
pub use error::ModelError;
pub use model::LanguageModel;
pub use ngram::NgramModel;
pub use probing::{ProbingConfig, ProbingContext};
