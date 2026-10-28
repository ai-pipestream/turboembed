//! Document language detection with a hashed character n-gram linear
//! classifier and script priors (#1525).
//!
//! - [`NgramModel`] is the trained model: hashed character n-gram features
//!   and a linear softmax classifier over language codes.
//! - [`NgramDetector`] is the annotator that reads a `View` and writes
//!   `opennlp:language` with the model's best language and confidence.

pub mod error;
pub mod ngram;

pub use error::ModelError;
pub use ngram::NgramModel;
