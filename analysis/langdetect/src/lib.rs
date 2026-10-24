//! Document language detection, ported from OpenNLP 3.x's
//! `LanguageDetectorME` onto the analysis API.
//!
//! - [`LanguageModel`] opens an OpenNLP language detector model
//!   (`langdetect-183.bin` and others `LanguageDetectorFactory` wrote).
//! - [`ContextGenerator`] is `DefaultLanguageDetectorContextGenerator`:
//!   the five normalizers OpenNLP runs first, then the distinct 1 to 3
//!   code unit n-grams of the lowercased text, in the order the JVM hands
//!   them to the model.
//! - [`ProbingContext`] and [`ProbingConfig`] reproduce the per-chunk
//!   context generation, cumulative `ngramCounts` `HashMap` bucket order,
//!   and `seenEnough` stopping rule of `probingPredictLanguages`.
//! - [`LanguageDetector`] is `predictLanguages`, `predictLanguage`, and
//!   `probingPredictLanguages`, and, as an annotator, reads a `View` and
//!   writes `opennlp:language`: every language of the model, best first,
//!   with its probability.
//!
//! The model is evaluated with `turbo-maxent`, whose arithmetic matches
//! the JVM's, over the n-grams in the JVM's order, so the probabilities
//! of the golden texts are the JVM's bit for bit. docs/langdetect.md
//! describes the port, its parity tests and, under "Differences from
//! OpenNLP", where it differs.

pub mod context;
pub mod detector;
pub mod error;
pub mod model;
pub mod probing;

pub use context::ContextGenerator;
pub use detector::{LanguageDetector, NO_EVIDENCE, Prediction, Ranking, Scratch};
pub use error::ModelError;
pub use model::LanguageModel;
pub use probing::{ProbingConfig, ProbingContext};
