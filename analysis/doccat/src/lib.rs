//! Document categorization, ported from OpenNLP 3.x onto the analysis API.
//!
//! - [`features`] is the document categorizer's feature generators,
//!   `BagOfWordsFeatureGenerator` and `NGramFeatureGenerator`.
//! - [`svm`] is the SVM categorizer of `opennlp-ml-libsvm`
//!   (`DocumentCategorizerSVM`): [`SvmDoccatModel`] opens a model exported
//!   as a LIBSVM text model and a sidecar, and [`SvmCategorizer`] is
//!   `categorize` and, as an annotator, reads `opennlp:tokens` and writes
//!   `turbo:categories`. Its LIBSVM predictor ([`svm::libsvm`]) is a port
//!   of LIBSVM's Java prediction code.
//! - [`sentiment`] is `SentimentME` (`opennlp.tools.sentiment`), a maxent
//!   classifier over the tokens and, with `EmojiSentimentFactory`, their
//!   emoji features: [`SentimentModel`] opens its model file and
//!   [`SentimentMe`] reads `opennlp:tokens` and writes the `sentiment`
//!   instance of `turbo:categories`.
//!
//! docs/doccat.md describes the model files, the parity tests and their
//! tolerances, and why the predictor is a port rather than linfa-svm;
//! docs/sentiment.md describes the sentiment detector.

pub mod features;
mod hash;
pub mod java;
pub mod sentiment;
pub mod svm;

pub use features::FeatureGenerator;
pub use sentiment::{SentimentFactory, SentimentMe, SentimentModel};
pub use svm::{ModelError, Scratch, SvmCategorizer, SvmDoccatModel};
