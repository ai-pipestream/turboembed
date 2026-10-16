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
//!
//! docs/doccat.md describes the model files, the parity tests and their
//! tolerances, and why the predictor is a port rather than linfa-svm.

pub mod features;
mod hash;
pub mod java;
pub mod svm;

pub use features::FeatureGenerator;
pub use svm::{ModelError, Scratch, SvmCategorizer, SvmDoccatModel};
