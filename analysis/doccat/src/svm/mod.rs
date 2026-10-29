//! OpenNLP's SVM document categorizer (`opennlp-ml-libsvm`,
//! `DocumentCategorizerSVM`) at prediction, over a model exported as a
//! LIBSVM text model and a sidecar.

pub mod categorizer;
pub mod error;
pub mod libsvm;
pub mod model;
pub mod sidecar;
mod text;
pub mod vector;
pub mod weighting;

pub use categorizer::SvmCategorizer;
pub use error::{File, LineError, ModelError};
pub use model::{SvmDoccatModel, Vote};
pub use sidecar::{Sidecar, Vocabulary};
pub use vector::Scratch;
pub use weighting::{FeatureSelection, Scaling, TermWeighting};
