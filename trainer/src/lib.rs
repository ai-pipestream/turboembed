//! Training for turbo bundle heads, in Rust, with no Python.
//!
//! Stage R1 (#1518): heads on a frozen encoder. A head is trained on the
//! per-word features the encoder gives for a corpus, and writes exactly
//! the tensors the core reads (`turbo::head::tagger::tensors`), so the
//! head a run holds in memory and the same head loaded from a bundle
//! score identically.
//!
//! Every op is generic over [`Real`]: training runs in f32, and the
//! gradient checks run the same code in f64 against finite differences.
//! A run is deterministic: the seed fixes the initial weights and the
//! data order, and gradients are summed in a fixed order whatever the
//! thread count (a batch's examples run in parallel, each into its own
//! buffer), so two runs give identical bytes.

pub mod adamw;
pub mod adapter;
mod batch;
pub mod crf;
pub mod fit;
pub mod fit_lemma;
pub mod fit_parse;
pub mod lemma;
pub mod linear;
pub mod logspace;
pub mod lstm;
pub mod parser;
pub mod real;
pub mod rng;
mod shapes;
pub mod tagger;
pub mod xent;

pub use fit::{Config, EarlyStop, Example, Report, batch_gradients, fit, fit_with};
pub use fit_lemma::{LemmaExample, fit_lemmatizer, fit_lemmatizer_with, mine_trees};
pub use fit_parse::{ParseExample, fit_parser, fit_parser_with};
pub use lemma::Lemmatizer;
pub use parser::Parser;
pub use real::Real;
pub use tagger::Tagger;
