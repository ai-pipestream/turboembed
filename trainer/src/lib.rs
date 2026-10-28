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
//! data order, gradients are summed in a fixed order, and nothing runs
//! on more than one thread, so two runs give identical bytes.

pub mod adamw;
pub mod adapter;
pub mod crf;
pub mod fit;
pub mod fit_parse;
pub mod linear;
pub mod logspace;
pub mod lstm;
pub mod parser;
pub mod real;
pub mod rng;
mod shapes;
pub mod tagger;
pub mod xent;

pub use fit::{Config, Example, Report, fit};
pub use fit_parse::{ParseExample, fit_parser};
pub use parser::Parser;
pub use real::Real;
pub use tagger::Tagger;
