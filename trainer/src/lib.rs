//! Training for turbo bundle heads, in Rust, with no Python.
//!
//! Stage R1 (#1518): heads on a frozen encoder. A head is trained on the
//! per-word features the encoder gives for a corpus, and writes exactly
//! the tensors the core reads (`turbo::head::tagger::tensors`), so the
//! head a run holds in memory and the same head loaded from a bundle
//! score identically. Stage R2 (#1519): the encoder itself in training
//! (`encoder`), on the CPU, alone or under a head on the same loop.
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
pub mod coref;
pub mod crf;
pub mod encoder;
pub mod fit;
pub mod fit_coref;
pub mod fit_lemma;
pub mod fit_parse;
pub mod fit_relation;
pub mod fit_span;
pub mod langid;
pub mod lemma;
pub mod linear;
pub mod logspace;
pub mod lstm;
pub mod mix;
pub mod parser;
pub mod real;
pub mod relation;
pub mod rng;
pub mod scale;
mod shapes;
pub mod span;
pub mod tagger;
pub mod tally;
pub mod xent;

pub use encoder::tuned::{Tuned, TunedExample, fit_tuned, fit_tuned_with, tuned_gradients};
pub use fit::{Config, Draw, EarlyStop, Example, Report, batch_gradients, fit, fit_with};
pub use fit_coref::{CorefExample, fit_coref, fit_coref_with};
pub use fit_lemma::{LemmaExample, fit_lemmatizer, fit_lemmatizer_with, mine_trees};
pub use fit_parse::{ParseExample, fit_parser, fit_parser_with};
pub use fit_relation::{RelationExample, fit_relations};
pub use fit_span::{SpanExample, fit_spans};
pub use lemma::Lemmatizer;
pub use parser::Parser;
pub use real::Real;
pub use relation::Relations;
pub use scale::{LossScale, Scaler};
pub use span::Spans;
pub use tagger::Tagger;
