//! The maxent tokenizer, a port of OpenNLP's `TokenizerME`.
//!
//! The text is first cut into whitespace units ([`units`]). A unit of one
//! UTF-16 code unit, or one the model's alphanumeric pattern accepts when
//! its optimization is on ([`alnum`]), is a token as it stands. In every
//! other unit the classifier decides at each code unit position whether a
//! token boundary falls there, from the features of
//! `DefaultTokenContextGenerator` ([`context`]). A token's confidence is
//! the product of the decisions inside it. [`TokenizerModel`] reads the
//! model file; [`TokenizerMe`] tokenizes a `&str` and is the annotator
//! that writes `Tokens` and `Confidence<Tokens>`, with the crate's
//! [`TokenizerOptions`](crate::TokenizerOptions) and its shared sentence
//! handling ([`unit`](crate::unit)).
//!
//! The tokens and probabilities are those of the Java, checked against
//! files the JVM wrote, with one difference: Java can split between the
//! two halves of a character outside the Basic Multilingual Plane, and
//! here such a split is not taken (see [`TokenizerMe::tokenize_with`]).

pub mod alnum;
mod annotator;
pub mod context;
pub mod model;
pub mod tokenizer;
pub mod units;

pub use alnum::{AlphaNumeric, PatternError};
pub use context::{AffixLimits, Features, UnitFacts, token_context};
pub use model::{ModelError, TokenizerModel};
pub use tokenizer::{Scratch, TokenizerMe};
pub use units::WhitespaceUnits;
