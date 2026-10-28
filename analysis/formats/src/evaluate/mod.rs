//! Deterministic evaluation metrics and cross-validation folds.
//!
//! True positives use multiset intersection, so each prediction consumes at
//! most one matching reference and recall cannot exceed 1. Component
//! evaluators compare typed predictions with exact-offset reference samples.

pub mod components;
pub mod measure;
pub mod partition;

pub use components::{
    ChunkerEvaluator, DetokenizerEvaluator, DocumentCategorizerEvaluator, EvalError, LanguageDetectorEvaluator,
    LemmatizerEvaluator, NameFinderEvaluator, PosEvaluator, SentenceDetectorEvaluator, TokenizerEvaluator, Verdict,
};
pub use measure::{FMeasure, LabeledSpan, Mean, count_true_positives, precision, recall};
pub use partition::{CrossValidation, Fold};
