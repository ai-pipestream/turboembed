//! Evaluation: OpenNLP's scores (`FMeasure`, `Mean`), the evaluators of
//! its components, and the folds of cross-validation.
//!
//! - [`measure`]: precision, recall and F-measure, and the arithmetic
//!   mean the accuracies are.
//! - [`components`]: an evaluator per component, scoring a prediction
//!   against a reference sample.
//! - [`partition`]: `CrossValidationPartitioner`'s folds.
//!
//! # What OpenNLP does here
//!
//! - `FMeasure` counts, for each reference, every prediction equal to it,
//!   so repeated predictions all count and recall can exceed 1; and it
//!   removes a match of the previous reference when a reference matches
//!   nothing ([`measure::count_true_positives`]).
//! - The F-measure is -1, not 0 or NaN, when precision and recall are 0;
//!   the static `precision` and `recall` are NaN on no predictions or no
//!   references, while the running scores are 0.
//! - An evaluator that throws on a prediction has counted part of it: the
//!   part-of-speech and lemma evaluators count the tokens before the
//!   missing prediction, the span evaluators count the whole sample.
//! - The name finder evaluator scores an untyped reference name as type
//!   `default`, but judges the sample against the untyped name, so such a
//!   sample is never correctly classified.
//! - The detokenizer evaluator compares the detokenized text with the
//!   sample's text, which the token format ends with a space, so no
//!   sample of that format with a token is right (an empty line is).
//! - With one partition, cross-validation trains on the samples at odd
//!   places and tests on all of them.
//!
//! The tests compare every evaluator but one with the `evaluate` goldens
//! the JVM wrote. `DetokenizerEvaluator` is in `opennlp-tools`, which the
//! goldens image does not build, so it has unit tests and no JVM golden.

pub mod components;
pub mod measure;
pub mod partition;

pub use components::{
    ChunkerEvaluator, DetokenizerEvaluator, DocumentCategorizerEvaluator, EvalError, LanguageDetectorEvaluator,
    LemmatizerEvaluator, NameFinderEvaluator, PosEvaluator, SentenceDetectorEvaluator, TokenizerEvaluator, Verdict,
};
pub use measure::{FMeasure, LabeledSpan, Mean, count_true_positives, precision, recall};
pub use partition::{CrossValidation, Fold};
