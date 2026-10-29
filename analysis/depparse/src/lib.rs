//! Typed dependency trees, CoNLL-U sample alignment, and UAS/LAS evaluation.
//! Dependency parsing models and document annotation use the biaffine parser
//! head on the shared encoder (`turbo-core` / `turbo-tagger`, #1514).
//! Node indexes are zero-based; `None` is the artificial root's outgoing arc.
//!
//! Word and token graph indexes cannot be mixed:
//! ```compile_fail
//! use turbo_analysis::Ix;
//! use turbo_analysis_layers::{Tokens, Words};
//! use turbo_depparse::DependencyGraph;
//! fn invalid(graph: &DependencyGraph<Words>, token: Ix<Tokens>) {
//!     graph.children(token).unwrap();
//! }
//! ```
mod evaluation;
mod graph;
mod held_out;
mod sample;

pub use evaluation::{AttachmentScores, DependencyCrossValidator, DependencyEvaluator, DependencyParser};
pub use graph::{ArcDirection, DependencyArc, DependencyGraph, PathStep};
pub use held_out::{EvaluationSubset, HeldOutEvaluation};
pub use sample::DependencySample;

/// A checked public boundary or invalid dependency graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;
pub(crate) fn error(e: impl std::fmt::Display) -> Error {
    Error(e.to_string())
}
