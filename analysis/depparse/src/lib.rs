//! Typed dependency trees and the OpenNLP 3.x greedy arc-standard parser.
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
mod annotator;
mod context;
mod evaluation;
mod graph;
mod model;
mod sample;
mod transition;

pub use annotator::{DependencyAnnotator, DependencyBackend};
pub use context::{DependencyContextGenerator, ParserInput};
pub use evaluation::{AttachmentScores, DependencyCrossValidator, DependencyEvaluator};
pub use graph::{ArcDirection, DependencyArc, DependencyGraph, PathStep};
pub use model::{DependencyModel, DependencyParser, DependencyParserME};
pub use sample::{DependencyEventStream, DependencySample};
pub use transition::{ArcStandardOracle, ArcStandardState, Position, Transition};

/// A checked public boundary, invalid model, or unavailable transition.
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

pub mod feedforward;
pub use feedforward::{
    Activation, FeatureLayout, FeedforwardContext, FeedforwardDependencyModel, FeedforwardDependencyParser,
    FeedforwardDependencyTrainer, FeedforwardSettings, NeuralWeights, Vocabulary,
};
