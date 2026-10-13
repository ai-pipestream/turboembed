//! The evaluation harness for retrieval pipelines (issue #154).
//!
//! This is step 1: the measures, the significance tests, the record a run
//! is written as, and the gate that compares two sets of records. Reading
//! the evaluation sets, running a pipeline over them and the `turbo-eval`
//! binary come in later steps.
//!
//! - [`metrics`]: nDCG@10 and recall@k at document level.
//! - [`spans`]: how well the top k chunks cover the relevant byte spans.
//! - [`significance`]: a paired bootstrap interval and a paired
//!   permutation p value, drawn from `java.util.Random` with a fixed seed.
//! - [`cost`]: the chunk length distribution.
//! - [`record`]: one evaluation record per set and pipeline, as JSON.
//! - [`set`]: an evaluation set once it is read.
//! - [`gate`]: candidate records against baseline records.

pub mod cost;
pub mod gate;
pub mod metrics;
pub mod record;
pub mod set;
pub mod significance;
pub mod spans;

/// Errors are messages, as in the rest of the workspace.
pub type Result<T> = std::result::Result<T, String>;

/// The cutoffs recall is reported at, at document and span level.
pub const RECALL_KS: [usize; 4] = [1, 5, 10, 100];
