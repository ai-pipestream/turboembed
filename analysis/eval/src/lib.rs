//! The evaluation harness for retrieval pipelines (issue #154).
//!
//! Steps 1 and 2 are in: the measures, the significance tests, the
//! record a run is written as, the gate that compares two sets of
//! records, and the evaluation sets, fetched by checksum and read into
//! typed values. Running a pipeline over them, the baselines and the
//! `turbo-eval` binary come in later steps.
//!
//! - [`metrics`]: nDCG@10 and recall@k at document level.
//! - [`spans`]: how well the top k chunks cover the relevant byte spans.
//! - [`significance`]: a paired bootstrap interval and a paired
//!   permutation p value, drawn from `java.util.Random` with a fixed seed.
//! - [`cost`]: the chunk length distribution.
//! - [`record`]: one evaluation record per set and pipeline, as JSON.
//! - [`set`]: an evaluation set once it is read.
//! - [`sets`]: each set's published files, licence and span unit, one
//!   file per set, and the licence and digest checks before a load.
//! - [`read`]: the readers of the sets' own formats, one per format,
//!   which convert span judgements to UTF-8 byte offsets.
//! - [`fetch`]: downloading a set into the cache root and checking each
//!   file's SHA-256.
//! - [`gate`]: candidate records against baseline records.

pub mod cost;
pub mod fetch;
pub mod gate;
pub mod metrics;
pub mod read;
pub mod record;
pub mod set;
pub mod sets;
pub mod significance;
pub mod spans;

/// Errors are messages, as in the rest of the workspace.
pub type Result<T> = std::result::Result<T, String>;

/// The cutoffs recall is reported at, at document and span level.
pub const RECALL_KS: [usize; 4] = [1, 5, 10, 100];
