//! The evaluation harness for retrieval pipelines (issue #154).
//!
//! A [`pipeline::Pipeline`] runs over one evaluation set: it analyzes
//! each document (sentences, structure, the embedder's subwords), chunks
//! it, embeds each chunk once and each query the same way, and ranks
//! every query against every chunk by exact cosine. The run is scored at
//! document and span level, its cost measured stage by stage, and the
//! result written as a record; the gate compares two sets of records.
//! The `turbo-eval` binary runs both. Reading the evaluation sets is
//! The sets are fetched by checksum and read into typed values, and
//! plug into a run at [`run::SetLoader`].
//!
//! - [`analysis`]: the layers each document is given before chunking.
//! - [`chunkers`]: the chunkers behind one trait: fixed token windows,
//!   sentence windows, the OpenNLP sandbox's semantic chunker, and
//!   turbo-chunker's optimizer.
//! - [`embedder`]: the embedder behind one trait, and the one that runs
//!   a bundle through a turbo session.
//! - [`retrieve`]: exact cosine over all chunk vectors; a document
//!   scores its best chunk.
//! - [`pipeline`]: the stages together, with their cost, and the record.
//! - [`run`]: the `run` and `gate` commands as functions.
//! - [`source`]: the commit a record names, and when it was made.
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
//! - [`model_gate`]: a trained model against the reference model for the
//!   same task on the same held-out samples, and its record.

pub use turbo_calibration::metrics as calibration;

pub mod analysis;
pub mod chunkers;
pub mod cost;
pub mod embedder;
pub mod fetch;
pub mod gate;
pub mod hybrid;
pub mod lexical;
pub mod metrics;
pub mod model_gate;
pub mod pipeline;
pub mod read;
pub mod record;
pub mod retrieve;
pub mod run;
pub mod set;
pub mod sets;
pub mod significance;
pub mod source;
pub mod spans;

/// Errors are messages, as in the rest of the workspace.
pub type Result<T> = std::result::Result<T, String>;

/// The cutoffs recall is reported at, at document and span level.
pub const RECALL_KS: [usize; 4] = [1, 5, 10, 100];
