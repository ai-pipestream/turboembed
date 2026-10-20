//! An exact vector index for bounded collections that live next to
//! inference: label and schema-field candidates, entity-linking candidates
//! for a loaded catalog, glossary candidates before a heavier scorer. It
//! holds its rows in memory and is not a corpus index: there is no
//! persistence, no update and no delete.
//!
//! - An [`IndexBuilder`] takes rows from `Vectors<L>` layers or from raw
//!   slices, each with the caller's metadata, and builds a
//!   [`VectorIndex`], which is immutable and shared across threads.
//! - Every index records the [`Space`] of its vectors: the model that made
//!   them, their dimension and whether they are unit length. A query names
//!   its own space, and a query from another model, dimension or
//!   normalization is refused with an [`IndexError`], never scored.
//! - Top-k is exact: every row is scored, by dot product or cosine
//!   ([`Metric`]), and ties keep insertion order.
//! - Zero vectors, values that are not finite and vectors too long to
//!   score in `f32` are refused on insert and on query.
//! - With the `turboquant` feature, `IndexBuilder::build_quantized` stores
//!   the rows as TurboQuant codes instead, 2 to 4 bits a coordinate, in a
//!   `QuantizedIndex` that answers the same queries with estimated scores.
//! - A [`TokenIndex`] holds several token vectors a candidate (the subword
//!   vectors of a chunk, a section or a document) and scores a query's
//!   token vectors against them by late interaction (MaxSim, as in
//!   ColBERT), with the candidate token that won each query token.
//!
//! A score is an `f32` dot product summed in one fixed order (16 lanes,
//! element `i` in lane `i % 16`, each lane a chain of fused multiply-adds,
//! the lanes then added pairwise: `j + 8`, `j + 4`, `j + 2`, `j + 1`).
//! The AVX-512, AVX2 and scalar paths all follow that order, so a score
//! is the same bits on every machine and every path. A cosine index
//! stores each row divided by its length (computed in `f64`), divides the
//! query the same way, and scores the dot product of the two, clamped to
//! `[-1, 1]`.

mod build;
mod index;
mod kernel;
mod late;
#[cfg(feature = "turboquant")]
mod quant;
mod rows;
mod topk;

#[cfg(test)]
mod tests;

pub use build::IndexBuilder;
pub use index::VectorIndex;
pub use late::{LateHit, Prune, Source, TokenIndex, TokenIndexBuilder, TokenMatch};
#[cfg(feature = "turboquant")]
pub use quant::{QuantizedIndex, TurboQuant};

use std::fmt;

use turbo_analysis::ModelRef;
use turbo_analysis_layers::Vectors;

/// How rows are compared with a query.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Metric {
    /// The dot product of the query and the row as given.
    Dot,
    /// The cosine of the angle between the query and the row.
    Cosine,
}

/// Whether the vectors of a space are unit length, as an encoder that
/// normalizes its output makes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Normalization {
    /// Every vector has length 1, to within [`UNIT_TOLERANCE`].
    Unit,
    /// Vectors are of any nonzero length.
    None,
}

/// How far from 1 the length of a vector in a [`Normalization::Unit`]
/// space may be: an FP16 encoder's normalized output is within it.
pub const UNIT_TOLERANCE: f64 = 1e-3;

/// The longest vector the index scores. Two vectors of at most this
/// length have a dot product of at most `2^100` in magnitude, and so does
/// every partial sum of it, far inside `f32`.
pub const MAX_LENGTH: f64 = (1u64 << 50) as f64;

/// The embedding space vectors come from: the model, the dimension and the
/// normalization. Vectors of two spaces are not comparable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Space {
    pub model: ModelRef,
    pub dim: u32,
    pub normalization: Normalization,
}

impl Space {
    pub fn new(model: ModelRef, dim: u32, normalization: Normalization) -> Space {
        Space { model, dim, normalization }
    }

    /// The space of a vectors layer. The layer does not say whether its
    /// producer normalized, so the caller does.
    pub fn of<L>(vectors: &Vectors<L>, normalization: Normalization) -> Space {
        Space { model: vectors.model.clone(), dim: vectors.dim, normalization }
    }
}

/// One result of a query: a row of the index, in insertion order from 0,
/// and its score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub row: u32,
    pub score: f32,
}

/// Which vector an error is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    /// The row of this number, inserted or about to be.
    Row(usize),
    /// The query of this number in a batch, 0 for a single query; for a
    /// late-interaction query, its token of this number.
    Query(usize),
    /// The late-interaction candidate of this number, added or about to be.
    Candidate(usize),
    /// Token `.1` of late-interaction candidate `.0`.
    Token(usize, usize),
}

impl fmt::Display for Which {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Which::Row(i) => write!(f, "row {i}"),
            Which::Query(i) => write!(f, "query {i}"),
            Which::Candidate(c) => write!(f, "candidate {c}"),
            Which::Token(c, i) => write!(f, "token {i} of candidate {c}"),
        }
    }
}

/// Why the index refused a vector, a layer or a query.
#[derive(Clone, Debug, PartialEq)]
pub enum IndexError {
    /// The space has dimension 0.
    ZeroDimension,
    /// The vectors are from another model than the index's.
    Model { index: Box<ModelRef>, other: Box<ModelRef> },
    /// The vector, layer or batch has another dimension than the index's.
    Dimension { index: u32, other: usize },
    /// The query's space is normalized another way than the index's.
    Normalization { index: Normalization, other: Normalization },
    /// A vector of zeros, which has no direction.
    Zero(Which),
    /// A value that is NaN or infinite, at this position of the vector.
    NonFinite(Which, usize),
    /// A vector in a unit space whose length is not 1.
    NotUnit(Which, f64),
    /// A vector longer than [`MAX_LENGTH`].
    TooLong(Which, f64),
    /// The layer fails its own check.
    Layer(String),
    /// More rows than a `u32` numbers.
    Full,
    /// A TurboQuant bit width other than 2 to 4.
    Bits(u8),
    /// TurboQuant storage refused the rows or a query, for the reason
    /// given (a dimension that is not a positive multiple of 8 or past
    /// 16384, a calibration that could not be fitted).
    Quantized(String),
    /// A late-interaction candidate or query with no token vectors.
    NoTokens(Which),
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexError::ZeroDimension => write!(f, "a space of dimension 0"),
            IndexError::Model { index, other } => write!(
                f,
                "vectors from model {} (revision {}, manifest {}), the index holds model {} (revision {}, manifest {})",
                other.id, other.revision, other.sha256, index.id, index.revision, index.sha256
            ),
            IndexError::Dimension { index, other } => {
                write!(f, "vectors of dimension {other}, the index holds dimension {index}")
            }
            IndexError::Normalization { index, other } => {
                write!(f, "a query with normalization {other:?}, the index holds {index:?}")
            }
            IndexError::Zero(w) => write!(f, "{w} is all zeros"),
            IndexError::NonFinite(w, at) => write!(f, "{w} has a value that is not finite at {at}"),
            IndexError::NotUnit(w, len) => write!(f, "{w} has length {len} in a unit space"),
            IndexError::TooLong(w, len) => write!(f, "{w} has length {len}, more than {MAX_LENGTH}"),
            IndexError::Layer(detail) => write!(f, "the vectors layer fails its check: {detail}"),
            IndexError::Full => write!(f, "more rows than the index numbers"),
            IndexError::Bits(bits) => write!(f, "{bits} bits a coordinate, TurboQuant codes 2 to 4"),
            IndexError::Quantized(detail) => write!(f, "TurboQuant storage refused: {detail}"),
            IndexError::NoTokens(w) => write!(f, "{w} has no token vectors"),
        }
    }
}

impl std::error::Error for IndexError {}
