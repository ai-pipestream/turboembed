//! Centroid vectors and prototype (label centroid) classification over
//! turboembed embeddings (`#321`).
//!
//! Generalizes `org.apache.opennlp.grpc.chunk.Centroids` and
//! `ChunkEmbedProcessor` from `opennlp-sandbox` (`OPENNLP-1833-grpc-uber-server`):
//!
//! - [`centroid`] computes the element-wise arithmetic mean of a non-empty
//!   slice of equal-length `f32` vectors in `f64`, with optional post-mean
//!   L2 normalization ([`CentroidNorm`]).
//! - [`LabelCentroids`] represents a set of class labels where each label is
//!   defined by a set of example sentences (or vectors) averaged into one
//!   prototype centroid vector. Inference is one `f64` dot product per label
//!   followed by temperature-scaled softmax (single-label [`Categories`]) or
//!   scaled sigmoid (multi-label).
//! - Centroids load from and save to a plain typed binary format (`TCENT001`)
//!   with no pickle and no JSON trickery.

use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::Arc;

use turbo_analysis::{Document, Error, Ix, Key, Layer, ModelRef, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Categories, CategoryScope, IncivilityLabel, IncivilityScores, Sentences, Vectors};

use crate::subwords::SubwordAnnotator;
use crate::vectors::{Input, VectorAnnotator};

const CENTROID_MAGIC: &[u8; 8] = b"TCENT001";

/// Post-aggregation normalization applied to an arithmetic mean centroid,
/// matching `org.apache.opennlp.grpc.v1.VectorNormalization` in
/// `Centroids.java`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum CentroidNorm {
    /// Leave the element-wise arithmetic mean unnormalized.
    #[default]
    None = 0,
    /// Scale the arithmetic mean to unit L2 length; fails if the mean vector
    /// has zero norm.
    L2 = 1,
}

impl CentroidNorm {
    fn from_byte(b: u8) -> Result<CentroidNorm, String> {
        match b {
            0 => Ok(CentroidNorm::None),
            1 => Ok(CentroidNorm::L2),
            other => Err(format!("unrecognized centroid normalization code {other}")),
        }
    }
}

/// Computes the element-wise centroid (mean) vector of `vectors` in `f64`
/// and optionally L2-normalizes the mean, matching
/// `org.apache.opennlp.grpc.chunk.Centroids.centroid`.
///
/// Returns `Ok(None)` when `vectors` is empty, or `Err` when vectors do not
/// share a positive dimension or when `CentroidNorm::L2` is requested on a
/// zero-norm mean vector.
pub fn centroid<V: AsRef<[f32]>>(vectors: &[V], norm: CentroidNorm) -> Result<Option<Vec<f32>>, String> {
    if vectors.is_empty() {
        return Ok(None);
    }
    let dim = vectors[0].as_ref().len();
    if dim == 0 {
        return Err("centroid vectors must have a non-zero dimension".into());
    }
    let mut sums = vec![0.0f64; dim];
    for (idx, v) in vectors.iter().enumerate() {
        let slice = v.as_ref();
        if slice.len() != dim {
            return Err(format!("vector {idx} has length {}, expected {dim}", slice.len()));
        }
        for (s, &val) in sums.iter_mut().zip(slice) {
            if !val.is_finite() {
                return Err(format!("vector {idx} contains a non-finite value"));
            }
            *s += f64::from(val);
        }
    }
    let count = vectors.len() as f64;
    for s in &mut sums {
        *s /= count;
    }
    let divisor = match norm {
        CentroidNorm::None => 1.0f64,
        CentroidNorm::L2 => {
            let sq_norm: f64 = sums.iter().map(|&v| v * v).sum();
            let d = sq_norm.sqrt();
            if d == 0.0 || !d.is_finite() {
                return Err("embedding centroid has zero norm and cannot be L2-normalized".into());
            }
            d
        }
    };
    let out: Vec<f32> = sums.into_iter().map(|v| (v / divisor) as f32).collect();
    Ok(Some(out))
}

/// Prototype (label centroid) classifier over turboembed vectors.
///
/// Each label in `tags` is represented by one centroid vector of length
/// `dim`. Inference computes one `f64` dot product per label.
#[derive(Clone, Debug)]
pub struct LabelCentroids {
    model: ModelRef,
    dim: u32,
    tags: Arc<TagSet>,
    /// Row-major `[tags.len(), dim]` centroid vectors.
    centroids: Vec<f32>,
    temperature: f64,
    norm: CentroidNorm,
    hash: u64,
}

impl PartialEq for LabelCentroids {
    fn eq(&self, other: &Self) -> bool {
        self.model == other.model
            && self.dim == other.dim
            && self.tags.name == other.tags.name
            && self.tags.len() == other.tags.len()
            && (0..self.tags.len()).all(|i| self.tags.label(Tag(i as u16)) == other.tags.label(Tag(i as u16)))
            && self.centroids == other.centroids
            && self.temperature == other.temperature
            && self.norm == other.norm
            && self.hash == other.hash
    }
}

impl LabelCentroids {
    /// Builds `LabelCentroids` from pre-computed row-major centroid vectors
    /// (`[tags.len(), dim]`).
    pub fn from_centroids(
        model: ModelRef,
        dim: u32,
        tags: Arc<TagSet>,
        centroids: Vec<f32>,
        temperature: f64,
        norm: CentroidNorm,
    ) -> Result<LabelCentroids, String> {
        if dim == 0 {
            return Err("centroid dimension must be at least 1".into());
        }
        if tags.is_empty() {
            return Err("LabelCentroids requires at least one label in its TagSet".into());
        }
        if tags.len() > usize::from(u16::MAX) + 1 {
            return Err("centroid label count exceeds the u16 tag range".into());
        }
        let mut labels = std::collections::HashSet::new();
        for i in 0..tags.len() {
            let label = tags.label(Tag(i as u16));
            if label.trim().is_empty() || !labels.insert(label) {
                return Err("centroid labels must be nonblank and unique".into());
            }
        }
        let expected =
            tags.len().checked_mul(dim as usize).ok_or_else(|| "centroid dimensions overflow usize".to_string())?;
        if centroids.len() != expected {
            return Err(format!(
                "centroids buffer has {} elements, expected {} ({} labels * dim {})",
                centroids.len(),
                expected,
                tags.len(),
                dim
            ));
        }
        if let Some(k) = centroids.iter().position(|v| !v.is_finite()) {
            return Err(format!("centroid element {k} is non-finite"));
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(format!("temperature {temperature} must be finite and positive"));
        }
        let mut out = LabelCentroids { model, dim, tags, centroids, temperature, norm, hash: 0 };
        // Include the scoring revision while keeping TCENT001 bytes compatible.
        out.hash = fnv1a(&out.to_bytes()) ^ 0x4345_4e54_0000_0002;
        Ok(out)
    }

    pub fn model(&self) -> &ModelRef {
        &self.model
    }

    pub fn dim(&self) -> u32 {
        self.dim
    }

    pub fn tags(&self) -> &Arc<TagSet> {
        &self.tags
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    pub fn norm(&self) -> CentroidNorm {
        self.norm
    }

    pub fn config_hash(&self) -> u64 {
        self.hash
    }

    /// The centroid vector of `tag`.
    pub fn centroid_of(&self, tag: Tag) -> &[f32] {
        let d = self.dim as usize;
        let k = tag.0 as usize;
        &self.centroids[k * d..(k + 1) * d]
    }
}

/// Helper to attach a `Categories<L>` layer to a document with provenance.
pub fn add_categories<L: CategoryScope>(
    doc: &mut Document,
    key: &Key<Categories<L>>,
    categories: Categories<L>,
    by: &'static str,
) -> Result<(), Error> {
    doc.add(key, categories, Provenance::by(by))
}

mod codec;
mod examples;
mod math;
mod scoring;
mod wire;
pub use examples::category_sentence_centroids;
pub(crate) use math::{argmax_tag, fnv1a, sigmoid, softmax_in_place};
pub(crate) use wire::{Cursor, write_str};
#[cfg(test)]
mod tests;

#[cfg(test)]
mod boundary_tests;
