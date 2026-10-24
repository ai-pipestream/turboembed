//! Fixed-weight classifier heads and span-pooling head annotators over
//! turboembed vectors (`#321`).
//!
//! - [`ClassifierHead`] runs a deterministic 1-layer linear or 2-layer MLP
//!   classifier head in `f64` over `f32` embedding vectors, with softmax for
//!   single-label ([`HeadMode::SingleLabel`]) and independent sigmoid for
//!   multi-label ([`HeadMode::MultiLabel`]).
//! - Weights load from and save to a plain typed binary format (`THEAD001`)
//!   with no pickle and no JSON trickery.
//! - [`ScoringHead`] unifies [`ClassifierHead`] and [`LabelCentroids`](crate::centroids::LabelCentroids) so
//!   any vector or span-pooling annotator step can run either a fixed-weight
//!   head or a prototype centroid classifier.
//! - [`VectorHeadAnnotator`] scores an existing `Vectors<L>`
//!   (`opennlp:embeddings<L>`) layer into `Categories<L>`.
//! - [`SentenceSpanHead`], [`ChunkSpanHead`], [`OpinionSpanHead`], and
//!   [`AspectSpanHead`] pool exact subword piece ranges from `Subwords`
//!   (`opennlp:subwords`) through the stage 2 range write (`#263`,
//!   [`VectorAnnotator::embed_spans`](crate::vectors::VectorAnnotator::embed_spans)) so opinion and aspect heads pool only
//!   the opinion or aspect span rather than the full sentence.
//! - [`IncivilityHeadAnnotator`] and [`PragmaticsHeadAnnotator`] write the
//!   multi-label [`Incivility`](turbo_analysis_layers::Incivility) (`turbo:incivility`) and sentence-level
//!   [`Pragmatics`](turbo_analysis_layers::Pragmatics) (`turbo:pragmatics`) layers from `#319`.

mod annotators;
mod batch;
mod domain;
mod fit;
mod io;
#[cfg(test)]
mod tests;

use std::f64::consts::FRAC_1_SQRT_2;
use std::sync::Arc;

use turbo_analysis::{Error, Ix, Layer, ModelRef, Tag, TagSet};
use turbo_analysis_layers::{Categories, CategoryScope, Vectors};

pub use self::annotators::{
    AspectSpanHead, ChunkSpanHead, IncivilityHeadAnnotator, OpinionSpanHead, PragmaticsHeadAnnotator, ScoringHead,
    SentenceSpanHead, VectorHeadAnnotator,
};
use crate::centroids::{argmax_tag, fnv1a, sigmoid, softmax_in_place};

pub(super) const HEAD_MAGIC: &[u8; 8] = b"THEAD001";

/// Output scoring mode of a [`ClassifierHead`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum HeadMode {
    /// Single-label classification: logits are normalized with softmax so
    /// class probabilities sum to `1.0`.
    #[default]
    SingleLabel = 0,
    /// Multi-label classification: each logit is mapped independently through
    /// the logistic sigmoid into `0.0..=1.0`.
    MultiLabel = 1,
}

impl HeadMode {
    pub(super) fn from_byte(b: u8) -> Result<HeadMode, String> {
        match b {
            0 => Ok(HeadMode::SingleLabel),
            1 => Ok(HeadMode::MultiLabel),
            other => Err(format!("unrecognized head mode code {other}")),
        }
    }
}

/// Hidden-layer activation function for a 2-layer MLP [`ClassifierHead`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum HiddenActivation {
    /// No hidden activation (used for 1-layer linear heads).
    #[default]
    None = 0,
    /// Rectified linear unit: `max(0.0, x)`.
    Relu = 1,
    /// Gaussian error linear unit: `0.5 * x * (1.0 + erf(x / sqrt(2.0)))`.
    Gelu = 2,
    /// Hyperbolic tangent: `x.tanh()`.
    Tanh = 3,
}

impl HiddenActivation {
    pub(super) fn from_byte(b: u8) -> Result<HiddenActivation, String> {
        match b {
            0 => Ok(HiddenActivation::None),
            1 => Ok(HiddenActivation::Relu),
            2 => Ok(HiddenActivation::Gelu),
            3 => Ok(HiddenActivation::Tanh),
            other => Err(format!("unrecognized hidden activation code {other}")),
        }
    }

    /// Applies this activation function in `f64`.
    pub fn apply(self, x: f64) -> f64 {
        match self {
            HiddenActivation::None => x,
            HiddenActivation::Relu => x.max(0.0),
            HiddenActivation::Gelu => 0.5 * x * (1.0 + erf_f64(x * FRAC_1_SQRT_2)),
            HiddenActivation::Tanh => x.tanh(),
        }
    }
}

/// One dense affine transformation `y = W x + b` inside a [`ClassifierHead`].
#[derive(Clone, Debug, PartialEq)]
pub struct DenseLayer {
    pub in_dim: u32,
    pub out_dim: u32,
    /// Row-major `[out_dim, in_dim]` matrix in `f64`.
    pub weights: Vec<f64>,
    /// Bias vector `[out_dim]` in `f64`.
    pub bias: Vec<f64>,
}

impl DenseLayer {
    /// Constructs and validates a `DenseLayer`.
    pub fn new(in_dim: u32, out_dim: u32, weights: Vec<f64>, bias: Vec<f64>) -> Result<DenseLayer, String> {
        if in_dim == 0 || out_dim == 0 {
            return Err(format!("DenseLayer dimensions ({in_dim} -> {out_dim}) must be positive"));
        }
        let expected_w = (out_dim as usize)
            .checked_mul(in_dim as usize)
            .ok_or_else(|| "DenseLayer weight count overflows usize".to_string())?;
        if weights.len() != expected_w {
            return Err(format!(
                "DenseLayer weights has {} elements, expected {expected_w} ({out_dim} * {in_dim})",
                weights.len()
            ));
        }
        if bias.len() != out_dim as usize {
            return Err(format!("DenseLayer bias has {} elements, expected {out_dim}", bias.len()));
        }
        if let Some(k) = weights.iter().position(|w| !w.is_finite()) {
            return Err(format!("DenseLayer weight {k} is non-finite"));
        }
        if let Some(k) = bias.iter().position(|b| !b.is_finite()) {
            return Err(format!("DenseLayer bias {k} is non-finite"));
        }
        Ok(DenseLayer { in_dim, out_dim, weights, bias })
    }

    fn forward_f32(&self, x: &[f32], out: &mut [f64]) {
        let d_in = self.in_dim as usize;
        for ((slot, row), &b) in out.iter_mut().zip(self.weights.chunks_exact(d_in)).zip(&self.bias) {
            let mut sum = b;
            for (&w, &xi) in row.iter().zip(x) {
                sum += w * f64::from(xi);
            }
            *slot = sum;
        }
    }

    fn forward_f64(&self, x: &[f64], out: &mut [f64]) {
        let d_in = self.in_dim as usize;
        for ((slot, row), &b) in out.iter_mut().zip(self.weights.chunks_exact(d_in)).zip(&self.bias) {
            let mut sum = b;
            for (&w, &xi) in row.iter().zip(x) {
                sum += w * xi;
            }
            *slot = sum;
        }
    }
}

/// Fixed-weight linear or 2-layer MLP classifier head over turboembed vectors.
///
/// All accumulation, hidden activations, softmax, and sigmoid operations run
/// in deterministic `f64` arithmetic.
#[derive(Clone, Debug)]
pub struct ClassifierHead {
    pub(super) model: ModelRef,
    pub(super) tags: Arc<TagSet>,
    pub(super) mode: HeadMode,
    pub(super) hidden_activation: HiddenActivation,
    pub(super) layers: Vec<DenseLayer>,
    pub(super) hash: u64,
}

impl PartialEq for ClassifierHead {
    fn eq(&self, other: &Self) -> bool {
        self.model == other.model
            && self.tags.name == other.tags.name
            && self.tags.len() == other.tags.len()
            && (0..self.tags.len()).all(|i| self.tags.label(Tag(i as u16)) == other.tags.label(Tag(i as u16)))
            && self.mode == other.mode
            && self.hidden_activation == other.hidden_activation
            && self.layers == other.layers
            && self.hash == other.hash
    }
}

impl ClassifierHead {
    /// Builds a 1-layer linear classifier head `y = W x + b`.
    pub fn linear(
        model: ModelRef,
        in_dim: u32,
        tags: Arc<TagSet>,
        mode: HeadMode,
        weights: Vec<f64>,
        bias: Vec<f64>,
    ) -> Result<ClassifierHead, String> {
        if tags.is_empty() {
            return Err("ClassifierHead requires at least one label in its TagSet".into());
        }
        let layer = DenseLayer::new(in_dim, tags.len() as u32, weights, bias)?;
        let mut head = ClassifierHead {
            model,
            tags,
            mode,
            hidden_activation: HiddenActivation::None,
            layers: vec![layer],
            hash: 0,
        };
        head.hash = fnv1a(&head.to_bytes());
        Ok(head)
    }

    /// Builds a 2-layer MLP classifier head `y = W2 * sigma(W1 * x + b1) + b2`.
    #[allow(clippy::too_many_arguments)]
    pub fn mlp(
        model: ModelRef,
        in_dim: u32,
        hidden_dim: u32,
        tags: Arc<TagSet>,
        mode: HeadMode,
        hidden_activation: HiddenActivation,
        w1: Vec<f64>,
        b1: Vec<f64>,
        w2: Vec<f64>,
        b2: Vec<f64>,
    ) -> Result<ClassifierHead, String> {
        if tags.is_empty() {
            return Err("ClassifierHead requires at least one label in its TagSet".into());
        }
        if hidden_activation == HiddenActivation::None {
            return Err("MLP ClassifierHead requires a non-None hidden activation".into());
        }
        let l1 = DenseLayer::new(in_dim, hidden_dim, w1, b1)?;
        let l2 = DenseLayer::new(hidden_dim, tags.len() as u32, w2, b2)?;
        let mut head = ClassifierHead { model, tags, mode, hidden_activation, layers: vec![l1, l2], hash: 0 };
        head.hash = fnv1a(&head.to_bytes());
        Ok(head)
    }

    pub fn model(&self) -> &ModelRef {
        &self.model
    }

    pub fn in_dim(&self) -> u32 {
        self.layers[0].in_dim
    }

    pub fn out_dim(&self) -> u32 {
        self.tags.len() as u32
    }

    pub fn tags(&self) -> &Arc<TagSet> {
        &self.tags
    }

    pub fn mode(&self) -> HeadMode {
        self.mode
    }

    pub fn hidden_activation(&self) -> HiddenActivation {
        self.hidden_activation
    }

    pub fn layers(&self) -> &[DenseLayer] {
        &self.layers
    }

    pub fn config_hash(&self) -> u64 {
        self.hash
    }

    /// Computes the raw pre-activation output logits in `f64` for `vector`.
    pub fn logits(&self, vector: &[f32], out: &mut [f64]) -> Result<(), String> {
        let d_in = self.in_dim() as usize;
        if vector.len() != d_in {
            return Err(format!("input vector has length {}, expected {d_in}", vector.len()));
        }
        if out.len() != self.tags.len() {
            return Err(format!("output slice has length {}, expected {}", out.len(), self.tags.len()));
        }
        match self.layers.as_slice() {
            [l1] => {
                l1.forward_f32(vector, out);
            }
            [l1, l2] => {
                let mut hidden = vec![0.0f64; l1.out_dim as usize];
                l1.forward_f32(vector, &mut hidden);
                for h in &mut hidden {
                    *h = self.hidden_activation.apply(*h);
                }
                l2.forward_f64(&hidden, out);
            }
            _ => unreachable!("ClassifierHead always has 1 or 2 layers"),
        }
        Ok(())
    }

    /// Computes post-activation scores in `f64` (softmax probabilities for
    /// [`HeadMode::SingleLabel`] or independent sigmoid scores in `0.0..=1.0`
    /// for [`HeadMode::MultiLabel`]) into `scores`, returning the highest
    /// `(Tag, score)`.
    pub fn score(&self, vector: &[f32], scores: &mut [f64]) -> Result<(Tag, f64), String> {
        self.logits(vector, scores)?;
        match self.mode {
            HeadMode::SingleLabel => softmax_in_place(scores),
            HeadMode::MultiLabel => {
                for s in scores.iter_mut() {
                    *s = sigmoid(*s);
                }
            }
        }
        Ok(argmax_tag(scores))
    }

    /// Scores every item in `vectors` (`opennlp:embeddings<L>`) into
    /// [`Categories<L>`] (`turbo:categories<L>`).
    pub fn categorize_vectors<L: CategoryScope>(
        &self,
        annotator: &'static str,
        vectors: &Vectors<L>,
    ) -> Result<Categories<L>, Error> {
        let fail = |detail: String| Error::Annotator { annotator, detail };
        if vectors.model != self.model {
            return Err(fail(format!(
                "vectors are from model {} (revision {}), head is for {} (revision {})",
                vectors.model.id, vectors.model.revision, self.model.id, self.model.revision
            )));
        }
        if vectors.dim != self.in_dim() {
            return Err(fail(format!("vectors have dim {}, head expects dim {}", vectors.dim, self.in_dim())));
        }
        let n = vectors.len();
        let n_labels = self.tags.len();
        let empty_tag = self.tags.tag("neutral").unwrap_or(Tag(0));
        let empty_score = match self.mode {
            HeadMode::SingleLabel => 1.0 / n_labels as f64,
            HeadMode::MultiLabel => 0.0,
        };
        let empty_probs = vec![empty_score; n_labels];
        let mut out = Categories::with_capacity(Arc::clone(&self.tags), n);
        let mut buf = vec![0.0f64; n_labels];
        for i in 0..n {
            let ix = Ix::<L>::new(i as u32);
            if vectors.is_empty(ix) {
                out.push(empty_tag, &empty_probs);
            } else {
                let (tag, _) = self.score(vectors.row(ix), &mut buf).map_err(fail)?;
                out.push(tag, &buf);
            }
        }
        Ok(out)
    }
}

/// Rational approximation to `erf(x)` in `f64` (Abramowitz and Stegun
/// 7.1.26, maximum error `1.5e-7`). It leans on the platform's `exp`, as the
/// `Tanh` activation leans on its `tanh`, so bit-for-bit results across
/// platforms are not promised; the CPU and CUDA parity test allows `1e-5`.
fn erf_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-ax * ax).exp())
}
