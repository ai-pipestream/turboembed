//! Fixed-weight classifier heads and span-pooling head annotators over
//! turboembed vectors (`#321`).
//!
//! - [`ClassifierHead`] runs a deterministic 1-layer linear or 2-layer MLP
//!   classifier head in `f64` over `f32` embedding vectors, with softmax for
//!   single-label ([`HeadMode::SingleLabel`]) and independent sigmoid for
//!   multi-label ([`HeadMode::MultiLabel`]).
//! - Weights load from and save to a plain typed binary format (`THEAD001`)
//!   with no pickle and no JSON trickery.
//! - [`ScoringHead`] unifies [`ClassifierHead`] and [`LabelCentroids`] so
//!   any vector or span-pooling annotator step can run either a fixed-weight
//!   head or a prototype centroid classifier.
//! - [`VectorHeadAnnotator`] scores an existing `Vectors<L>`
//!   (`opennlp:embeddings<L>`) layer into `Categories<L>`.
//! - [`SentenceSpanHead`], [`ChunkSpanHead`], [`OpinionSpanHead`], and
//!   [`AspectSpanHead`] pool exact subword piece ranges from `Subwords`
//!   (`opennlp:subwords`) through the stage 2 range write (`#263`,
//!   [`VectorAnnotator::embed_spans`]) so opinion and aspect heads pool only
//!   the opinion or aspect span rather than the full sentence.
//! - [`IncivilityHeadAnnotator`] and [`PragmaticsHeadAnnotator`] write the
//!   multi-label [`Incivility`] (`turbo:incivility`) and sentence-level
//!   [`Pragmatics`] (`turbo:pragmatics`) layers from `#319`.

use std::f64::consts::FRAC_1_SQRT_2;
use std::fs;
use std::io::{self, Read, Write};
use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Execution, Ix, Layer, ModelRef, Refs, Span, Tag, TagSet};
use turbo_analysis_layers::{
    Aspect, Aspects, Categories, CategoryScope, Chunks, Incivility, IncivilityItem, IncivilityLabel, IncivilityScores,
    IncivilityTarget, Opinions, PolarityScores, Pragmatics, PragmaticsItem, Sentences, Subwords, Tokens, Vectors,
};

use crate::centroids::{Cursor, LabelCentroids, argmax_tag, fnv1a, sigmoid, softmax_in_place, write_str};
use crate::vectors::VectorAnnotator;
use crate::window::Which;

const HEAD_MAGIC: &[u8; 8] = b"THEAD001";

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
    fn from_byte(b: u8) -> Result<HeadMode, String> {
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
    fn from_byte(b: u8) -> Result<HiddenActivation, String> {
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
    model: ModelRef,
    tags: Arc<TagSet>,
    mode: HeadMode,
    hidden_activation: HiddenActivation,
    layers: Vec<DenseLayer>,
    hash: u64,
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

    /// Fits a deterministic 1-layer single-label linear head over labeled
    /// `f32` embedding vectors by minimizing L2-regularized multinomial
    /// cross-entropy loss in `f64` via full-batch gradient descent.
    pub fn fit_single_label<V: AsRef<[f32]>>(
        model: ModelRef,
        in_dim: u32,
        tags: Arc<TagSet>,
        examples: &[(V, Tag)],
        l2_reg: f64,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<ClassifierHead, String> {
        if examples.is_empty() {
            return Err("fit_single_label requires at least one training example".into());
        }
        if in_dim == 0 {
            return Err("in_dim must be at least 1".into());
        }
        if tags.is_empty() {
            return Err("tags must not be empty".into());
        }
        if !l2_reg.is_finite() || l2_reg < 0.0 {
            return Err(format!("l2_reg {l2_reg} must be finite and non-negative"));
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(format!("learning_rate {learning_rate} must be finite and positive"));
        }
        let d = in_dim as usize;
        let k = tags.len();
        for (idx, (vec, tag)) in examples.iter().enumerate() {
            if vec.as_ref().len() != d {
                return Err(format!("example {idx} vector has length {}, expected {d}", vec.as_ref().len()));
            }
            if tag.0 as usize >= k {
                return Err(format!("example {idx} tag {} is outside TagSet of size {k}", tag.0));
            }
        }
        let mut weights = vec![0.0f64; k * d];
        let mut bias = vec![0.0f64; k];
        let mut grad_w = vec![0.0f64; k * d];
        let mut grad_b = vec![0.0f64; k];
        let mut probs = vec![0.0f64; k];
        let inv_n = 1.0 / examples.len() as f64;

        for _ in 0..epochs {
            grad_w.fill(0.0);
            grad_b.fill(0.0);
            for (vec, gold) in examples {
                let x = vec.as_ref();
                for (c, (row, &b)) in weights.chunks_exact(d).zip(&bias).enumerate() {
                    let mut z = b;
                    for (&w, &xi) in row.iter().zip(x) {
                        z += w * f64::from(xi);
                    }
                    probs[c] = z;
                }
                softmax_in_place(&mut probs);
                probs[gold.0 as usize] -= 1.0;
                for (c, &err) in probs.iter().enumerate() {
                    grad_b[c] += err;
                    let gw_row = &mut grad_w[c * d..(c + 1) * d];
                    for (gw, &xi) in gw_row.iter_mut().zip(x) {
                        *gw += err * f64::from(xi);
                    }
                }
            }
            for (w, &gw) in weights.iter_mut().zip(&grad_w) {
                let g = gw * inv_n + l2_reg * *w;
                *w -= learning_rate * g;
            }
            for (b, &gb) in bias.iter_mut().zip(&grad_b) {
                *b -= learning_rate * (gb * inv_n);
            }
        }
        Self::linear(model, in_dim, tags, HeadMode::SingleLabel, weights, bias)
    }

    /// Fits a deterministic 1-layer multi-label linear head over labeled
    /// `f32` embedding vectors by minimizing L2-regularized binary
    /// cross-entropy loss per label in `f64` via full-batch gradient descent.
    pub fn fit_multi_label<V: AsRef<[f32]>, B: AsRef<[bool]>>(
        model: ModelRef,
        in_dim: u32,
        tags: Arc<TagSet>,
        examples: &[(V, B)],
        l2_reg: f64,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<ClassifierHead, String> {
        if examples.is_empty() {
            return Err("fit_multi_label requires at least one training example".into());
        }
        if in_dim == 0 {
            return Err("in_dim must be at least 1".into());
        }
        if tags.is_empty() {
            return Err("tags must not be empty".into());
        }
        if !l2_reg.is_finite() || l2_reg < 0.0 {
            return Err(format!("l2_reg {l2_reg} must be finite and non-negative"));
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(format!("learning_rate {learning_rate} must be finite and positive"));
        }
        let d = in_dim as usize;
        let k = tags.len();
        for (idx, (vec, y)) in examples.iter().enumerate() {
            if vec.as_ref().len() != d {
                return Err(format!("example {idx} vector has length {}, expected {d}", vec.as_ref().len()));
            }
            if y.as_ref().len() != k {
                return Err(format!("example {idx} target has {} labels, expected {k}", y.as_ref().len()));
            }
        }
        let mut weights = vec![0.0f64; k * d];
        let mut bias = vec![0.0f64; k];
        let mut grad_w = vec![0.0f64; k * d];
        let mut grad_b = vec![0.0f64; k];
        let inv_n = 1.0 / examples.len() as f64;

        for _ in 0..epochs {
            grad_w.fill(0.0);
            grad_b.fill(0.0);
            for (vec, y) in examples {
                let x = vec.as_ref();
                let targets = y.as_ref();
                for (c, ((row, &b), &target)) in weights.chunks_exact(d).zip(&bias).zip(targets).enumerate() {
                    let mut z = b;
                    for (&w, &xi) in row.iter().zip(x) {
                        z += w * f64::from(xi);
                    }
                    let pred = sigmoid(z);
                    let err = pred - if target { 1.0 } else { 0.0 };
                    grad_b[c] += err;
                    let gw_row = &mut grad_w[c * d..(c + 1) * d];
                    for (gw, &xi) in gw_row.iter_mut().zip(x) {
                        *gw += err * f64::from(xi);
                    }
                }
            }
            for (w, &gw) in weights.iter_mut().zip(&grad_w) {
                let g = gw * inv_n + l2_reg * *w;
                *w -= learning_rate * g;
            }
            for (b, &gb) in bias.iter_mut().zip(&grad_b) {
                *b -= learning_rate * (gb * inv_n);
            }
        }
        Self::linear(model, in_dim, tags, HeadMode::MultiLabel, weights, bias)
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

    /// Maps this head's labels to [`IncivilityLabel`]s when configured as a
    /// [`HeadMode::MultiLabel`] incivility head.
    fn incivility_label_map(&self, annotator: &'static str) -> Result<Vec<IncivilityLabel>, Error> {
        let fail = |detail: String| Error::Annotator { annotator, detail };
        if self.mode != HeadMode::MultiLabel {
            return Err(fail("incivility head must have HeadMode::MultiLabel".into()));
        }
        let mut map = Vec::with_capacity(self.tags.len());
        for i in 0..self.tags.len() {
            let label = self.tags.label(Tag(i as u16));
            let parsed = IncivilityLabel::from_name(label)
                .ok_or_else(|| fail(format!("head label {label:?} is not a valid IncivilityLabel")))?;
            map.push(parsed);
        }
        Ok(map)
    }

    /// Scores `vectors` (`Vectors<Sentences>`) with a multi-label incivility
    /// head and writes [`Incivility`] (`turbo:incivility`).
    pub fn score_incivility_sentences(
        &self,
        annotator: &'static str,
        tokens: &Tokens,
        sentences: &Sentences,
        vectors: &Vectors<Sentences>,
    ) -> Result<Incivility, Error> {
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
        if vectors.len() != sentences.len() {
            return Err(fail(format!(
                "sentence vectors has {} items, sentences has {}",
                vectors.len(),
                sentences.len()
            )));
        }
        let label_map = self.incivility_label_map(annotator)?;
        let mut out = Incivility::new();
        let mut buf = vec![0.0f64; self.tags.len()];
        for i in 0..sentences.len() {
            let ix = Ix::<Sentences>::new(i as u32);
            let mut item_scores = IncivilityScores::default();
            if !vectors.is_empty(ix) {
                self.score(vectors.row(ix), &mut buf).map_err(fail)?;
                for (&lbl, &val) in label_map.iter().zip(&buf) {
                    item_scores.set(lbl, val);
                }
            }
            out.try_push(
                tokens,
                Some(sentences),
                None,
                None,
                IncivilityItem { target: IncivilityTarget::Sentence(ix), scores: item_scores, evidence: &[] },
            )
            .map_err(fail)?;
        }
        Ok(out)
    }

    /// Scores `vectors` (`Vectors<Chunks>`) with a multi-label incivility head
    /// and writes [`Incivility`] (`turbo:incivility`).
    pub fn score_incivility_chunks(
        &self,
        annotator: &'static str,
        tokens: &Tokens,
        chunks: &Chunks,
        vectors: &Vectors<Chunks>,
    ) -> Result<Incivility, Error> {
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
        if vectors.len() != chunks.len() {
            return Err(fail(format!("chunk vectors has {} items, chunks has {}", vectors.len(), chunks.len())));
        }
        let label_map = self.incivility_label_map(annotator)?;
        let mut out = Incivility::new();
        let mut buf = vec![0.0f64; self.tags.len()];
        for i in 0..chunks.len() {
            let ix = Ix::<Chunks>::new(i as u32);
            let mut item_scores = IncivilityScores::default();
            if !vectors.is_empty(ix) {
                self.score(vectors.row(ix), &mut buf).map_err(fail)?;
                for (&lbl, &val) in label_map.iter().zip(&buf) {
                    item_scores.set(lbl, val);
                }
            }
            out.try_push(
                tokens,
                None,
                Some(chunks),
                None,
                IncivilityItem { target: IncivilityTarget::Chunk(ix), scores: item_scores, evidence: &[] },
            )
            .map_err(fail)?;
        }
        Ok(out)
    }

    /// Scores `vectors` (`Vectors<Sentences>`) with a sarcasm/irony head and
    /// writes [`Pragmatics`] (`turbo:pragmatics`) with `evidence: None`,
    /// recording that the score came from the embedding head alone (`#324`).
    pub fn score_pragmatics_sentences(
        &self,
        annotator: &'static str,
        sentences: &Sentences,
        tokens: &Tokens,
        vectors: &Vectors<Sentences>,
    ) -> Result<Pragmatics, Error> {
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
        if vectors.len() != sentences.len() {
            return Err(fail(format!(
                "sentence vectors has {} items, sentences has {}",
                vectors.len(),
                sentences.len()
            )));
        }
        let pos_idx = self
            .tags
            .tag("sarcastic")
            .or_else(|| self.tags.tag("ironic"))
            .or_else(|| self.tags.tag("sarcasm"))
            .unwrap_or(Tag(0))
            .0 as usize;
        let mut out = Pragmatics::new();
        let mut buf = vec![0.0f64; self.tags.len()];
        for i in 0..sentences.len() {
            let ix = Ix::<Sentences>::new(i as u32);
            let score = if vectors.is_empty(ix) {
                0.0
            } else {
                self.score(vectors.row(ix), &mut buf).map_err(fail)?;
                buf[pos_idx].clamp(0.0, 1.0)
            };
            out.try_push(sentences, tokens, PragmaticsItem { score, evidence: None }).map_err(fail)?;
        }
        Ok(out)
    }

    /// Serializes this head to the plain typed `THEAD001` binary format.
    pub fn write_to<W: Write>(&self, mut w: W) -> io::Result<()> {
        w.write_all(HEAD_MAGIC)?;
        w.write_all(&[self.mode as u8, self.hidden_activation as u8, self.layers.len() as u8])?;
        write_str(&mut w, &self.model.id)?;
        write_str(&mut w, &self.model.revision)?;
        write_str(&mut w, &self.model.sha256)?;
        write_str(&mut w, &self.tags.name)?;
        w.write_all(&(self.tags.len() as u32).to_le_bytes())?;
        for i in 0..self.tags.len() {
            write_str(&mut w, self.tags.label(Tag(i as u16)))?;
        }
        for layer in &self.layers {
            w.write_all(&layer.in_dim.to_le_bytes())?;
            w.write_all(&layer.out_dim.to_le_bytes())?;
            for &weight in &layer.weights {
                w.write_all(&weight.to_le_bytes())?;
            }
            for &b in &layer.bias {
                w.write_all(&b.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Serializes this head into an in-memory byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.write_to(&mut buf).expect("writing to Vec<u8> cannot fail");
        buf
    }

    /// Writes this head to `path` in the `THEAD001` binary format.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        fs::write(path, self.to_bytes())
    }

    /// Loads a `ClassifierHead` from a byte slice in the `THEAD001` binary
    /// format.
    pub fn from_bytes(bytes: &[u8]) -> Result<ClassifierHead, String> {
        let mut cur = Cursor::new(bytes);
        let magic = cur.read_array::<8>()?;
        if &magic != HEAD_MAGIC {
            return Err("invalid ClassifierHead header (expected THEAD001)".into());
        }
        let mode = HeadMode::from_byte(cur.read_u8()?)?;
        let hidden_activation = HiddenActivation::from_byte(cur.read_u8()?)?;
        let n_layers = cur.read_u8()? as usize;
        if n_layers == 0 || n_layers > 2 {
            return Err(format!("THEAD001 supports 1 or 2 layers, got {n_layers}"));
        }
        let model = ModelRef { id: cur.read_string()?, revision: cur.read_string()?, sha256: cur.read_string()? };
        let tag_set_name = cur.read_string()?;
        let n_labels = cur.read_u32()? as usize;
        if n_labels == 0 {
            return Err("THEAD001 file has 0 labels".into());
        }
        let mut labels = Vec::with_capacity(n_labels);
        for _ in 0..n_labels {
            labels.push(cur.read_string()?);
        }
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let tags = TagSet::new(&tag_set_name, &label_refs);
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let in_dim = cur.read_u32()?;
            let out_dim = cur.read_u32()?;
            let w_count = (out_dim as usize)
                .checked_mul(in_dim as usize)
                .ok_or_else(|| "THEAD001 layer dimensions overflow usize".to_string())?;
            let mut weights = Vec::with_capacity(w_count);
            for _ in 0..w_count {
                weights.push(cur.read_f64()?);
            }
            let mut bias = Vec::with_capacity(out_dim as usize);
            for _ in 0..out_dim as usize {
                bias.push(cur.read_f64()?);
            }
            layers.push(DenseLayer::new(in_dim, out_dim, weights, bias)?);
        }
        cur.finish()?;
        match layers.len() {
            1 => {
                if hidden_activation != HiddenActivation::None {
                    return Err("1-layer THEAD001 head must have HiddenActivation::None".into());
                }
                let l = layers.remove(0);
                if l.out_dim as usize != tags.len() {
                    return Err(format!("THEAD001 output dim {} does not match {} labels", l.out_dim, tags.len()));
                }
                Self::linear(model, l.in_dim, tags, mode, l.weights, l.bias)
            }
            2 => {
                let l2 = layers.pop().unwrap();
                let l1 = layers.pop().unwrap();
                if l1.out_dim != l2.in_dim {
                    return Err(format!(
                        "THEAD001 layer 1 out_dim {} does not match layer 2 in_dim {}",
                        l1.out_dim, l2.in_dim
                    ));
                }
                if l2.out_dim as usize != tags.len() {
                    return Err(format!("THEAD001 output dim {} does not match {} labels", l2.out_dim, tags.len()));
                }
                Self::mlp(
                    model,
                    l1.in_dim,
                    l1.out_dim,
                    tags,
                    mode,
                    hidden_activation,
                    l1.weights,
                    l1.bias,
                    l2.weights,
                    l2.bias,
                )
            }
            _ => unreachable!(),
        }
    }

    /// Reads a `ClassifierHead` from any reader in the `THEAD001` format.
    pub fn read_from<R: Read>(mut r: R) -> Result<ClassifierHead, String> {
        let mut buf = Vec::new();
        r.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        Self::from_bytes(&buf)
    }

    /// Loads a `ClassifierHead` from `path`.
    pub fn load(path: &Path) -> Result<ClassifierHead, String> {
        let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(&bytes)
    }
}

/// Unified scoring head wrapping either a fixed-weight [`ClassifierHead`] or
/// a prototype [`LabelCentroids`] classifier.
#[derive(Clone, Debug)]
pub enum ScoringHead {
    Classifier(Arc<ClassifierHead>),
    Centroids(Arc<LabelCentroids>),
}

impl From<Arc<ClassifierHead>> for ScoringHead {
    fn from(h: Arc<ClassifierHead>) -> Self {
        ScoringHead::Classifier(h)
    }
}

impl From<ClassifierHead> for ScoringHead {
    fn from(h: ClassifierHead) -> Self {
        ScoringHead::Classifier(Arc::new(h))
    }
}

impl From<Arc<LabelCentroids>> for ScoringHead {
    fn from(c: Arc<LabelCentroids>) -> Self {
        ScoringHead::Centroids(c)
    }
}

impl From<LabelCentroids> for ScoringHead {
    fn from(c: LabelCentroids) -> Self {
        ScoringHead::Centroids(Arc::new(c))
    }
}

impl ScoringHead {
    pub fn model(&self) -> &ModelRef {
        match self {
            ScoringHead::Classifier(h) => h.model(),
            ScoringHead::Centroids(c) => c.model(),
        }
    }

    pub fn in_dim(&self) -> u32 {
        match self {
            ScoringHead::Classifier(h) => h.in_dim(),
            ScoringHead::Centroids(c) => c.dim(),
        }
    }

    pub fn tags(&self) -> &Arc<TagSet> {
        match self {
            ScoringHead::Classifier(h) => h.tags(),
            ScoringHead::Centroids(c) => c.tags(),
        }
    }

    pub fn config_hash(&self) -> u64 {
        match self {
            ScoringHead::Classifier(h) => h.config_hash(),
            ScoringHead::Centroids(c) => c.config_hash() ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    /// Scores `vector` into `scores` (`f64` probabilities or multi-label
    /// scores) and returns the top `(Tag, score)`.
    pub fn score(&self, vector: &[f32], scores: &mut [f64]) -> Result<(Tag, f64), String> {
        match self {
            ScoringHead::Classifier(h) => h.score(vector, scores),
            ScoringHead::Centroids(c) => c.score_single_label(vector, scores),
        }
    }

    /// Scores every item of `vectors` (`opennlp:embeddings<L>`) into
    /// [`Categories<L>`] (`turbo:categories<L>`).
    pub fn categorize_vectors<L: CategoryScope>(
        &self,
        annotator: &'static str,
        vectors: &Vectors<L>,
    ) -> Result<Categories<L>, Error> {
        match self {
            ScoringHead::Classifier(h) => h.categorize_vectors(annotator, vectors),
            ScoringHead::Centroids(c) => c.categorize_vectors(annotator, vectors),
        }
    }

    /// Converts a probability distribution `probs` over this head's `TagSet`
    /// into three-way [`PolarityScores`] (`positive`, `negative`, `neutral`).
    pub fn polarity_scores_from_probs(&self, probs: &[f64]) -> PolarityScores {
        let mut pos = 0.0f64;
        let mut neg = 0.0f64;
        let mut neu = 0.0f64;
        let tags = self.tags();
        for (i, &p) in probs.iter().enumerate() {
            let label = tags.label(Tag(i as u16));
            if label.eq_ignore_ascii_case("positive") || label.eq_ignore_ascii_case("pos") {
                pos += p;
            } else if label.eq_ignore_ascii_case("negative") || label.eq_ignore_ascii_case("neg") {
                neg += p;
            } else if label.eq_ignore_ascii_case("neutral") || label.eq_ignore_ascii_case("neu") {
                neu += p;
            }
        }
        PolarityScores::new(pos, neg, neu)
    }
}

/// Annotator step that reads `Vectors<L>` (`opennlp:embeddings<L>`) and
/// writes `Categories<L>` (`turbo:categories<L>`) using a [`ScoringHead`].
#[derive(Clone)]
pub struct VectorHeadAnnotator<L: CategoryScope> {
    pub head: ScoringHead,
    _marker: PhantomData<fn() -> L>,
}

impl<L: CategoryScope> VectorHeadAnnotator<L> {
    pub fn new(head: impl Into<ScoringHead>) -> Self {
        VectorHeadAnnotator { head: head.into(), _marker: PhantomData }
    }
}

impl<L: CategoryScope> Annotator for VectorHeadAnnotator<L> {
    const ID: &'static str = "turbo:vector-head";
    type Reads = (L, Vectors<L>);
    type Writes = (Categories<L>,);

    fn annotate(&self, (_scope, vectors): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        Ok((self.head.categorize_vectors(Self::ID, vectors)?,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.head.model().clone())
    }

    fn config_hash(&self) -> u64 {
        self.head.config_hash()
    }
}

/// Span-pooling head annotator over `Sentences`: pools each sentence's
/// subword range from `Subwords` via [`VectorAnnotator::embed_spans`] (`#263`)
/// and writes `Categories<Sentences>`.
#[derive(Clone)]
pub struct SentenceSpanHead {
    pub vectors: Arc<VectorAnnotator>,
    pub head: ScoringHead,
}

impl SentenceSpanHead {
    pub fn new(vectors: Arc<VectorAnnotator>, head: impl Into<ScoringHead>) -> Self {
        SentenceSpanHead { vectors, head: head.into() }
    }
}

impl Annotator for SentenceSpanHead {
    const ID: &'static str = "turbo:sentence-span-head";
    type Reads = (Subwords, Sentences);
    type Writes = (Categories<Sentences>,);

    fn annotate(&self, (subwords, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let spans: Vec<Span> = (0..sentences.len()).map(|i| sentences.spans.get(Ix::new(i as u32))).collect();
        let pooled: Vectors<Sentences> =
            self.vectors.embed_spans(Self::ID, subwords, Some(&sentences.spans), sentences.spans.view(), &spans)?;
        Ok((self.head.categorize_vectors(Self::ID, &pooled)?,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.vectors.model().clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.vectors.execution().clone())
    }

    fn config_hash(&self) -> u64 {
        self.vectors.config(Which::Sentences) ^ self.head.config_hash()
    }
}

/// Span-pooling head annotator over `Chunks`: pools each chunk's subword
/// range from `Subwords` via [`VectorAnnotator::embed_spans`] (`#263`) and
/// writes `Categories<Chunks>`.
#[derive(Clone)]
pub struct ChunkSpanHead {
    pub vectors: Arc<VectorAnnotator>,
    pub head: ScoringHead,
}

impl ChunkSpanHead {
    pub fn new(vectors: Arc<VectorAnnotator>, head: impl Into<ScoringHead>) -> Self {
        ChunkSpanHead { vectors, head: head.into() }
    }
}

impl Annotator for ChunkSpanHead {
    const ID: &'static str = "turbo:chunk-span-head";
    type Reads = (Subwords, Option<Sentences>, Chunks);
    type Writes = (Categories<Chunks>,);

    fn annotate(&self, (subwords, sentences, chunks): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let spans: Vec<Span> = (0..chunks.len()).map(|i| chunks.spans.get(Ix::new(i as u32))).collect();
        let pooled: Vectors<Chunks> =
            self.vectors.embed_spans(Self::ID, subwords, sentences.map(|s| &s.spans), chunks.spans.view(), &spans)?;
        Ok((self.head.categorize_vectors(Self::ID, &pooled)?,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.vectors.model().clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.vectors.execution().clone())
    }

    fn config_hash(&self) -> u64 {
        self.vectors.config(Which::Chunks) ^ self.head.config_hash()
    }
}

/// Span-pooling head annotator over `Opinions` (`turbo:opinions`): pools the
/// exact subword range of each opinion expression (`opinions.span(i, tokens)`),
/// never the whole sentence, from the sentence windows via `#263`
/// ([`VectorAnnotator::embed_spans`]), and writes `Categories<Opinions>`.
#[derive(Clone)]
pub struct OpinionSpanHead {
    pub vectors: Arc<VectorAnnotator>,
    pub head: ScoringHead,
}

impl OpinionSpanHead {
    pub fn new(vectors: Arc<VectorAnnotator>, head: impl Into<ScoringHead>) -> Self {
        OpinionSpanHead { vectors, head: head.into() }
    }

    /// Pools the exact subword range of every opinion span in `opinions`
    /// within the sentence windows (when `sentences` is `Some`).
    pub fn pool_opinions(
        &self,
        subwords: &Subwords,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        opinions: &Opinions,
    ) -> Result<Vectors<Opinions>, Error> {
        let spans: Vec<Span> = (0..opinions.len()).map(|i| opinions.span(Ix::new(i as u32), tokens)).collect();
        self.vectors.embed_spans(Self::ID, subwords, sentences.map(|s| &s.spans), tokens.spans.view(), &spans)
    }
}

impl Annotator for OpinionSpanHead {
    const ID: &'static str = "turbo:opinion-span-head";
    type Reads = (Subwords, Tokens, Option<Sentences>, Opinions);
    type Writes = (Categories<Opinions>,);

    fn annotate(&self, (subwords, tokens, sentences, opinions): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let pooled = self.pool_opinions(subwords, tokens, sentences, opinions)?;
        Ok((self.head.categorize_vectors(Self::ID, &pooled)?,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.vectors.model().clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.vectors.execution().clone())
    }

    fn config_hash(&self) -> u64 {
        self.vectors.config(Which::Sentences) ^ self.head.config_hash()
    }
}

/// Span-pooling head annotator over `Aspects` (`turbo:aspects`): pools the
/// exact subword range of each aspect term (`aspects.span(i, tokens, sentences)`,
/// which is the term's token range for explicit aspects or the sentence span
/// for implicit aspects) from the sentence windows via `#263`
/// ([`VectorAnnotator::embed_spans`]), and writes `Categories<Aspects>`.
#[derive(Clone)]
pub struct AspectSpanHead {
    pub vectors: Arc<VectorAnnotator>,
    pub head: ScoringHead,
}

impl AspectSpanHead {
    pub fn new(vectors: Arc<VectorAnnotator>, head: impl Into<ScoringHead>) -> Self {
        AspectSpanHead { vectors, head: head.into() }
    }

    /// Pools the exact subword range of every aspect in `aspects` within the
    /// sentence windows.
    pub fn pool_aspects(
        &self,
        subwords: &Subwords,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        aspects: &Aspects,
    ) -> Result<Vectors<Aspects>, Error> {
        let mut spans = Vec::with_capacity(aspects.len());
        for i in 0..aspects.len() {
            let ix = Ix::<Aspects>::new(i as u32);
            let span = aspects.span(ix, tokens, sentences).ok_or_else(|| Error::Annotator {
                annotator: Self::ID,
                detail: format!("implicit aspect {i} requires a Sentences layer to resolve its span"),
            })?;
            spans.push(span);
        }
        self.vectors.embed_spans(Self::ID, subwords, sentences.map(|s| &s.spans), tokens.spans.view(), &spans)
    }

    /// Pools each aspect's exact subword range and populates its three-way
    /// [`PolarityScores`] (`positive`, `negative`, `neutral`) from `self.head`.
    pub fn score_aspects_polarity(
        &self,
        subwords: &Subwords,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        opinions: Option<&Opinions>,
        aspects: &Aspects,
    ) -> Result<Aspects, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        let pooled = self.pool_aspects(subwords, tokens, sentences, aspects)?;
        let n_labels = self.head.tags().len();
        let mut probs = vec![0.0f64; n_labels];
        let mut out = Aspects::on(
            aspects.tokens_key,
            aspects.sentences_key,
            aspects.opinions_key,
            Arc::clone(&aspects.categories),
        );
        for i in 0..aspects.len() {
            let ix = Ix::<Aspects>::new(i as u32);
            let cur = aspects.get(ix);
            let polarity = if pooled.is_empty(ix) {
                PolarityScores::new(0.0, 0.0, 1.0)
            } else {
                self.head.score(pooled.row(ix), &mut probs).map_err(fail)?;
                self.head.polarity_scores_from_probs(&probs)
            };
            out.try_push(
                tokens,
                sentences,
                opinions,
                Aspect { target: cur.target, category: cur.category, polarity, opinions: cur.opinions },
            )
            .map_err(fail)?;
        }
        Ok(out)
    }
}

impl Annotator for AspectSpanHead {
    const ID: &'static str = "turbo:aspect-span-head";
    type Reads = (Subwords, Tokens, Option<Sentences>, Aspects);
    type Writes = (Categories<Aspects>,);

    fn annotate(&self, (subwords, tokens, sentences, aspects): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let pooled = self.pool_aspects(subwords, tokens, sentences, aspects)?;
        Ok((self.head.categorize_vectors(Self::ID, &pooled)?,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.vectors.model().clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.vectors.execution().clone())
    }

    fn config_hash(&self) -> u64 {
        self.vectors.config(Which::Sentences) ^ self.head.config_hash()
    }
}

/// Multi-label incivility annotator (`turbo:incivility-head`) reading
/// `(Tokens, Sentences, Vectors<Sentences>)` and writing [`Incivility`]
/// (`turbo:incivility`).
#[derive(Clone)]
pub struct IncivilityHeadAnnotator(pub Arc<ClassifierHead>);

impl Annotator for IncivilityHeadAnnotator {
    const ID: &'static str = "turbo:incivility-head";
    type Reads = (Tokens, Sentences, Vectors<Sentences>);
    type Writes = (Incivility,);

    fn annotate(&self, (tokens, sentences, vectors): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        Ok((self.0.score_incivility_sentences(Self::ID, tokens, sentences, vectors)?,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.0.model().clone())
    }

    fn config_hash(&self) -> u64 {
        self.0.config_hash()
    }
}

/// Sentence-level sarcasm and irony head annotator (`turbo:pragmatics-head`)
/// reading `(Tokens, Sentences, Vectors<Sentences>)` and writing
/// [`Pragmatics`] (`turbo:pragmatics`) with `evidence: None` (`#324`).
#[derive(Clone)]
pub struct PragmaticsHeadAnnotator(pub Arc<ClassifierHead>);

impl Annotator for PragmaticsHeadAnnotator {
    const ID: &'static str = "turbo:pragmatics-head";
    type Reads = (Tokens, Sentences, Vectors<Sentences>);
    type Writes = (Pragmatics,);

    fn annotate(&self, (tokens, sentences, vectors): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        Ok((self.0.score_pragmatics_sentences(Self::ID, sentences, tokens, vectors)?,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.0.model().clone())
    }

    fn config_hash(&self) -> u64 {
        self.0.config_hash()
    }
}

/// High-accuracy rational approximation to `erf(x)` in `f64` (Abramowitz and
/// Stegun 7.1.26, maximum error `1.5e-7`, deterministic across platforms).
fn erf_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-ax * ax).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model() -> ModelRef {
        ModelRef { id: "tiny-bert".into(), revision: "rev1".into(), sha256: "0123456789abcdef".into() }
    }

    #[test]
    fn linear_and_mlp_heads_round_trip_and_score_deterministically() {
        let tags = TagSet::new("sentiment", &["positive", "negative", "neutral"]);
        let linear = ClassifierHead::linear(
            test_model(),
            3,
            Arc::clone(&tags),
            HeadMode::SingleLabel,
            vec![
                2.0, -1.0, 0.0, // positive
                -1.0, 2.0, 0.0, // negative
                0.0, 0.0, 1.5, // neutral
            ],
            vec![0.1, -0.1, 0.0],
        )
        .unwrap();

        let bytes = linear.to_bytes();
        let loaded = ClassifierHead::from_bytes(&bytes).unwrap();
        assert_eq!(linear, loaded);

        let mut probs1 = [0.0f64; 3];
        let mut probs2 = [0.0f64; 3];
        let (t1, p1) = linear.score(&[1.0, -0.5, 0.0], &mut probs1).unwrap();
        let (t2, p2) = loaded.score(&[1.0, -0.5, 0.0], &mut probs2).unwrap();
        assert_eq!(t1, Tag(0));
        assert_eq!(t1, t2);
        assert_eq!(p1.to_bits(), p2.to_bits());
        assert_eq!(probs1, probs2);
        assert!((probs1.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        // 2-layer MLP with GELU activation and MultiLabel sigmoid mode.
        let inciv_names = IncivilityLabel::ALL.map(|l| l.name());
        let inciv_tags = TagSet::new("incivility", &inciv_names);
        let mlp = ClassifierHead::mlp(
            test_model(),
            2,
            3,
            inciv_tags,
            HeadMode::MultiLabel,
            HiddenActivation::Gelu,
            vec![1.0, 0.5, -0.5, 1.0, 0.2, -0.2],
            vec![0.0, 0.1, -0.1],
            vec![
                1.2, -0.4, 0.0, // insult
                0.8, 0.2, -0.1, // profanity
                -0.5, 0.9, 0.3, // threat
                0.1, -0.2, 0.7, // identity_attack
                -0.3, 0.4, -0.2, // intolerance
                0.9, 0.9, 0.1, // severe_toxicity
            ],
            vec![-0.2, -0.5, -1.0, -0.8, -0.9, -1.2],
        )
        .unwrap();

        let mlp_bytes = mlp.to_bytes();
        let mlp_loaded = ClassifierHead::from_bytes(&mlp_bytes).unwrap();
        assert_eq!(mlp, mlp_loaded);

        let mut s1 = [0.0f64; 6];
        let mut s2 = [0.0f64; 6];
        mlp.score(&[1.5, -0.5], &mut s1).unwrap();
        mlp_loaded.score(&[1.5, -0.5], &mut s2).unwrap();
        assert_eq!(s1, s2);
        assert!(s1.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn fit_single_and_multi_label_converge_deterministically() {
        let tags = TagSet::new("sentiment", &["positive", "negative"]);
        let examples: &[([f32; 2], Tag)] =
            &[([1.0, 0.1], Tag(0)), ([0.9, 0.2], Tag(0)), ([-1.0, -0.1], Tag(1)), ([-0.8, -0.2], Tag(1))];
        let h1 =
            ClassifierHead::fit_single_label(test_model(), 2, Arc::clone(&tags), examples, 1e-3, 150, 0.5).unwrap();
        let h2 =
            ClassifierHead::fit_single_label(test_model(), 2, Arc::clone(&tags), examples, 1e-3, 150, 0.5).unwrap();
        assert_eq!(h1, h2);

        let mut probs = [0.0f64; 2];
        let (tag_pos, p_pos) = h1.score(&[0.95, 0.15], &mut probs).unwrap();
        assert_eq!(tag_pos, Tag(0));
        assert!(p_pos > 0.9, "p_pos={p_pos}");

        let (tag_neg, p_neg) = h1.score(&[-0.95, -0.15], &mut probs).unwrap();
        assert_eq!(tag_neg, Tag(1));
        assert!(p_neg > 0.9, "p_neg={p_neg}");
    }
}
