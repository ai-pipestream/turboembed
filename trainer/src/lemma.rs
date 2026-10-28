//! An edit-tree lemmatizer head in training: the weights of
//! `turbo::head::lemmatizer::tensors`, in that order, with its loss and
//! gradients on one sentence of per-word encoder features.
//!
//! The loss is each word's cross-entropy over the head's trees, for the
//! words whose gold tree the head has (a tree too rare to keep teaches
//! nothing). The forward is the core's: the adapter's word vectors, each
//! joined by its tag's embedding when the head reads tags, then one
//! linear layer. Training reads the gold tags; inference reads the tagger
//! head's.

use turbo::head::EditTreeLemmatizer;
use turbo::head::lemmatizer::tensors;
use turbo::manifest::{HeadKind, HeadSpec};
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::adapter::{self, Shape};
use crate::real::Real;
use crate::rng::Rng;
use crate::shapes::{checked_shapes, fits};
use crate::{linear, xent};

/// The weights of one head, `params[i]` named `names[i]`.
#[derive(Debug, Clone)]
pub struct Lemmatizer<R> {
    spec: HeadSpec,
    hidden: usize,
    names: Vec<String>,
    /// Shaped as `tensors` says at construction and never reshaped.
    params: Vec<Vec<R>>,
}

/// The activations of one sentence, kept for backward.
#[derive(Debug, Default)]
pub struct Work<R> {
    adapter: adapter::Work<R>,
    joined: Vec<R>,
    scores: Vec<R>,
}

impl<R: Real> Work<R> {
    /// The tree scores [n, trees] of the last forward.
    pub fn scores(&self) -> &[R] {
        &self.scores
    }
}

impl<R: Real> Lemmatizer<R> {
    /// A head with seeded initial weights: Xavier-uniform matrices and tag
    /// embeddings, zero biases but for each LSTM forget gate's (1).
    pub fn new(spec: &HeadSpec, hidden: usize, rng: &mut Rng) -> Result<Self> {
        let shapes = checked_shapes(spec, hidden, tensors(spec, hidden))?;
        let mut params = Vec::with_capacity(shapes.len());
        for ((suffix, shape), n) in &shapes {
            let n = *n;
            let v = if shape.len() == 2 {
                let limit = (6.0 / (shape[0] + shape[1]) as f64).sqrt();
                (0..n).map(|_| R::from_f64(rng.uniform(limit))).collect()
            } else if suffix.starts_with("lstm.") {
                let h = n / 4;
                (0..n).map(|i| if (h..2 * h).contains(&i) { R::ONE } else { R::ZERO }).collect()
            } else {
                vec![R::ZERO; n]
            };
            params.push(v);
        }
        Self::with_params(spec, hidden, params)
    }

    /// A head from given weights, in `tensors` order.
    pub fn with_params(spec: &HeadSpec, hidden: usize, params: Vec<Vec<R>>) -> Result<Self> {
        let shapes = checked_shapes(spec, hidden, tensors(spec, hidden))?;
        if spec.kind != HeadKind::EditTreeLemmatizer {
            return Err(invalid(format!("trainer: head {:?} is not an edit-tree lemmatizer", spec.name)));
        }
        fits(spec, &shapes, &params)?;
        let names = shapes.into_iter().map(|((s, _), _)| s).collect();
        Ok(Lemmatizer { spec: spec.clone(), hidden, names, params })
    }

    /// The weights, in `tensors` order.
    pub fn params(&self) -> &[Vec<R>] {
        &self.params
    }

    /// Tensor i's weights, to change in place; its length is fixed.
    pub fn param_mut(&mut self, i: usize) -> &mut [R] {
        &mut self.params[i]
    }

    /// Replaces every weight, refused unless each tensor keeps its length.
    pub fn set_params(&mut self, params: Vec<Vec<R>>) -> Result<()> {
        if params.len() != self.params.len() || params.iter().zip(&self.params).any(|(a, b)| a.len() != b.len()) {
            return Err(invalid(format!("trainer: weights do not match the tensors of head {:?}", self.spec.name)));
        }
        self.params = params;
        Ok(())
    }

    /// One optimizer step on the weights; returns the gradient norm
    /// before clipping.
    pub fn step(&mut self, opt: &mut AdamW<R>, grads: &mut [Vec<R>], decay: &[bool]) -> Result<f64> {
        opt.step(&mut self.params, grads, decay)
    }

    pub fn spec(&self) -> &HeadSpec {
        &self.spec
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Whether tensor i takes weight decay: matrices do; biases and the
    /// tag embeddings do not.
    pub fn decays(&self) -> Vec<bool> {
        self.names.iter().map(|n| n.ends_with(".w") || n.ends_with(".u") || n == "out.weight").collect()
    }

    pub fn zero_grads(&self) -> Vec<Vec<R>> {
        self.params.iter().map(|p| vec![R::ZERO; p.len()]).collect()
    }

    fn shape(&self) -> Shape {
        let (layers, h) = self.spec.adapter.map_or((0, 0), |a| (a.bilstm.layers as usize, a.bilstm.hidden as usize));
        Shape { hidden: self.hidden, layers, h }
    }

    /// (tag embedding width, index of `pos.embedding`), 0 and None
    /// without a tag input.
    fn pos(&self) -> (usize, Option<usize>) {
        match &self.spec.pos_input {
            Some(p) => (p.dim as usize, Some(6 * self.shape().layers)),
            None => (0, None),
        }
    }

    /// The tree scores [n, trees] of features [n, hidden] and, with a tag
    /// input, each word's tag; kept in `w`.
    pub fn forward(&self, features: &[R], tags: &[u32], w: &mut Work<R>) -> Result<usize> {
        if self.hidden == 0 || !features.len().is_multiple_of(self.hidden) {
            return Err(invalid(format!("trainer: {} features are not rows of {}", features.len(), self.hidden)));
        }
        let n = features.len() / self.hidden;
        let (dim, emb) = self.pos();
        let wanted = if emb.is_some() { n } else { 0 };
        if tags.len() != wanted {
            return Err(invalid(format!("trainer: {} tags for {n} words, {wanted} wanted", tags.len())));
        }
        let shape = self.shape();
        adapter::forward(shape, &self.params, features, &mut w.adapter);
        let x = adapter::output(features, &w.adapter);
        let width = shape.width();
        w.joined.clear();
        for i in 0..n {
            w.joined.extend_from_slice(&x[i * width..][..width]);
            if let Some(e) = emb {
                let table = &self.params[e];
                let t = tags[i] as usize;
                if (t + 1) * dim > table.len() {
                    return Err(invalid(format!("trainer: tag {t} past the tag table")));
                }
                w.joined.extend_from_slice(&table[t * dim..][..dim]);
            }
        }
        let o = self.params.len() - 2;
        let trees = self.spec.labels.len();
        w.scores.clear();
        w.scores.resize(n * trees, R::ZERO);
        linear::forward(&self.params[o], &self.params[o + 1], &w.joined, width + dim, trees, &mut w.scores);
        Ok(n)
    }

    /// The summed loss over the words with a gold tree (`gold[i]`, None
    /// for a word whose tree the head does not have), and its gradients
    /// added to `grads` (shaped as `zero_grads` gives); gradients of any
    /// other shape are refused before anything runs.
    pub fn loss(
        &self,
        features: &[R],
        tags: &[u32],
        gold: &[Option<u32>],
        grads: &mut [Vec<R>],
        w: &mut Work<R>,
    ) -> Result<f64> {
        if grads.len() != self.params.len() || grads.iter().zip(&self.params).any(|(g, p)| g.len() != p.len()) {
            return Err(invalid(format!("trainer: gradients do not match the tensors of head {:?}", self.spec.name)));
        }
        let n = self.forward(features, tags, w)?;
        if gold.len() != n {
            return Err(invalid(format!("trainer: {} gold trees for {n} words", gold.len())));
        }
        let trees = self.spec.labels.len();
        let rows: Vec<usize> = (0..n).filter(|&i| gold[i].is_some()).collect();
        let picked: Vec<R> = rows.iter().flat_map(|&i| w.scores[i * trees..][..trees].iter().copied()).collect();
        let want: Vec<u32> = rows.iter().map(|&i| gold[i].expect("filtered")).collect();
        let mut dpicked = vec![R::ZERO; picked.len()];
        let total = xent::loss(&picked, &want, trees, &mut dpicked)?;
        let mut dscores = vec![R::ZERO; n * trees];
        for (k, &i) in rows.iter().enumerate() {
            dscores[i * trees..][..trees].copy_from_slice(&dpicked[k * trees..][..trees]);
        }
        let shape = self.shape();
        let (dim, emb) = self.pos();
        let width = shape.width();
        let joined_width = width + dim;
        let o = self.params.len() - 2;
        let mut djoined = vec![R::ZERO; n * joined_width];
        let [dw, db] = grads.get_disjoint_mut([o, o + 1]).expect("distinct");
        linear::backward(&self.params[o], &w.joined, &dscores, joined_width, trees, dw, db, Some(&mut djoined));
        if let Some(e) = emb {
            for (i, &t) in tags.iter().enumerate() {
                let src = &djoined[i * joined_width + width..][..dim];
                for (g, d) in grads[e][t as usize * dim..][..dim].iter_mut().zip(src) {
                    *g += *d;
                }
            }
        }
        if shape.layers > 0 {
            let dx: Vec<R> = (0..n).flat_map(|i| djoined[i * joined_width..][..width].iter().copied()).collect();
            adapter::backward(shape, &self.params, features, dx, grads, &w.adapter, false);
        }
        Ok(total)
    }
}

impl Lemmatizer<f32> {
    /// The weights as the core reads them, by suffix, in `tensors` order.
    pub fn tensors(&self) -> Vec<(String, Vec<f32>)> {
        self.names.iter().cloned().zip(self.params.iter().cloned()).collect()
    }

    /// The core's lemmatizer over these weights: the inference path itself.
    pub fn to_core(&self) -> Result<EditTreeLemmatizer> {
        let get = |s: &str| self.names.iter().position(|n| n == s).map(|i| self.params[i].clone());
        EditTreeLemmatizer::from_tensors(&self.spec, self.hidden, &get)
    }
}
