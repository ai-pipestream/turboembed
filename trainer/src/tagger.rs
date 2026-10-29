//! A token tagger head in training: the weights of
//! `turbo::head::tagger::tensors`, in that order, with its loss and
//! gradients on one sequence of per-word encoder features.
//!
//! The features are the words already pooled the way the head's
//! `word_pooling` says (the feature cache does that with the core's own
//! pooling), so a sequence is [words, hidden] and the encoder takes no
//! part in training.

use turbo::head::TokenTagger;
use turbo::head::crf::{Allowed, allowed};
use turbo::head::tagger::tensors;
use turbo::manifest::{Decode, HeadKind, HeadSpec};
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::adapter::{self, Shape};
use crate::real::Real;
use crate::rng::Rng;
use crate::shapes::{checked_shapes, fits};
use crate::{crf, linear, xent};

/// The weights of one head, `params[i]` named `names[i]`.
#[derive(Debug, Clone)]
pub struct Tagger<R> {
    spec: HeadSpec,
    hidden: usize,
    names: Vec<String>,
    /// Shaped as `tensors` says at construction and never reshaped.
    params: Vec<Vec<R>>,
    allowed: Allowed,
}

/// The activations of one sequence, kept for backward.
#[derive(Debug, Default)]
pub struct Work<R> {
    adapter: adapter::Work<R>,
    scores: Vec<R>,
    dscores: Vec<R>,
}

impl<R: Real> Work<R> {
    /// The label scores [n, labels] of the last forward.
    pub fn scores(&self) -> &[R] {
        &self.scores
    }
}

impl<R: Real> Tagger<R> {
    /// A head with seeded initial weights: Xavier-uniform matrices, zero
    /// biases but for each LSTM forget gate's (1), zero CRF scores.
    pub fn new(spec: &HeadSpec, hidden: usize, rng: &mut Rng) -> Result<Self> {
        let shapes = checked_shapes(spec, hidden, tensors(spec, hidden))?;
        let mut params = Vec::with_capacity(shapes.len());
        for ((suffix, shape), n) in &shapes {
            let n = *n;
            let v = if shape.len() == 2 && !suffix.starts_with("crf.") {
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
        if spec.kind != HeadKind::TokenTagger {
            return Err(invalid(format!("trainer: head {:?} is not a token tagger", spec.name)));
        }
        let shapes = checked_shapes(spec, hidden, tensors(spec, hidden))?;
        fits(spec, &shapes, &params)?;
        let (_, allowed) = allowed(spec.scheme, &spec.labels)?;
        let names = shapes.into_iter().map(|((s, _), _)| s).collect();
        Ok(Tagger { spec: spec.clone(), hidden, names, params, allowed })
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
        let lens: Vec<usize> = self.params.iter().map(Vec::len).collect();
        if params.len() != lens.len() || params.iter().zip(&lens).any(|(p, &n)| p.len() != n) {
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

    /// Whether tensor i takes weight decay: matrices do, biases and CRF
    /// scores do not.
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

    /// The label scores [n, labels] of features [n, hidden], kept in `w`.
    pub fn forward(&self, features: &[R], w: &mut Work<R>) -> Result<usize> {
        if !features.len().is_multiple_of(self.hidden) {
            return Err(invalid(format!("trainer: {} features are not rows of {}", features.len(), self.hidden)));
        }
        let n = features.len() / self.hidden;
        let shape = self.shape();
        adapter::forward(shape, &self.params, features, &mut w.adapter);
        let (layers, width) = (shape.layers, shape.width());
        let labels = self.spec.labels.len();
        let o = 6 * layers;
        let x = adapter::output(features, &w.adapter);
        w.scores.clear();
        w.scores.resize(n * labels, R::ZERO);
        linear::forward(&self.params[o], &self.params[o + 1], x, width, labels, &mut w.scores);
        Ok(n)
    }

    /// The loss of one sequence against `gold`, and its gradients added
    /// to `grads` (shaped as [`zero_grads`](Self::zero_grads) gives);
    /// gradients of any other shape are refused before anything runs.
    pub fn loss(&self, features: &[R], gold: &[u32], grads: &mut [Vec<R>], w: &mut Work<R>) -> Result<f64> {
        if grads.len() != self.params.len() || grads.iter().zip(&self.params).any(|(g, p)| g.len() != p.len()) {
            return Err(invalid(format!("trainer: gradients do not match the tensors of head {:?}", self.spec.name)));
        }
        let n = self.forward(features, w)?;
        if gold.len() != n {
            return Err(invalid(format!("trainer: {} gold labels for {n} words", gold.len())));
        }
        let labels = self.spec.labels.len();
        let shape = self.shape();
        let o = 6 * shape.layers;
        w.dscores.clear();
        w.dscores.resize(n * labels, R::ZERO);
        let loss = match self.spec.decode {
            Decode::Crf => {
                let p = crf::Params {
                    transitions: &self.params[o + 2],
                    start: &self.params[o + 3],
                    end: &self.params[o + 4],
                };
                let [t, s, e] = grads.get_disjoint_mut([o + 2, o + 3, o + 4]).expect("distinct");
                let g = crf::Grad { transitions: t, start: s, end: e };
                crf::nll(p, &self.allowed, &w.scores, gold, labels, &mut w.dscores, g)?
            }
            _ => xent::loss(&w.scores, gold, labels, &mut w.dscores)?,
        };
        let (layers, width) = (shape.layers, shape.width());
        let x = adapter::output(features, &w.adapter);
        let mut dx = vec![R::ZERO; if layers == 0 { 0 } else { n * width }];
        let [dw, db] = grads.get_disjoint_mut([o, o + 1]).expect("distinct");
        let dxo = if layers == 0 { None } else { Some(&mut dx[..]) };
        linear::backward(&self.params[o], x, &w.dscores, width, labels, dw, db, dxo);
        adapter::backward(shape, &self.params, features, dx, grads, &w.adapter);
        Ok(loss)
    }
}

impl Tagger<f32> {
    /// The weights as the core reads them, by suffix, in `tensors` order.
    pub fn tensors(&self) -> Vec<(String, Vec<f32>)> {
        self.names.iter().cloned().zip(self.params.iter().cloned()).collect()
    }

    /// The core's tagger over these weights: the inference path itself.
    pub fn to_token_tagger(&self) -> Result<TokenTagger> {
        let get = |s: &str| self.names.iter().position(|n| n == s).map(|i| self.params[i].clone());
        TokenTagger::from_tensors(&self.spec, self.hidden, &get)
    }
}
