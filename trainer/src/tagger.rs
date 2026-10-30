//! A token tagger head in training: the weights of
//! `turbo::head::tagger::tensors`, in that order, with its loss and
//! gradients on one sequence of per-word encoder features.
//!
//! The features are the words already pooled the way the head's
//! `word_pooling` says (the feature cache does that with the core's own
//! pooling), so a sequence is [words, hidden] and the encoder takes no
//! part in training. A head with a `hash_embed` also reads each word's
//! text, hashed as the core hashes it (`turbo::head::hash_embed`), and
//! learns its tables.

use turbo::head::crf::{Allowed, allowed};
use turbo::head::hash_embed::{self, SEEDS};
use turbo::head::tagger::tensors;
use turbo::head::{SentenceBoundary, TokenSplitter, TokenTagger};
use turbo::manifest::{Decode, HeadSpec};
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
    /// Features joined with each word's hashed embedding, with a hash_embed.
    input: Vec<R>,
    rows: Vec<[[u32; SEEDS]; 4]>,
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
            let v = if suffix.starts_with("hash.") {
                // Small, so the sum of a word's rows starts near the
                // scale of one Xavier row.
                (0..n).map(|_| R::from_f64(rng.uniform(0.05))).collect()
            } else if shape.len() == 2 && !suffix.starts_with("crf.") {
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
        if !spec.kind.scores_words() {
            return Err(invalid(format!("trainer: head {:?} does not score words as a token tagger does", spec.name)));
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

    /// Every tensor, to step in place: each keeps its length, as
    /// [`set_params`](Self::set_params) checks.
    pub fn params_mut(&mut self) -> &mut [Vec<R>] {
        &mut self.params
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

    /// The width of the encoder features the head reads.
    pub fn hidden(&self) -> usize {
        self.hidden
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

    /// Values a word's hashed embedding adds (0 without a hash_embed).
    fn hashed(&self) -> usize {
        self.spec.hash_embed.map_or(0, |h| 4 * h.dim as usize)
    }

    fn shape(&self) -> Shape {
        let (layers, h) = self.spec.adapter.map_or((0, 0), |a| (a.bilstm.layers as usize, a.bilstm.hidden as usize));
        Shape { hidden: self.hidden + self.hashed(), layers, h }
    }

    /// Where the output layer's tensors start: after the adapter's and
    /// the hash tables'.
    fn out_at(&self) -> usize {
        6 * self.shape().layers + if self.spec.hash_embed.is_some() { 4 } else { 0 }
    }

    /// The label scores [n, labels] of features [n, hidden], kept in `w`;
    /// refused for a head that reads each word's text.
    pub fn forward(&self, features: &[R], w: &mut Work<R>) -> Result<usize> {
        self.forward_forms(features, None, w)
    }

    /// As [`forward`](Self::forward), with each word's text, which a head
    /// with a `hash_embed` reads and any other head ignores.
    pub fn forward_forms(&self, features: &[R], forms: Option<&[&str]>, w: &mut Work<R>) -> Result<usize> {
        if !features.len().is_multiple_of(self.hidden) {
            return Err(invalid(format!("trainer: {} features are not rows of {}", features.len(), self.hidden)));
        }
        let n = features.len() / self.hidden;
        self.join(features, forms, n, w)?;
        let shape = self.shape();
        let x0: &[R] = if self.spec.hash_embed.is_some() { &w.input } else { features };
        adapter::forward(shape, &self.params, x0, &mut w.adapter);
        let width = shape.width();
        let labels = self.spec.labels.len();
        let o = self.out_at();
        let x = adapter::output(x0, &w.adapter);
        w.scores.clear();
        w.scores.resize(n * labels, R::ZERO);
        linear::forward(&self.params[o], &self.params[o + 1], x, width, labels, &mut w.scores);
        Ok(n)
    }

    /// With a hash_embed, each word's features joined with the sum of
    /// its hashed rows into `w.input`, and the rows kept for backward. A
    /// sentence of no words needs no forms, as in the core.
    fn join(&self, features: &[R], forms: Option<&[&str]>, n: usize, w: &mut Work<R>) -> Result<()> {
        let Some(h) = self.spec.hash_embed else { return Ok(()) };
        if n == 0 {
            w.rows.clear();
            w.input.clear();
            return Ok(());
        }
        let forms = forms.ok_or_else(|| invalid("trainer: the head reads each word's text"))?;
        if forms.len() != n {
            return Err(invalid(format!("trainer: {} forms for {n} words", forms.len())));
        }
        let (hidden, d) = (self.hidden, h.dim as usize);
        let width = hidden + 4 * d;
        let t = 6 * self.shape().layers;
        w.rows.clear();
        w.rows.extend(forms.iter().map(|f| hash_embed::rows(&h, f)));
        w.input.clear();
        w.input.resize(n * width, R::ZERO);
        for i in 0..n {
            let row = &mut w.input[i * width..][..width];
            row[..hidden].copy_from_slice(&features[i * hidden..][..hidden]);
            for (k, seeds) in w.rows[i].iter().enumerate() {
                let dst = &mut row[hidden + k * d..][..d];
                for &r in seeds {
                    let src = &self.params[t + k][r as usize * d..][..d];
                    dst.iter_mut().zip(src).for_each(|(a, &b)| *a += b);
                }
            }
        }
        Ok(())
    }

    /// The loss of one sequence against `gold`, and its gradients added
    /// to `grads` (shaped as [`zero_grads`](Self::zero_grads) gives);
    /// gradients of any other shape are refused before anything runs.
    pub fn loss(&self, features: &[R], gold: &[u32], grads: &mut [Vec<R>], w: &mut Work<R>) -> Result<f64> {
        self.loss_forms(features, None, gold, grads, w)
    }

    /// As [`loss`](Self::loss), with each word's text (see
    /// [`forward_forms`](Self::forward_forms)).
    pub fn loss_forms(
        &self,
        features: &[R],
        forms: Option<&[&str]>,
        gold: &[u32],
        grads: &mut [Vec<R>],
        w: &mut Work<R>,
    ) -> Result<f64> {
        self.loss_inner(features, forms, gold, grads, w, None)
    }

    /// As [`loss_forms`](Self::loss_forms), the loss's gradient with
    /// respect to `features` also written to `dfeatures` ([words, hidden]),
    /// for what feeds the head (a learned mix of layers, say).
    pub fn loss_input(
        &self,
        features: &[R],
        forms: Option<&[&str]>,
        gold: &[u32],
        grads: &mut [Vec<R>],
        w: &mut Work<R>,
        dfeatures: &mut Vec<R>,
    ) -> Result<f64> {
        self.loss_inner(features, forms, gold, grads, w, Some(dfeatures))
    }

    fn loss_inner(
        &self,
        features: &[R],
        forms: Option<&[&str]>,
        gold: &[u32],
        grads: &mut [Vec<R>],
        w: &mut Work<R>,
        dfeatures: Option<&mut Vec<R>>,
    ) -> Result<f64> {
        if grads.len() != self.params.len() || grads.iter().zip(&self.params).any(|(g, p)| g.len() != p.len()) {
            return Err(invalid(format!("trainer: gradients do not match the tensors of head {:?}", self.spec.name)));
        }
        let n = self.forward_forms(features, forms, w)?;
        if gold.len() != n {
            return Err(invalid(format!("trainer: {} gold labels for {n} words", gold.len())));
        }
        let labels = self.spec.labels.len();
        let shape = self.shape();
        let o = self.out_at();
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
        let hashed = self.spec.hash_embed.is_some();
        let (layers, width) = (shape.layers, shape.width());
        let x0: &[R] = if hashed { &w.input } else { features };
        let x = adapter::output(x0, &w.adapter);
        let input = dfeatures.is_some();
        let mut dx = vec![R::ZERO; if layers == 0 && !hashed && !input { 0 } else { n * width }];
        let [dw, db] = grads.get_disjoint_mut([o, o + 1]).expect("distinct");
        let dxo = if dx.is_empty() { None } else { Some(&mut dx[..]) };
        linear::backward(&self.params[o], x, &w.dscores, width, labels, dw, db, dxo);
        let dx0 = adapter::backward(shape, &self.params, x0, dx, grads, &w.adapter, hashed || input);
        if let Some(d) = dfeatures {
            // The input's rows are the features, then a hashed head's
            // embeddings of the word's text.
            let width0 = dx0.len() / n.max(1);
            d.clear();
            d.extend((0..n).flat_map(|i| dx0[i * width0..][..self.hidden].iter().copied()));
        }
        if let Some(h) = self.spec.hash_embed {
            let (hidden, d, t) = (self.hidden, h.dim as usize, 6 * layers);
            let width0 = hidden + 4 * d;
            for (i, word) in w.rows.iter().enumerate() {
                for (k, seeds) in word.iter().enumerate() {
                    let src = &dx0[i * width0 + hidden + k * d..][..d];
                    for &r in seeds {
                        let dst = &mut grads[t + k][r as usize * d..][..d];
                        dst.iter_mut().zip(src).for_each(|(a, &b)| *a += b);
                    }
                }
            }
        }
        Ok(loss)
    }
}

impl Tagger<f32> {
    /// The weights as the core reads them, by suffix, in `tensors` order.
    pub fn tensors(&self) -> Vec<(String, Vec<f32>)> {
        self.names.iter().cloned().zip(self.params.iter().cloned()).collect()
    }

    /// The core's tagger over these weights: the inference path itself.
    /// A sentence boundary or token split head scores through the same
    /// tagger (`TokenTagger::scoring_words`); its own type decodes.
    pub fn to_token_tagger(&self) -> Result<TokenTagger> {
        TokenTagger::scoring_words(&self.spec, self.hidden, &|s| self.tensor(s))
    }

    /// The core's sentence boundary head over these weights.
    pub fn to_boundary(&self) -> Result<SentenceBoundary> {
        SentenceBoundary::from_tensors(&self.spec, self.hidden, &|s| self.tensor(s))
    }

    /// The core's token split head over these weights.
    pub fn to_splitter(&self) -> Result<TokenSplitter> {
        TokenSplitter::from_tensors(&self.spec, self.hidden, &|s| self.tensor(s))
    }

    fn tensor(&self, suffix: &str) -> Option<Vec<f32>> {
        self.names.iter().position(|n| n == suffix).map(|i| self.params[i].clone())
    }
}
