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
    language: Option<u32>,
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
            let v = if suffix.starts_with("hash.") || suffix == "language.embedding" {
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
        self.names
            .iter()
            .map(|n| n.ends_with(".w") || n.ends_with(".u") || n == "out.weight" || n == "language.embedding")
            .collect()
    }

    pub fn zero_grads(&self) -> Vec<Vec<R>> {
        self.params.iter().map(|p| vec![R::ZERO; p.len()]).collect()
    }

    fn hashed(&self) -> usize {
        self.spec.hash_embed.map_or(0, |h| 4 * h.dim as usize)
    }

    fn language(&self) -> usize {
        self.spec.language.as_ref().map_or(0, |language| language.dim as usize)
    }

    fn joined(&self) -> bool {
        self.spec.hash_embed.is_some() || self.spec.language.is_some()
    }

    fn adapter_at(&self) -> usize {
        usize::from(self.spec.language.is_some())
    }

    fn shape(&self) -> Shape {
        let (layers, h) = self.spec.adapter.map_or((0, 0), |a| (a.bilstm.layers as usize, a.bilstm.hidden as usize));
        Shape { hidden: self.hidden + self.language() + self.hashed(), layers, h }
    }

    fn out_at(&self) -> usize {
        self.adapter_at() + 6 * self.shape().layers + if self.spec.hash_embed.is_some() { 4 } else { 0 }
    }

    /// The label scores [n, labels] of features [n, hidden], kept in `w`;
    /// refused for a head that reads each word's text.
    pub fn forward(&self, features: &[R], w: &mut Work<R>) -> Result<usize> {
        self.forward_forms(features, None, w)
    }

    /// As [`forward`](Self::forward), with each word's text, which a head
    /// with a `hash_embed` reads and any other head ignores.
    pub fn forward_forms(&self, features: &[R], forms: Option<&[&str]>, w: &mut Work<R>) -> Result<usize> {
        self.forward_language(features, forms, None, w)
    }

    /// As [`forward_forms`](Self::forward_forms), with the language-table row.
    pub fn forward_language(
        &self,
        features: &[R],
        forms: Option<&[&str]>,
        language: Option<u32>,
        w: &mut Work<R>,
    ) -> Result<usize> {
        if !features.len().is_multiple_of(self.hidden) {
            return Err(invalid(format!("trainer: {} features are not rows of {}", features.len(), self.hidden)));
        }
        let n = features.len() / self.hidden;
        let language = self.language_row(language)?;
        self.join(features, forms, language, n, w)?;
        let shape = self.shape();
        let x0: &[R] = if self.joined() { &w.input } else { features };
        adapter::forward(shape, &self.params[self.adapter_at()..], x0, &mut w.adapter);
        let width = shape.width();
        let labels = self.spec.labels.len();
        let o = self.out_at();
        let x = adapter::output(x0, &w.adapter);
        w.scores.clear();
        w.scores.resize(n * labels, R::ZERO);
        linear::forward(&self.params[o], &self.params[o + 1], x, width, labels, &mut w.scores);
        Ok(n)
    }

    fn language_row(&self, row: Option<u32>) -> Result<Option<usize>> {
        let Some(language) = &self.spec.language else {
            return match row {
                Some(_) => Err(invalid("trainer: a language given to a head without a language input")),
                None => Ok(None),
            };
        };
        let row = match row {
            Some(row) => row as usize,
            None if language.languages.len() == 1 => 0,
            None => return Err(invalid("trainer: a multilingual example needs a language")),
        };
        if row >= language.languages.len() {
            return Err(invalid(format!("trainer: language row {row} is out of range")));
        }
        Ok(Some(row))
    }

    /// Joins encoder features, the selected language and hashed word text.
    fn join(
        &self,
        features: &[R],
        forms: Option<&[&str]>,
        language: Option<usize>,
        n: usize,
        w: &mut Work<R>,
    ) -> Result<()> {
        if !self.joined() {
            return Ok(());
        }
        if n == 0 {
            w.rows.clear();
            w.input.clear();
            return Ok(());
        }
        if let Some(h) = self.spec.hash_embed {
            let forms = forms.ok_or_else(|| invalid("trainer: the head reads each word's text"))?;
            if forms.len() != n {
                return Err(invalid(format!("trainer: {} forms for {n} words", forms.len())));
            }
            w.rows.clear();
            w.rows.extend(forms.iter().map(|form| hash_embed::rows(&h, form)));
        }
        let (hidden, language_width, hash_width) = (self.hidden, self.language(), self.hashed());
        let width = hidden + language_width + hash_width;
        let t = self.adapter_at() + 6 * self.shape().layers;
        w.input.clear();
        w.input.resize(n * width, R::ZERO);
        for i in 0..n {
            let row = &mut w.input[i * width..][..width];
            row[..hidden].copy_from_slice(&features[i * hidden..][..hidden]);
            if let Some(language) = language {
                let d = language_width;
                row[hidden..hidden + d].copy_from_slice(&self.params[0][language * d..][..d]);
            }
            for (k, seeds) in w.rows.get(i).into_iter().flatten().enumerate() {
                let d = hash_width / 4;
                let dst = &mut row[hidden + language_width + k * d..][..d];
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
        self.loss_language(features, forms, None, gold, grads, w)
    }

    /// As [`loss_forms`](Self::loss_forms), with the language-table row.
    pub fn loss_language(
        &self,
        features: &[R],
        forms: Option<&[&str]>,
        language: Option<u32>,
        gold: &[u32],
        grads: &mut [Vec<R>],
        w: &mut Work<R>,
    ) -> Result<f64> {
        w.language = language;
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
        w.language = None;
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
        let n = self.forward_language(features, forms, w.language, w)?;
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
        let joined = self.joined();
        let (layers, width) = (shape.layers, shape.width());
        let x0: &[R] = if joined { &w.input } else { features };
        let x = adapter::output(x0, &w.adapter);
        let input = dfeatures.is_some();
        let mut dx = vec![R::ZERO; if layers == 0 && !joined && !input { 0 } else { n * width }];
        let [dw, db] = grads.get_disjoint_mut([o, o + 1]).expect("distinct");
        let dxo = if dx.is_empty() { None } else { Some(&mut dx[..]) };
        linear::backward(&self.params[o], x, &w.dscores, width, labels, dw, db, dxo);
        let adapter_at = self.adapter_at();
        let dx0 = adapter::backward(
            shape,
            &self.params[adapter_at..],
            x0,
            dx,
            &mut grads[adapter_at..],
            &w.adapter,
            joined || input,
        );
        if let Some(d) = dfeatures {
            let width0 = dx0.len() / n.max(1);
            d.clear();
            d.extend((0..n).flat_map(|i| dx0[i * width0..][..self.hidden].iter().copied()));
        }
        if let Some(row) = self.language_row(w.language)? {
            let (hidden, d) = (self.hidden, self.language());
            let width0 = hidden + d + self.hashed();
            for i in 0..n {
                let src = &dx0[i * width0 + hidden..][..d];
                grads[0][row * d..][..d].iter_mut().zip(src).for_each(|(a, &b)| *a += b);
            }
        }
        if let Some(h) = self.spec.hash_embed {
            let (hidden, d, t) = (self.hidden + self.language(), h.dim as usize, adapter_at + 6 * layers);
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
