//! A span classifier head in training: the weights of
//! `turbo::head::span::tensors`, in that order, with its loss and
//! gradients on one sequence of per-word encoder features.
//!
//! Every span of 1 to `span.max_words` words is one softmax over the
//! labels, its gold label the span's own when it is a gold span and `O`
//! otherwise; the loss sums them all. The forward is the core's
//! (`turbo::head::span`): a span reads its first and last word vectors
//! and its width embedding, joined, through one linear layer.

use turbo::head::SpanClassifier;
use turbo::head::span::tensors;
use turbo::manifest::{HeadKind, HeadSpec};
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::adapter::{self, Shape};
use crate::logspace::softmax;
use crate::real::{Real, dot};
use crate::rng::Rng;
use crate::shapes::{checked_shapes, fits};

/// A gold span: words `first..=last` of its sequence and its label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoldSpan {
    pub first: u32,
    pub last: u32,
    pub label: u32,
}

/// The weights of one head, `params[i]` named `names[i]`.
#[derive(Debug, Clone)]
pub struct Spans<R> {
    spec: HeadSpec,
    hidden: usize,
    names: Vec<String>,
    /// Shaped as `tensors` says at construction and never reshaped.
    params: Vec<Vec<R>>,
    none: u32,
}

/// The adapter's activations of one sequence, kept for backward.
#[derive(Debug, Default)]
pub struct Work<R> {
    adapter: adapter::Work<R>,
}

impl<R: Real> Spans<R> {
    /// A head with seeded initial weights: Xavier-uniform matrices and
    /// width embeddings, zero biases but for each LSTM forget gate's (1).
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
        if spec.kind != HeadKind::SpanClassifier {
            return Err(invalid(format!("trainer: head {:?} is not a span classifier", spec.name)));
        }
        let shapes = checked_shapes(spec, hidden, tensors(spec, hidden))?;
        fits(spec, &shapes, &params)?;
        let none = spec.labels.iter().position(|l| l == "O").ok_or_else(|| invalid("trainer: no label \"O\""))?;
        let names = shapes.into_iter().map(|((s, _), _)| s).collect();
        Ok(Spans { spec: spec.clone(), hidden, names, params, none: none as u32 })
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

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Whether tensor i takes weight decay: matrices do; biases and the
    /// width embeddings do not.
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

    fn max_words(&self) -> usize {
        self.spec.span.map_or(0, |s| s.max_words as usize)
    }

    fn width_dim(&self) -> usize {
        self.spec.span.map_or(0, |s| s.width_dim as usize)
    }

    /// The loss of one sequence of features [n, hidden] against its gold
    /// spans, and its gradients added to `grads` (shaped as `zero_grads`
    /// gives). Returns the loss and the number of spans it sums over. A
    /// gold span longer than `span.max_words`, out of the sequence, or
    /// given twice is refused, as are gradients of another shape; a
    /// refusal leaves `grads` as they were.
    pub fn loss(
        &self,
        features: &[R],
        gold: &[GoldSpan],
        grads: &mut [Vec<R>],
        w: &mut Work<R>,
    ) -> Result<(f64, usize)> {
        if grads.len() != self.params.len() || grads.iter().zip(&self.params).any(|(g, p)| g.len() != p.len()) {
            return Err(invalid(format!("trainer: gradients do not match the tensors of head {:?}", self.spec.name)));
        }
        if !features.len().is_multiple_of(self.hidden) || features.iter().any(|v| !v.to_f64().is_finite()) {
            return Err(invalid(format!("trainer: features are not finite rows of {}", self.hidden)));
        }
        let n = features.len() / self.hidden;
        let (max, labels) = (self.max_words(), self.spec.labels.len());
        let mut target = vec![self.none; n * max];
        for g in gold {
            let (i, j) = (g.first as usize, g.last as usize);
            let fits = i <= j && j < n && j - i < max && (g.label as usize) < labels && g.label != self.none;
            if !fits || target[i * max + (j - i)] != self.none {
                return Err(invalid(format!("trainer: gold span {g:?} of a sequence of {n} words")));
            }
            target[i * max + (j - i)] = g.label;
        }
        let shape = self.shape();
        adapter::forward(shape, &self.params, features, &mut w.adapter);
        let v = adapter::output(features, &w.adapter);
        let (width, dim, o) = (shape.width(), self.width_dim(), 6 * shape.layers);
        let cols = 2 * width + dim;
        let (emb, ow, ob) = (&self.params[o], &self.params[o + 1], &self.params[o + 2]);
        let mut local = self.zero_grads();
        let mut dv = vec![R::ZERO; n * width];
        let (mut total, mut count) = (0f64, 0usize);
        let mut x = vec![R::ZERO; cols];
        let mut row = vec![0f64; labels];
        for i in 0..n {
            for j in i..n.min(i + max) {
                let k = j - i;
                x[..width].copy_from_slice(&v[i * width..][..width]);
                x[width..2 * width].copy_from_slice(&v[j * width..][..width]);
                x[2 * width..].copy_from_slice(&emb[k * dim..][..dim]);
                for (l, r) in row.iter_mut().enumerate() {
                    *r = (ob[l] + dot(&ow[l * cols..][..cols], &x)).to_f64();
                }
                let g = target[i * max + k] as usize;
                let gold_score = row[g];
                let (m, ln_z) = softmax(&mut row);
                total += ln_z - (gold_score - m);
                count += 1;
                row[g] -= 1.0;
                for (l, &d) in row.iter().enumerate() {
                    let d = R::from_f64(d);
                    local[o + 2][l] += d;
                    for (c, &xc) in x.iter().enumerate() {
                        local[o + 1][l * cols + c] += d * xc;
                        let back = d * ow[l * cols + c];
                        match c {
                            c if c < width => dv[i * width + c] += back,
                            c if c < 2 * width => dv[j * width + c - width] += back,
                            c => local[o][k * dim + c - 2 * width] += back,
                        }
                    }
                }
            }
        }
        if !total.is_finite() || local.iter().flatten().any(|g| !g.to_f64().is_finite()) {
            return Err(invalid("trainer: the span loss or a gradient is not finite"));
        }
        adapter::backward(shape, &self.params, features, dv, &mut local, &w.adapter, false);
        for (g, l) in grads.iter_mut().zip(local) {
            g.iter_mut().zip(l).for_each(|(a, b)| *a += b);
        }
        Ok((total, count))
    }
}

impl Spans<f32> {
    /// The weights as the core reads them, by suffix, in `tensors` order.
    pub fn tensors(&self) -> Vec<(String, Vec<f32>)> {
        self.names.iter().cloned().zip(self.params.iter().cloned()).collect()
    }

    /// The core's span classifier over these weights: the inference path.
    pub fn to_core(&self) -> Result<SpanClassifier> {
        let get = |s: &str| self.names.iter().position(|n| n == s).map(|i| self.params[i].clone());
        SpanClassifier::from_tensors(&self.spec, self.hidden, &get)
    }
}
