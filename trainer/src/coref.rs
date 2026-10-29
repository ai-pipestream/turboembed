//! A coreference ranker head in training: the weights of
//! `turbo::head::coref::tensors`, in that order, with its loss and
//! gradients on one document of per-word encoder features and its
//! mentions.
//!
//! The loss is Lee et al.'s (2017): for each mention, minus the log of
//! the probability its scored antecedents (the dummy among them, scoring
//! 0) give to the gold ones, which are the earlier mentions of its chain
//! within reach, or the dummy when there are none. The forward is the
//! core's (`turbo::head::coref`).

use turbo::head::CorefRanker;
use turbo::head::coref::{bucket, check_mentions, tensors};
use turbo::manifest::{HeadKind, HeadSpec};
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::adapter::{self, Shape};
use crate::real::{Real, dot};
use crate::rng::Rng;
use crate::shapes::{checked_shapes, fits};

/// The weights of one head, `params[i]` named `names[i]`.
#[derive(Debug, Clone)]
pub struct Ranker<R> {
    spec: HeadSpec,
    hidden: usize,
    names: Vec<String>,
    /// Shaped as `tensors` says at construction and never reshaped.
    params: Vec<Vec<R>>,
}

/// The adapter's activations of one document, kept for backward.
#[derive(Debug, Default)]
pub struct Work<R> {
    adapter: adapter::Work<R>,
}

/// The log of the sum of the exponentials of `v`, centred on its largest
/// value, and each term's share of the sum into `p`.
fn shares(v: &[f64], p: &mut Vec<f64>) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    p.clear();
    p.extend(v.iter().map(|x| (x - m).exp()));
    let z: f64 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= z);
    m + z.ln()
}

impl<R: Real> Ranker<R> {
    /// A head with seeded initial weights: Xavier-uniform matrices and
    /// width embeddings, zero biases and distance scores but for each LSTM
    /// forget gate's (1).
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
        if spec.kind != HeadKind::CorefRanker {
            return Err(invalid(format!("trainer: head {:?} is not a coreference ranker", spec.name)));
        }
        let shapes = checked_shapes(spec, hidden, tensors(spec, hidden))?;
        fits(spec, &shapes, &params)?;
        let names = shapes.into_iter().map(|((s, _), _)| s).collect();
        Ok(Ranker { spec: spec.clone(), hidden, names, params })
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

    /// Whether tensor i takes weight decay: matrices do; biases, width
    /// embeddings and distance scores do not.
    pub fn decays(&self) -> Vec<bool> {
        self.names
            .iter()
            .map(|n| n.ends_with(".w") || n.ends_with(".u") || n == "mention.weight" || n == "pair.weight")
            .collect()
    }

    pub fn zero_grads(&self) -> Vec<Vec<R>> {
        self.params.iter().map(|p| vec![R::ZERO; p.len()]).collect()
    }

    fn shape(&self) -> Shape {
        let (layers, h) = self.spec.adapter.map_or((0, 0), |a| (a.bilstm.layers as usize, a.bilstm.hidden as usize));
        Shape { hidden: self.hidden, layers, h }
    }

    /// The loss of one document of features [n, hidden] whose `mentions`
    /// (in order of first word then last, as the core takes them) are in
    /// the chains `chain` (a chain id per mention), and its gradients
    /// added to `grads` (shaped as `zero_grads` gives). Returns the loss
    /// and the number of mentions. A refusal leaves `grads` as they were.
    pub fn loss(
        &self,
        features: &[R],
        mentions: &[(u32, u32)],
        chain: &[u32],
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
        let (span, coref) = (self.spec.span.expect("validated"), self.spec.coref.expect("validated"));
        let (max, e, d, k) =
            (span.max_words as usize, span.width_dim as usize, coref.dim as usize, coref.antecedents as usize);
        check_mentions(mentions, n, max)?;
        if chain.len() != mentions.len() {
            return Err(invalid(format!("trainer: {} chain ids for {} mentions", chain.len(), mentions.len())));
        }
        let shape = self.shape();
        adapter::forward(shape, &self.params, features, &mut w.adapter);
        let v = adapter::output(features, &w.adapter);
        let (width, o) = (shape.width(), 6 * shape.layers);
        let cols = 2 * width + e;
        let (emb, mw, mb, pw, dist) =
            (&self.params[o], &self.params[o + 1], &self.params[o + 2], &self.params[o + 3], &self.params[o + 4]);
        let m = mentions.len();
        // Each mention's input row and vector.
        let mut xs = vec![R::ZERO; m * cols];
        let mut g = vec![R::ZERO; m * d];
        for (a, &(i, j)) in mentions.iter().enumerate() {
            let (i, j) = (i as usize, j as usize);
            let x = &mut xs[a * cols..][..cols];
            x[..width].copy_from_slice(&v[i * width..][..width]);
            x[width..2 * width].copy_from_slice(&v[j * width..][..width]);
            x[2 * width..].copy_from_slice(&emb[(j - i) * e..][..e]);
            for q in 0..d {
                g[a * d + q] = (mb[q] + dot(&mw[q * cols..][..cols], x)).tanh();
            }
        }
        let mut pg = vec![R::ZERO; m * d];
        for b in 0..m {
            for q in 0..d {
                pg[b * d + q] = dot(&pw[q * d..][..d], &g[b * d..][..d]);
            }
        }
        let mut local = self.zero_grads();
        let (mut dg, mut dpg) = (vec![R::ZERO; m * d], vec![R::ZERO; m * d]);
        let mut total = 0f64;
        let (mut s, mut all, mut gold) = (Vec::new(), Vec::new(), Vec::new());
        for a in 0..m {
            let reach = k.min(a);
            // Candidate 0 is the dummy; candidate c + 1 is mention a - 1 - c.
            s.clear();
            s.push(0.0);
            for c in 0..reach {
                let b = a - 1 - c;
                s.push((dot(&g[a * d..][..d], &pg[b * d..][..d]) + dist[bucket((a - b) as u32)]).to_f64());
            }
            let ok: Vec<bool> = (0..=reach).map(|c| c > 0 && chain[a - c] == chain[a]).collect();
            let none = !ok.iter().any(|&x| x);
            let golden: Vec<f64> = s
                .iter()
                .zip(&ok)
                .enumerate()
                .filter(|&(c, (_, &y))| y || (none && c == 0))
                .map(|(_, (&v, _))| v)
                .collect();
            total += shares(&s, &mut all) - shares(&golden, &mut gold);
            let mut gi = gold.iter();
            for c in 0..=reach {
                let p_gold = if ok[c] || (none && c == 0) { *gi.next().expect("one share each") } else { 0.0 };
                let delta = R::from_f64(all[c] - p_gold);
                if c == 0 {
                    continue;
                }
                let b = a - 1 - (c - 1);
                local[o + 4][bucket((a - b) as u32)] += delta;
                for q in 0..d {
                    dg[a * d + q] += delta * pg[b * d + q];
                    dpg[b * d + q] += delta * g[a * d + q];
                }
            }
        }
        // Through P: pg_b = P g_b.
        for b in 0..m {
            for q in 0..d {
                let dq = dpg[b * d + q];
                for r in 0..d {
                    local[o + 3][q * d + r] += dq * g[b * d + r];
                    dg[b * d + r] += dq * pw[q * d + r];
                }
            }
        }
        // Through tanh and the mention layer, to the words and widths.
        let mut dv = vec![R::ZERO; n * width];
        for (a, &(i, j)) in mentions.iter().enumerate() {
            let (i, j) = (i as usize, j as usize);
            let x = &xs[a * cols..][..cols];
            for q in 0..d {
                let gq = g[a * d + q];
                let dz = dg[a * d + q] * (R::ONE - gq * gq);
                local[o + 2][q] += dz;
                for (c, &xc) in x.iter().enumerate() {
                    local[o + 1][q * cols + c] += dz * xc;
                    let back = dz * mw[q * cols + c];
                    match c {
                        c if c < width => dv[i * width + c] += back,
                        c if c < 2 * width => dv[j * width + c - width] += back,
                        c => local[o][(j - i) * e + c - 2 * width] += back,
                    }
                }
            }
        }
        if !total.is_finite() {
            return Err(invalid("trainer: the coreference loss is not finite"));
        }
        // Checked once the adapter's share is in: its backward can
        // overflow, and nothing reaches `grads` unless all of it is finite.
        adapter::backward(shape, &self.params, features, dv, &mut local, &w.adapter, false);
        adapter::commit(grads, &local, "coreference")?;
        Ok((total, m))
    }
}

impl Ranker<f32> {
    /// The weights as the core reads them, by suffix, in `tensors` order.
    pub fn tensors(&self) -> Vec<(String, Vec<f32>)> {
        self.names.iter().cloned().zip(self.params.iter().cloned()).collect()
    }

    /// The core's ranker over these weights: the inference path.
    pub fn to_core(&self) -> Result<CorefRanker> {
        let get = |s: &str| self.names.iter().position(|n| n == s).map(|i| self.params[i].clone());
        CorefRanker::from_tensors(&self.spec, self.hidden, &get)
    }
}
