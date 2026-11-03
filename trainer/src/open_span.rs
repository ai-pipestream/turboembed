//! An open-label span typing head (`HEAD_OPEN_SPAN`) in training: optional
//! BiLSTM adapter, candidate span boundary + width projection with leaky
//! ReLU, caller label projection + direct encoder cosine similarity over
//! centered unit vectors, and per-span softmax cross-entropy against outside
//! (`O`, logit `0.0`) plus the `L` caller labels.

use turbo::head::OpenSpanMatcher;
use turbo::head::open_span::{LEAKY_SLOPE, tensors};
use turbo::manifest::{HeadKind, HeadSpec};
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::adapter::{self, Shape};
use crate::logspace::softmax;
use crate::real::{Real, dot};
use crate::rng::Rng;
use crate::shapes::{checked_shapes, fits};
use crate::span::GoldSpan;

/// Weight applied to positive gold entity spans relative to outside (`O`)
/// candidate spans in the per-span cross-entropy loss.
const POS_SPAN_WEIGHT: f64 = 4.0;

/// Trainable weights of one `HEAD_OPEN_SPAN` head.
///
/// Tensor layout in `params` (matching [`tensors`]):
/// - `0..6 * layers`: BiLSTM adapter tensors
/// - `o + 0`: `span.width` `[max_words, dim]`
/// - `o + 1`: `span.weight` `[dim, 2 * width + dim]`
/// - `o + 2`: `span.bias` `[dim]`
/// - `o + 3`: `label.weight` `[dim, hidden]`
/// - `o + 4`: `label.bias` `[dim]`
/// - `o + 5`: `out.bias` `[3]` (`[bias, start_sim_scale, end_sim_scale]`)
/// - `o + 6`: `center.weight` `[2, hidden]` (`[center_x, center_q]`)
#[derive(Debug, Clone)]
pub struct OpenSpans<R> {
    spec: HeadSpec,
    hidden: usize,
    names: Vec<String>,
    params: Vec<Vec<R>>,
}

/// The adapter's activations of one sequence, kept for backward.
#[derive(Debug, Default)]
pub struct Work<R> {
    adapter: adapter::Work<R>,
}

impl<R: Real> OpenSpans<R> {
    /// A head with seeded initial weights: Xavier-uniform matrices and
    /// width embeddings, zero biases (except LSTM forget gates at 1,
    /// `out.bias` at `[0, 3, 3]`, and `center.weight` at 0).
    pub fn new(spec: &HeadSpec, hidden: usize, rng: &mut Rng) -> Result<Self> {
        let shapes = checked_shapes(spec, hidden, tensors(spec, hidden))?;
        let mut params = Vec::with_capacity(shapes.len());
        for ((suffix, shape), n) in &shapes {
            let n = *n;
            let v = if suffix == "center.weight" {
                vec![R::ZERO; n]
            } else if shape.len() == 2 {
                let limit = (6.0 / (shape[0] + shape[1]) as f64).sqrt();
                (0..n).map(|_| R::from_f64(rng.uniform(limit))).collect()
            } else if suffix.starts_with("lstm.") {
                let h = n / 4;
                (0..n).map(|i| if (h..2 * h).contains(&i) { R::ONE } else { R::ZERO }).collect()
            } else if suffix == "out.bias" {
                vec![R::ZERO, R::from_f64(3.0), R::from_f64(3.0)]
            } else {
                vec![R::ZERO; n]
            };
            params.push(v);
        }
        Self::with_params(spec, hidden, params)
    }

    /// A head from given weights, in `tensors` order.
    pub fn with_params(spec: &HeadSpec, hidden: usize, params: Vec<Vec<R>>) -> Result<Self> {
        if spec.kind != HeadKind::OpenSpan {
            return Err(invalid(format!("trainer: head {:?} is not an open-span head", spec.name)));
        }
        let shapes = checked_shapes(spec, hidden, tensors(spec, hidden))?;
        fits(spec, &shapes, &params)?;
        let names = shapes.into_iter().map(|((s, _), _)| s).collect();
        Ok(OpenSpans { spec: spec.clone(), hidden, names, params })
    }

    /// Sets `center.weight` (`[center_x, center_q]`, each of length `hidden`).
    pub fn set_centers(&mut self, center_x: &[R], center_q: &[R]) {
        if let Some(idx) = self.names.iter().position(|n| n == "center.weight")
            && center_x.len() == self.hidden
            && center_q.len() == self.hidden
        {
            self.params[idx][..self.hidden].copy_from_slice(center_x);
            self.params[idx][self.hidden..2 * self.hidden].copy_from_slice(center_q);
        }
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
    /// before clipping. `center.weight` holds fixed empirical centers and is
    /// not stepped by AdamW.
    pub fn step(&mut self, opt: &mut AdamW<R>, grads: &mut [Vec<R>], decay: &[bool]) -> Result<f64> {
        if let Some(idx) = self.names.iter().position(|n| n == "center.weight")
            && let Some(g) = grads.get_mut(idx)
        {
            g.fill(R::ZERO);
        }
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

    /// Whether tensor i takes weight decay: projection matrices do; biases,
    /// width embeddings, and empirical centers do not.
    pub fn decays(&self) -> Vec<bool> {
        self.names
            .iter()
            .map(|n| n.ends_with(".w") || n.ends_with(".u") || n == "span.weight" || n == "label.weight")
            .collect()
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

    /// The loss of one sequence of features `[n, hidden]` and label vectors
    /// `[L, hidden]` (`L >= 1`) against its gold spans, and its gradients
    /// added to `grads` (shaped as `zero_grads` gives). Returns the loss and
    /// the number of candidate spans it sums over.
    pub fn loss(
        &self,
        features: &[R],
        label_features: &[R],
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
        if label_features.is_empty()
            || !label_features.len().is_multiple_of(self.hidden)
            || label_features.iter().any(|v| !v.to_f64().is_finite())
        {
            return Err(invalid(format!("trainer: label features are not finite rows of {}", self.hidden)));
        }
        let n = features.len() / self.hidden;
        let num_labels = label_features.len() / self.hidden;
        let max = self.max_words();
        let mut target = vec![0usize; n * max];
        for g in gold {
            let (i, j, l) = (g.first as usize, g.last as usize, g.label as usize);
            let fits = i <= j && j < n && j - i < max && l < num_labels;
            if !fits || target[i * max + (j - i)] != 0 {
                return Err(invalid(format!(
                    "trainer: gold span {g:?} of a sequence of {n} words and {num_labels} labels"
                )));
            }
            target[i * max + (j - i)] = l + 1;
        }
        let shape = self.shape();
        adapter::forward(shape, &self.params, features, &mut w.adapter);
        let v = adapter::output(features, &w.adapter);
        let (width, dim, o) = (shape.width(), self.width_dim(), 6 * shape.layers);
        let cols = 2 * width + dim;
        let (emb, sw, sb) = (&self.params[o], &self.params[o + 1], &self.params[o + 2]);
        let (lw, lb, ob, cw) = (&self.params[o + 3], &self.params[o + 4], &self.params[o + 5], &self.params[o + 6]);
        let (cx, cq) = (&cw[..self.hidden], &cw[self.hidden..2 * self.hidden]);

        let mut x_tilde = Vec::with_capacity(n * self.hidden);
        let mut x_inv_r = Vec::with_capacity(n);
        for i in 0..n {
            let (t, ir) = center_norm(&features[i * self.hidden..][..self.hidden], cx);
            x_tilde.extend(t);
            x_inv_r.push(ir);
        }
        let mut q_tilde = Vec::with_capacity(num_labels * self.hidden);
        let mut q_inv_r = Vec::with_capacity(num_labels);
        for l in 0..num_labels {
            let (t, ir) = center_norm(&label_features[l * self.hidden..][..self.hidden], cq);
            q_tilde.extend(t);
            q_inv_r.push(ir);
        }

        let mut sim = vec![R::ZERO; n * num_labels];
        for i in 0..n {
            let xt = &x_tilde[i * self.hidden..][..self.hidden];
            for l in 0..num_labels {
                let qt = &q_tilde[l * self.hidden..][..self.hidden];
                sim[i * num_labels + l] = dot(xt, qt);
            }
        }

        let mut p = vec![R::ZERO; num_labels * dim];
        for l in 0..num_labels {
            let qt = &q_tilde[l * self.hidden..][..self.hidden];
            for d in 0..dim {
                p[l * dim + d] = lb[d] + dot(&lw[d * self.hidden..][..self.hidden], qt);
            }
        }

        let mut local = self.zero_grads();
        let mut dv = vec![R::ZERO; n * width];
        let mut dp = vec![R::ZERO; num_labels * dim];
        let mut d_sim = vec![R::ZERO; n * num_labels];
        let (mut total, mut count) = (0f64, 0usize);
        let mut x = vec![R::ZERO; cols];
        let mut h = vec![R::ZERO; dim];
        let mut slopes = vec![R::ZERO; dim];
        let mut dh = vec![R::ZERO; dim];
        let mut row = vec![0f64; num_labels + 1];
        let neg_slope = R::from_f64(LEAKY_SLOPE as f64);

        for i in 0..n {
            let sim_i = &sim[i * num_labels..][..num_labels];
            for j in i..n.min(i + max) {
                let sim_j = &sim[j * num_labels..][..num_labels];
                let k = j - i;
                x[..width].copy_from_slice(&v[i * width..][..width]);
                x[width..2 * width].copy_from_slice(&v[j * width..][..width]);
                x[2 * width..].copy_from_slice(&emb[k * dim..][..dim]);
                for d in 0..dim {
                    let pre = sb[d] + dot(&sw[d * cols..][..cols], &x);
                    let s = if pre.to_f64() >= 0.0 { R::ONE } else { neg_slope };
                    slopes[d] = s;
                    h[d] = s * pre;
                }
                row[0] = 0.0;
                for l in 0..num_labels {
                    let base = ob[0] + ob[1] * sim_i[l] + ob[2] * sim_j[l];
                    row[l + 1] = (base + dot(&h, &p[l * dim..][..dim])).to_f64();
                }
                let g = target[i * max + k];
                let w_span = if g != 0 { POS_SPAN_WEIGHT } else { 1.0 };
                let gold_score = row[g];
                let (m, ln_z) = softmax(&mut row);
                total += w_span * (ln_z - (gold_score - m));
                count += 1;
                row[g] -= 1.0;

                dh.fill(R::ZERO);
                for l in 0..num_labels {
                    let dz = R::from_f64(w_span * row[l + 1]);
                    local[o + 5][0] += dz;
                    local[o + 5][1] += dz * sim_i[l];
                    local[o + 5][2] += dz * sim_j[l];
                    d_sim[i * num_labels + l] += dz * ob[1];
                    d_sim[j * num_labels + l] += dz * ob[2];
                    let pl = &p[l * dim..][..dim];
                    let dpl = &mut dp[l * dim..][..dim];
                    for d in 0..dim {
                        dpl[d] += dz * h[d];
                        dh[d] += dz * pl[d];
                    }
                }
                for d in 0..dim {
                    let dpre = dh[d] * slopes[d];
                    local[o + 2][d] += dpre;
                    let sw_row = &sw[d * cols..][..cols];
                    for (c, &xc) in x.iter().enumerate() {
                        local[o + 1][d * cols + c] += dpre * xc;
                        let back = dpre * sw_row[c];
                        match c {
                            c if c < width => dv[i * width + c] += back,
                            c if c < 2 * width => dv[j * width + c - width] += back,
                            c => local[o][k * dim + c - 2 * width] += back,
                        }
                    }
                }
            }
        }
        let mut d_q_tilde = vec![R::ZERO; num_labels * self.hidden];
        for l in 0..num_labels {
            let qt = &q_tilde[l * self.hidden..][..self.hidden];
            let dqt = &mut d_q_tilde[l * self.hidden..][..self.hidden];
            for d in 0..dim {
                let dpl = dp[l * dim + d];
                local[o + 4][d] += dpl;
                let lw_row = &lw[d * self.hidden..][..self.hidden];
                let dlw = &mut local[o + 3][d * self.hidden..][..self.hidden];
                for col in 0..self.hidden {
                    dlw[col] += dpl * qt[col];
                    dqt[col] += dpl * lw_row[col];
                }
            }
        }
        let mut d_x_tilde = vec![R::ZERO; n * self.hidden];
        for i in 0..n {
            let xt = &x_tilde[i * self.hidden..][..self.hidden];
            let dxt = &mut d_x_tilde[i * self.hidden..][..self.hidden];
            for l in 0..num_labels {
                let ds = d_sim[i * num_labels + l];
                let qt = &q_tilde[l * self.hidden..][..self.hidden];
                let dqt = &mut d_q_tilde[l * self.hidden..][..self.hidden];
                for col in 0..self.hidden {
                    dxt[col] += ds * qt[col];
                    dqt[col] += ds * xt[col];
                }
            }
            center_norm_back(dxt, xt, x_inv_r[i], &mut local[o + 6][..self.hidden]);
        }
        for l in 0..num_labels {
            let qt = &q_tilde[l * self.hidden..][..self.hidden];
            let dqt = &d_q_tilde[l * self.hidden..][..self.hidden];
            center_norm_back(dqt, qt, q_inv_r[l], &mut local[o + 6][self.hidden..2 * self.hidden]);
        }
        if !total.is_finite() {
            return Err(invalid("trainer: the open-span loss is not finite"));
        }
        adapter::backward(shape, &self.params, features, dv, &mut local, &w.adapter, false);
        adapter::commit(grads, &local, "open_span")?;
        Ok((total, count))
    }
}

impl OpenSpans<f32> {
    /// The weights as the core reads them, by suffix, in `tensors` order.
    pub fn tensors(&self) -> Vec<(String, Vec<f32>)> {
        self.names.iter().cloned().zip(self.params.iter().cloned()).collect()
    }

    /// The core's open-span matcher over these weights: the inference path.
    pub fn to_core(&self) -> Result<OpenSpanMatcher> {
        let get = |s: &str| self.names.iter().position(|n| n == s).map(|i| self.params[i].clone());
        OpenSpanMatcher::from_tensors(&self.spec, self.hidden, &get)
    }
}

fn center_norm<R: Real>(src: &[R], center: &[R]) -> (Vec<R>, R) {
    let norm = src.iter().map(|v| v.to_f64() * v.to_f64()).sum::<f64>().sqrt().max(1e-6);
    let inv = R::from_f64(1.0 / norm);
    let u: Vec<R> = src.iter().zip(center).map(|(&x, &c)| x * inv - c).collect();
    let r = (u.iter().map(|v| v.to_f64() * v.to_f64()).sum::<f64>() + 1e-6).sqrt();
    let inv_r = R::from_f64(1.0 / r);
    (u.into_iter().map(|v| v * inv_r).collect(), inv_r)
}

fn center_norm_back<R: Real>(d_tilde: &[R], tilde: &[R], inv_r: R, d_center: &mut [R]) {
    let proj = dot(d_tilde, tilde);
    for ((dc, &dt), &t) in d_center.iter_mut().zip(d_tilde).zip(tilde) {
        *dc -= (dt - proj * t) * inv_r;
    }
}
