//! A biaffine parser head in training: the weights of
//! `turbo::head::parser::tensors`, in that order, with its loss and
//! gradients on one sentence of per-word encoder features.
//!
//! The loss is the parser's two cross-entropies summed over the words:
//! each word's head among every node but itself, and its relation under
//! its gold head (Dozat and Manning; SuPar trains the same pair). The
//! forward is the core's (`turbo::head::parser`): root vector as node 0,
//! LeakyReLU(0.1) MLPs, arc biaffine with a dependent-side bias,
//! relation biaffine with a bias on both sides.

use turbo::head::BiaffineParser;
use turbo::head::parser::{LEAKY_SLOPE, tensors};
use turbo::manifest::{HeadKind, HeadSpec};
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::adapter::{self, Shape};
use crate::linear;
use crate::real::Real;
use crate::rng::Rng;
use crate::shapes::{checked_shapes, fits};

/// The weights of one head, `params[i]` named `names[i]`.
#[derive(Debug, Clone)]
pub struct Parser<R> {
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
    /// [n + 1, width]: the root, then the words.
    nodes: Vec<R>,
    /// The four MLP outputs [n + 1, A or R]: arc dep, arc head, rel dep, rel head.
    views: [Vec<R>; 4],
    /// U head_h [n + 1, A + 1].
    projected: Vec<R>,
    arcs: Vec<R>,
    n: usize,
}

impl<R: Real> Work<R> {
    /// The arc scores [n, n + 1] of the last forward, laid out as the
    /// core's `BiaffineParser::arcs` gives them.
    pub fn arcs(&self) -> &[R] {
        &self.arcs
    }
}

fn leaky<R: Real>(v: &mut [R]) {
    let slope = R::from_f64(LEAKY_SLOPE as f64);
    v.iter_mut().filter(|x| **x < R::ZERO).for_each(|x| *x *= slope);
}

impl<R: Real> Parser<R> {
    /// A head with seeded initial weights: Xavier-uniform matrices and
    /// root, zero biases but for each LSTM forget gate's (1), zero
    /// biaffine tensors (as SuPar starts them).
    pub fn new(spec: &HeadSpec, hidden: usize, rng: &mut Rng) -> Result<Self> {
        let shapes = checked_shapes(spec, hidden, tensors(spec, hidden))?;
        let mut params = Vec::with_capacity(shapes.len());
        for ((suffix, shape), n) in &shapes {
            let n = *n;
            let v = if suffix == "root" || (shape.len() == 2 && !suffix.ends_with("arc.weight")) {
                let fan = if shape.len() == 2 { shape[0] + shape[1] } else { shape[0] + 1 };
                let limit = (6.0 / fan as f64).sqrt();
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
        if spec.kind != HeadKind::BiaffineParser {
            return Err(invalid(format!("trainer: head {:?} is not a biaffine parser", spec.name)));
        }
        fits(spec, &shapes, &params)?;
        let names = shapes.into_iter().map(|((s, _), _)| s).collect();
        Ok(Parser { spec: spec.clone(), hidden, names, params })
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

    /// Whether tensor i takes weight decay: matrices do; biases, the
    /// root and the biaffine tensors do not.
    pub fn decays(&self) -> Vec<bool> {
        self.names
            .iter()
            .map(|n| {
                n.ends_with(".w")
                    || n.ends_with(".u")
                    || (n.ends_with(".weight") && !n.starts_with("arc.") && !n.starts_with("rel."))
            })
            .collect()
    }

    pub fn zero_grads(&self) -> Vec<Vec<R>> {
        self.params.iter().map(|p| vec![R::ZERO; p.len()]).collect()
    }

    fn shape(&self) -> Shape {
        let (layers, h) = self.spec.adapter.map_or((0, 0), |a| (a.bilstm.layers as usize, a.bilstm.hidden as usize));
        Shape { hidden: self.hidden, layers, h }
    }

    /// (A, R): the arc and relation MLP widths.
    fn widths(&self) -> (usize, usize) {
        let b = self.spec.biaffine.expect("validated");
        (b.arc as usize, b.rel as usize)
    }

    /// The arc scores [n, n + 1] of features [n, hidden], kept in `w`.
    pub fn forward(&self, features: &[R], w: &mut Work<R>) -> Result<usize> {
        if !features.len().is_multiple_of(self.hidden) {
            return Err(invalid(format!("trainer: {} features are not rows of {}", features.len(), self.hidden)));
        }
        let n = features.len() / self.hidden;
        let shape = self.shape();
        adapter::forward(shape, &self.params, features, &mut w.adapter);
        let width = shape.width();
        let o = 6 * shape.layers;
        w.nodes.clear();
        w.nodes.extend_from_slice(&self.params[o]);
        w.nodes.extend_from_slice(adapter::output(features, &w.adapter));
        let (a, r) = self.widths();
        let rows = n + 1;
        for (k, out) in [a, a, r, r].into_iter().enumerate() {
            let v = &mut w.views[k];
            v.clear();
            v.resize(rows * out, R::ZERO);
            linear::forward(&self.params[o + 1 + 2 * k], &self.params[o + 2 + 2 * k], &w.nodes, width, out, v);
            leaky(v);
        }
        let u = &self.params[o + 9];
        w.projected.clear();
        w.projected.resize(rows * (a + 1), R::ZERO);
        for h in 0..rows {
            let y = &w.views[1][h * a..][..a];
            for i in 0..=a {
                let mut s = R::ZERO;
                for (p, q) in u[i * a..][..a].iter().zip(y) {
                    s += *p * *q;
                }
                w.projected[h * (a + 1) + i] = s;
            }
        }
        w.arcs.clear();
        w.arcs.resize(n * rows, R::ZERO);
        for d in 1..rows {
            let x = &w.views[0][d * a..][..a];
            for h in 0..rows {
                let p = &w.projected[h * (a + 1)..][..a + 1];
                let mut s = p[a];
                for (u, v) in x.iter().zip(p) {
                    s += *u * *v;
                }
                w.arcs[(d - 1) * rows + h] = s;
            }
        }
        w.n = n;
        Ok(n)
    }

    /// The loss of one sentence against its gold heads (0 the root) and
    /// relations, and its gradients added to `grads` (shaped as
    /// `zero_grads` gives); gradients of any other shape are refused
    /// before anything runs.
    pub fn loss(
        &self,
        features: &[R],
        heads: &[u32],
        rels: &[u32],
        grads: &mut [Vec<R>],
        w: &mut Work<R>,
    ) -> Result<f64> {
        if grads.len() != self.params.len() || grads.iter().zip(&self.params).any(|(g, p)| g.len() != p.len()) {
            return Err(invalid(format!("trainer: gradients do not match the tensors of head {:?}", self.spec.name)));
        }
        if features.iter().any(|v| !v.to_f64().is_finite()) {
            return Err(invalid("trainer: a parser feature is not finite"));
        }
        let n = self.forward(features, w)?;
        let labels = self.spec.labels.len();
        if heads.len() != n || rels.len() != n {
            return Err(invalid(format!("trainer: {} heads and {} relations for {n} words", heads.len(), rels.len())));
        }
        if heads.iter().enumerate().any(|(i, &h)| h as usize > n || h as usize == i + 1)
            || rels.iter().any(|&l| l as usize >= labels)
        {
            return Err(invalid("trainer: a gold head out of range or on itself, or a relation past the labels"));
        }
        let (a, r) = self.widths();
        let rows = n + 1;
        let shape = self.shape();
        let (width, o) = (shape.width(), 6 * shape.layers);
        // Gradients gather here and reach `grads` only when every one,
        // and the loss, is finite, so a refused sentence leaves them as
        // they were.
        let mut local = self.zero_grads();
        let mut total = 0f64;
        // Arc cross-entropy over every head but the word itself, centred
        // on the largest score (as `xent::loss`), so a large common offset
        // does not swallow the normaliser.
        let mut darcs = vec![R::ZERO; n * rows];
        for d in 1..rows {
            let row = &w.arcs[(d - 1) * rows..][..rows];
            let m = (0..rows).filter(|&h| h != d).map(|h| row[h].to_f64()).fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = (0..rows).filter(|&h| h != d).map(|h| (row[h].to_f64() - m).exp()).sum();
            let g = heads[d - 1] as usize;
            total += z.ln() - (row[g].to_f64() - m);
            for h in (0..rows).filter(|&h| h != d) {
                let p = (row[h].to_f64() - m).exp() / z;
                darcs[(d - 1) * rows + h] = R::from_f64(p - if h == g { 1.0 } else { 0.0 });
            }
        }
        let mut dviews: [Vec<R>; 4] = std::array::from_fn(|k| vec![R::ZERO; w.views[k].len()]);
        let mut dproj = vec![R::ZERO; rows * (a + 1)];
        for d in 1..rows {
            for h in 0..rows {
                let g = darcs[(d - 1) * rows + h];
                let p = &w.projected[h * (a + 1)..][..a + 1];
                for i in 0..a {
                    dviews[0][d * a + i] += g * p[i];
                    dproj[h * (a + 1) + i] += g * w.views[0][d * a + i];
                }
                dproj[h * (a + 1) + a] += g;
            }
        }
        {
            let u = &self.params[o + 9];
            let du = &mut local[o + 9];
            for h in 0..rows {
                for i in 0..=a {
                    let g = dproj[h * (a + 1) + i];
                    for j in 0..a {
                        du[i * a + j] += g * w.views[1][h * a + j];
                        dviews[1][h * a + j] += g * u[i * a + j];
                    }
                }
            }
        }
        // Relation cross-entropy under each gold head.
        let k = r + 1;
        let wr = &self.params[o + 10];
        let one = |v: &[R], i: usize| if i < r { v[i] } else { R::ONE };
        let mut scores = vec![R::ZERO; labels];
        for d in 1..rows {
            let hd = heads[d - 1] as usize;
            let x = &w.views[2][d * r..][..r];
            let y = &w.views[3][hd * r..][..r];
            for (l, s) in scores.iter_mut().enumerate() {
                let m = &wr[l * k * k..][..k * k];
                let mut t = R::ZERO;
                for i in 0..k {
                    let mut inner = R::ZERO;
                    for j in 0..k {
                        inner += m[i * k + j] * one(y, j);
                    }
                    t += one(x, i) * inner;
                }
                *s = t;
            }
            let mx = scores.iter().map(|v| v.to_f64()).fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = scores.iter().map(|v| (v.to_f64() - mx).exp()).sum();
            let g = rels[d - 1] as usize;
            total += z.ln() - (scores[g].to_f64() - mx);
            let dw = &mut local[o + 10];
            for (l, s) in scores.iter().enumerate() {
                let gl = R::from_f64((s.to_f64() - mx).exp() / z - if l == g { 1.0 } else { 0.0 });
                let m = &wr[l * k * k..][..k * k];
                for i in 0..k {
                    let xi = one(x, i);
                    for j in 0..k {
                        let yj = one(y, j);
                        dw[l * k * k + i * k + j] += gl * xi * yj;
                        if i < r {
                            dviews[2][d * r + i] += gl * m[i * k + j] * yj;
                        }
                        if j < r {
                            dviews[3][hd * r + j] += gl * xi * m[i * k + j];
                        }
                    }
                }
            }
        }
        // Through LeakyReLU and the four MLPs to the nodes.
        let slope = R::from_f64(LEAKY_SLOPE as f64);
        let mut dnodes = vec![R::ZERO; rows * width];
        for (kk, out) in [a, a, r, r].into_iter().enumerate() {
            for (g, v) in dviews[kk].iter_mut().zip(&w.views[kk]) {
                if *v < R::ZERO {
                    *g *= slope;
                }
            }
            let (wi, bi) = (o + 1 + 2 * kk, o + 2 + 2 * kk);
            let [dw, db] = local.get_disjoint_mut([wi, bi]).expect("distinct");
            linear::backward(&self.params[wi], &w.nodes, &dviews[kk], width, out, dw, db, Some(&mut dnodes));
        }
        for (g, d) in local[o].iter_mut().zip(&dnodes[..width]) {
            *g += *d;
        }
        if shape.layers > 0 {
            adapter::backward(shape, &self.params, features, dnodes[width..].to_vec(), &mut local, &w.adapter, false);
        }
        if !total.is_finite() || !local.iter().flatten().all(|v| v.to_f64().is_finite()) {
            return Err(invalid("trainer: the parser's loss or a gradient is not finite"));
        }
        for (g, l) in grads.iter_mut().zip(local) {
            for (a, b) in g.iter_mut().zip(l) {
                *a += b;
            }
        }
        Ok(total)
    }
}

impl Parser<f32> {
    /// The weights as the core reads them, by suffix, in `tensors` order.
    pub fn tensors(&self) -> Vec<(String, Vec<f32>)> {
        self.names.iter().cloned().zip(self.params.iter().cloned()).collect()
    }

    /// The core's parser over these weights: the inference path itself.
    pub fn to_core(&self) -> Result<BiaffineParser> {
        let get = |s: &str| self.names.iter().position(|n| n == s).map(|i| self.params[i].clone());
        BiaffineParser::from_tensors(&self.spec, self.hidden, &get)
    }
}
