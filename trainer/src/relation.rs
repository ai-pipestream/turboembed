//! A relation classifier head in training: the weights of
//! `turbo::head::relation::tensors`, in that order, with its loss and
//! gradients on one sequence of per-word encoder features and its
//! mentions.
//!
//! Every ordered pair of distinct entities adds its adaptive threshold
//! loss, `none` the threshold, so a pair may hold several relations. The
//! forward is the core's (`turbo::head::relation`): each entity's mentions
//! pooled by logsumexp, its type embedding joined, through the head and
//! tail LeakyReLU(0.1) MLPs, then the biaffine with a bias on both sides.

use turbo::head::RelationClassifier;
use turbo::head::parser::LEAKY_SLOPE;
use turbo::head::relation::{Mention, check, tensors};
use turbo::manifest::{HeadKind, HeadSpec, NO_RELATION};
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::adapter::{self, Shape};
use crate::linear;
use crate::logspace::logsumexp;
use crate::real::{Real, dot};
use crate::rng::Rng;
use crate::shapes::{checked_shapes, fits};

/// A gold relation: entity `head` to entity `tail` of its sequence, and
/// its label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoldRelation {
    pub head: u32,
    pub tail: u32,
    pub label: u32,
}

/// The weights of one head, `params[i]` named `names[i]`.
#[derive(Debug, Clone)]
pub struct Relations<R> {
    spec: HeadSpec,
    hidden: usize,
    names: Vec<String>,
    /// Shaped as `tensors` says at construction and never reshaped.
    params: Vec<Vec<R>>,
    none: u32,
}

/// The activations of one sequence, kept for backward.
#[derive(Debug, Default)]
pub struct Work<R> {
    adapter: adapter::Work<R>,
    /// [e, 2 * width + type_dim]: each entity's joined vector.
    mentions: Vec<R>,
    /// Each mention's softmax weights in its entity's pooling.
    weights: Vec<R>,
    /// The head and tail views [m, mlp], after LeakyReLU.
    views: [Vec<R>; 2],
    /// W_l [tail_b; 1] for every tail b and label l, [m, labels, mlp + 1].
    projected: Vec<R>,
}

impl<R: Real> Relations<R> {
    /// A head with seeded initial weights: Xavier-uniform matrices and
    /// type embeddings, zero biases but for each LSTM forget gate's (1),
    /// and a zero biaffine tensor (as the parser starts its own).
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
        if spec.kind != HeadKind::RelationClassifier {
            return Err(invalid(format!("trainer: head {:?} is not a relation classifier", spec.name)));
        }
        let shapes = checked_shapes(spec, hidden, tensors(spec, hidden))?;
        fits(spec, &shapes, &params)?;
        let none = spec.labels.iter().position(|l| l == NO_RELATION).ok_or_else(|| invalid("trainer: no \"none\""))?;
        let names = shapes.into_iter().map(|((s, _), _)| s).collect();
        Ok(Relations { spec: spec.clone(), hidden, names, params, none: none as u32 })
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

    /// Whether tensor i takes weight decay: matrices do; biases, the type
    /// embeddings and the biaffine tensor do not.
    pub fn decays(&self) -> Vec<bool> {
        let decays = |n: &str| n.ends_with(".w") || n.ends_with(".u") || n == "head.weight" || n == "tail.weight";
        self.names.iter().map(|n| decays(n)).collect()
    }

    pub fn zero_grads(&self) -> Vec<Vec<R>> {
        self.params.iter().map(|p| vec![R::ZERO; p.len()]).collect()
    }

    fn shape(&self) -> Shape {
        let (layers, h) = self.spec.adapter.map_or((0, 0), |a| (a.bilstm.layers as usize, a.bilstm.hidden as usize));
        Shape { hidden: self.hidden, layers, h }
    }

    /// (mlp, entity types, type_dim).
    fn widths(&self) -> (usize, usize, usize) {
        let r = self.spec.relation.as_ref().expect("validated");
        (r.mlp as usize, r.entity_labels.len(), r.type_dim as usize)
    }

    /// The loss of one sequence of features [n, hidden] with its mentions
    /// and entity types (as `turbo::head::relation` takes them) against
    /// its gold relations, and its gradients added to `grads` (shaped as
    /// `zero_grads` gives). Returns the loss and the number of entity
    /// pairs it sums over.
    ///
    /// Each pair's loss is the adaptive threshold loss with `none` as the
    /// threshold (Zhou et al. 2021): each gold label against the others
    /// and `none`, plus `none` against every label the pair does not hold.
    /// Refused: what the core refuses of the mentions and types; a gold
    /// relation on entities out of range or on itself, labelled `none` or
    /// past the labels, or given twice; gradients of another shape. A
    /// refusal leaves `grads` as they were.
    pub fn loss(
        &self,
        features: &[R],
        mentions: &[Mention],
        types: &[u32],
        gold: &[GoldRelation],
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
        let (mlp, kinds, dim) = self.widths();
        let labels = self.spec.labels.len();
        check(mentions, types, kinds, n).map_err(|e| invalid(format!("trainer: {}", e.message)))?;
        let m = types.len();
        let pairs = RelationClassifier::pair_count(m).ok_or_else(|| invalid(format!("trainer: {m} entities")))?;
        let none = self.none as usize;
        let mut target = vec![false; m * m * labels];
        for g in gold {
            let (a, b, l) = (g.head as usize, g.tail as usize, g.label as usize);
            let fits = a < m && b < m && a != b && l < labels && l != none;
            if !fits || std::mem::replace(&mut target[(a * m + b) * labels + l], true) {
                return Err(invalid(format!("trainer: gold relation {g:?} of {m} entities")));
            }
        }
        if m < 2 {
            return Ok((0.0, 0));
        }
        let shape = self.shape();
        adapter::forward(shape, &self.params, features, &mut w.adapter);
        let v = adapter::output(features, &w.adapter);
        let (width, o) = (shape.width(), 6 * shape.layers);
        let ti = o;
        let hw = o + usize::from(kinds > 0);
        let (cols, k) = (2 * width + dim, mlp + 1);
        pool(v, width, mentions, m, cols, &mut w.mentions, &mut w.weights);
        if dim > 0 {
            for (row, &t) in w.mentions.chunks_exact_mut(cols).zip(types) {
                row[2 * width..].copy_from_slice(&self.params[ti][t as usize * dim..][..dim]);
            }
        }
        let slope = R::from_f64(LEAKY_SLOPE as f64);
        for (s, view) in w.views.iter_mut().enumerate() {
            view.clear();
            view.resize(m * mlp, R::ZERO);
            linear::forward(&self.params[hw + 2 * s], &self.params[hw + 2 * s + 1], &w.mentions, cols, mlp, view);
            view.iter_mut().filter(|x| **x < R::ZERO).for_each(|x| *x *= slope);
        }
        let wr = &self.params[hw + 4];
        let one = |t: &[R], j: usize| if j < mlp { t[j] } else { R::ONE };
        w.projected.clear();
        w.projected.resize(m * labels * k, R::ZERO);
        for b in 0..m {
            let y = &w.views[1][b * mlp..][..mlp];
            for (r, p) in w.projected[b * labels * k..][..labels * k].iter_mut().enumerate() {
                *p = dot(&wr[r * k..][..mlp], y) + wr[r * k + mlp];
            }
        }
        // Gradients gather here and reach `grads` only when all of them,
        // and the loss, are finite.
        let mut local = self.zero_grads();
        let mut dviews: [Vec<R>; 2] = [vec![R::ZERO; m * mlp], vec![R::ZERO; m * mlp]];
        let mut dproj = vec![R::ZERO; m * labels * k];
        let (mut total, mut row, mut d) = (0f64, vec![0f64; labels], vec![0f64; labels]);
        for a in 0..m {
            let x = &w.views[0][a * mlp..][..mlp];
            for b in (0..m).filter(|&b| b != a) {
                let p = &w.projected[b * labels * k..][..labels * k];
                for (l, r) in row.iter_mut().enumerate() {
                    *r = (dot(x, &p[l * k..][..mlp]) + p[l * k + mlp]).to_f64();
                }
                let held = &target[(a * m + b) * labels..][..labels];
                total += threshold_loss(&row, held, none, &mut d);
                for (l, &dl) in d.iter().enumerate().filter(|(_, dl)| **dl != 0.0) {
                    let dl = R::from_f64(dl);
                    for i in 0..k {
                        dproj[(b * labels + l) * k + i] += dl * one(x, i);
                        if i < mlp {
                            dviews[0][a * mlp + i] += dl * p[l * k + i];
                        }
                    }
                }
            }
        }
        if !total.is_finite() {
            return Err(invalid("trainer: the relation loss is not finite"));
        }
        for b in 0..m {
            let y = &w.views[1][b * mlp..][..mlp];
            for r in 0..labels * k {
                let dr = dproj[b * labels * k + r];
                for j in 0..k {
                    local[hw + 4][r * k + j] += dr * one(y, j);
                    if j < mlp {
                        dviews[1][b * mlp + j] += dr * wr[r * k + j];
                    }
                }
            }
        }
        let mut dentities = vec![R::ZERO; m * cols];
        for (s, dview) in dviews.iter_mut().enumerate() {
            dview.iter_mut().zip(&w.views[s]).filter(|(_, v)| **v < R::ZERO).for_each(|(g, _)| *g *= slope);
            let (wi, bi) = (hw + 2 * s, hw + 2 * s + 1);
            let [dw, db] = local.get_disjoint_mut([wi, bi]).expect("distinct");
            linear::backward(&self.params[wi], &w.mentions, dview, cols, mlp, dw, db, Some(&mut dentities));
        }
        if dim > 0 {
            for (de, &t) in dentities.chunks_exact(cols).zip(types) {
                let row = &mut local[ti][t as usize * dim..][..dim];
                row.iter_mut().zip(&de[2 * width..]).for_each(|(a, b)| *a += *b);
            }
        }
        // Each mention's share of its entity's logsumexp is its softmax
        // weight among the entity's mentions.
        let mut dv = vec![R::ZERO; n * width];
        for (mi, x) in mentions.iter().enumerate() {
            let de = &dentities[x.entity as usize * cols..][..2 * width];
            for (side, word) in [x.first, x.last].into_iter().enumerate() {
                let (wt, at) = (&w.weights[(mi * 2 + side) * width..][..width], word as usize * width);
                for j in 0..width {
                    dv[at + j] += de[side * width + j] * wt[j];
                }
            }
        }
        // Checked once the adapter's share is in: its backward can
        // overflow, and nothing reaches `grads` unless all of it is finite.
        adapter::backward(shape, &self.params, features, dv, &mut local, &w.adapter, false);
        adapter::commit(grads, &local, "relation")?;
        Ok((total, pairs))
    }
}

/// The adaptive threshold loss of one pair's scores `row` with its gold
/// labels `held` and threshold label `none`, and its gradient by score
/// into `d`: each held label's cross-entropy among the held labels and
/// `none`, then `none`'s among itself and every label not held.
fn threshold_loss(row: &[f64], held: &[bool], none: usize, d: &mut [f64]) -> f64 {
    d.fill(0.0);
    let (mut total, positives) = (0f64, held.iter().filter(|&&h| h).count());
    if positives > 0 {
        let lse = logsumexp(row.iter().enumerate().filter(|&(l, _)| held[l] || l == none).map(|(_, &v)| v));
        for (l, &v) in row.iter().enumerate().filter(|&(l, _)| held[l] || l == none) {
            if held[l] {
                total += lse - v;
                d[l] -= 1.0;
            }
            d[l] += positives as f64 * (v - lse).exp();
        }
    }
    let lse = logsumexp(row.iter().enumerate().filter(|&(l, _)| !held[l]).map(|(_, &v)| v));
    total += lse - row[none];
    for (l, &v) in row.iter().enumerate().filter(|&(l, _)| !held[l]) {
        d[l] += (v - lse).exp() - if l == none { 1.0 } else { 0.0 };
    }
    total
}

/// The entities' joined vectors [e, cols] into `out` (the type part left
/// zero) as the core pools them, and each mention's softmax weights
/// among its entity's mentions, [mentions, 2, width], into `weights`.
fn pool<R: Real>(
    v: &[R],
    width: usize,
    mentions: &[Mention],
    e: usize,
    cols: usize,
    out: &mut Vec<R>,
    weights: &mut Vec<R>,
) {
    let mut max = vec![f64::NEG_INFINITY; e * 2 * width];
    let value = |m: &Mention, side: usize, j: usize| v[[m.first, m.last][side] as usize * width + j].to_f64();
    for m in mentions {
        for side in 0..2 {
            for j in 0..width {
                let k = (m.entity as usize * 2 + side) * width + j;
                max[k] = max[k].max(value(m, side, j));
            }
        }
    }
    let mut sum = vec![0f64; e * 2 * width];
    for m in mentions {
        for side in 0..2 {
            for j in 0..width {
                let k = (m.entity as usize * 2 + side) * width + j;
                sum[k] += (value(m, side, j) - max[k]).exp();
            }
        }
    }
    out.clear();
    out.resize(e * cols, R::ZERO);
    for i in 0..e {
        for j in 0..2 * width {
            let k = i * 2 * width + j;
            out[i * cols + j] = R::from_f64(max[k] + sum[k].ln());
        }
    }
    weights.clear();
    for m in mentions {
        for side in 0..2 {
            for j in 0..width {
                let k = (m.entity as usize * 2 + side) * width + j;
                weights.push(R::from_f64((value(m, side, j) - max[k]).exp() / sum[k]));
            }
        }
    }
}

impl Relations<f32> {
    /// The weights as the core reads them, by suffix, in `tensors` order.
    pub fn tensors(&self) -> Vec<(String, Vec<f32>)> {
        self.names.iter().cloned().zip(self.params.iter().cloned()).collect()
    }

    /// The core's relation classifier over these weights: the inference
    /// path.
    pub fn to_core(&self) -> Result<RelationClassifier> {
        let get = |s: &str| self.names.iter().position(|n| n == s).map(|i| self.params[i].clone());
        RelationClassifier::from_tensors(&self.spec, self.hidden, &get)
    }
}
