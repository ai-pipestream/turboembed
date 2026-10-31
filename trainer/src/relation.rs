//! A relation classifier head in training: the weights of
//! `turbo::head::relation::tensors`, in that order, with its loss and
//! gradients on one sequence of per-word encoder features and its
//! mentions.
//!
//! Every ordered pair of distinct mentions is one softmax over the
//! labels, its gold label the pair's own when it is a gold relation and
//! `none` otherwise; the loss sums them all. The forward is the core's
//! (`turbo::head::relation`): each mention's first and last word vectors
//! and type embedding, joined, through the head and tail LeakyReLU(0.1)
//! MLPs, then the biaffine with a bias on both sides.

use turbo::head::RelationClassifier;
use turbo::head::parser::LEAKY_SLOPE;
use turbo::head::relation::{Mention, tensors};
use turbo::manifest::{HeadKind, HeadSpec, NO_RELATION};
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::adapter::{self, Shape};
use crate::linear;
use crate::logspace::softmax;
use crate::real::{Real, dot};
use crate::rng::Rng;
use crate::shapes::{checked_shapes, fits};

/// A gold relation: mention `head` to mention `tail` of its sequence, and
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
    /// [m, 2 * width + type_dim]: each mention's joined vector.
    mentions: Vec<R>,
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
    /// against its gold relations, and its gradients added to `grads`
    /// (shaped as `zero_grads` gives). Returns the loss and the number of
    /// pairs it sums over. A mention the core would refuse, a gold
    /// relation on a mention pair that is not one (out of range or a
    /// mention on itself), labelled `none` or past the labels, or given
    /// twice, and gradients of another shape, are refused; a refusal
    /// leaves `grads` as they were.
    pub fn loss(
        &self,
        features: &[R],
        mentions: &[Mention],
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
        let (mlp, types, dim) = self.widths();
        let labels = self.spec.labels.len();
        for (i, x) in mentions.iter().enumerate() {
            let typed = if types == 0 { x.entity == 0 } else { (x.entity as usize) < types };
            if x.first > x.last || x.last as usize >= n || !typed {
                return Err(invalid(format!("trainer: mention {i} {x:?} of a sequence of {n} words")));
            }
        }
        let m = mentions.len();
        let pairs = RelationClassifier::pair_count(m).ok_or_else(|| invalid(format!("trainer: {m} mentions")))?;
        let mut target = vec![self.none; m * m];
        for g in gold {
            let (a, b) = (g.head as usize, g.tail as usize);
            let fits = a < m && b < m && a != b && (g.label as usize) < labels && g.label != self.none;
            if !fits || target[a * m + b] != self.none {
                return Err(invalid(format!("trainer: gold relation {g:?} of {m} mentions")));
            }
            target[a * m + b] = g.label;
        }
        if m < 2 {
            return Ok((0.0, 0));
        }
        let shape = self.shape();
        adapter::forward(shape, &self.params, features, &mut w.adapter);
        let v = adapter::output(features, &w.adapter);
        let (width, o) = (shape.width(), 6 * shape.layers);
        let ti = o;
        let hw = o + usize::from(types > 0);
        let (cols, k) = (2 * width + dim, mlp + 1);
        w.mentions.clear();
        for x in mentions {
            w.mentions.extend_from_slice(&v[x.first as usize * width..][..width]);
            w.mentions.extend_from_slice(&v[x.last as usize * width..][..width]);
            if dim > 0 {
                w.mentions.extend_from_slice(&self.params[ti][x.entity as usize * dim..][..dim]);
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
        let (mut total, mut row) = (0f64, vec![0f64; labels]);
        for a in 0..m {
            let x = &w.views[0][a * mlp..][..mlp];
            for b in (0..m).filter(|&b| b != a) {
                let p = &w.projected[b * labels * k..][..labels * k];
                for (l, r) in row.iter_mut().enumerate() {
                    *r = (dot(x, &p[l * k..][..mlp]) + p[l * k + mlp]).to_f64();
                }
                let g = target[a * m + b] as usize;
                let gold_score = row[g];
                let (mx, ln_z) = softmax(&mut row);
                total += ln_z - (gold_score - mx);
                row[g] -= 1.0;
                for (l, &d) in row.iter().enumerate() {
                    let d = R::from_f64(d);
                    for i in 0..k {
                        dproj[(b * labels + l) * k + i] += d * one(x, i);
                        if i < mlp {
                            dviews[0][a * mlp + i] += d * p[l * k + i];
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
                let d = dproj[b * labels * k + r];
                for j in 0..k {
                    local[hw + 4][r * k + j] += d * one(y, j);
                    if j < mlp {
                        dviews[1][b * mlp + j] += d * wr[r * k + j];
                    }
                }
            }
        }
        let mut dmentions = vec![R::ZERO; m * cols];
        for (s, dview) in dviews.iter_mut().enumerate() {
            dview.iter_mut().zip(&w.views[s]).filter(|(_, v)| **v < R::ZERO).for_each(|(g, _)| *g *= slope);
            let (wi, bi) = (hw + 2 * s, hw + 2 * s + 1);
            let [dw, db] = local.get_disjoint_mut([wi, bi]).expect("distinct");
            linear::backward(&self.params[wi], &w.mentions, dview, cols, mlp, dw, db, Some(&mut dmentions));
        }
        let mut dv = vec![R::ZERO; n * width];
        for (x, d) in mentions.iter().zip(dmentions.chunks_exact(cols)) {
            let (f, l) = (x.first as usize * width, x.last as usize * width);
            dv[f..f + width].iter_mut().zip(&d[..width]).for_each(|(a, b)| *a += *b);
            dv[l..l + width].iter_mut().zip(&d[width..2 * width]).for_each(|(a, b)| *a += *b);
            if dim > 0 {
                let t = &mut local[ti][x.entity as usize * dim..][..dim];
                t.iter_mut().zip(&d[2 * width..]).for_each(|(a, b)| *a += *b);
            }
        }
        // Checked once the adapter's share is in: its backward can
        // overflow, and nothing reaches `grads` unless all of it is finite.
        adapter::backward(shape, &self.params, features, dv, &mut local, &w.adapter, false);
        adapter::commit(grads, &local, "relation")?;
        Ok((total, pairs))
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
