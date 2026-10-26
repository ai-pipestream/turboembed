//! GLiNER2's heads after the encoder, for relations, as the reference
//! computes them in evaluation mode (dropout off):
//!
//! - `count_pred`: Linear(D, 2D), ReLU, Linear(2D, 20) on a relation's
//!   `[P]` vector; the argmax is how many instances it has.
//! - `count_embed` (`count_lstm_v2`): a GRU over the first K rows of a
//!   position table, started from each field's `[R]` vector; each step's
//!   state plus that `[R]` vector; projected to 128, two post-norm
//!   transformer layers (4 heads, ReLU feed-forward of 256, epsilon 1e-5)
//!   over the two fields of each instance; then
//!   Linear(128 + D, D), ReLU, Linear(D, D), ReLU, Linear(D, D) on the
//!   transformer output followed by its input. Each result is one query.
//! - `span_rep` (`SpanMarkerV0`): start and end projections
//!   (Linear(D, 4D), ReLU, Linear(4D, D)) of each word, and for the span
//!   of words `i..=i + w` (w below the maximum width),
//!   Linear(4D, D)(ReLU(Linear(2D, 4D)(ReLU([start_i; end_{i+w}])))). A
//!   query's logit for a span is its dot product with that.
//!
//! The span logits are computed as the reference's `score_queries`
//! factors them: the first span layer splits into its start and end
//! columns, applied once per word, and the last folds into each query.

use turbo::head::Linear;
use turbo::safetensors::{Dtype, File};

use crate::dot::dot;
use crate::projection::{Projection, Scratch, mlp_into, relu};

/// Two linear layers with a ReLU between, as `create_mlp` and
/// `create_projection_layer` build them.
struct Mlp(Vec<Linear>);

impl Mlp {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let mut v = x.to_vec();
        for (k, l) in self.0.iter().enumerate() {
            let mut out = vec![0.0; l.out_dim];
            l.apply(&v, &mut out);
            if k + 1 < self.0.len() {
                out.iter_mut().for_each(|x| *x = x.max(0.0));
            }
            v = out;
        }
        v
    }
}

struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Norm {
    fn apply(&self, row: &mut [f32]) {
        let n = row.len() as f64;
        let mean = row.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        let var = row.iter().map(|&x| (f64::from(x) - mean).powi(2)).sum::<f64>() / n;
        let scale = (var + 1e-5).sqrt().recip();
        for ((x, &w), &b) in row.iter_mut().zip(&self.weight).zip(&self.bias) {
            *x = ((f64::from(*x) - mean) * scale * f64::from(w) + f64::from(b)) as f32;
        }
    }
}

/// One `nn.TransformerEncoderLayer`, post-norm, ReLU.
struct Layer {
    in_proj: Linear,
    out_proj: Linear,
    linear1: Linear,
    linear2: Linear,
    norm1: Norm,
    norm2: Norm,
}

const ATTENTION_HEADS: usize = 4;

impl Layer {
    /// `x` is `n` rows of the model width; attention runs across them.
    fn apply(&self, x: &mut [f32], n: usize) {
        let m = self.out_proj.out_dim;
        let hd = m / ATTENTION_HEADS;
        let mut qkv = vec![0.0; n * 3 * m];
        for (row, out) in x.chunks_exact(m).zip(qkv.chunks_exact_mut(3 * m)) {
            self.in_proj.apply(row, out);
        }
        let scale = (hd as f32).sqrt().recip();
        let mut ctx = vec![0.0; n * m];
        for h in 0..ATTENTION_HEADS {
            for i in 0..n {
                let q = &qkv[i * 3 * m + h * hd..][..hd];
                let mut s: Vec<f32> = (0..n)
                    .map(|j| {
                        q.iter().zip(&qkv[j * 3 * m + m + h * hd..][..hd]).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let top = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                s.iter_mut().for_each(|v| *v = (*v - top).exp());
                let sum: f32 = s.iter().sum();
                for (j, p) in s.iter().enumerate() {
                    let v = &qkv[j * 3 * m + 2 * m + h * hd..][..hd];
                    for (c, &vv) in ctx[i * m + h * hd..][..hd].iter_mut().zip(v) {
                        *c += p / sum * vv;
                    }
                }
            }
        }
        let mut tmp = vec![0.0; m];
        let mut hidden = vec![0.0; self.linear1.out_dim];
        for (row, c) in x.chunks_exact_mut(m).zip(ctx.chunks_exact(m)) {
            self.out_proj.apply(c, &mut tmp);
            row.iter_mut().zip(&tmp).for_each(|(x, a)| *x += a);
            self.norm1.apply(row);
            self.linear1.apply(row, &mut hidden);
            hidden.iter_mut().for_each(|x| *x = x.max(0.0));
            self.linear2.apply(&hidden, &mut tmp);
            row.iter_mut().zip(&tmp).for_each(|(x, a)| *x += a);
            self.norm2.apply(row);
        }
    }
}

/// The relation heads of a GLiNER2 checkpoint.
pub struct Heads {
    pub width: usize,
    pub max_width: usize,
    count_pred: Mlp,
    positions: Vec<f32>,
    gru_ih: Linear,
    gru_hh: Linear,
    in_projector: Linear,
    layers: Vec<Layer>,
    /// Projection, so all of a relation's queries go through it at once.
    out_projector: [Projection; 3],
    /// The span start and end projections, packed for many rows.
    start: [Projection; 2],
    end: [Projection; 2],
    /// The first span layer's start and end columns, and its bias.
    first_start: Projection,
    first_end: Projection,
    first_bias: Vec<f32>,
    last: Linear,
    /// `last` folded into queries: its weights as a layer from the model
    /// width to the span hidden width, without bias.
    fold: Projection,
}

fn linear(f: &File, name: &str) -> Result<Linear, String> {
    let w = f.get(&format!("{name}.weight"), Dtype::F32, 2).map_err(|e| e.message)?;
    let b = f.get(&format!("{name}.bias"), Dtype::F32, 1).map_err(|e| e.message)?;
    let (out, inp) = (w.shape[0] as usize, w.shape[1] as usize);
    Linear::new(inp, out, w.f32s(), b.f32s()).map_err(|e| format!("{name}: {}", e.message))
}

fn mlp(f: &File, prefix: &str, indexes: &[usize]) -> Result<Mlp, String> {
    let layers = indexes.iter().map(|i| linear(f, &format!("{prefix}.{i}"))).collect::<Result<Vec<_>, _>>()?;
    if layers.windows(2).any(|w| w[0].out_dim != w[1].in_dim) {
        return Err(format!("{prefix}: layer widths do not chain"));
    }
    Ok(Mlp(layers))
}

fn norm(f: &File, name: &str) -> Result<Norm, String> {
    let w = f.get(&format!("{name}.weight"), Dtype::F32, 1).map_err(|e| e.message)?.f32s();
    let b = f.get(&format!("{name}.bias"), Dtype::F32, 1).map_err(|e| e.message)?.f32s();
    if w.len() != b.len() || !w.iter().chain(&b).all(|x| x.is_finite()) {
        return Err(format!("{name}: weight and bias must be finite and of one length"));
    }
    Ok(Norm { weight: w, bias: b })
}

/// `last` as the layer that folds a query into it: `[in][out]` read as
/// `[out][in]`, from `out_dim` to `in_dim`, packed.
fn fold(last: &Linear) -> Result<Projection, String> {
    let zero = vec![0.0; last.in_dim];
    let l = Linear::new(last.out_dim, last.in_dim, transpose(last), zero).map_err(|e| e.message)?;
    Projection::new(&l)
}

fn transpose(l: &Linear) -> Vec<f32> {
    let mut t = vec![0.0; l.weight().len()];
    for (o, row) in l.weight().chunks_exact(l.in_dim).enumerate() {
        for (i, &v) in row.iter().enumerate() {
            t[i * l.out_dim + o] = v;
        }
    }
    t
}

/// A two-layer MLP's layers, packed.
fn packed(m: &Mlp) -> Result<[Projection; 2], String> {
    Ok([Projection::new(&m.0[0])?, Projection::new(&m.0[1])?])
}

/// Splits a `[out, 2 * half]` layer into its two `[out, half]` column
/// blocks, without bias.
fn columns(l: &Linear) -> Result<(Linear, Linear), String> {
    if !l.in_dim.is_multiple_of(2) {
        return Err("span_rep out_project.0: odd input width".into());
    }
    let half = l.in_dim / 2;
    let mut a = Vec::with_capacity(l.out_dim * half);
    let mut b = Vec::with_capacity(l.out_dim * half);
    for row in l.weight().chunks_exact(l.in_dim) {
        a.extend_from_slice(&row[..half]);
        b.extend_from_slice(&row[half..]);
    }
    let zero = vec![0.0; l.out_dim];
    let make = |w| Linear::new(half, l.out_dim, w, zero.clone()).map_err(|e| e.message);
    Ok((make(a)?, make(b)?))
}

impl Heads {
    /// The heads from a checkpoint's tensors, unprefixed, every shape
    /// checked against the encoder width `width`.
    pub fn load(f: &File, width: usize, max_width: usize) -> Result<Heads, String> {
        let span = "span_rep.span_rep_layer";
        let first = linear(f, &format!("{span}.out_project.0"))?;
        let (first_start, first_end) = columns(&first)?;
        let pos = f.get("count_embed.pos_embedding.weight", Dtype::F32, 2).map_err(|e| e.message)?;
        let t = "count_embed.transformer";
        let layers = (0..2)
            .map(|i| {
                let p = format!("{t}.transformer.layers.{i}");
                let attn_w = f.get(&format!("{p}.self_attn.in_proj_weight"), Dtype::F32, 2).map_err(|e| e.message)?;
                let attn_b = f.get(&format!("{p}.self_attn.in_proj_bias"), Dtype::F32, 1).map_err(|e| e.message)?;
                let in_proj =
                    Linear::new(attn_w.shape[1] as usize, attn_w.shape[0] as usize, attn_w.f32s(), attn_b.f32s())
                        .map_err(|e| e.message)?;
                Ok(Layer {
                    in_proj,
                    out_proj: linear(f, &format!("{p}.self_attn.out_proj"))?,
                    linear1: linear(f, &format!("{p}.linear1"))?,
                    linear2: linear(f, &format!("{p}.linear2"))?,
                    norm1: norm(f, &format!("{p}.norm1"))?,
                    norm2: norm(f, &format!("{p}.norm2"))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let last = linear(f, &format!("{span}.out_project.3"))?;
        let heads = Heads {
            width,
            max_width,
            count_pred: mlp(f, "count_pred", &[0, 2])?,
            positions: pos.f32s(),
            gru_ih: gru(f, "ih")?,
            gru_hh: gru(f, "hh")?,
            in_projector: linear(f, &format!("{t}.in_projector"))?,
            layers,
            out_projector: mlp(f, &format!("{t}.out_projector"), &[0, 2, 4])?
                .0
                .iter()
                .map(Projection::new)
                .collect::<Result<Vec<_>, String>>()?
                .try_into()
                .map_err(|_| "out_projector: three layers".to_string())?,
            start: packed(&mlp(f, &format!("{span}.project_start"), &[0, 3])?)?,
            end: packed(&mlp(f, &format!("{span}.project_end"), &[0, 3])?)?,
            first_bias: first.bias().to_vec(),
            first_start: Projection::new(&first_start)?,
            first_end: Projection::new(&first_end)?,
            fold: fold(&last)?,
            last,
        };
        heads.check(pos.shape[1] as usize)?;
        Ok(heads)
    }

    /// Every layer's widths against the next one's and the encoder width,
    /// so no product in the heads meets a row of another length. Each
    /// [`Linear`] is finite by construction; the norms and the position
    /// table are checked here.
    fn check(&self, pos_width: usize) -> Result<(), String> {
        let d = self.width;
        let m = self.in_projector.out_dim;
        let hidden = self.first_bias.len();
        let layer_ok = |l: &Layer| {
            l.in_proj.in_dim == m
                && l.in_proj.out_dim == 3 * m
                && l.out_proj.in_dim == m
                && l.out_proj.out_dim == m
                && l.linear1.in_dim == m
                && l.linear2.in_dim == l.linear1.out_dim
                && l.linear2.out_dim == m
                && [&l.norm1, &l.norm2].iter().all(|n| n.weight.len() == m)
        };
        let ok = d > 0
            && pos_width == d
            && !self.positions.is_empty()
            && self.positions.len().is_multiple_of(d)
            && self.positions.iter().all(|x| x.is_finite())
            && self.count_pred.0[0].in_dim == d
            && self.count_pred.0[1].out_dim > 0
            && self.gru_ih.in_dim == d
            && self.gru_ih.out_dim == 3 * d
            && self.gru_hh.in_dim == d
            && self.gru_hh.out_dim == 3 * d
            && self.in_projector.in_dim == d
            && m.is_multiple_of(ATTENTION_HEADS)
            && self.layers.iter().all(layer_ok)
            && self.out_projector[0].in_dim == m + d
            && self.out_projector[1].in_dim == self.out_projector[0].out_dim
            && self.out_projector[2].in_dim == self.out_projector[1].out_dim
            && self.out_projector[2].out_dim == d
            && [&self.start, &self.end]
                .iter()
                .all(|p| p[0].in_dim == d && p[1].in_dim == p[0].out_dim && p[1].out_dim == d)
            && self.first_start.in_dim == d
            && self.first_end.in_dim == d
            && self.first_start.out_dim == hidden
            && self.last.in_dim == hidden
            && self.last.out_dim == d
            && self.max_width > 0;
        if ok { Ok(()) } else { Err("GLiNER2 relation head shapes do not match the encoder width".into()) }
    }
}

mod execute;

/// The span layer's hidden rows of one text.
#[derive(Default)]
pub struct Spans {
    rows: Vec<f32>,
    pub n: usize,
    pub width: usize,
    hidden: usize,
}

fn gru(f: &File, side: &str) -> Result<Linear, String> {
    let w = f.get(&format!("count_embed.gru.weight_{side}_l0"), Dtype::F32, 2).map_err(|e| e.message)?;
    let b = f.get(&format!("count_embed.gru.bias_{side}_l0"), Dtype::F32, 1).map_err(|e| e.message)?;
    Linear::new(w.shape[1] as usize, w.shape[0] as usize, w.f32s(), b.f32s()).map_err(|e| e.message)
}

/// The logistic function in single precision, as the reference applies it.
pub fn sigmoid(x: f32) -> f32 {
    turbo::head::sigmoid(x)
}
