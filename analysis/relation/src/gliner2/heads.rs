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
    out_projector: Mlp,
    start: Mlp,
    end: Mlp,
    /// The first span layer's start and end columns, and its bias.
    first_start: Linear,
    first_end: Linear,
    first_bias: Vec<f32>,
    last: Linear,
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
    Ok(Norm { weight: w, bias: b })
}

/// Splits a `[out, 2 * half]` layer into its two `[out, half]` column
/// blocks, without bias.
fn columns(l: &Linear) -> Result<(Linear, Linear), String> {
    let half = l.in_dim / 2;
    let mut a = Vec::with_capacity(l.out_dim * half);
    let mut b = Vec::with_capacity(l.out_dim * half);
    for row in l.weight.chunks_exact(l.in_dim) {
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
        let heads = Heads {
            width,
            max_width,
            count_pred: mlp(f, "count_pred", &[0, 2])?,
            positions: pos.f32s(),
            gru_ih: gru(f, "ih")?,
            gru_hh: gru(f, "hh")?,
            in_projector: linear(f, &format!("{t}.in_projector"))?,
            layers,
            out_projector: mlp(f, &format!("{t}.out_projector"), &[0, 2, 4])?,
            start: mlp(f, &format!("{span}.project_start"), &[0, 3])?,
            end: mlp(f, &format!("{span}.project_end"), &[0, 3])?,
            first_bias: first.bias.clone(),
            first_start,
            first_end,
            last: linear(f, &format!("{span}.out_project.3"))?,
        };
        heads.check(pos.shape[1] as usize)?;
        Ok(heads)
    }

    fn check(&self, pos_width: usize) -> Result<(), String> {
        let d = self.width;
        let m = self.in_projector.out_dim;
        let ok = pos_width == d
            && self.count_pred.0[0].in_dim == d
            && self.gru_ih.in_dim == d
            && self.gru_ih.out_dim == 3 * d
            && self.gru_hh.in_dim == d
            && self.gru_hh.out_dim == 3 * d
            && self.in_projector.in_dim == d
            && m.is_multiple_of(ATTENTION_HEADS)
            && self
                .layers
                .iter()
                .all(|l| l.in_proj.in_dim == m && l.in_proj.out_dim == 3 * m && l.out_proj.out_dim == m)
            && self.out_projector.0[0].in_dim == m + d
            && self.out_projector.0.last().is_some_and(|l| l.out_dim == d)
            && self.start.0[0].in_dim == d
            && self.end.0[0].in_dim == d
            && self.first_start.in_dim == d
            && self.last.out_dim == d
            && self.max_width > 0;
        if ok { Ok(()) } else { Err("GLiNER2 relation head shapes do not match the encoder width".into()) }
    }

    /// The count head's logits on a relation's `[P]` vector.
    pub fn count_logits(&self, p: &[f32]) -> Vec<f32> {
        self.count_pred.apply(p)
    }

    /// How many instances a relation has: the count head's argmax, the
    /// first of equal values, on its `[P]` vector.
    pub fn count(&self, p: &[f32]) -> usize {
        let logits = self.count_logits(p);
        let mut best = 0;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        best.min(self.positions.len() / self.width)
    }

    /// The `k` instances' head and tail queries, `[k][2][D]` flattened,
    /// from the two `[R]` vectors.
    pub fn queries(&self, fields: [&[f32]; 2], k: usize) -> Vec<f32> {
        let d = self.width;
        let mut states: Vec<Vec<f32>> = fields.iter().map(|f| f.to_vec()).collect();
        let mut out = Vec::with_capacity(k * 2 * d);
        let (mut gi, mut gh) = (vec![0.0; 3 * d], vec![0.0; 3 * d]);
        let mut steps = vec![vec![0.0; 2 * d]; k];
        for (t, step) in steps.iter_mut().enumerate() {
            self.gru_ih.apply(&self.positions[t * d..(t + 1) * d], &mut gi);
            for (m, h) in states.iter_mut().enumerate() {
                self.gru_hh.apply(h, &mut gh);
                for j in 0..d {
                    let r = sigmoid(gi[j] + gh[j]);
                    let z = sigmoid(gi[d + j] + gh[d + j]);
                    let n = (gi[2 * d + j] + r * gh[2 * d + j]).tanh();
                    h[j] = (1.0 - z) * n + z * h[j];
                }
                for (s, (&a, &b)) in step[m * d..(m + 1) * d].iter_mut().zip(h.iter().zip(fields[m])) {
                    *s = a + b;
                }
            }
        }
        let m = self.in_projector.out_dim;
        for y in &steps {
            let mut x = vec![0.0; 2 * m];
            for f in 0..2 {
                self.in_projector.apply(&y[f * d..(f + 1) * d], &mut x[f * m..(f + 1) * m]);
            }
            for l in &self.layers {
                l.apply(&mut x, 2);
            }
            for f in 0..2 {
                let mut cat = x[f * m..(f + 1) * m].to_vec();
                cat.extend_from_slice(&y[f * d..(f + 1) * d]);
                out.extend(self.out_projector.apply(&cat));
            }
        }
        out
    }

    /// The span layer's hidden rows for every span of the `n` word vectors
    /// `words` (`[n][D]`): `[n][max_width][4D]`, empty rows for spans past
    /// the last word.
    pub fn spans(&self, words: &[f32], n: usize) -> Spans {
        let (d, w) = (self.width, self.max_width);
        let hidden = self.first_bias.len();
        let mut start = vec![0.0; n * hidden];
        let mut end = vec![0.0; n * hidden];
        for i in 0..n {
            let h = &words[i * d..(i + 1) * d];
            let relu = |v: Vec<f32>| v.into_iter().map(|x| x.max(0.0)).collect::<Vec<_>>();
            self.first_start.apply(&relu(self.start.apply(h)), &mut start[i * hidden..(i + 1) * hidden]);
            self.first_end.apply(&relu(self.end.apply(h)), &mut end[i * hidden..(i + 1) * hidden]);
        }
        let mut rows = vec![0.0; n * w * hidden];
        for i in 0..n {
            for k in 0..w.min(n - i) {
                let row = &mut rows[(i * w + k) * hidden..][..hidden];
                let (s, e) = (&start[i * hidden..][..hidden], &end[(i + k) * hidden..][..hidden]);
                for (((r, &a), &b), &c) in row.iter_mut().zip(s).zip(e).zip(&self.first_bias) {
                    *r = (a + b + c).max(0.0);
                }
            }
        }
        Spans { rows, n, width: w, hidden }
    }

    /// One query's logits over the spans, `[n][max_width]`, with spans
    /// past the last word at negative infinity.
    pub fn logits(&self, spans: &Spans, query: &[f32]) -> Vec<f32> {
        let h = spans.hidden;
        let mut folded = vec![0.0; h];
        for (o, &q) in query.iter().enumerate() {
            for (f, &w) in folded.iter_mut().zip(&self.last.weight[o * h..(o + 1) * h]) {
                *f += q * w;
            }
        }
        let bias: f32 = query.iter().zip(&self.last.bias).map(|(a, b)| a * b).sum();
        let mut out = vec![f32::NEG_INFINITY; spans.n * spans.width];
        for i in 0..spans.n {
            for k in 0..spans.width.min(spans.n - i) {
                let row = &spans.rows[(i * spans.width + k) * h..][..h];
                out[i * spans.width + k] = row.iter().zip(&folded).map(|(a, b)| a * b).sum::<f32>() + bias;
            }
        }
        out
    }
}

/// The span layer's hidden rows of one text.
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
    1.0 / (1.0 + (-x).exp())
}
