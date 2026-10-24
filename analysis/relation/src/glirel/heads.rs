//! GLiREL's layers after the encoder, as the reference computes them in
//! evaluation mode (dropout off), with H the head width and E the
//! encoder's:
//!
//! - `token_rep_layer.projection`: Linear(E, H) on every encoder row read.
//! - `rnn`: one bidirectional LSTM layer (H/2 per direction, gates in
//!   PyTorch's order input, forget, cell, output) over the text's words.
//! - `span_marker` (`SpanMarkerV1`): start and end projections
//!   (Linear(H, 4H), ReLU, Linear(4H, H)) of each word; an entity of words
//!   `s..=e` is Linear(4H, H)(ReLU(Linear(2H, 4H)(ReLU([start_s; end_e])))).
//! - `out_project`: a pair of entities is
//!   Linear(4H, H)(ReLU(Linear(2H, 4H)([head; tail]))).
//! - `prompt_rep_layer`: Linear(H, 4H), ReLU, Linear(4H, H) on each label's
//!   mean of its `[REL]` and first label subword.
//! - the scorer: a pair's logit for a label is the dot product of the two.
//!
//! The pair layer's first matrix splits into its head and tail columns,
//! applied once per entity, and its last folds into each label, as
//! GLiNER2's span head is factored.

use turbo::head::Linear;
use turbo::safetensors::{Dtype, File};

use crate::kernel::{Packed, apply, dot};

/// A layer's columns `from..from + width` of a `[out, in]` matrix, without
/// bias, packed.
fn columns(l: &Linear, from: usize, width: usize) -> Result<Packed, String> {
    let w = l.weight().chunks_exact(l.in_dim).flat_map(|row| row[from..from + width].iter().copied()).collect();
    Packed::new(&Linear::new(width, l.out_dim, w, vec![0.0; l.out_dim]).map_err(|e| e.message)?)
}

fn linear(f: &File, name: &str) -> Result<Linear, String> {
    let w = f.get(&format!("{name}.weight"), Dtype::F32, 2).map_err(|e| e.message)?;
    let b = f.get(&format!("{name}.bias"), Dtype::F32, 1).map_err(|e| e.message)?;
    Linear::new(w.shape[1] as usize, w.shape[0] as usize, w.f32s(), b.f32s())
        .map_err(|e| format!("{name}: {}", e.message))
}

/// Linear, ReLU, Linear, as `create_projection_layer` builds it.
fn mlp(f: &File, name: &str) -> Result<[Packed; 2], String> {
    Ok([Packed::new(&linear(f, &format!("{name}.0"))?)?, Packed::new(&linear(f, &format!("{name}.3"))?)?])
}

fn relu(mut v: Vec<f32>) -> Vec<f32> {
    v.iter_mut().for_each(|x| *x = x.max(0.0));
    v
}

fn run(m: &[Packed; 2], x: &[f32], n: usize) -> Vec<f32> {
    m[1].rows(&relu(m[0].rows(x, n)), n)
}

use turbo::head::sigmoid;

/// One direction of the LSTM: the input product with both biases, and the
/// recurrent product.
struct Direction {
    input: Packed,
    recurrent: Linear,
}

impl Direction {
    fn load(f: &File, suffix: &str) -> Result<Direction, String> {
        let get = |n: &str, rank| f.get(&format!("rnn.lstm.{n}_l0{suffix}"), Dtype::F32, rank).map_err(|e| e.message);
        let (wi, wh) = (get("weight_ih", 2)?, get("weight_hh", 2)?);
        let bias: Vec<f32> =
            get("bias_ih", 1)?.f32s().iter().zip(get("bias_hh", 1)?.f32s()).map(|(a, b)| a + b).collect();
        let input = Linear::new(wi.shape[1] as usize, wi.shape[0] as usize, wi.f32s(), bias).map_err(|e| e.message)?;
        let out = wh.shape[0] as usize;
        let recurrent = Linear::new(wh.shape[1] as usize, out, wh.f32s(), vec![0.0; out]).map_err(|e| e.message)?;
        Ok(Direction { input: Packed::new(&input)?, recurrent })
    }

    /// The hidden state at each of the `n` words, `[n][L]`, from the first
    /// word on or, `reverse`, from the last back.
    fn states(&self, x: &[f32], n: usize, reverse: bool) -> Vec<f32> {
        let l = self.recurrent.in_dim;
        let pre = self.input.rows(x, n);
        let (mut h, mut c, mut g) = (vec![0.0; l], vec![0.0; l], vec![0.0; 4 * l]);
        let mut out = vec![0.0; n * l];
        for k in 0..n {
            let t = if reverse { n - 1 - k } else { k };
            apply(&self.recurrent, &h, &mut g);
            let p = &pre[t * 4 * l..(t + 1) * 4 * l];
            for j in 0..l {
                let i = sigmoid(p[j] + g[j]);
                let fg = sigmoid(p[l + j] + g[l + j]);
                let cell = (p[2 * l + j] + g[2 * l + j]).tanh();
                let o = sigmoid(p[3 * l + j] + g[3 * l + j]);
                c[j] = fg * c[j] + i * cell;
                h[j] = o * c[j].tanh();
            }
            out[t * l..(t + 1) * l].copy_from_slice(&h);
        }
        out
    }
}

/// The heads of a GLiREL checkpoint.
pub struct Heads {
    /// The encoder width E and the head width H.
    pub encoder_width: usize,
    pub width: usize,
    projection: Packed,
    lstm: [Direction; 2],
    start: [Packed; 2],
    end: [Packed; 2],
    /// The entity layer's first matrix, by its start and end columns.
    entity_first: [Packed; 2],
    entity_bias: Vec<f32>,
    entity_last: Packed,
    /// The pair layer's first matrix, by its head and tail columns.
    pair_first: [Packed; 2],
    pair_bias: Vec<f32>,
    pair_last: Linear,
    prompt: [Packed; 2],
}

impl Heads {
    /// The heads from a checkpoint's tensors, every width checked against
    /// the encoder width `encoder_width` and against the next layer's.
    pub fn load(f: &File, encoder_width: usize) -> Result<Heads, String> {
        let projection = linear(f, "token_rep_layer.projection")?;
        let h = projection.out_dim;
        let m = "span_rep_layer.rel_rep_layer";
        let entity = linear(f, &format!("{m}.span_marker.out_project.0"))?;
        let pair = linear(f, &format!("{m}.out_project.0"))?;
        let even = |l: &Linear| l.in_dim == 2 * h;
        if !even(&entity) || !even(&pair) {
            return Err("GLiREL entity and pair layers must read two head-width rows".into());
        }
        let heads = Heads {
            encoder_width,
            width: h,
            lstm: [Direction::load(f, "")?, Direction::load(f, "_reverse")?],
            start: mlp(f, &format!("{m}.span_marker.project_start"))?,
            end: mlp(f, &format!("{m}.span_marker.project_end"))?,
            entity_first: [columns(&entity, 0, h)?, columns(&entity, h, h)?],
            entity_bias: entity.bias().to_vec(),
            entity_last: Packed::new(&linear(f, &format!("{m}.span_marker.out_project.3"))?)?,
            pair_first: [columns(&pair, 0, h)?, columns(&pair, h, h)?],
            pair_bias: pair.bias().to_vec(),
            pair_last: linear(f, &format!("{m}.out_project.3"))?,
            prompt: mlp(f, "prompt_rep_layer")?,
            projection: Packed::new(&projection)?,
        };
        heads.check()?;
        Ok(heads)
    }

    fn check(&self) -> Result<(), String> {
        let (e, h) = (self.encoder_width, self.width);
        let l = h / 2;
        let mlp_ok = |p: &[Packed; 2]| p[0].in_dim == h && p[1].in_dim == p[0].out_dim && p[1].out_dim == h;
        let ok = h > 0
            && h.is_multiple_of(2)
            && self.projection.in_dim == e
            && self.lstm.iter().all(|d| {
                d.input.in_dim == h
                    && d.input.out_dim == 4 * l
                    && d.recurrent.in_dim == l
                    && d.recurrent.out_dim == 4 * l
            })
            && [&self.start, &self.end, &self.prompt].into_iter().all(mlp_ok)
            && self.entity_first[0].out_dim == self.entity_bias.len()
            && self.entity_last.in_dim == self.entity_bias.len()
            && self.entity_last.out_dim == h
            && self.pair_first[0].out_dim == self.pair_bias.len()
            && self.pair_last.in_dim == self.pair_bias.len()
            && self.pair_last.out_dim == h;
        if ok { Ok(()) } else { Err("GLiREL head widths do not chain or do not match the encoder width".into()) }
    }

    /// `n` encoder rows projected to the head width.
    ///
    /// # Panics
    /// When `rows` is not `n` rows of the encoder width.
    pub fn project(&self, rows: &[f32], n: usize) -> Vec<f32> {
        self.projection.rows(rows, n)
    }

    /// The LSTM over the `n` projected words: `[n][H]`, each word's forward
    /// state then its backward state.
    ///
    /// # Panics
    /// When `words` is not `n` rows of the head width.
    pub fn lstm(&self, words: &[f32], n: usize) -> Vec<f32> {
        let l = self.width / 2;
        let (fwd, bwd) = (self.lstm[0].states(words, n, false), self.lstm[1].states(words, n, true));
        let mut out = Vec::with_capacity(n * 2 * l);
        for t in 0..n {
            out.extend_from_slice(&fwd[t * l..(t + 1) * l]);
            out.extend_from_slice(&bwd[t * l..(t + 1) * l]);
        }
        out
    }

    /// Each entity's representation, `[k][H]`, from the LSTM's rows and the
    /// entities' first and last words.
    ///
    /// # Panics
    /// When `lstm` is not whole rows of the head width or a word is past
    /// its end.
    pub fn entities(&self, lstm: &[f32], spans: &[(usize, usize)]) -> Vec<f32> {
        let h = self.width;
        assert!(lstm.len().is_multiple_of(h), "LSTM rows of the head width");
        let n = lstm.len() / h;
        let rows = |pick: fn(&(usize, usize)) -> usize| -> Vec<f32> {
            spans.iter().flat_map(|s| lstm[pick(s) * h..(pick(s) + 1) * h].iter().copied()).collect()
        };
        assert!(spans.iter().all(|&(s, e)| s <= e && e < n), "entity words inside the text");
        let k = spans.len();
        let start = relu(run(&self.start, &rows(|s| s.0), k));
        let end = relu(run(&self.end, &rows(|s| s.1), k));
        let (a, b) = (self.entity_first[0].rows(&start, k), self.entity_first[1].rows(&end, k));
        let hidden: Vec<f32> = a
            .chunks_exact(self.entity_bias.len())
            .zip(b.chunks_exact(self.entity_bias.len()))
            .flat_map(|(a, b)| a.iter().zip(b).zip(&self.entity_bias).map(|((a, b), c)| (a + b + c).max(0.0)))
            .collect();
        self.entity_last.rows(&hidden, k)
    }

    /// Each label's representation, `[m][H]`, from its mean of `[REL]` and
    /// first label subword, already projected.
    pub fn types(&self, means: &[f32], m: usize) -> Vec<f32> {
        run(&self.prompt, means, m)
    }

    /// Each pair's logit for each label, `[pairs][m]`.
    ///
    /// # Panics
    /// When `entities` or `types` are not whole rows of the head width, or
    /// a pair names an entity past the last.
    pub fn logits(&self, entities: &[f32], pairs: &[(usize, usize)], types: &[f32]) -> Vec<f32> {
        let (h, f) = (self.width, self.pair_bias.len());
        assert!(entities.len().is_multiple_of(h) && types.len().is_multiple_of(h), "rows of the head width");
        let (k, m) = (entities.len() / h, types.len() / h);
        assert!(pairs.iter().all(|&(a, b)| a < k && b < k), "pairs of the entities given");
        let (head, tail) = (self.pair_first[0].rows(entities, k), self.pair_first[1].rows(entities, k));
        // The last layer folded into each label: logit = relu(..) . W^T t + t . b.
        let mut folded = vec![0.0; m * f];
        for (t, fold) in types.chunks_exact(h).zip(folded.chunks_exact_mut(f)) {
            for (o, &q) in t.iter().enumerate() {
                for (x, &w) in fold.iter_mut().zip(&self.pair_last.weight()[o * f..(o + 1) * f]) {
                    *x += q * w;
                }
            }
        }
        let bias: Vec<f32> = types.chunks_exact(h).map(|t| dot(t, self.pair_last.bias())).collect();
        let mut hidden = vec![0.0; f];
        let mut out = Vec::with_capacity(pairs.len() * m);
        for &(a, b) in pairs {
            let (x, y) = (&head[a * f..(a + 1) * f], &tail[b * f..(b + 1) * f]);
            for (((z, &x), &y), &c) in hidden.iter_mut().zip(x).zip(y).zip(&self.pair_bias) {
                *z = (x + y + c).max(0.0);
            }
            out.extend(folded.chunks_exact(f).zip(&bias).map(|(w, &c)| dot(&hidden, w) + c));
        }
        out
    }
}
