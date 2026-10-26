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

use crate::dot::dot;
use crate::projection::{Projection, Scratch, mlp_into, relu};

/// A layer's columns `from..from + width` of a `[out, in]` matrix, without
/// bias, packed.
fn columns(l: &Linear, from: usize, width: usize) -> Result<Projection, String> {
    let w = l.weight().chunks_exact(l.in_dim).flat_map(|row| row[from..from + width].iter().copied()).collect();
    Projection::new(&Linear::new(width, l.out_dim, w, vec![0.0; l.out_dim]).map_err(|e| e.message)?)
}

fn linear(f: &File, name: &str) -> Result<Linear, String> {
    let w = f.get(&format!("{name}.weight"), Dtype::F32, 2).map_err(|e| e.message)?;
    let b = f.get(&format!("{name}.bias"), Dtype::F32, 1).map_err(|e| e.message)?;
    Linear::new(w.shape[1] as usize, w.shape[0] as usize, w.f32s(), b.f32s())
        .map_err(|e| format!("{name}: {}", e.message))
}

/// Linear, ReLU, Linear, as `create_projection_layer` builds it.
fn mlp(f: &File, name: &str) -> Result<[Projection; 2], String> {
    Ok([Projection::new(&linear(f, &format!("{name}.0"))?)?, Projection::new(&linear(f, &format!("{name}.3"))?)?])
}

use turbo::head::sigmoid;

/// One direction of the LSTM: the input product with both biases, and the
/// recurrent product.
struct Direction {
    input: Projection,
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
        Ok(Direction { input: Projection::new(&input)?, recurrent })
    }

    /// The hidden state at each of the `n` words, `[n][L]`, from the first
    /// word on or, `reverse`, from the last back.
    fn states_into(&self, x: &[f32], n: usize, reverse: bool, buffers: &mut [Vec<f32>], out: &mut Vec<f32>) {
        let [pre, h, c, g] = buffers else { unreachable!("four LSTM buffers") };
        let l = self.recurrent.in_dim;
        self.input.rows_into(x, n, pre);
        h.resize(l, 0.0);
        h.fill(0.0);
        c.resize(l, 0.0);
        c.fill(0.0);
        g.resize(4 * l, 0.0);
        out.resize(n * l, 0.0);
        for k in 0..n {
            let t = if reverse { n - 1 - k } else { k };
            self.recurrent.apply(h, g);
            let p = &pre[t * 4 * l..(t + 1) * 4 * l];
            for j in 0..l {
                let i = sigmoid(p[j] + g[j]);
                let fg = sigmoid(p[l + j] + g[l + j]);
                let cell = (p[2 * l + j] + g[2 * l + j]).tanh();
                let o = sigmoid(p[3 * l + j] + g[3 * l + j]);
                c[j] = fg * c[j] + i * cell;
                h[j] = o * c[j].tanh();
            }
            out[t * l..(t + 1) * l].copy_from_slice(h);
        }
    }
}

/// The heads of a GLiREL checkpoint.
pub struct Heads {
    /// The encoder width E and the head width H.
    pub encoder_width: usize,
    pub width: usize,
    projection: Projection,
    lstm: [Direction; 2],
    start: [Projection; 2],
    end: [Projection; 2],
    /// The entity layer's first matrix, by its start and end columns.
    entity_first: [Projection; 2],
    entity_bias: Vec<f32>,
    entity_last: Projection,
    /// The pair layer's first matrix, by its head and tail columns.
    pair_first: [Projection; 2],
    pair_bias: Vec<f32>,
    pair_last: Linear,
    prompt: [Projection; 2],
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
            entity_last: Projection::new(&linear(f, &format!("{m}.span_marker.out_project.3"))?)?,
            pair_first: [columns(&pair, 0, h)?, columns(&pair, h, h)?],
            pair_bias: pair.bias().to_vec(),
            pair_last: linear(f, &format!("{m}.out_project.3"))?,
            prompt: mlp(f, "prompt_rep_layer")?,
            projection: Projection::new(&projection)?,
        };
        heads.check()?;
        Ok(heads)
    }

    fn check(&self) -> Result<(), String> {
        let (e, h) = (self.encoder_width, self.width);
        let l = h / 2;
        let mlp_ok = |p: &[Projection; 2]| p[0].in_dim == h && p[1].in_dim == p[0].out_dim && p[1].out_dim == h;
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
}

mod execute;
