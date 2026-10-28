//! ModernBERT and nomic-bert (TURBO_FAMILY_MODERNBERT,
//! TURBO_FAMILY_NOMIC_BERT, #1461) on the BERT encoder's rows, packing,
//! pooling and kernels: what differs is the layer loop.
//!
//! Both rotate Q and K by each token's position (its column in its row)
//! in the QKV product's epilogue, over weights packed pair-adjacent once
//! per model (kernels/pairs.rs), and both have a gated MLP whose epilogue
//! writes act(gate) * up from one product of twice the width. No
//! projection has a bias: the model's zero vector stands in, and adding
//! it leaves every bit.
//!
//! ModernBERT is pre-norm: x += attn(attn_ln(x)), x += mlp(ffn_ln(x)),
//! with the normalized rows in a second buffer, xn, and the residual kept
//! in x; layer 0 has no attn_ln and reads x. Its norms have no bias. Its
//! layers alternate: layer l attends to the whole row iff l % global_every
//! is 0, with the global rotary base, and otherwise each query to the
//! keys within window / 2 of it, with the local base. final_norm follows
//! the last layer. nomic-bert is post-norm in BERT's order, every layer
//! global, its embeddings the word row plus the token type row.

use crate::backend::{
    TURBO_ACTIVATION_SILU, TURBO_FAMILY_MODERNBERT, TURBO_FAMILY_NOMIC_BERT, TURBO_MODERNBERT_ATTN_LN_WEIGHT,
    TURBO_MODERNBERT_EMBEDDING_TENSORS, TURBO_MODERNBERT_EMBEDDINGS_LN_WEIGHT, TURBO_MODERNBERT_FFN_LN_WEIGHT,
    TURBO_MODERNBERT_FINAL_NORM_WEIGHT, TURBO_MODERNBERT_LAYER_TENSORS, TURBO_NOMIC_ATTN_LN_BIAS,
    TURBO_NOMIC_ATTN_LN_WEIGHT, TURBO_NOMIC_EMBEDDING_TENSORS, TURBO_NOMIC_EMBEDDINGS_LN_BIAS,
    TURBO_NOMIC_EMBEDDINGS_LN_WEIGHT, TURBO_NOMIC_FFN_LN_BIAS, TURBO_NOMIC_FFN_LN_WEIGHT, TURBO_NOMIC_LAYER_TENSORS,
    TURBO_NOMIC_TOKEN_TYPE_EMBEDDINGS, turbo_backend_model,
};
use crate::cpu::kernels::{Act, Epilogue, Gemm, Packed, Rope, zeroed};
use crate::cpu::pool::Pool;

use super::job::{Bufs, Step};
use super::{Encoder, TOKENS};

#[cfg(test)]
mod tests;

/// The family's parts beyond a BERT's.
pub(super) struct Rotary {
    modern: bool,
    act: Act,
    /// The core's cos/sin table, [2 (global, local)][2 (cos, sin)]
    /// [positions][pairs]: the model's, which outlives the session.
    table: *const f32,
    positions: usize,
    pairs: usize,
    /// A local layer's keys each side of a query; usize::MAX with no
    /// local layer.
    window: usize,
    global_every: usize,
    /// ModernBERT's normalized rows, [tokens, hidden]; empty for nomic.
    pub(super) xn: Vec<f32>,
}

impl Rotary {
    /// The parts of a model of a rotary family whose sessions hold up to
    /// `tokens` tokens; None for any other family. Err is the bytes it
    /// could not allocate.
    pub(super) fn new(d: &turbo_backend_model, tokens: usize) -> Result<Option<Rotary>, usize> {
        if !matches!(d.family, TURBO_FAMILY_MODERNBERT | TURBO_FAMILY_NOMIC_BERT) {
            return Ok(None);
        }
        let modern = d.family == TURBO_FAMILY_MODERNBERT;
        let xn = if modern { zeroed(tokens.checked_mul(d.hidden as usize).ok_or(usize::MAX)?)? } else { Vec::new() };
        Ok(Some(Rotary {
            modern,
            act: if d.activation == TURBO_ACTIVATION_SILU { Act::Silu } else { Act::GeluErf },
            table: d.rope_table,
            positions: d.max_positions as usize,
            pairs: d.rope_pairs as usize,
            window: if d.attention_window == 0 { usize::MAX } else { d.attention_window as usize / 2 },
            global_every: d.global_every as usize,
            xn,
        }))
    }

    /// Whether layer l attends to the whole row, with the global base.
    fn global(&self, l: usize) -> bool {
        self.global_every == 0 || l.is_multiple_of(self.global_every)
    }

    /// The encoder's embeddings and layers, through x as the last layer
    /// leaves it, each step on `go`.
    pub(super) fn layers(
        &self,
        e: &Encoder,
        packed: &Packed,
        bufs: Bufs,
        tokens: usize,
        pool: &mut Pool,
        go: &dyn Fn(&mut Pool, Step, usize),
    ) {
        let (h, n_i) = (e.hidden, e.intermediate);
        let zero = &packed.zero[..];
        let chunks = tokens.div_ceil(TOKENS);
        let linear = |pool: &mut Pool, g: Gemm| go(pool, Step::Linear(g), g.tasks());
        let gemm = |x, ldx, w, y, ldy, epilogue| Gemm { x, ldx, rows: tokens, w, bias: zero, y, ldy, epilogue };
        let (emb, per) = if self.modern {
            (TURBO_MODERNBERT_EMBEDDING_TENSORS, TURBO_MODERNBERT_LAYER_TENSORS)
        } else {
            (TURBO_NOMIC_EMBEDDING_TENSORS, TURBO_NOMIC_LAYER_TENSORS)
        };
        let w = |l: usize, r: u32| e.tensor((emb + l as u32 * per + r) as usize);
        let embed = if self.modern {
            Step::Embed { ty: None, w: e.tensor(TURBO_MODERNBERT_EMBEDDINGS_LN_WEIGHT as usize), b: &zero[..h] }
        } else {
            Step::Embed {
                ty: Some(e.tensor(TURBO_NOMIC_TOKEN_TYPE_EMBEDDINGS as usize)),
                w: e.tensor(TURBO_NOMIC_EMBEDDINGS_LN_WEIGHT as usize),
                b: e.tensor(TURBO_NOMIC_EMBEDDINGS_LN_BIAS as usize),
            }
        };
        go(pool, embed, chunks);
        let half = self.positions * self.pairs;
        for l in 0..e.layers {
            let pl = &packed.layers[l];
            let global = self.global(l);
            // SAFETY: the table holds both bases' cos and sin.
            let cos = unsafe { self.table.add(if global { 0 } else { 2 * half }) };
            let rope = Rope {
                hidden: h,
                d: h / e.heads,
                cos,
                sin: unsafe { cos.add(half) },
                at: e.at.as_ptr(),
                seq: e.seq,
                t_out: bufs.kt,
                stride: e.stride,
                slot: e.slot.as_ptr(),
            };
            let window = if global { usize::MAX } else { self.window };
            let attention = Step::Attention { window };
            let blocks = e.heads * e.blocks[e.batch];
            if self.modern {
                // Layer 0 has no attention norm: its QKV reads x.
                let src = if l == 0 {
                    bufs.x
                } else {
                    go(pool, Step::NormInto(w(l, TURBO_MODERNBERT_ATTN_LN_WEIGHT), &zero[..h]), chunks);
                    bufs.xn
                };
                linear(pool, gemm(src, h, &pl.qkv, bufs.qv, 3 * h, Epilogue::Rope(rope)));
                go(pool, attention, blocks);
                linear(pool, gemm(bufs.ctx, h, &pl.out, bufs.x, h, Epilogue::Residual));
                go(pool, Step::NormInto(w(l, TURBO_MODERNBERT_FFN_LN_WEIGHT), &zero[..h]), chunks);
                linear(pool, gemm(bufs.xn, h, &pl.ffn_in, bufs.ffn, n_i, Epilogue::Gated(self.act)));
                linear(pool, gemm(bufs.ffn, n_i, &pl.ffn_out, bufs.x, h, Epilogue::Residual));
            } else {
                linear(pool, gemm(bufs.x, h, &pl.qkv, bufs.qv, 3 * h, Epilogue::Rope(rope)));
                go(pool, attention, blocks);
                linear(pool, gemm(bufs.ctx, h, &pl.out, bufs.x, h, Epilogue::Residual));
                go(pool, Step::Norm(w(l, TURBO_NOMIC_ATTN_LN_WEIGHT), w(l, TURBO_NOMIC_ATTN_LN_BIAS)), chunks);
                linear(pool, gemm(bufs.x, h, &pl.ffn_in, bufs.ffn, n_i, Epilogue::Gated(self.act)));
                linear(pool, gemm(bufs.ffn, n_i, &pl.ffn_out, bufs.x, h, Epilogue::Residual));
                go(pool, Step::Norm(w(l, TURBO_NOMIC_FFN_LN_WEIGHT), w(l, TURBO_NOMIC_FFN_LN_BIAS)), chunks);
            }
        }
        if self.modern {
            go(pool, Step::Norm(e.tensor(TURBO_MODERNBERT_FINAL_NORM_WEIGHT as usize), &zero[..h]), chunks);
        }
    }
}
