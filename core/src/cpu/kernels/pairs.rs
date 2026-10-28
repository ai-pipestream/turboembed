//! Pair-adjacent packing for the rotary families (#1461): the weights laid
//! out once at load so that the two outputs a rotation or a gated MLP
//! combines are neighbouring columns of one product, and the epilogues
//! that combine them as the product's tile is written.
//!
//! RoPE (rotate_half) turns column c of a head with column c + d/2. A
//! head's scores q . k do not depend on the order of its columns as long
//! as Q and K share it, so the Q and K rows of each head are packed as
//! (0, d/2), (1, d/2 + 1), .., and V's as stored: the context comes out
//! in the stored order, which the output projection reads. A gated MLP's
//! output j is act(gate_j) * up_j, so gate row j and up row j are packed
//! side by side, and the epilogue writes one column from the two.

use crate::backend::{
    TURBO_FAMILY_MODERNBERT, TURBO_MODERNBERT_ATTN_OUT_WEIGHT, TURBO_MODERNBERT_EMBEDDING_TENSORS,
    TURBO_MODERNBERT_FFN_IN_WEIGHT, TURBO_MODERNBERT_FFN_OUT_WEIGHT, TURBO_MODERNBERT_LAYER_TENSORS,
    TURBO_MODERNBERT_QKV_WEIGHT, TURBO_NOMIC_ATTN_OUT_WEIGHT, TURBO_NOMIC_EMBEDDING_TENSORS,
    TURBO_NOMIC_FFN_GATE_WEIGHT, TURBO_NOMIC_FFN_OUT_WEIGHT, TURBO_NOMIC_FFN_UP_WEIGHT, TURBO_NOMIC_LAYER_TENSORS,
    TURBO_NOMIC_QKV_WEIGHT, turbo_backend_model,
};

use super::rowwise::{gelu, silu};
use super::{Isa, Packed, PackedLayer, Panels, zeroed};

/// A gated MLP's activation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::cpu) enum Act {
    /// GELU by erf: GeGLU.
    GeluErf,
    /// x * sigmoid(x): SwiGLU.
    Silu,
}

/// The rotation in the QKV product's epilogue, over pair-adjacent Q and K
/// columns: y1 = x1 cos - x2 sin, y2 = x2 cos + x1 sin, cos and sin of
/// the token's position and the pair. Q (columns 0..hidden) and V
/// (2 hidden..3 hidden) go to y; K goes transposed to t_out, as
/// Epilogue::Split puts it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::cpu) struct Rope {
    pub(in crate::cpu) hidden: usize,
    /// A head's width.
    pub(in crate::cpu) d: usize,
    /// The layer's base's cos and sin, [positions][d / 2] each.
    pub(in crate::cpu) cos: *const f32,
    pub(in crate::cpu) sin: *const f32,
    /// Each token's place in its rows, [batch, seq]: its position is
    /// at % seq.
    pub(in crate::cpu) at: *const u32,
    pub(in crate::cpu) seq: usize,
    pub(in crate::cpu) t_out: *mut f32,
    pub(in crate::cpu) stride: usize,
    pub(in crate::cpu) slot: *const u32,
}

impl Rope {
    /// Rows r..r + rows by columns col..col + cols of the product, its
    /// sums in acc at row stride nr, plus `b`.
    ///
    /// # Safety
    /// col and cols are even, the pairs of one head; y holds the rows at
    /// ldy, t_out and slot as Split's; the table holds every position
    /// of the rows; no other thread touches this tile.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(super) unsafe fn tile(
        &self,
        acc: &[f32],
        nr: usize,
        r: usize,
        rows: usize,
        col: usize,
        cols: usize,
        b: &[f32],
        y: *mut f32,
        ldy: usize,
    ) {
        let (h, half) = (self.hidden, self.d / 2);
        for i in 0..rows {
            let tok = r + i;
            // SAFETY: as the contract says, for every pointer below.
            unsafe {
                let pos = *self.at.add(tok) as usize % self.seq;
                let (cos, sin) = (self.cos.add(pos * half), self.sin.add(pos * half));
                let a = &acc[i * nr..][..cols];
                for j in (0..cols).step_by(2) {
                    let c = col + j;
                    let (x1, x2) = (a[j] + b[j], a[j + 1] + b[j + 1]);
                    if c >= 2 * h {
                        *y.add(tok * ldy + c) = x1;
                        *y.add(tok * ldy + c + 1) = x2;
                        continue;
                    }
                    let p = (c % self.d) / 2;
                    let (cs, sn) = (*cos.add(p), *sin.add(p));
                    let (y1, y2) = (x1 * cs - x2 * sn, x2 * cs + x1 * sn);
                    if c < h {
                        *y.add(tok * ldy + c) = y1;
                        *y.add(tok * ldy + c + 1) = y2;
                    } else {
                        let s = *self.slot.add(tok) as usize;
                        *self.t_out.add((c - h) * self.stride + s) = y1;
                        *self.t_out.add((c + 1 - h) * self.stride + s) = y2;
                    }
                }
            }
        }
    }
}

/// The gated MLP's epilogue: y[t, c / 2] = act(sum_c + b_c) * (sum_c+1 +
/// b_c+1) for the even columns c of the tile, so y is half as wide as
/// the product.
///
/// # Safety
/// col and cols are even; y holds the rows at ldy of at least (col +
/// cols) / 2 columns; no other thread touches this tile's half.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) unsafe fn gated(
    act: Act,
    acc: &[f32],
    nr: usize,
    r: usize,
    rows: usize,
    col: usize,
    cols: usize,
    b: &[f32],
    y: *mut f32,
    ldy: usize,
) {
    for i in 0..rows {
        // SAFETY: as the contract says.
        let out = unsafe { std::slice::from_raw_parts_mut(y.add((r + i) * ldy + col / 2), cols / 2) };
        let a = &acc[i * nr..][..cols];
        for (j, o) in out.iter_mut().enumerate() {
            let (g, u) = (a[2 * j] + b[2 * j], a[2 * j + 1] + b[2 * j + 1]);
            *o = match act {
                Act::GeluErf => gelu(g),
                Act::Silu => silu(g),
            } * u;
        }
    }
}

/// W's rows [3 hidden, k] with each head's Q and K rows taken in pairs
/// (c, c + d/2), V's as stored.
pub(in crate::cpu) fn qkv_rows(w: &[f32], hidden: usize, d: usize) -> Vec<&[f32]> {
    let row = |o: usize| &w[o * hidden..][..hidden];
    let mut rows = Vec::with_capacity(3 * hidden);
    for third in 0..2 {
        for head in (0..hidden).step_by(d) {
            for c in 0..d / 2 {
                rows.push(row(third * hidden + head + c));
                rows.push(row(third * hidden + head + c + d / 2));
            }
        }
    }
    rows.extend((2 * hidden..3 * hidden).map(row));
    rows
}

/// Gate row j then up row j, for each j: both [n, k].
pub(in crate::cpu) fn gated_rows<'a>(gate: &'a [f32], up: &'a [f32], k: usize) -> Vec<&'a [f32]> {
    gate.chunks_exact(k).zip(up.chunks_exact(k)).flat_map(|(g, u)| [g, u]).collect()
}

impl Packed {
    /// A ModernBERT's or nomic-bert's linear layers, `tensors` in the
    /// family's TURBO_MODERNBERT_* or TURBO_NOMIC_* order, pair-adjacent
    /// as the module says, with zero biases. Err is the bytes it could not
    /// allocate.
    pub(in crate::cpu) fn rotary(d: &turbo_backend_model, tensors: &[&[f32]], isa: Isa) -> Result<Packed, usize> {
        let (h, i) = (d.hidden as usize, d.intermediate as usize);
        let head = h / d.heads as usize;
        let modern = d.family == TURBO_FAMILY_MODERNBERT;
        let (emb, per) = if modern {
            (TURBO_MODERNBERT_EMBEDDING_TENSORS, TURBO_MODERNBERT_LAYER_TENSORS)
        } else {
            (TURBO_NOMIC_EMBEDDING_TENSORS, TURBO_NOMIC_LAYER_TENSORS)
        };
        let mut layers = Vec::new();
        layers.try_reserve_exact(d.layers as usize).map_err(|_| usize::MAX)?;
        for l in 0..d.layers {
            let t = |r: u32| tensors[(emb + l * per + r) as usize];
            let (qkv, out, (gate, up), ffn_out) = if modern {
                let wi = t(TURBO_MODERNBERT_FFN_IN_WEIGHT);
                (
                    t(TURBO_MODERNBERT_QKV_WEIGHT),
                    t(TURBO_MODERNBERT_ATTN_OUT_WEIGHT),
                    wi.split_at(i * h),
                    t(TURBO_MODERNBERT_FFN_OUT_WEIGHT),
                )
            } else {
                (
                    t(TURBO_NOMIC_QKV_WEIGHT),
                    t(TURBO_NOMIC_ATTN_OUT_WEIGHT),
                    (t(TURBO_NOMIC_FFN_GATE_WEIGHT), t(TURBO_NOMIC_FFN_UP_WEIGHT)),
                    t(TURBO_NOMIC_FFN_OUT_WEIGHT),
                )
            };
            layers.push(PackedLayer {
                qkv: Panels::pack(&qkv_rows(qkv, h, head), h, isa)?,
                qkv_bias: zeroed(3 * h)?,
                out: Panels::pack(&[out], h, isa)?,
                ffn_in: Panels::pack(&gated_rows(gate, up, h), h, isa)?,
                ffn_out: Panels::pack(&[ffn_out], i, isa)?,
            });
        }
        Ok(Packed { isa, layers, zero: zeroed((3 * h).max(2 * i))? })
    }
}

#[cfg(test)]
#[path = "pairs_tests.rs"]
mod tests;
