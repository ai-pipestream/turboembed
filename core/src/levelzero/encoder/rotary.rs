//! ModernBERT and nomic-bert (TURBO_FAMILY_MODERNBERT,
//! TURBO_FAMILY_NOMIC_BERT, #1461) on a listed GPU: the layer loop of
//! the CPU's encoder (cpu/encoder/rotary/) over the F32 linear kernels
//! and the rotary kernels of encoder.cl.
//!
//! Both families rotate Q and K by each token's position (its column in
//! its row) with the core's cos/sin table, copied to the device at load,
//! and have a gated MLP, act(gate) * up. No projection has a bias.
//! ModernBERT is pre-norm, its normalized rows in xn and the residual in
//! x, layer 0 without an attention norm, its norms without a bias; layer
//! l attends to the whole row iff l % global_every is 0, with the global
//! base, and otherwise each query to the keys within attention_window / 2
//! of it, with the local base; final_norm follows the last layer.
//! nomic-bert is post-norm in BERT's order, every layer global.
//!
//! The session computes in F32, at MODEL and EXACT: the linear layers run
//! on linear_sg (or linear, for widths not a multiple of 16). FASTEST is
//! refused, as for DeBERTa-v2, never run in F32 in its place.

use std::ffi::c_void;

use super::kernels::Kernels;
use super::session::Session;
use super::*;
use crate::TURBO_PRECISION_FASTEST;
use crate::backend::{
    TURBO_ACTIVATION_GELU_ERF, TURBO_ACTIVATION_SILU, TURBO_FAMILY_MODERNBERT, TURBO_FAMILY_NOMIC_BERT,
    TURBO_MLP_GATED, TURBO_MODERNBERT_ATTN_LN_WEIGHT, TURBO_MODERNBERT_ATTN_OUT_WEIGHT,
    TURBO_MODERNBERT_EMBEDDING_TENSORS, TURBO_MODERNBERT_EMBEDDINGS_LN_WEIGHT, TURBO_MODERNBERT_FFN_IN_WEIGHT,
    TURBO_MODERNBERT_FFN_LN_WEIGHT, TURBO_MODERNBERT_FFN_OUT_WEIGHT, TURBO_MODERNBERT_FINAL_NORM_WEIGHT,
    TURBO_MODERNBERT_LAYER_TENSORS, TURBO_MODERNBERT_QKV_WEIGHT, TURBO_MODERNBERT_WORD_EMBEDDINGS,
    TURBO_NOMIC_ATTN_LN_BIAS, TURBO_NOMIC_ATTN_LN_WEIGHT, TURBO_NOMIC_ATTN_OUT_WEIGHT, TURBO_NOMIC_EMBEDDING_TENSORS,
    TURBO_NOMIC_EMBEDDINGS_LN_BIAS, TURBO_NOMIC_EMBEDDINGS_LN_WEIGHT, TURBO_NOMIC_FFN_GATE_WEIGHT,
    TURBO_NOMIC_FFN_LN_BIAS, TURBO_NOMIC_FFN_LN_WEIGHT, TURBO_NOMIC_FFN_OUT_WEIGHT, TURBO_NOMIC_FFN_UP_WEIGHT,
    TURBO_NOMIC_LAYER_TENSORS, TURBO_NOMIC_QKV_WEIGHT, TURBO_NOMIC_TOKEN_TYPE_EMBEDDINGS, TURBO_NOMIC_WORD_EMBEDDINGS,
    TURBO_NORM_LAYER, TURBO_NORM_LAYER_NO_BIAS, turbo_backend_model,
};
use crate::levelzero::gpu::{Arg, Context, Kernel, Queue, Res, fail, fail_field};
use crate::status::{INVALID_ARGUMENT, INVALID_STRUCT_SIZE, UNSUPPORTED, UNSUPPORTED_OPTION};

/// Whether `family` is one of the rotary families.
pub(super) fn is(family: u32) -> bool {
    matches!(family, TURBO_FAMILY_MODERNBERT | TURBO_FAMILY_NOMIC_BERT)
}

/// The tensors the description must hold.
pub(super) fn tensor_count(d: &turbo_backend_model) -> u64 {
    let (emb, per) = bases(d);
    emb as u64 + d.layers as u64 * per as u64
}

fn bases(d: &turbo_backend_model) -> (u32, u32) {
    if d.family == TURBO_FAMILY_MODERNBERT {
        (TURBO_MODERNBERT_EMBEDDING_TENSORS, TURBO_MODERNBERT_LAYER_TENSORS)
    } else {
        (TURBO_NOMIC_EMBEDDING_TENSORS, TURBO_NOMIC_LAYER_TENSORS)
    }
}

/// Ok when this backend runs the model `d` describes, as the CPU's
/// encoder checks it (cpu/rotary.rs); else why not.
pub(super) fn check_load(d: &turbo_backend_model) -> Res<()> {
    let need = std::mem::offset_of!(turbo_backend_model, global_every) + size_of::<u32>();
    if (d.struct_size as usize) < need {
        let msg = format!("turbo_backend_model of {} bytes: a rotary model needs {need}", d.struct_size);
        return Err(fail(INVALID_STRUCT_SIZE, msg));
    }
    let modern = d.family == TURBO_FAMILY_MODERNBERT;
    let (act, norm) = if modern {
        (TURBO_ACTIVATION_GELU_ERF, TURBO_NORM_LAYER_NO_BIAS)
    } else {
        (TURBO_ACTIVATION_SILU, TURBO_NORM_LAYER)
    };
    if (d.mlp, d.activation, d.norm) != (TURBO_MLP_GATED, act, norm) {
        let msg = format!(
            "mlp {}, activation {}, norm {}: the levelzero backend runs this family with {TURBO_MLP_GATED}, {act}, \
             {norm}",
            d.mlp, d.activation, d.norm
        );
        return Err(fail(UNSUPPORTED, msg));
    }
    let head = d.hidden.checked_div(d.heads).unwrap_or(0);
    if head == 0
        || !d.hidden.is_multiple_of(d.heads)
        || !head.is_multiple_of(2)
        || d.rope_pairs * 2 != head
        || d.rope_table.is_null()
    {
        let msg = format!(
            "hidden {}, heads {}, rope_pairs {}: a rotary head is an even width, twice rope_pairs, with a table",
            d.hidden, d.heads, d.rope_pairs
        );
        return Err(fail(INVALID_ARGUMENT, msg));
    }
    if (d.attention_window == 0) != (d.global_every == 0) || (modern && d.attention_window == 0) {
        let msg = format!("attention_window {}, global_every {}", d.attention_window, d.global_every);
        return Err(fail(INVALID_ARGUMENT, msg));
    }
    if d.position_offset != 0 {
        return Err(fail(
            INVALID_ARGUMENT,
            format!("position_offset {}: a rotary model counts from 0", d.position_offset),
        ));
    }
    Ok(())
}

/// Ok unless the session asks for FASTEST, which this backend has no F16
/// rotary path for yet.
pub(super) fn check_precision(family: u32, precision: u32) -> Res<()> {
    if precision == TURBO_PRECISION_FASTEST {
        let name = if family == TURBO_FAMILY_MODERNBERT { "FAMILY_MODERNBERT" } else { "FAMILY_NOMIC_BERT" };
        return Err(fail_field(
            UNSUPPORTED_OPTION,
            3,
            format!(
                "precision: FASTEST on {name}: the levelzero backend runs this encoder in F32 alone; ask for MODEL or EXACT"
            ),
        ));
    }
    Ok(())
}

/// The bytes of the core's cos/sin table.
pub(super) fn table_bytes(d: &turbo_backend_model) -> usize {
    4 * d.max_positions as usize * d.rope_pairs as usize * size_of::<f32>()
}

/// The core's cos/sin table copied to a new device allocation.
pub(super) fn upload_table(c: &Context, d: &turbo_backend_model) -> Res<*mut c_void> {
    let bytes = table_bytes(d);
    let table = c.alloc_device(bytes)?;
    let copied = (|| {
        let mut q = c.lock_queue()?;
        // SAFETY: the core's table holds `bytes` while the model is loaded.
        let appended = unsafe { c.copy(&mut q, table, d.rope_table as *const c_void, bytes) };
        let synced = c.sync(&mut q);
        appended?;
        synced
    })();
    if let Err(e) = copied {
        c.free(table, "the rotary table");
        return Err(e);
    }
    Ok(table)
}

/// The rotary kernels, one set per session.
pub(super) struct RotaryKernels {
    embed: Kernel,
    norm: Kernel,
    add: Kernel,
    rope: Kernel,
    gated: Kernel,
    attention: Kernel,
}

impl RotaryKernels {
    /// The local memory the attention kernel keeps for its own use.
    pub(super) fn attention_local_bytes(&self) -> Res<u32> {
        self.attention.local_bytes()
    }

    pub(super) fn new(c: &Context) -> Res<RotaryKernels> {
        Ok(RotaryKernels {
            embed: c.kernel("rotary_embed", [BLOCK, 1, 1])?,
            norm: c.kernel("rotary_norm", [BLOCK, 1, 1])?,
            add: c.kernel("rotary_add", [WIDE, 1, 1])?,
            rope: c.kernel("rotary_rope", [WIDE, 1, 1])?,
            gated: c.kernel("rotary_gated", [WIDE, 1, 1])?,
            attention: c.kernel("rotary_attention", [BLOCK, 1, 1])?,
        })
    }
}

/// Bytes of local memory rotary_attention takes for rows of seq tokens:
/// the query's head and the row's scores.
pub(super) fn attention_local(seq: u32, head_dim: u32) -> u64 {
    4 * (head_dim as u64 + seq as u64)
}

impl Session {
    /// The rotary encoder over the packed rows through `layers` layers,
    /// then the pooling.
    pub(super) fn encode_rotary(&self, q: &mut Queue, rk: &RotaryKernels, layers: u32) -> Res<u64> {
        use Arg::*;
        let c = self.ctx();
        let d = &self.model().desc;
        let w = &self.weights.tensors;
        let k: &Kernels = &self.kernels;
        let (batch, tokens) = (self.batch, self.tokens);
        let (h, n_i) = (d.hidden, d.intermediate);
        let head_dim = h / d.heads;
        let eps = d.layer_norm_eps as f32;
        let modern = d.family == TURBO_FAMILY_MODERNBERT;
        let (emb, per) = bases(d);
        let t = |r: u32| w[r as usize];
        let lw = |l: u32, r: u32| w[(emb + l * per + r) as usize];
        let n = tokens as u64 * h as u64;
        // A linear layer without a bias, y = x W^T, x [tokens, n_in].
        let linear = |q: &mut Queue, x: u64, weight: u64, n_in: u32, n_out: u32, y: u64, what: &str| -> Res<()> {
            let args = [
                Ptr(x),
                Ptr(weight),
                Ptr(0),
                Ptr(y),
                I32(tokens as i32),
                I32(n_out as i32),
                I32(n_in as i32),
                I32(0),
                I32(n_in as i32),
            ];
            if n_in.is_multiple_of(16) {
                k.linear_sg.launch(c, q, what, &args, [n_out.div_ceil(SG_N), tokens.div_ceil(SG_T), 1])
            } else {
                k.linear.launch(c, q, what, &args, [n_out.div_ceil(TILE), tokens.div_ceil(TILE), 1])
            }
        };
        let norm = |q: &mut Queue, x: u64, y: u64, nw: u64, nb: u64, what: &str| {
            let args = [Ptr(x), Ptr(y), Ptr(nw), Ptr(nb), F32(eps), I32(h as i32)];
            rk.norm.launch(c, q, what, &args, [tokens, 1, 1])
        };
        let add = |q: &mut Queue, what: &str| {
            let args = [Ptr(self.x), Ptr(self.tmp), U64(n)];
            rk.add.launch(c, q, what, &args, [elementwise_groups(n), 1, 1])
        };

        // The lookup, which also copies the mask, rows and positions.
        let staging = self.staging as u64;
        let sn = self.max_batch as u64 * self.max_seq as u64 * 4;
        let (word, ty, ew, eb) = if modern {
            (t(TURBO_MODERNBERT_WORD_EMBEDDINGS), 0, t(TURBO_MODERNBERT_EMBEDDINGS_LN_WEIGHT), 0)
        } else {
            (
                t(TURBO_NOMIC_WORD_EMBEDDINGS),
                t(TURBO_NOMIC_TOKEN_TYPE_EMBEDDINGS),
                t(TURBO_NOMIC_EMBEDDINGS_LN_WEIGHT),
                t(TURBO_NOMIC_EMBEDDINGS_LN_BIAS),
            )
        };
        let args = [
            Ptr(staging),
            Ptr(staging + sn),
            Ptr(staging + 2 * sn),
            I32(self.has_types as i32),
            Ptr(staging + 3 * sn),
            Ptr(staging + 4 * sn),
            I32(batch as i32),
            Ptr(word),
            Ptr(ty),
            Ptr(ew),
            Ptr(eb),
            F32(eps),
            I32(h as i32),
            Ptr(self.x),
            Ptr(self.packed_mask),
            Ptr(self.rows),
            Ptr(self.positions),
        ];
        rk.embed.launch(c, q, "the embedding lookup", &args, [tokens, 1, 1])?;

        let half = d.max_positions as u64 * d.rope_pairs as u64 * 4;
        let table = self.model().rope as u64;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let (ffn_gate, ffn_up, ld) = if modern {
            (self.ffn, self.ffn + n_i as u64 * 4, 2 * n_i)
        } else {
            (self.ffn, self.ffn + tokens as u64 * n_i as u64 * 4, n_i)
        };
        for l in 0..layers.min(d.layers) {
            let global = d.global_every == 0 || l.is_multiple_of(d.global_every);
            let cos = table + if global { 0 } else { 2 * half };
            // The projections: from xn after ModernBERT's attention norm
            // (layer 0 has none and reads x), from x for nomic-bert.
            let src = if modern && l > 0 {
                let nw = lw(l, TURBO_MODERNBERT_ATTN_LN_WEIGHT);
                norm(q, self.x, self.xn, nw, 0, "the attention norm")?;
                self.xn
            } else {
                self.x
            };
            let qkv_w = lw(l, if modern { TURBO_MODERNBERT_QKV_WEIGHT } else { TURBO_NOMIC_QKV_WEIGHT });
            linear(q, src, qkv_w, h, 3 * h, self.qkv, "the query, key and value projection")?;
            let pairs = tokens as u64 * 2 * (h as u64 / 2);
            let args = [
                Ptr(self.qkv),
                Ptr(self.positions),
                Ptr(cos),
                Ptr(cos + half),
                I32(h as i32),
                I32(head_dim as i32),
                I32(tokens as i32),
            ];
            rk.rope.launch(c, q, "the rotation", &args, [elementwise_groups(pairs), 1, 1])?;
            let window = if global || d.attention_window == 0 { -1 } else { (d.attention_window / 2) as i32 };
            let args = [
                Ptr(self.qkv),
                Ptr(self.packed_mask),
                Ptr(self.rows),
                I32(h as i32),
                I32(head_dim as i32),
                F32(scale),
                I32(window),
                Ptr(self.att),
                Local(attention_local(self.longest, head_dim) as usize),
            ];
            rk.attention.launch(c, q, "attention", &args, [self.longest, d.heads, batch])?;
            let out_w = lw(l, if modern { TURBO_MODERNBERT_ATTN_OUT_WEIGHT } else { TURBO_NOMIC_ATTN_OUT_WEIGHT });
            linear(q, self.att, out_w, h, h, self.tmp, "the attention output projection")?;
            add(q, "the attention residual")?;
            if modern {
                norm(q, self.x, self.xn, lw(l, TURBO_MODERNBERT_FFN_LN_WEIGHT), 0, "the feed-forward norm")?;
                linear(
                    q,
                    self.xn,
                    lw(l, TURBO_MODERNBERT_FFN_IN_WEIGHT),
                    h,
                    2 * n_i,
                    self.ffn,
                    "the feed-forward input",
                )?;
            } else {
                let (nw, nb) = (lw(l, TURBO_NOMIC_ATTN_LN_WEIGHT), lw(l, TURBO_NOMIC_ATTN_LN_BIAS));
                norm(q, self.x, self.x, nw, nb, "the attention LayerNorm")?;
                linear(q, self.x, lw(l, TURBO_NOMIC_FFN_GATE_WEIGHT), h, n_i, ffn_gate, "the feed-forward gate")?;
                linear(q, self.x, lw(l, TURBO_NOMIC_FFN_UP_WEIGHT), h, n_i, ffn_up, "the feed-forward up projection")?;
            }
            let act = if d.activation == TURBO_ACTIVATION_SILU { 2 } else { 1 };
            let args = [
                Ptr(ffn_gate),
                I32(ld as i32),
                Ptr(ffn_up),
                I32(ld as i32),
                Ptr(self.mid),
                I32(n_i as i32),
                I32(tokens as i32),
                I32(act),
            ];
            let gated = tokens as u64 * n_i as u64;
            rk.gated.launch(c, q, "the gated activation", &args, [elementwise_groups(gated), 1, 1])?;
            let out_w = lw(l, if modern { TURBO_MODERNBERT_FFN_OUT_WEIGHT } else { TURBO_NOMIC_FFN_OUT_WEIGHT });
            linear(q, self.mid, out_w, n_i, h, self.tmp, "the feed-forward output")?;
            add(q, "the feed-forward residual")?;
            if !modern {
                let (nw, nb) = (lw(l, TURBO_NOMIC_FFN_LN_WEIGHT), lw(l, TURBO_NOMIC_FFN_LN_BIAS));
                norm(q, self.x, self.x, nw, nb, "the feed-forward LayerNorm")?;
            }
        }
        if modern && layers >= d.layers {
            norm(q, self.x, self.x, t(TURBO_MODERNBERT_FINAL_NORM_WEIGHT), 0, "the final norm")?;
        }
        self.pool_out(q)?;
        Ok(0)
    }
}
