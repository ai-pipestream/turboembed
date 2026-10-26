//! The encoder's lookup and attention launches, by family: the BERT
//! encoder's, or a DeBERTa-v2 model's (deberta.rs).

use super::kernels::*;
use super::session::*;
use super::*;
use crate::backend::{TURBO_DEBERTA_EMBEDDINGS_LN_BIAS, TURBO_DEBERTA_EMBEDDINGS_LN_WEIGHT};
use crate::levelzero::gpu::{Arg, Queue, Res};

impl Session {
    /// The embedding lookup over the packed rows into x, which also copies
    /// the mask and the row table to the device.
    pub(super) fn lookup(&self, q: &mut Queue) -> Res<()> {
        use Arg::*;
        let c = self.ctx();
        let d = &self.model().desc;
        let w = &self.weights.tensors;
        let k = &self.kernels;
        let (batch, tokens, h) = (self.batch, self.tokens, d.hidden);
        let eps = d.layer_norm_eps as f32;
        let n = self.max_batch as u64 * self.max_seq as u64 * 4;
        let staging = self.staging as u64;
        if let Some(dk) = &k.deberta {
            let args = [
                Ptr(staging),
                Ptr(staging + 3 * n),
                Ptr(staging + 4 * n),
                I32(batch as i32),
                Ptr(w[WORD]),
                Ptr(w[TURBO_DEBERTA_EMBEDDINGS_LN_WEIGHT as usize]),
                Ptr(w[TURBO_DEBERTA_EMBEDDINGS_LN_BIAS as usize]),
                F32(eps),
                I32(h as i32),
                Ptr(self.x),
                Ptr(self.packed_mask),
                Ptr(self.rows),
            ];
            return dk.embed.launch(c, q, "the embedding lookup", &args, [tokens, 1, 1]);
        }
        let args = [
            Ptr(staging),
            Ptr(staging + n),
            Ptr(staging + 2 * n),
            I32(self.has_types as i32),
            Ptr(staging + 3 * n),
            Ptr(staging + 4 * n),
            I32(batch as i32),
            Ptr(w[WORD]),
            Ptr(w[POSITION]),
            Ptr(w[TOKEN_TYPE]),
            Ptr(w[EMB_LN_W]),
            Ptr(w[EMB_LN_B]),
            F32(eps),
            I32(h as i32),
            Ptr(self.x),
            Ptr(self.packed_mask),
            Ptr(self.rows),
            I32(tokens as i32),
            Ptr(self.xh),
        ];
        if tokens < FEW_TOKENS {
            k.embed_layer_norm_group.launch(c, q, "the embedding lookup", &args, [tokens, 1, 1])
        } else {
            k.embed_layer_norm.launch(c, q, "the embedding lookup", &args, [tokens.div_ceil(ROWS), 1, 1])
        }
    }

    /// Layer `l`'s attention from qkv into att.
    pub(super) fn attend(&self, q: &mut Queue, l: u32) -> Res<()> {
        use Arg::*;
        let c = self.ctx();
        let d = &self.model().desc;
        let k = &self.kernels;
        let (batch, h) = (self.batch, d.hidden);
        let head_dim = h / d.heads;
        let xmx = k.linear_dpas.is_some();
        if let Some(dk) = &k.deberta {
            let scale = (1.0 / (3.0 * head_dim as f64).sqrt()) as f32;
            let (pos, span) = (self.weights.relative[l as usize], self.max_seq as i32 - 1);
            if let Some((a, queries)) = &dk.tiled {
                let args = [
                    Ptr(self.qkv),
                    Ptr(self.packed_mask),
                    Ptr(self.rows),
                    I32(h as i32),
                    F32(scale),
                    Ptr(self.att),
                    Ptr(pos),
                    Ptr(self.buckets),
                    I32(span),
                ];
                return a.launch(c, q, "attention", &args, [self.longest.div_ceil(*queries), d.heads, batch]);
            }
            let args = [
                Ptr(self.qkv),
                Ptr(self.packed_mask),
                Ptr(self.rows),
                I32(h as i32),
                I32(head_dim as i32),
                F32(scale),
                Ptr(self.att),
                Ptr(pos),
                Ptr(self.buckets),
                I32(span),
                Local(attention_local_bytes(self.longest, head_dim) as usize),
            ];
            return dk.general.launch(c, q, "attention", &args, [self.longest, d.heads, batch]);
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        match &k.attention {
            Attention::Xmx(a, staged) => {
                let args =
                    [Ptr(self.qkv), Ptr(self.packed_mask), Ptr(self.rows), I32(h as i32), F32(scale), Ptr(self.att)];
                // The staged kernel where the longest row fills its group.
                let (a, sgs) = match staged {
                    Some(s) if self.longest >= 16 * ATT2_SUBGROUPS => (s, ATT2_SUBGROUPS),
                    _ => (a, ATT_SUBGROUPS),
                };
                let groups = [self.longest.div_ceil(16 * sgs), d.heads, batch];
                a.launch(c, q, "attention", &args, groups)
            }
            Attention::Tiled(a) => {
                let args =
                    [Ptr(self.qkv), Ptr(self.packed_mask), Ptr(self.rows), I32(h as i32), F32(scale), Ptr(self.att)];
                a.launch(c, q, "attention", &args, [self.longest.div_ceil(QUERIES), d.heads, batch])
            }
            Attention::General(a) => {
                let args = [
                    Ptr(self.qkv),
                    Ptr(self.packed_mask),
                    Ptr(self.rows),
                    I32(h as i32),
                    I32(head_dim as i32),
                    F32(scale),
                    Ptr(if xmx { 0 } else { self.att }),
                    Ptr(if xmx { self.att } else { 0 }),
                    Local(attention_local_bytes(self.longest, head_dim) as usize),
                ];
                a.launch(c, q, "attention", &args, [self.longest, d.heads, batch])
            }
        }
    }
}
