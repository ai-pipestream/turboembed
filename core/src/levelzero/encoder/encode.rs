//! The encoder's launches over the packed rows.

use super::kernels::*;
use super::session::*;
use super::*;
use crate::TURBO_NORMALIZE_L2;
use crate::backend::{TURBO_BERT_EMBEDDING_TENSORS, TURBO_BERT_LAYER_TENSORS};
use crate::levelzero::gpu::{Arg, Queue, Res};

impl Session {
    /// The encoder over the packed rows through `layers` layers, appended
    /// to the queue: all of them for a run, fewer for the tuner, whose
    /// output is not read. Returns the oneDNN primitives it built.
    pub(super) fn encode(&self, q: &mut Queue, layers: u32) -> Res<u64> {
        let c = self.ctx();
        let d = &self.model().desc;
        let w = &self.weights.tensors;
        let k = &self.kernels;
        let (batch, tokens) = (self.batch, self.tokens);
        let (h, inter) = (d.hidden, d.intermediate);
        let eps = d.layer_norm_eps as f32;
        let head_dim = h / d.heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let layer = |l: u32, r: u32| w[(TURBO_BERT_EMBEDDING_TENSORS + l * TURBO_BERT_LAYER_TENSORS + r) as usize];
        use Arg::*;
        // Attention on the matrix engines reads F16 projections and writes
        // an F16 context.
        let half_attention = matches!(k.attention, Attention::Xmx(..));
        // Where the run's token bin takes linear=onednn, oneDNN takes the
        // linear layers of a batch past the few-token kernels: at FASTEST
        // in F16 with its own LayerNorm after the projections back to the
        // hidden width, in F32 with the backend's; the two queues are
        // ordered by waiting.
        #[cfg(feature = "levelzero-onednn")]
        let dnnl = if self.linear.at(tokens) == Linear::Onednn && tokens > FEW_TOKENS_DPAS { c.dnnl() } else { None };
        #[cfg(not(feature = "levelzero-onednn"))]
        let dnnl: Option<()> = None;
        // oneDNN's tokens, the batch's binned, and the primitives the run
        // built: some the first time at a bin, none after.
        #[cfg(feature = "levelzero-onednn")]
        let dm = onednn_tokens(tokens, self.max_batch * self.max_seq);
        let built = std::cell::Cell::new(0u64);
        #[cfg(feature = "levelzero-onednn")]
        let count = |(signal, b): (u64, bool)| {
            built.set(built.get() + b as u64);
            signal
        };
        // Few tokens: a group per token keeps each LayerNorm short.
        let few_tokens = tokens < FEW_TOKENS;
        let xmx = k.linear_dpas.is_some();
        // A linear layer: `which` of the layer's four weights, for the F16
        // copy at FASTEST, and the F32 weight; in F32 its sums split over
        // its terms into at most `splits` parts, and at FASTEST unsplit,
        // whether it writes F16. x is F16 at FASTEST, else F32. Returns
        // the parts.
        let linear = |q: &mut Queue,
                      x: u64,
                      n_in: u32,
                      (l, which, weight): (u32, usize, u64),
                      bias: u64,
                      n_out: u32,
                      y: u64,
                      flags: i32,
                      (splits, to_half): (u32, bool),
                      what: &str|
         -> Res<u32> {
            #[cfg(feature = "levelzero-onednn")]
            if let Some(dn) = dnnl {
                // F16 from the packed weights where the output is F16, F32
                // from the layer's own [n_out, n_in] weights in an F32
                // session; the backend's kernels for the rest.
                let packed = match &self.half {
                    Some(half) if to_half => Some(half.packed[l as usize][which]),
                    None => Some(weight),
                    Some(_) => None,
                };
                if let Some(packed) = packed {
                    let signal = count(dn.matmul(&crate::levelzero::onednn::Matmul {
                        a: x,
                        packed,
                        bias: if flags & LINEAR_BIAS != 0 { bias } else { 0 },
                        residual: 0,
                        c: y,
                        m: dm,
                        k: n_in,
                        n: n_out,
                        gelu: flags & LINEAR_GELU != 0,
                        f32: self.half.is_none(),
                        wait: q.last_event() as u64,
                    })?);
                    q.pending_wait = signal as crate::levelzero::ze::Handle;
                    let _ = (splits, what);
                    return Ok(1);
                }
            }
            if let (Some(kd), Some(half)) = (&k.linear_dpas, &self.half) {
                let (tm, tn, wm, wn, kernel) = if tokens <= FEW_TOKENS_DPAS {
                    (FEW_TM, DPAS_TN, FEW_WM, FEW_WN, &kd[2 + to_half as usize])
                } else {
                    let (tm, tn, wm, wn) = k.dpas;
                    (tm, tn, wm, wn, &kd[to_half as usize])
                };
                let args = [
                    Ptr(x),
                    Ptr(half.layers[l as usize][which]),
                    Ptr(bias),
                    Ptr(y),
                    I32(tokens as i32),
                    I32(n_out as i32),
                    I32(n_in as i32),
                    I32(flags),
                ];
                // Unsplit, so each output's sum runs in one order whatever
                // the batch.
                let groups = [n_out.div_ceil(tn * wn), tokens.div_ceil(tm * wm), 1];
                return kernel.launch(c, q, what, &args, groups).map(|()| 1);
            }
            // More than a handful of tokens, in F32: from the transposed
            // weights, unsplit.
            if let Some(f32t) = &self.f32t
                && tokens > 8
            {
                let args = [
                    Ptr(x),
                    Ptr(f32t.layers[l as usize][which]),
                    Ptr(bias),
                    Ptr(y),
                    I32(tokens as i32),
                    I32(n_out as i32),
                    I32(n_in as i32),
                    I32(flags),
                ];
                let groups = [n_out.div_ceil(SGEMM_TN * SGEMM_WN), tokens.div_ceil(SGEMM_TM * SGEMM_WM), 1];
                return k.linear_sgemm.launch(c, q, what, &args, groups).map(|()| 1);
            }
            // A split takes an equal share of the terms, a multiple of 16.
            let splits = if n_in.is_multiple_of(splits * 16) { splits } else { 1 };
            let k_len = n_in / splits;
            if tokens <= 8
                && let Some(share) = gemv_share(k_len)
            {
                let args = [
                    Ptr(x),
                    Ptr(weight),
                    Ptr(bias),
                    Ptr(y),
                    I32(tokens as i32),
                    I32(n_out as i32),
                    I32(n_in as i32),
                    I32(flags),
                    I32(k_len as i32),
                    I32(share as i32),
                ];
                return k.linear_gemv.launch(c, q, what, &args, [n_out.div_ceil(32), 1, splits]).map(|()| splits);
            }
            if k_len.is_multiple_of(16) {
                let groups = [n_out.div_ceil(SG_N), tokens.div_ceil(SG_T), splits];
                let args = [
                    Ptr(x),
                    Ptr(weight),
                    Ptr(bias),
                    Ptr(y),
                    I32(tokens as i32),
                    I32(n_out as i32),
                    I32(n_in as i32),
                    I32(flags),
                    I32(k_len as i32),
                ];
                return k.linear_sg.launch(c, q, what, &args, groups).map(|()| splits);
            }
            let groups = [n_out.div_ceil(TILE), tokens.div_ceil(TILE), splits];
            let args = [
                Ptr(x),
                Ptr(weight),
                Ptr(bias),
                Ptr(y),
                I32(tokens as i32),
                I32(n_out as i32),
                I32(n_in as i32),
                I32(flags),
                I32(k_len as i32),
            ];
            k.linear.launch(c, q, what, &args, groups).map(|()| splits)
        };
        let add_ln = |q: &mut Queue, bias: u64, lnw: u64, lnb: u64, parts: u32, what: &str| {
            let args = [
                Ptr(self.x),
                Ptr(self.tmp),
                Ptr(bias),
                Ptr(lnw),
                Ptr(lnb),
                F32(eps),
                I32(h as i32),
                I32(parts as i32),
                I32(tokens as i32),
                Ptr(self.xh),
            ];
            if few_tokens {
                k.add_layer_norm_group.launch(c, q, what, &args, [tokens, 1, 1])
            } else {
                k.add_layer_norm_v4.as_ref().unwrap_or(&k.add_layer_norm).launch(
                    c,
                    q,
                    what,
                    &args,
                    [tokens.div_ceil(ROWS), 1, 1],
                )
            }
        };

        // At FASTEST, a projection back to the hidden width with the
        // LayerNorm after it, in one kernel, from F16 act onto the F16
        // residual stream; false where that kernel does not run, for a
        // hidden width wider than a group spans.
        let fused = |q: &mut Queue, act: u64, n_in: u32, (l, which): (u32, usize), (bias, lnw, lnb), what: &str| {
            #[cfg(feature = "levelzero-onednn")]
            if let (Some(dn), Some(half)) = (dnnl, &self.half) {
                // The projection with its residual added, into a scratch
                // the run is done with, then the LayerNorm onto the F16
                // residual stream; the last layer's also widened to F32.
                let scratch = if which == 1 { self.ffn } else { self.att };
                let signal = count(dn.matmul(&crate::levelzero::onednn::Matmul {
                    a: act,
                    packed: half.packed[l as usize][which],
                    bias,
                    residual: self.xh,
                    c: scratch,
                    m: dm,
                    k: n_in,
                    n: h,
                    gelu: false,
                    f32: false,
                    wait: q.last_event() as u64,
                })?);
                let signal = count(dn.layer_norm(&crate::levelzero::onednn::LayerNorm {
                    src: scratch,
                    gamma: lnw,
                    beta: lnb,
                    eps,
                    dst: self.xh,
                    m: dm,
                    n: h,
                    wait: signal,
                })?);
                q.pending_wait = signal as crate::levelzero::ze::Handle;
                if l + 1 == d.layers && which == 3 {
                    let n = tokens as u64 * h as u64;
                    let args = [Ptr(self.xh), U64(n), Ptr(self.x)];
                    k.widen_f16.launch(c, q, what, &args, [elementwise_groups(n), 1, 1])?;
                }
                return Ok(true);
            }
            // In F32, oneDNN's projection with its bias and the residual
            // added, then the LayerNorm from that alone.
            #[cfg(feature = "levelzero-onednn")]
            if let (Some(dn), None, Some(kln)) = (dnnl, &self.half, &k.add_layer_norm_v4) {
                let weight = layer(l, if which == 1 { ATTN_OUT_WEIGHT } else { FFN_OUT_WEIGHT });
                let signal = count(dn.matmul(&crate::levelzero::onednn::Matmul {
                    a: act,
                    packed: weight,
                    bias,
                    residual: self.x,
                    c: self.tmp,
                    m: dm,
                    k: n_in,
                    n: h,
                    gelu: false,
                    f32: true,
                    wait: q.last_event() as u64,
                })?);
                q.pending_wait = signal as crate::levelzero::ze::Handle;
                let args = [
                    Ptr(self.x),
                    Ptr(self.tmp),
                    Ptr(0),
                    Ptr(lnw),
                    Ptr(lnb),
                    F32(eps),
                    I32(h as i32),
                    I32(0),
                    I32(tokens as i32),
                    Ptr(0),
                ];
                kln.launch(c, q, what, &args, [tokens.div_ceil(ROWS), 1, 1])?;
                return Ok(true);
            }
            let (Some(kln), Some(half)) = (&k.linear_dpas_layer_norm, &self.half) else { return Ok(false) };
            let args = [
                Ptr(act),
                Ptr(half.layers[l as usize][which]),
                Ptr(bias),
                // The residual stream is F16; the last layer's also goes to
                // x in F32, for the pooling.
                Ptr(if l + 1 == d.layers && which == 3 { self.x } else { 0 }),
                Ptr(self.xh),
                Ptr(lnw),
                Ptr(lnb),
                F32(eps),
                I32(tokens as i32),
                I32(h as i32),
                I32(n_in as i32),
            ];
            // Fewer tokens than a full group of blocks: one block a group,
            // so at most one block's tail computes rows past the last token. A row's
            // sums run in one order either way.
            let (tm, blocks) = k.ln;
            let (kln, rows) = if tokens < tm * blocks { (&kln[0], tm) } else { (&kln[1], tm * blocks) };
            kln.launch(c, q, what, &args, [tokens.div_ceil(rows), 1, 1]).map(|()| true)
        };
        let n = self.max_batch as u64 * self.max_seq as u64 * 4;
        let staging = self.staging as u64;
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
        if few_tokens {
            k.embed_layer_norm_group.launch(c, q, "the embedding lookup", &args, [tokens, 1, 1])?;
        } else {
            k.embed_layer_norm.launch(c, q, "the embedding lookup", &args, [tokens.div_ceil(ROWS), 1, 1])?;
        }
        for l in 0..layers.min(d.layers) {
            // The linear layers that read the hidden states read them in F16 at
            // FASTEST, as the LayerNorms leave them.
            let x = if xmx { self.xh } else { self.x };
            let (qkv_w, qkv_b) = self.weights.qkv[l as usize];
            linear(
                q,
                x,
                h,
                (l, 0, qkv_w),
                qkv_b,
                3 * h,
                self.qkv,
                LINEAR_BIAS,
                (1, half_attention),
                "the query, key and value projection",
            )?;
            match &k.attention {
                Attention::Xmx(a, staged) => {
                    let args = [
                        Ptr(self.qkv),
                        Ptr(self.packed_mask),
                        Ptr(self.rows),
                        I32(h as i32),
                        F32(scale),
                        Ptr(self.att),
                    ];
                    // The staged kernel where the longest row fills its group.
                    let (a, sgs) = match staged {
                        Some(s) if self.longest >= 16 * ATT2_SUBGROUPS => (s, ATT2_SUBGROUPS),
                        _ => (a, ATT_SUBGROUPS),
                    };
                    let groups = [self.longest.div_ceil(16 * sgs), d.heads, batch];
                    a.launch(c, q, "attention", &args, groups)?;
                }
                Attention::Tiled(a) => {
                    let args = [
                        Ptr(self.qkv),
                        Ptr(self.packed_mask),
                        Ptr(self.rows),
                        I32(h as i32),
                        F32(scale),
                        Ptr(self.att),
                    ];
                    a.launch(c, q, "attention", &args, [self.longest.div_ceil(QUERIES), d.heads, batch])?;
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
                    a.launch(c, q, "attention", &args, [self.longest, d.heads, batch])?;
                }
            }
            let wo = (l, 1, layer(l, ATTN_OUT_WEIGHT));
            let ln = (layer(l, ATTN_OUT_BIAS), layer(l, ATTN_LN_WEIGHT), layer(l, ATTN_LN_BIAS));
            if !fused(q, self.att, h, (l, 1), ln, "the attention output and LayerNorm")? {
                let parts =
                    linear(q, self.att, h, wo, 0, h, self.tmp, 0, (1, false), "the attention output projection")?;
                add_ln(q, ln.0, ln.1, ln.2, parts, "the attention LayerNorm")?;
            }
            // Large batches: the whole block in one kernel, its middle kept
            // in local memory.
            if let (Some((kmlp, blocks)), Some(half)) = (&k.linear_dpas_mlp, &self.half)
                && tokens >= MLP_TOKENS
                && dnnl.is_none()
            {
                let args = [
                    Ptr(half.layers[l as usize][2]),
                    Ptr(layer(l, FFN_IN_BIAS)),
                    Ptr(half.layers[l as usize][3]),
                    Ptr(layer(l, FFN_OUT_BIAS)),
                    Ptr(if l + 1 == d.layers { self.x } else { 0 }),
                    Ptr(self.xh),
                    Ptr(layer(l, FFN_LN_WEIGHT)),
                    Ptr(layer(l, FFN_LN_BIAS)),
                    F32(eps),
                    I32(tokens as i32),
                    I32(h as i32),
                    I32(inter as i32),
                    Local(mlp_local_bytes(h, *blocks) as usize),
                ];
                let groups = [tokens.div_ceil(DPAS_LN_TM * blocks), 1, 1];
                kmlp.launch(c, q, "the feed-forward block", &args, groups)?;
                continue;
            }
            let (wi, bi) = ((l, 2, layer(l, FFN_IN_WEIGHT)), layer(l, FFN_IN_BIAS));
            linear(
                q,
                x,
                h,
                wi,
                bi,
                inter,
                self.ffn,
                LINEAR_BIAS | LINEAR_GELU,
                (1, true),
                "the feed-forward input and GELU",
            )?;
            let wf = (l, 3, layer(l, FFN_OUT_WEIGHT));
            let ln = (layer(l, FFN_OUT_BIAS), layer(l, FFN_LN_WEIGHT), layer(l, FFN_LN_BIAS));
            if !fused(q, self.ffn, inter, (l, 3), ln, "the feed-forward output and LayerNorm")? {
                let parts =
                    linear(q, self.ffn, inter, wf, 0, h, self.tmp, 0, (SPLITS, false), "the feed-forward output")?;
                add_ln(q, ln.0, ln.1, ln.2, parts, "the feed-forward LayerNorm")?;
            }
        }
        // A vector per row, or per range after embed_write_ranges, whose
        // ranges the kernel reads from the staging after the row table.
        let vectors = if self.ranges > 0 {
            let args = [
                Ptr(self.x),
                Ptr(self.packed_mask),
                Ptr(self.rows),
                Ptr(self.ranges_at() as u64),
                I32(h as i32),
                I32(self.output_dim as i32),
                Ptr(self.output.ptr as u64),
            ];
            let groups = [self.ranges, self.output_dim.div_ceil(16), 1];
            k.pool_ranges.launch(c, q, "range pooling", &args, groups)?;
            self.ranges
        } else {
            let args = [
                Ptr(self.x),
                Ptr(self.packed_mask),
                Ptr(self.rows),
                I32(h as i32),
                I32(self.output_dim as i32),
                I32(self.pooling as i32),
                Ptr(self.output.ptr as u64),
            ];
            k.pool.launch(c, q, "pooling", &args, [batch, self.output_dim.div_ceil(16), 1])?;
            batch
        };
        if self.normalize == TURBO_NORMALIZE_L2 {
            let args = [Ptr(self.output.ptr as u64), I32(self.output_dim as i32)];
            k.normalize.launch(c, q, "normalization", &args, [vectors, 1, 1])?;
        }
        Ok(built.get())
    }
}
