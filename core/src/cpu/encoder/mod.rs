//! A BERT encoder in F32 on the host: the family TURBO_FAMILY_BERT names.
//!
//! Embeddings are the sum of the word, position and token type rows,
//! LayerNorm'd. Each layer is self-attention over the row's live tokens
//! (Q, K and V projections, scaled dot products, softmax, the output
//! projection), a residual and LayerNorm, then a feed-forward block with
//! GELU (erf), a residual and LayerNorm. Then each row is pooled (mean over
//! its mask, its first token, or its last live token), cut to output_dim,
//! and L2-normalized when asked.
//!
//! Rows written with ranges (`write_ranges`) are pooled by range instead:
//! each range's vector is the mean over its columns' live tokens of the
//! final hidden states, read where the encoder left them, with the same
//! sum, cut and normalization as a row's mean.
//!
//! The rows of a batch are packed: row r's positions up to its last live
//! token sit one after another with the other rows', so padding past that
//! token is never computed. No output depends on it: attention skips
//! masked keys, and no pooling reads past the last live token. Positions
//! are the row's own column indices, as upstream numbers them.
//!
//! A run is a sequence of steps, each split into tasks that the session's
//! threads take (pool.rs): the embeddings and each LayerNorm by chunks of
//! tokens, each linear layer by tiles of its output (kernels/), and
//! attention by row, head and block of queries, and pooling by range by
//! groups of ranges. A task computes each of
//! its outputs whole, in the same order whichever thread takes it and
//! however many threads there are; no sum is split between tasks. So a
//! run gives the same bits for the same rows on any number of threads.
//!
//! Every buffer is allocated by `new`, for the session's largest batch and
//! its threads; `write` and `run` allocate nothing.

use crate::backend::{
    TURBO_BERT_EMBEDDING_TENSORS, TURBO_BERT_LAYER_TENSORS, turbo_backend_embed_rows, turbo_backend_model,
};
use crate::{TURBO_NORMALIZE_L2, TURBO_POOLING_CLS, TURBO_POOLING_LAST, turbo_token_range};

use super::kernels::{Attend, Epilogue, Gemm, LANES, Packed, zeroed};
use super::pool::Pool;

mod job;
#[cfg(test)]
mod tests;

use job::{Bufs, Job, Step, entry};

// TURBO_BERT_* in turbo_backend.h.
const WORD: usize = 0;
const POSITION: usize = 1;
const TOKEN_TYPE: usize = 2;
const EMB_LN_W: usize = 3;
const EMB_LN_B: usize = 4;
pub(super) const Q_W: usize = 0;
pub(super) const Q_B: usize = 1;
pub(super) const K_W: usize = 2;
pub(super) const K_B: usize = 3;
pub(super) const V_W: usize = 4;
pub(super) const V_B: usize = 5;
pub(super) const O_W: usize = 6;
const O_B: usize = 7;
const ATTN_LN_W: usize = 8;
const ATTN_LN_B: usize = 9;
pub(super) const FFN_IN_W: usize = 10;
const FFN_IN_B: usize = 11;
pub(super) const FFN_OUT_W: usize = 12;
const FFN_OUT_B: usize = 13;
const FFN_LN_W: usize = 14;
const FFN_LN_B: usize = 15;

/// Tokens per task of the steps that go token by token.
const TOKENS: usize = 16;
/// Queries per attention task.
const QUERIES: usize = 32;
/// Range pooling tasks per thread, so that threads that finish early take
/// more of the ranges, which differ in length.
const RANGE_TASKS_PER_THREAD: usize = 4;

/// What an encoder is allocated for.
#[derive(Clone, Copy)]
pub(super) struct Size {
    /// The most threads it runs on.
    pub threads: usize,
    pub max_batch: usize,
    pub max_seq: usize,
    /// The most ranges one `write_ranges` holds.
    pub max_ranges: usize,
}

pub(super) struct Encoder {
    layers: usize,
    hidden: usize,
    heads: usize,
    intermediate: usize,
    eps: f32,
    /// Every tensor in F32, in TURBO_BERT_* order: pointers into the
    /// core's weights, or into the model's converted copy, both of which
    /// outlive the session.
    tensors: Vec<(*const f32, usize)>,
    /// The model's linear layers, packed for the kernel this processor
    /// runs: the model's, which outlives the session.
    packed: *const Packed,

    max_seq: usize,
    /// The position table row a row's first token reads.
    position_offset: usize,
    // The written rows, [batch, seq] packed with no stride.
    batch: usize,
    seq: usize,
    ids: Vec<u32>,
    types: Vec<u32>,
    mask: Vec<u8>,
    pooling: u32,
    normalize: u32,
    output_dim: usize,
    /// The ranges the run pools by, as `write_ranges` copied them in, its
    /// capacity the session's max_ranges. Empty after `write`: the run
    /// then pools each row.
    ranges: Vec<turbo_token_range>,

    /// Each row's first packed token and its length through its last
    /// live token.
    rows: Vec<(usize, usize)>,
    /// Each packed token's place in ids.
    at: Vec<u32>,
    /// Each row's first slot in kt, and each packed token's slot: a row
    /// takes its length rounded up to LANES slots, the rest zero.
    padded: Vec<usize>,
    slot: Vec<u32>,
    /// The rows, longest first: attention takes the longest rows' tasks
    /// first, so no thread starts one near the end of the step.
    order: Vec<usize>,
    /// The attention tasks before each row of `order`, per head: the
    /// row's blocks of QUERIES queries. One longer than the batch.
    blocks: Vec<usize>,
    // Scratch: [tokens, hidden] for the hidden states and the attention
    // context; [tokens, 3 * hidden] for the Q and V projections (K's third
    // is left unwritten); [hidden, stride] for the K projection
    // transposed, a column per row of stride slots; and [tokens,
    // intermediate] for the feed-forward block.
    x: Vec<f32>,
    qv: Vec<f32>,
    kt: Vec<f32>,
    stride: usize,
    ctx: Vec<f32>,
    ffn: Vec<f32>,
    /// Each thread's attention scratch, `per_thread` floats apiece.
    scratch: Vec<f32>,
    per_thread: usize,
}

impl Encoder {
    /// An encoder for `size.max_batch` rows of `size.max_seq` tokens and
    /// up to `size.max_ranges` ranges over `tensors` and `packed`, which
    /// stay where they are while it lives, run on up to `size.threads`
    /// threads. Err is the bytes it could not allocate.
    pub(super) fn new(
        d: &turbo_backend_model,
        tensors: Vec<&[f32]>,
        packed: &Packed,
        size: Size,
    ) -> Result<Encoder, usize> {
        let Size { threads, max_batch, max_seq, max_ranges } = size;
        debug_assert_eq!(tensors.len(), (TURBO_BERT_EMBEDDING_TENSORS + d.layers * TURBO_BERT_LAYER_TENSORS) as usize);
        debug_assert_eq!(packed.layers.len(), d.layers as usize);
        let (h, i) = (d.hidden as usize, d.intermediate as usize);
        let tokens = max_batch.checked_mul(max_seq).ok_or(usize::MAX)?;
        let wide = |n: usize| tokens.checked_mul(n).ok_or(usize::MAX);
        let per_thread = Attend::scratch(max_seq, h / d.heads as usize);
        let stride = max_batch.checked_mul(max_seq.next_multiple_of(LANES)).ok_or(usize::MAX)?;
        let mut ranges = Vec::new();
        ranges.try_reserve_exact(max_ranges).map_err(|_| max_ranges.saturating_mul(size_of::<turbo_token_range>()))?;
        Ok(Encoder {
            layers: d.layers as usize,
            hidden: h,
            heads: d.heads as usize,
            intermediate: i,
            eps: d.layer_norm_eps as f32,
            tensors: tensors.iter().map(|t| (t.as_ptr(), t.len())).collect(),
            packed,
            max_seq,
            position_offset: d.position_offset as usize,
            batch: 0,
            seq: 0,
            ids: zeroed(tokens)?,
            types: zeroed(tokens)?,
            mask: zeroed(tokens)?,
            pooling: 0,
            normalize: 0,
            output_dim: 0,
            ranges,
            rows: zeroed(max_batch)?,
            at: zeroed(tokens)?,
            padded: zeroed(max_batch)?,
            slot: zeroed(tokens)?,
            order: zeroed(max_batch)?,
            blocks: zeroed(max_batch + 1)?,
            x: zeroed(wide(h)?)?,
            qv: zeroed(wide(3 * h)?)?,
            kt: zeroed(stride.checked_mul(h).ok_or(usize::MAX)?)?,
            stride,
            ctx: zeroed(wide(h)?)?,
            ffn: zeroed(wide(i)?)?,
            scratch: zeroed(threads.checked_mul(per_thread).ok_or(usize::MAX)?)?,
            per_thread,
        })
    }

    fn tensor(&self, i: usize) -> &[f32] {
        let (p, n) = self.tensors[i];
        // SAFETY: a tensor of the model, which outlives the session.
        unsafe { std::slice::from_raw_parts(p, n) }
    }

    fn layer(&self, l: usize, r: usize) -> &[f32] {
        self.tensor(TURBO_BERT_EMBEDDING_TENSORS as usize + l * TURBO_BERT_LAYER_TENSORS as usize + r)
    }

    pub(super) fn normalize(&self) -> u32 {
        self.normalize
    }

    /// Copy in the rows the core checked, dropping their stride; the run
    /// pools each row.
    pub(super) fn write(&mut self, r: &turbo_backend_embed_rows, ids: &[i32], mask: &[i32], types: Option<&[i32]>) {
        self.copy_rows(r, ids, mask, types);
        self.ranges.clear();
    }

    /// Copy in the rows and the ranges the core checked; the run pools
    /// each range. There are at most the session's max_ranges, so the
    /// copy fits in the memory `new` gave it.
    pub(super) fn write_ranges(
        &mut self,
        r: &turbo_backend_embed_rows,
        ids: &[i32],
        mask: &[i32],
        types: Option<&[i32]>,
        ranges: &[turbo_token_range],
    ) {
        debug_assert!(!ranges.is_empty() && ranges.len() <= self.ranges.capacity());
        self.copy_rows(r, ids, mask, types);
        self.ranges.clear();
        self.ranges.extend_from_slice(ranges);
    }

    fn copy_rows(&mut self, r: &turbo_backend_embed_rows, ids: &[i32], mask: &[i32], types: Option<&[i32]>) {
        let (b, s, stride) = (r.batch as usize, r.seq as usize, r.row_stride as usize);
        debug_assert!(s <= self.max_seq && b * s <= self.ids.len());
        for row in 0..b {
            let (src, dst) = (row * stride, row * s);
            for p in 0..s {
                self.ids[dst + p] = ids[src + p] as u32;
                self.mask[dst + p] = mask[src + p] as u8;
                self.types[dst + p] = types.map_or(0, |t| t[src + p] as u32);
            }
        }
        self.batch = b;
        self.seq = s;
        self.pooling = r.pooling;
        self.normalize = r.normalize;
        self.output_dim = r.output_dim as usize;
    }

    /// Run the written rows into `out`, [batch, output_dim] packed, or
    /// [ranges, output_dim] after `write_ranges`, on `pool`, which has at
    /// most the threads the encoder was made for.
    pub(super) fn run(&mut self, pool: &mut Pool, out: &mut [f32]) {
        debug_assert!(pool.threads() * self.per_thread <= self.scratch.len());
        let tokens = self.pack();
        let bufs = Bufs {
            x: self.x.as_mut_ptr(),
            qv: self.qv.as_mut_ptr(),
            kt: self.kt.as_mut_ptr(),
            ctx: self.ctx.as_mut_ptr(),
            ffn: self.ffn.as_mut_ptr(),
            scratch: self.scratch.as_mut_ptr(),
        };
        let e = &*self;
        // SAFETY: the model's, which outlives the session.
        let packed = unsafe { &*e.packed };
        let entry = entry(packed.isa);
        let (h, n_i) = (e.hidden, e.intermediate);
        let go = |pool: &mut Pool, step: Step, tasks: usize| {
            let job = Job { e, bufs, step, tokens };
            // SAFETY: entry is the kernel of the instruction set the
            // weights were packed for, which this processor has; job's
            // buffers are the encoder's, sized for the batch, and each
            // task writes only its own part of them (see Job).
            pool.run(tasks, &|t, th| unsafe { entry(&job, t, th) });
        };
        let linear = |pool: &mut Pool, g: Gemm| go(pool, Step::Linear(g), g.tasks());
        let chunks = tokens.div_ceil(TOKENS);
        go(pool, Step::Lookup, chunks);
        for l in 0..e.layers {
            let (pl, w) = (&packed.layers[l], |r| e.layer(l, r));
            let gemm = |x, ldx, w, bias, y, ldy, epilogue| Gemm { x, ldx, rows: tokens, w, bias, y, ldy, epilogue };
            let split = Epilogue::Split { lo: h, hi: 2 * h, t_out: bufs.kt, stride: e.stride, slot: e.slot.as_ptr() };
            linear(pool, gemm(bufs.x, h, &pl.qkv, &pl.qkv_bias, bufs.qv, 3 * h, split));
            go(pool, Step::Attention, e.heads * e.blocks[e.batch]);
            linear(pool, gemm(bufs.ctx, h, &pl.out, w(O_B), bufs.x, h, Epilogue::Residual));
            go(pool, Step::Norm(w(ATTN_LN_W), w(ATTN_LN_B)), chunks);
            linear(pool, gemm(bufs.x, h, &pl.ffn_in, w(FFN_IN_B), bufs.ffn, n_i, Epilogue::Gelu));
            linear(pool, gemm(bufs.ffn, n_i, &pl.ffn_out, w(FFN_OUT_B), bufs.x, h, Epilogue::Residual));
            go(pool, Step::Norm(w(FFN_LN_W), w(FFN_LN_B)), chunks);
        }
        if e.ranges.is_empty() {
            self.pool(out);
            return;
        }
        let count = e.ranges.len();
        debug_assert!(out.len() >= count * e.output_dim);
        let per_task = count.div_ceil(pool.threads() * RANGE_TASKS_PER_THREAD);
        go(pool, Step::PoolRanges { out: out.as_mut_ptr(), per_task }, count.div_ceil(per_task));
    }

    /// Each row's place among the packed tokens, each token's place in
    /// the rows, and the order attention takes the rows in. Returns the
    /// packed tokens.
    fn pack(&mut self) -> usize {
        let (mut t, mut slot) = (0, 0);
        for r in 0..self.batch {
            let m = &self.mask[r * self.seq..(r + 1) * self.seq];
            let len = m.iter().rposition(|&v| v != 0).map_or(0, |p| p + 1);
            self.rows[r] = (t, len);
            self.padded[r] = slot;
            for p in 0..len {
                self.at[t + p] = (r * self.seq + p) as u32;
                self.slot[t + p] = (slot + p) as u32;
            }
            // The slots past the row's tokens, which attention reads and
            // weighs by 0: zero, not what an earlier run left there.
            let lp = len.next_multiple_of(LANES);
            for col in self.kt.chunks_exact_mut(self.stride) {
                col[slot + len..slot + lp].fill(0.0);
            }
            t += len;
            slot += lp;
        }
        let (rows, order) = (&self.rows, &mut self.order[..self.batch]);
        for (i, o) in order.iter_mut().enumerate() {
            *o = i;
        }
        // In place: sort_unstable allocates nothing.
        order.sort_unstable_by_key(|&r| std::cmp::Reverse(rows[r].1));
        self.blocks[0] = 0;
        for (i, &r) in order.iter().enumerate() {
            self.blocks[i + 1] = self.blocks[i] + rows[r].1.div_ceil(QUERIES);
        }
        t
    }

    /// Pool each row, cut it to output_dim, and normalize it. The cut
    /// comes first, so an L2-normalized vector is unit length at the
    /// width the caller asked for.
    fn pool(&self, out: &mut [f32]) {
        let (h, od) = (self.hidden, self.output_dim);
        for r in 0..self.batch {
            let (start, len) = self.rows[r];
            let dst = &mut out[r * od..][..od];
            match self.pooling {
                TURBO_POOLING_CLS => dst.copy_from_slice(&self.x[start * h..][..od]),
                TURBO_POOLING_LAST => dst.copy_from_slice(&self.x[(start + len - 1) * h..][..od]),
                _ => self.mean(&self.x, r, 0..len, dst),
            }
            self.finish(dst);
        }
    }

    /// The mean of row `r`'s live tokens in columns `cols`, which end at
    /// or before the row's packed length, from the hidden states `x`,
    /// into `dst`: output_dim values. As upstream's mean pooling divides
    /// the masked sum by the mask's sum: summed in F32 in column order,
    /// then scaled by one over the count. At least one token is live.
    #[inline(always)]
    fn mean(&self, x: &[f32], r: usize, cols: std::ops::Range<usize>, dst: &mut [f32]) {
        let (h, od) = (self.hidden, dst.len());
        let start = self.rows[r].0;
        dst.fill(0.0);
        let mask = &self.mask[r * self.seq..][..cols.end];
        let mut n = 0u32;
        for p in cols.filter(|&p| mask[p] != 0) {
            for (d, &x) in dst.iter_mut().zip(&x[(start + p) * h..][..od]) {
                *d += x;
            }
            n += 1;
        }
        debug_assert!(n > 0, "row {r}: no live token to pool");
        let inv = 1.0 / n as f32;
        for d in dst.iter_mut() {
            *d *= inv;
        }
    }

    /// Normalize a pooled vector as the write asked.
    #[inline(always)]
    fn finish(&self, dst: &mut [f32]) {
        if self.normalize == TURBO_NORMALIZE_L2 {
            // As upstream: divided by the norm, or by 1e-12 when the
            // norm is smaller.
            let norm = dst.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt().max(1e-12);
            let inv = (1.0 / norm) as f32;
            for d in dst.iter_mut() {
                *d *= inv;
            }
        }
    }

    /// The final hidden states of the last run, [packed tokens, hidden],
    /// and where each row's tokens start among them and how many there
    /// are through its last live token.
    #[cfg(test)]
    fn hidden_states(&self) -> (&[f32], &[(usize, usize)]) {
        (&self.x, &self.rows[..self.batch])
    }
}
