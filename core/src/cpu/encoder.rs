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
//! The rows of a batch are packed: row r's positions up to its last live
//! token sit one after another with the other rows', so padding past that
//! token is never computed. No output depends on it: attention skips
//! masked keys, and no pooling reads past the last live token. Positions
//! are the row's own column indices, as upstream numbers them.
//!
//! A run is a sequence of steps, each split into tasks that the session's
//! threads take (pool.rs): the embeddings and each LayerNorm by chunks of
//! tokens, each linear layer by tiles of its output (kernels.rs), and
//! attention by row, head and block of queries. A task computes each of
//! its outputs whole, in the same order whichever thread takes it and
//! however many threads there are; no sum is split between tasks. So a
//! run gives the same bits for the same rows on any number of threads.
//!
//! Every buffer is allocated by `new`, for the session's largest batch and
//! its threads; `write` and `run` allocate nothing.

use crate::backend::{
    TURBO_BERT_EMBEDDING_TENSORS, TURBO_BERT_LAYER_TENSORS, turbo_backend_embed_rows, turbo_backend_model,
};
use crate::{TURBO_NORMALIZE_L2, TURBO_POOLING_CLS, TURBO_POOLING_LAST};

use super::kernels::{Attend, Epilogue, Gemm, Isa, LANES, Micro, Packed, layer_norm, zeroed};
use super::pool::Pool;

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
    // The written rows, [batch, seq] packed with no stride.
    batch: usize,
    seq: usize,
    ids: Vec<u32>,
    types: Vec<u32>,
    mask: Vec<u8>,
    pooling: u32,
    normalize: u32,
    output_dim: usize,

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
    /// An encoder for `max_batch` rows of `max_seq` tokens over `tensors`
    /// and `packed`, which stay where they are while it lives, run on up
    /// to `threads` threads. Err is the bytes it could not allocate.
    pub(super) fn new(
        d: &turbo_backend_model,
        tensors: Vec<&[f32]>,
        packed: &Packed,
        threads: usize,
        max_batch: usize,
        max_seq: usize,
    ) -> Result<Encoder, usize> {
        debug_assert_eq!(tensors.len(), (TURBO_BERT_EMBEDDING_TENSORS + d.layers * TURBO_BERT_LAYER_TENSORS) as usize);
        debug_assert_eq!(packed.layers.len(), d.layers as usize);
        let (h, i) = (d.hidden as usize, d.intermediate as usize);
        let tokens = max_batch.checked_mul(max_seq).ok_or(usize::MAX)?;
        let wide = |n: usize| tokens.checked_mul(n).ok_or(usize::MAX);
        let per_thread = Attend::scratch(max_seq, h / d.heads as usize);
        let stride = max_batch.checked_mul(max_seq.next_multiple_of(LANES)).ok_or(usize::MAX)?;
        Ok(Encoder {
            layers: d.layers as usize,
            hidden: h,
            heads: d.heads as usize,
            intermediate: i,
            eps: d.layer_norm_eps as f32,
            tensors: tensors.iter().map(|t| (t.as_ptr(), t.len())).collect(),
            packed,
            max_seq,
            batch: 0,
            seq: 0,
            ids: zeroed(tokens)?,
            types: zeroed(tokens)?,
            mask: zeroed(tokens)?,
            pooling: 0,
            normalize: 0,
            output_dim: 0,
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

    /// Copy in the rows the core checked, dropping their stride.
    pub(super) fn write(&mut self, r: &turbo_backend_embed_rows, ids: &[i32], mask: &[i32], types: Option<&[i32]>) {
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

    /// Run the written rows into `out`, [batch, output_dim] packed, on
    /// `pool`, which has at most the threads the encoder was made for.
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
        self.pool(out);
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
                // Mean over the tokens whose mask is 1, as upstream's
                // mean pooling divides the masked sum by the mask's sum.
                _ => {
                    dst.fill(0.0);
                    let mask = &self.mask[r * self.seq..][..len];
                    let mut n = 0u32;
                    for (p, _) in mask.iter().enumerate().filter(|(_, m)| **m != 0) {
                        for (d, &x) in dst.iter_mut().zip(&self.x[(start + p) * h..][..od]) {
                            *d += x;
                        }
                        n += 1;
                    }
                    let inv = 1.0 / n as f32;
                    for d in dst.iter_mut() {
                        *d *= inv;
                    }
                }
            }
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
    }
}

/// The encoder's scratch, written by the tasks of a step.
#[derive(Clone, Copy)]
struct Bufs {
    x: *mut f32,
    qv: *mut f32,
    kt: *mut f32,
    ctx: *mut f32,
    ffn: *mut f32,
    scratch: *mut f32,
}

/// One step of a run.
enum Step<'a> {
    /// The embeddings of each token, LayerNorm'd, into x.
    Lookup,
    Linear(Gemm<'a>),
    /// Attention within each row and head, from qv and kt into ctx.
    Attention,
    /// x = LayerNorm(x), with these weights and biases.
    Norm(&'a [f32], &'a [f32]),
}

/// A step and what its tasks read and write.
///
/// Every task of a step writes a part of the output no other task of the
/// step touches, and reads only what no task of the step writes, apart
/// from its own part in place: a chunk of tokens' rows of x (Lookup,
/// Norm); a tile of rows by columns of the product's output (Linear); the
/// ctx rows of a block of one row's queries in one head's columns, and the
/// thread's own scratch (Attention). Steps are one after another: the
/// pool returns only when every task of a step is done.
struct Job<'a> {
    e: &'a Encoder,
    bufs: Bufs,
    step: Step<'a>,
    tokens: usize,
}

// SAFETY: the raw pointers in a Job are shared between the pool's
// threads under the rule above, so no two threads touch the same float
// while either writes it; the rest is read only.
unsafe impl Sync for Job<'_> {}

impl Job<'_> {
    /// # Safety
    /// As entry().
    #[inline(always)]
    unsafe fn task<M: Micro>(&self, t: usize, th: usize) {
        // SAFETY: per step, under the rule on Job.
        unsafe {
            match &self.step {
                Step::Lookup => self.lookup(t),
                Step::Linear(g) => g.task::<M>(t),
                Step::Attention => self.attention::<M>(t, th),
                Step::Norm(w, b) => self.norm(t, w, b),
            }
        }
    }

    /// The tokens of chunk t.
    fn chunk(&self, t: usize) -> std::ops::Range<usize> {
        t * TOKENS..((t + 1) * TOKENS).min(self.tokens)
    }

    /// Token `tok`'s row of x.
    ///
    /// # Safety
    /// No other thread touches it.
    // A row of the encoder's x, which the job holds by pointer: see Job.
    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    unsafe fn x_row(&self, tok: usize) -> &mut [f32] {
        let h = self.e.hidden;
        // SAFETY: tok is under the packed tokens, which x holds.
        unsafe { std::slice::from_raw_parts_mut(self.bufs.x.add(tok * h), h) }
    }

    #[inline(always)]
    unsafe fn lookup(&self, t: usize) {
        let (e, h) = (self.e, self.e.hidden);
        let (word, pos, ty) = (e.tensor(WORD), e.tensor(POSITION), e.tensor(TOKEN_TYPE));
        let (ln_w, ln_b) = (e.tensor(EMB_LN_W), e.tensor(EMB_LN_B));
        for tok in self.chunk(t) {
            let at = e.at[tok] as usize;
            let (id, ty_id, p) = (e.ids[at] as usize, e.types[at] as usize, at % e.seq);
            let (w, ps, tt) = (&word[id * h..][..h], &pos[p * h..][..h], &ty[ty_id * h..][..h]);
            // SAFETY: this chunk's token.
            let dst = unsafe { self.x_row(tok) };
            for i in 0..h {
                dst[i] = w[i] + ps[i] + tt[i];
            }
            layer_norm(dst, ln_w, ln_b, e.eps);
        }
    }

    #[inline(always)]
    unsafe fn norm(&self, t: usize, w: &[f32], b: &[f32]) {
        for tok in self.chunk(t) {
            // SAFETY: this chunk's token.
            layer_norm(unsafe { self.x_row(tok) }, w, b, self.e.eps);
        }
    }

    /// Task t: head t % heads of a block of one row's queries.
    #[inline(always)]
    unsafe fn attention<M: Micro>(&self, t: usize, th: usize) {
        let e = self.e;
        let (h, heads) = (e.hidden, e.heads);
        let d = h / heads;
        let (head, block) = (t % heads, t / heads);
        let i = e.blocks[..=e.batch].partition_point(|&b| b <= block) - 1;
        let r = e.order[i];
        let (start, len) = e.rows[r];
        let q0 = (block - e.blocks[i]) * QUERIES;
        let a = Attend {
            // SAFETY: the row's tokens are under the packed tokens, and
            // its slots in each column of kvt under stride.
            qv: unsafe { self.bufs.qv.add(start * 3 * h) },
            ld: 3 * h,
            q_at: head * d,
            v_at: 2 * h + head * d,
            kt: unsafe { self.bufs.kt.add(head * d * e.stride + e.padded[r]) },
            stride: e.stride,
            d,
            len,
            mask: &e.mask[r * e.seq..][..len],
            scale: 1.0 / (d as f32).sqrt(),
            fma: M::FMA,
            ctx: unsafe { self.bufs.ctx.add(start * h) },
            ldc: h,
            c_at: head * d,
        };
        // SAFETY: th is under the pool's threads, each with its scratch.
        let scratch = unsafe { std::slice::from_raw_parts_mut(self.bufs.scratch.add(th * e.per_thread), e.per_thread) };
        // SAFETY: this task's block of queries in this head; the
        // processor has M's instruction set (entry()).
        unsafe { a.run::<M::V>(q0, (q0 + QUERIES).min(len), scratch) };
    }
}

/// Task t of a job, on thread th, with one instruction set's kernels.
///
/// # Safety
/// The job's weights were packed for that instruction set and the
/// processor has it; the job follows the rule on Job.
type Entry = unsafe fn(&Job, usize, usize);

fn entry(isa: Isa) -> Entry {
    match isa {
        Isa::Portable => entry_portable,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => entry_avx2,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => entry_avx512,
    }
}

unsafe fn entry_portable(j: &Job, t: usize, th: usize) {
    // SAFETY: as Entry.
    unsafe { j.task::<super::kernels::Portable>(t, th) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn entry_avx2(j: &Job, t: usize, th: usize) {
    // SAFETY: as Entry.
    unsafe { j.task::<super::kernels::Avx2>(t, th) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx2,fma")]
unsafe fn entry_avx512(j: &Job, t: usize, th: usize) {
    // SAFETY: as Entry.
    unsafe { j.task::<super::kernels::Avx512>(t, th) }
}

#[cfg(test)]
mod tests {
    use super::super::kernels::{Panels, exp, gelu};
    use super::*;
    use crate::{TURBO_DTYPE_F32, TURBO_POOLING_MEAN};

    /// erf by its Taylor series in F64, which converges everywhere and is
    /// exact to rounding for |x| under 3.
    fn erf_series(x: f64) -> f64 {
        let z = x * x;
        let (mut term, mut sum, mut n) = (x, x, 0.0);
        loop {
            n += 1.0;
            term *= -z / n;
            let t = term / (2.0 * n + 1.0);
            sum += t;
            if t.abs() < 1e-17 * sum.abs().max(1e-300) {
                break;
            }
        }
        sum * std::f64::consts::FRAC_2_SQRT_PI
    }

    #[test]
    fn gelu_matches_its_definition() {
        for (x, want) in
            [(0.5, 0.520_499_877_813_046_5), (1.0, 0.842_700_792_949_714_9), (2.0, 0.995_322_265_018_952_7)]
        {
            assert!((erf_series(x) - want).abs() < 1e-15, "the series at {x}");
        }
        // Under -5 the series itself loses 1e-9 to cancellation.
        let mut worst = 0.0f64;
        for i in -5000..=6000 {
            let x = (i as f32 / 1000.0) as f64;
            let want = 0.5 * x * (1.0 + erf_series(x / std::f64::consts::SQRT_2));
            let got = gelu(x as f32) as f64;
            assert!((got - want).abs() <= 1e-6 * want.abs().max(1e-3), "gelu({x}) is {got}, not {want}");
            if want.abs() > 1e-3 {
                worst = worst.max((got - want).abs() / want.abs());
            }
        }
        println!("gelu's worst relative error on [-5, 6] where it is over 1e-3: {worst:.2e}");
        assert!(worst < 5e-7, "gelu's relative error reaches {worst:.2e}");
        assert_eq!(gelu(1e30), 1e30);
        assert_eq!(gelu(-1e30), 0.0);
        assert_eq!(gelu(0.0), 0.0);
    }

    #[test]
    fn exp_is_within_two_ulps() {
        let mut worst = 0.0f64;
        // Over the range it is not 0, and under the clamp at 88.
        for i in -79_999..=87_900 {
            let x = i as f32 / 1000.0 - 0.000_37;
            let (got, want) = (exp(x) as f64, (x as f64).exp());
            worst = worst.max(((got - want) / want).abs());
        }
        assert!(worst < 2.0 * f32::EPSILON as f64, "exp's relative error reaches {worst}");
        assert_eq!(exp(-80.01), 0.0);
        assert!(exp(-79.99) >= f32::MIN_POSITIVE);
        assert_eq!(exp(f32::NEG_INFINITY), 0.0);
    }

    /// Every instruction set this processor has.
    fn isas() -> Vec<Isa> {
        #[cfg(target_arch = "x86_64")]
        let all = vec![Isa::Portable, Isa::Avx2, Isa::Avx512];
        #[cfg(not(target_arch = "x86_64"))]
        let all = vec![Isa::Portable];
        all.into_iter().filter(|i| i.available()).collect()
    }

    fn values(n: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|i| (((i * 7919 + seed * 104_729) % 2001) as f32 / 1000.0 - 1.0) * scale).collect()
    }

    /// Every kernel and epilogue against the plain sum, for shapes with
    /// and without remainders in each dimension, over every task.
    #[test]
    fn linear_kernels_match_the_plain_sum() {
        for (rows, n_in, n_out) in
            [(1, 1, 1), (3, 7, 5), (4, 8, 2), (9, 37, 11), (13, 64, 130), (5, 1536, 9), (70, 40, 97)]
        {
            let x = values(rows * n_in, 1, 1.0);
            let w = values(n_out * n_in, 2, 1.0);
            let b = values(n_out, 3, 1.0);
            let y0 = values(rows * n_out, 4, 1.0);
            for isa in isas() {
                let panels = Panels::pack(&[&w], n_in, isa).unwrap();
                // Split's transposed columns, a third to two thirds: each
                // row's slot one past its index.
                let (lo, hi, stride) = (n_out / 3, 2 * n_out / 3, rows + 3);
                let slot: Vec<u32> = (1..=rows as u32).collect();
                let mut kt = vec![f32::NAN; (hi - lo) * stride];
                let split = Epilogue::Split { lo, hi, t_out: kt.as_mut_ptr(), stride, slot: slot.as_ptr() };
                for epilogue in [Epilogue::Gelu, Epilogue::Residual, split] {
                    let mut y = y0.clone();
                    y.extend([f32::NAN; 3]);
                    let g = Gemm {
                        x: x.as_ptr(),
                        ldx: n_in,
                        rows,
                        w: &panels,
                        bias: &b,
                        y: y.as_mut_ptr(),
                        ldy: n_out,
                        epilogue,
                    };
                    for t in 0..g.tasks() {
                        // SAFETY: the buffers are the shapes given, and
                        // the kernel is one this processor has.
                        unsafe { run_task(isa, &g, t) };
                    }
                    for t in 0..rows {
                        for o in 0..n_out {
                            let s = b[o] as f64
                                + (0..n_in).map(|i| x[t * n_in + i] as f64 * w[o * n_in + i] as f64).sum::<f64>();
                            let (want, got) = match epilogue {
                                Epilogue::Gelu => (gelu(s as f32) as f64, y[t * n_out + o]),
                                Epilogue::Residual => (y0[t * n_out + o] as f64 + s, y[t * n_out + o]),
                                Epilogue::Split { lo, hi, .. } if (lo..hi).contains(&o) => {
                                    assert_eq!(y[t * n_out + o], y0[t * n_out + o], "these go to t_out only");
                                    (s, kt[(o - lo) * stride + t + 1])
                                }
                                Epilogue::Split { .. } => (s, y[t * n_out + o]),
                            };
                            let got = got as f64;
                            let tol = 1e-4 * (1.0 + want.abs()) + if epilogue == Epilogue::Gelu { 1e-4 } else { 0.0 };
                            assert!(
                                (got - want).abs() < tol,
                                "{isa:?} {epilogue:?} {rows}x{n_in}x{n_out} [{t}, {o}]: {got} vs {want}"
                            );
                        }
                    }
                    assert!(y[rows * n_out..].iter().all(|v| v.is_nan()), "nothing past the output is written");
                }
            }
        }
    }

    unsafe fn run_task(isa: Isa, g: &Gemm, t: usize) {
        unsafe {
            match isa {
                Isa::Portable => g.task::<super::super::kernels::Portable>(t),
                #[cfg(target_arch = "x86_64")]
                Isa::Avx2 => avx2(g, t),
                #[cfg(target_arch = "x86_64")]
                Isa::Avx512 => avx512(g, t),
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2(g: &Gemm, t: usize) {
        unsafe { g.task::<super::super::kernels::Avx2>(t) }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512vl,avx2,fma")]
    unsafe fn avx512(g: &Gemm, t: usize) {
        unsafe { g.task::<super::super::kernels::Avx512>(t) }
    }

    /// A small BERT with synthetic weights: 2 layers, hidden 96 in 3
    /// heads of 32, intermediate 192.
    struct Model {
        desc: turbo_backend_model,
        tensors: Vec<Vec<f32>>,
    }

    impl Model {
        fn new() -> Model {
            Model::shaped(2, 96, 3, 192, 128)
        }

        fn shaped(layers: usize, h: usize, heads: u32, i: usize, positions: usize) -> Model {
            let vocab = 500usize;
            let mut shapes = vec![vocab * h, positions * h, 2 * h, h, h];
            for _ in 0..layers {
                shapes.extend([h * h, h, h * h, h, h * h, h, h * h, h, h, h, i * h, i, h * i, h, h, h]);
            }
            let tensors =
                shapes.iter().enumerate().map(|(n, &len)| values(len, n, if len > h { 0.1 } else { 0.5 })).collect();
            let desc = turbo_backend_model {
                struct_size: size_of::<turbo_backend_model>() as u32,
                family: crate::backend::TURBO_FAMILY_BERT,
                dtype: TURBO_DTYPE_F32,
                layers: layers as u32,
                hidden: h as u32,
                heads,
                intermediate: i as u32,
                vocab_size: vocab as u32,
                max_positions: positions as u32,
                token_types: 2,
                layer_norm_eps: 1e-12,
                tensor_count: shapes.len() as u32,
                reserved: 0,
                tensors: std::ptr::null(),
                format: crate::backend::TURBO_FORMAT_SAFETENSORS,
                graph_input: crate::backend::TURBO_INPUT_TOKEN_IDS,
                graph_output: crate::backend::TURBO_OUTPUT_HIDDEN_STATES,
                compute_dtype: 0,
                fixed_seq: 0,
                fixed_batch: 0,
                artifact: std::ptr::null(),
                artifact_bytes: 0,
            };
            Model { desc, tensors }
        }

        fn slices(&self) -> Vec<&[f32]> {
            self.tensors.iter().map(Vec::as_slice).collect()
        }

        fn packed(&self, isa: Isa) -> Packed {
            let t = self.slices();
            let layer = |l: usize, r: usize| t[5 + l * 16 + r];
            Packed::new(&self.desc, layer, isa).unwrap()
        }

        /// The batch's vectors, run on `threads` threads with `isa`.
        fn run(&self, isa: Isa, threads: usize, ids: &[i32], mask: &[i32], batch: usize, seq: usize) -> Vec<f32> {
            self.timed(isa, threads, ids, mask, batch, seq, 2).0
        }

        /// As run, `runs` times; also the fastest run's time.
        #[allow(clippy::too_many_arguments)]
        fn timed(
            &self,
            isa: Isa,
            threads: usize,
            ids: &[i32],
            mask: &[i32],
            batch: usize,
            seq: usize,
            runs: usize,
        ) -> (Vec<f32>, std::time::Duration) {
            let packed = self.packed(isa);
            let mut pool = Pool::new(threads);
            assert_eq!(pool.threads(), threads);
            let mut e = Encoder::new(&self.desc, self.slices(), &packed, threads, batch, seq).unwrap();
            let rows = turbo_backend_embed_rows {
                struct_size: size_of::<turbo_backend_embed_rows>() as u32,
                batch: batch as u32,
                seq: seq as u32,
                row_stride: seq as u32,
                ids: ids.as_ptr(),
                mask: mask.as_ptr(),
                types: std::ptr::null(),
                pooling: TURBO_POOLING_MEAN,
                normalize: TURBO_NORMALIZE_L2,
                output_dim: self.desc.hidden,
                reserved: 0,
            };
            let mut out = vec![f32::NAN; batch * self.desc.hidden as usize];
            // More than once, to see later runs on the same session agree.
            let mut first = Vec::new();
            let mut best = std::time::Duration::MAX;
            for _ in 0..runs {
                let t = std::time::Instant::now();
                e.write(&rows, ids, mask, None);
                e.run(&mut pool, &mut out);
                best = best.min(t.elapsed());
                if first.is_empty() {
                    first = out.clone();
                }
            }
            assert!(bits(&first) == bits(&out), "{isa:?} on {threads} threads: a later run differs");
            (out, best)
        }
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|f| f.to_bits()).collect()
    }

    /// Rows of assorted lengths, some longer than an attention block, one
    /// with masked tokens between live ones, and padding.
    fn batch() -> (Vec<i32>, Vec<i32>, usize, usize) {
        let (batch, seq) = (9, 100);
        let lens = [100, 1, 37, 64, 65, 2, 33, 90, 17];
        let (mut ids, mut mask) = (vec![0; batch * seq], vec![0; batch * seq]);
        for (r, &n) in lens.iter().enumerate() {
            for p in 0..n {
                ids[r * seq + p] = ((r * 131 + p * 17) % 500) as i32;
                mask[r * seq + p] = 1;
            }
        }
        for p in [3, 10, 11, 40] {
            mask[2 * seq + p] = 0;
        }
        (ids, mask, batch, seq)
    }

    /// The same batch gives the same bits on any number of threads, and
    /// with any kernel that fuses multiply and add.
    #[test]
    fn vectors_are_the_same_bits_on_any_number_of_threads() {
        let m = Model::new();
        let (ids, mask, b, s) = batch();
        let most = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        for isa in isas() {
            let want = m.run(isa, 1, &ids, &mask, b, s);
            assert!(want.iter().all(|v| v.is_finite()));
            for threads in [2, 3, 4, 7, 16, most] {
                let got = m.run(isa, threads, &ids, &mask, b, s);
                assert!(bits(&got) == bits(&want), "{isa:?}: {threads} threads differ from 1");
            }
        }
        let fused: Vec<Vec<u32>> = isas()
            .into_iter()
            .filter(|&i| i != Isa::Portable || cfg!(target_arch = "aarch64"))
            .map(|i| bits(&m.run(i, 3, &ids, &mask, b, s)))
            .collect();
        assert!(fused.windows(2).all(|w| w[0] == w[1]), "the fused kernels differ");
    }

    /// How the run's time goes with threads and kernels, on a batch of
    /// 32 rows of up to 256 tokens, 1353 live, at all-MiniLM-L6-v2's shape
    /// and the small bundle's.
    #[test]
    #[ignore = "measures; run with --ignored --nocapture"]
    fn speed_by_threads_and_kernel() {
        let (batch, seq) = (32, 256);
        let mut lens: Vec<usize> = (0..batch).map(|r| if r == 0 { seq } else { 6 + (r * 29) % 53 }).collect();
        lens[batch - 1] += 1353 - lens.iter().sum::<usize>();
        let (mut ids, mut mask) = (vec![0; batch * seq], vec![0; batch * seq]);
        for (r, &n) in lens.iter().enumerate() {
            for p in 0..n {
                ids[r * seq + p] = ((r * 131 + p * 17) % 500) as i32;
                mask[r * seq + p] = 1;
            }
        }
        let most = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        for (name, m) in
            [("MiniLM shape", Model::shaped(6, 384, 12, 1536, 512)), ("small", Model::shaped(2, 32, 4, 64, 512))]
        {
            for isa in isas() {
                for threads in [1, 2, most] {
                    let (_, t) = m.timed(isa, threads, &ids, &mask, batch, seq, 8);
                    println!("{name}: {isa:?}, {threads} threads: {:.2} ms", t.as_secs_f64() * 1e3);
                }
            }
        }
    }
}
