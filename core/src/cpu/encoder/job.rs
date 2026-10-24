//! A run's steps, the tasks each splits into, and the kernels a task
//! runs with.

use crate::cpu::kernels::{Attend, Gemm, Isa, Micro, layer_norm};

use super::{EMB_LN_B, EMB_LN_W, Encoder, POSITION, QUERIES, TOKEN_TYPE, TOKENS, WORD};

/// The encoder's scratch, written by the tasks of a step.
#[derive(Clone, Copy)]
pub(super) struct Bufs {
    pub(super) x: *mut f32,
    pub(super) qv: *mut f32,
    pub(super) kt: *mut f32,
    pub(super) ctx: *mut f32,
    pub(super) ffn: *mut f32,
    pub(super) scratch: *mut f32,
}

/// One step of a run.
pub(super) enum Step<'a> {
    /// The embeddings of each token, LayerNorm'd, into x.
    Lookup,
    Linear(Gemm<'a>),
    /// Attention within each row and head, from qv and kt into ctx.
    Attention,
    /// x = LayerNorm(x), with these weights and biases.
    Norm(&'a [f32], &'a [f32]),
    /// Each written range's vector from x, `per_task` ranges a task,
    /// into `out`, [ranges, output_dim] packed.
    PoolRanges {
        out: *mut f32,
        per_task: usize,
    },
}

/// A step and what its tasks read and write.
///
/// Every task of a step writes a part of the output no other task of the
/// step touches, and reads only what no task of the step writes, apart
/// from its own part in place: a chunk of tokens' rows of x (Lookup,
/// Norm); a tile of rows by columns of the product's output (Linear); the
/// ctx rows of a block of one row's queries in one head's columns, and the
/// thread's own scratch (Attention); the output rows of a group of ranges,
/// reading x, which no task of the step writes (PoolRanges). Steps are
/// one after another: the pool returns only when every task of a step is
/// done.
pub(super) struct Job<'a> {
    pub(super) e: &'a Encoder,
    pub(super) bufs: Bufs,
    pub(super) step: Step<'a>,
    pub(super) tokens: usize,
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
                Step::PoolRanges { out, per_task } => self.pool_ranges(t, *out, *per_task),
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
            let (id, ty_id, p) = (e.ids[at] as usize, e.types[at] as usize, at % e.seq + e.position_offset);
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

    /// Task t: ranges t * per_task onward, up to per_task of them, each
    /// pooled whole, so a range's sum is one task's in column order.
    ///
    /// # Safety
    /// `out` holds a vector of output_dim values for every written range.
    #[inline(always)]
    unsafe fn pool_ranges(&self, t: usize, out: *mut f32, per_task: usize) {
        let e = self.e;
        let od = e.output_dim;
        // SAFETY: no task of this step writes x.
        let x = unsafe { std::slice::from_raw_parts(self.bufs.x, self.tokens * e.hidden) };
        let first = t * per_task;
        for (i, range) in e.ranges.iter().enumerate().skip(first).take(per_task) {
            let (row, len) = (range.row as usize, e.rows[range.row as usize].1);
            // Columns past the row's last live token were not packed:
            // they hold no live token to pool.
            let cols = range.start as usize..(range.end as usize).min(len);
            // SAFETY: range i's vector is this task's alone, inside out.
            let dst = unsafe { std::slice::from_raw_parts_mut(out.add(i * od), od) };
            e.mean(x, row, cols, dst);
            e.finish(dst);
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
            ahead: e.ahead,
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
pub(super) type Entry = unsafe fn(&Job, usize, usize);

pub(super) fn entry(isa: Isa) -> Entry {
    match isa {
        Isa::Portable => entry_portable,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => entry_avx2,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => entry_avx512,
    }
}

pub(super) unsafe fn entry_portable(j: &Job, t: usize, th: usize) {
    // SAFETY: as Entry.
    unsafe { j.task::<crate::cpu::kernels::Portable>(t, th) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn entry_avx2(j: &Job, t: usize, th: usize) {
    // SAFETY: as Entry.
    unsafe { j.task::<crate::cpu::kernels::Avx2>(t, th) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx2,fma")]
pub(super) unsafe fn entry_avx512(j: &Job, t: usize, th: usize) {
    // SAFETY: as Entry.
    unsafe { j.task::<crate::cpu::kernels::Avx512>(t, th) }
}
