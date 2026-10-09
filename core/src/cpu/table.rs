//! A static model on the host: the family TURBO_FAMILY_STATIC names.
//!
//! There is no encoder. A row's vector is its tokens' rows of the table,
//! each times the token's weight, averaged over the live tokens whose
//! weight is not 0; the zero vector when there are none. Then the row is
//! cut to output_dim and L2-normalized when asked. CLS pooling takes the
//! row's first column and LAST its last live token, each times its weight.
//!
//! `write` keeps only what the run reads: each row's live ids, in order,
//! one after another. A run splits the rows over the session's threads;
//! each row is summed by one task, token by token in the row's order, so
//! a run gives the same bits for the same rows on any number of threads.
//! The sum runs over the width in blocks of BLOCK values held in
//! registers, every token's block before the next block, so the table is
//! read once per token and each output is written once.
//!
//! Every buffer is allocated by `new`, for the session's largest batch and
//! its longest rows; `write` and `run` allocate nothing.

use crate::backend::{TURBO_STATIC_EMBEDDINGS, TURBO_STATIC_WEIGHTS, turbo_backend_embed_rows, turbo_backend_tensor};
use crate::{TURBO_DTYPE_BF16, TURBO_DTYPE_F16, TURBO_NORMALIZE_L2, TURBO_POOLING_CLS, TURBO_POOLING_LAST};

use super::kernels::Isa;
use super::pool::Pool;

/// Values of a vector summed at once, in registers: eight AVX2 registers,
/// four AVX-512 ones.
const BLOCK: usize = 64;

/// Live tokens a task should sum, at least, before the run splits rows
/// over threads: below it a task is shorter than handing it to a thread.
const TOKENS_PER_TASK: usize = 2048;

/// The table as the model holds it: the bytes the core verified, in their
/// stored dtype, read in place.
#[derive(Clone, Copy)]
pub(super) struct Table {
    rows: *const u8,
    weights: *const u8,
    /// TURBO_DTYPE_F32, F16 or BF16, for both tensors.
    dtype: u32,
    pub(super) dim: usize,
    vocab: usize,
}

// The core's verified bytes, unchanged until model_release.
unsafe impl Send for Table {}
unsafe impl Sync for Table {}

impl Table {
    /// The two tensors model_load was handed, in TURBO_STATIC_* order.
    pub(super) fn new(tensors: &[turbo_backend_tensor], dtype: u32) -> Table {
        let (e, w) = (&tensors[TURBO_STATIC_EMBEDDINGS as usize], &tensors[TURBO_STATIC_WEIGHTS as usize]);
        Table {
            rows: e.data as *const u8,
            weights: w.data as *const u8,
            dtype,
            dim: e.shape[1] as usize,
            vocab: e.shape[0] as usize,
        }
    }

    /// Token `id`'s weight as F32.
    #[inline(always)]
    fn weight(&self, id: usize) -> f32 {
        debug_assert!(id < self.vocab);
        unsafe {
            match self.dtype {
                TURBO_DTYPE_F16 => f16_to_f32(*(self.weights as *const u16).add(id)),
                TURBO_DTYPE_BF16 => f32::from_bits((*(self.weights as *const u16).add(id) as u32) << 16),
                _ => *(self.weights as *const f32).add(id),
            }
        }
    }
}

pub(super) struct Static {
    table: Table,
    isa: Isa,
    /// Every row's live ids, one row after another.
    ids: Vec<u32>,
    /// Row r's ids are ids[starts[r]..starts[r + 1]].
    starts: Vec<usize>,
    /// Each row's first column, live or not, for CLS pooling.
    first: Vec<u32>,
    batch: usize,
    pooling: u32,
    normalize: u32,
    output_dim: usize,
}

impl Static {
    /// A session's state for rows of up to `max_seq` tokens, `max_batch`
    /// of them. Err is the bytes that could not be allocated.
    pub(super) fn new(table: Table, isa: Isa, max_batch: usize, max_seq: usize) -> Result<Static, usize> {
        let tokens = max_batch.checked_mul(max_seq).ok_or(usize::MAX)?;
        let mut ids = Vec::new();
        ids.try_reserve_exact(tokens).map_err(|_| tokens.saturating_mul(4))?;
        ids.resize(tokens, 0);
        Ok(Static {
            table,
            isa,
            ids,
            starts: vec![0; max_batch + 1],
            first: vec![0; max_batch],
            batch: 0,
            pooling: 0,
            normalize: 0,
            output_dim: table.dim,
        })
    }

    pub(super) fn normalize(&self) -> u32 {
        self.normalize
    }

    /// Keep each row's live ids, in order, dropping the rest.
    pub(super) fn write(&mut self, r: &turbo_backend_embed_rows, ids: &[i32], mask: &[i32]) {
        let (b, s, stride) = (r.batch as usize, r.seq as usize, r.row_stride as usize);
        let mut n = 0;
        for row in 0..b {
            let at = row * stride;
            self.starts[row] = n;
            self.first[row] = ids[at] as u32;
            for p in 0..s {
                if mask[at + p] != 0 {
                    self.ids[n] = ids[at + p] as u32;
                    n += 1;
                }
            }
        }
        self.starts[b] = n;
        self.batch = b;
        self.pooling = r.pooling;
        self.normalize = r.normalize;
        self.output_dim = r.output_dim as usize;
    }

    /// Run the written rows into `out`, [batch, output_dim] packed, on
    /// `pool`.
    pub(super) fn run(&mut self, pool: &mut Pool, out: &mut [f32]) {
        let tokens = self.starts[self.batch];
        let rows_per_task = if pool.threads() == 1 || tokens < TOKENS_PER_TASK {
            self.batch
        } else {
            // About four tasks a thread, each of at least one row.
            self.batch.div_ceil(pool.threads() * 4).max(1)
        };
        let tasks = self.batch.div_ceil(rows_per_task);
        let job = Job { s: self, out: Out(out.as_mut_ptr()), rows_per_task };
        let entry = entry(self.isa);
        // SAFETY: entry is the kernel of an instruction set this processor
        // has (Isa::detect); each task writes only its own rows of out,
        // which holds batch x output_dim values.
        pool.run(tasks, &|t, _| unsafe { entry(&job, t) });
    }

    /// Row `r` into `dst`, output_dim values, fusing each multiply-add
    /// when FMA.
    #[inline(always)]
    fn row<const FMA: bool>(&self, r: usize, dst: &mut [f32]) {
        let ids = &self.ids[self.starts[r]..self.starts[r + 1]];
        let t = &self.table;
        match self.pooling {
            TURBO_POOLING_CLS => self.one(self.first[r] as usize, dst),
            TURBO_POOLING_LAST => match ids.last() {
                Some(&id) => self.one(id as usize, dst),
                None => dst.fill(0.0),
            },
            _ => {
                let n = ids.iter().filter(|&&id| t.weight(id as usize) != 0.0).count();
                if n == 0 {
                    dst.fill(0.0);
                } else {
                    match t.dtype {
                        TURBO_DTYPE_F16 => sum::<F16, FMA>(t, ids, dst),
                        TURBO_DTYPE_BF16 => sum::<Bf16, FMA>(t, ids, dst),
                        _ => sum::<F32, FMA>(t, ids, dst),
                    }
                    let inv = 1.0 / n as f32;
                    for d in dst.iter_mut() {
                        *d *= inv;
                    }
                }
            }
        }
        if self.normalize == TURBO_NORMALIZE_L2 {
            // As upstream: divided by the norm, or by 1e-12 when the norm
            // is smaller. The zero vector stays zero.
            let norm = dst.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt().max(1e-12);
            let inv = (1.0 / norm) as f32;
            for d in dst.iter_mut() {
                *d *= inv;
            }
        }
    }

    /// One token's row times its weight.
    fn one(&self, id: usize, dst: &mut [f32]) {
        let t = &self.table;
        let w = t.weight(id);
        let od = dst.len();
        match t.dtype {
            TURBO_DTYPE_F16 => F16::widen(t, id, 0, od, dst),
            TURBO_DTYPE_BF16 => Bf16::widen(t, id, 0, od, dst),
            _ => F32::widen(t, id, 0, od, dst),
        }
        for d in dst.iter_mut() {
            *d *= w;
        }
    }
}

/// A stored dtype of the table.
trait Stored {
    /// Values from..from + n of token id's row, as F32, into dst[..n].
    fn widen(t: &Table, id: usize, from: usize, n: usize, dst: &mut [f32]);
    /// Value j of token id's row, as F32.
    fn at(t: &Table, id: usize, j: usize) -> f32;
}

struct F32;
struct F16;
struct Bf16;

impl Stored for F32 {
    #[inline(always)]
    fn widen(t: &Table, id: usize, from: usize, n: usize, dst: &mut [f32]) {
        let row = unsafe { std::slice::from_raw_parts((t.rows as *const f32).add(id * t.dim + from), n) };
        dst[..n].copy_from_slice(row);
    }
    #[inline(always)]
    fn at(t: &Table, id: usize, j: usize) -> f32 {
        unsafe { *(t.rows as *const f32).add(id * t.dim + j) }
    }
}

impl Stored for F16 {
    #[inline(always)]
    fn widen(t: &Table, id: usize, from: usize, n: usize, dst: &mut [f32]) {
        for (j, d) in dst[..n].iter_mut().enumerate() {
            *d = F16::at(t, id, from + j);
        }
    }
    #[inline(always)]
    fn at(t: &Table, id: usize, j: usize) -> f32 {
        f16_to_f32(unsafe { *(t.rows as *const u16).add(id * t.dim + j) })
    }
}

impl Stored for Bf16 {
    #[inline(always)]
    fn widen(t: &Table, id: usize, from: usize, n: usize, dst: &mut [f32]) {
        for (j, d) in dst[..n].iter_mut().enumerate() {
            *d = Bf16::at(t, id, from + j);
        }
    }
    #[inline(always)]
    fn at(t: &Table, id: usize, j: usize) -> f32 {
        f32::from_bits((unsafe { *(t.rows as *const u16).add(id * t.dim + j) } as u32) << 16)
    }
}

/// The weighted sum of the rows of `ids` into `dst`, its first dst.len()
/// values: block by block of the width, each block's sum in registers
/// over every token, in the row's order.
#[inline(always)]
fn sum<S: Stored, const FMA: bool>(t: &Table, ids: &[u32], dst: &mut [f32]) {
    let od = dst.len();
    let mut from = 0;
    while from < od {
        let n = BLOCK.min(od - from);
        let mut acc = [0f32; BLOCK];
        if n == BLOCK {
            for &id in ids {
                let id = id as usize;
                let w = t.weight(id);
                if w == 0.0 {
                    continue;
                }
                for (j, a) in acc.iter_mut().enumerate() {
                    *a = madd::<FMA>(S::at(t, id, from + j), w, *a);
                }
            }
        } else {
            for &id in ids {
                let id = id as usize;
                let w = t.weight(id);
                if w == 0.0 {
                    continue;
                }
                for (j, a) in acc[..n].iter_mut().enumerate() {
                    *a = madd::<FMA>(S::at(t, id, from + j), w, *a);
                }
            }
        }
        dst[from..from + n].copy_from_slice(&acc[..n]);
        from += n;
    }
}

/// x * w + a, in one rounding when FMA: only where the instruction set
/// has it, since elsewhere mul_add is a library call.
#[inline(always)]
fn madd<const FMA: bool>(x: f32, w: f32, a: f32) -> f32 {
    if FMA { x.mul_add(w, a) } else { x * w + a }
}

/// What a task reads and where it writes.
struct Job<'a> {
    s: &'a Static,
    out: Out,
    rows_per_task: usize,
}

/// The run's output; each task writes only its own rows.
struct Out(*mut f32);
unsafe impl Sync for Out {}

impl Job<'_> {
    #[inline(always)]
    fn task<const FMA: bool>(&self, t: usize) {
        let s = self.s;
        let od = s.output_dim;
        let rows = t * self.rows_per_task..((t + 1) * self.rows_per_task).min(s.batch);
        for r in rows {
            // SAFETY: row r is this task's alone, inside the output.
            let dst = unsafe { std::slice::from_raw_parts_mut(self.out.0.add(r * od), od) };
            s.row::<FMA>(r, dst);
        }
    }
}

type Entry = unsafe fn(&Job, usize);

fn entry(isa: Isa) -> Entry {
    match isa {
        Isa::Portable => entry_portable,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => entry_avx2,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => entry_avx512,
    }
}

unsafe fn entry_portable(j: &Job, t: usize) {
    // Every aarch64 processor has FMA.
    j.task::<{ cfg!(target_arch = "aarch64") }>(t)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn entry_avx2(j: &Job, t: usize) {
    j.task::<true>(t)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx2,fma,f16c")]
unsafe fn entry_avx512(j: &Job, t: usize) {
    j.task::<true>(t)
}

/// An IEEE half to the single it names exactly.
#[inline(always)]
fn f16_to_f32(h: u16) -> f32 {
    super::f16_to_f32(h)
}
