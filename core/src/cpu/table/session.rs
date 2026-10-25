//! A static model's session: the rows written, and each vector summed
//! by one task.

use std::sync::Arc;

use crate::backend::turbo_backend_embed_rows;
use crate::turbo_token_range;
use crate::{TURBO_DTYPE_BF16, TURBO_DTYPE_F16, TURBO_DTYPE_F64, TURBO_DTYPE_I8, TURBO_NORMALIZE_L2};
use crate::{TURBO_POOLING_CLS, TURBO_POOLING_LAST};

use super::super::kernels::Isa;
use super::super::pool::Pool;
#[cfg(target_arch = "x86_64")]
use super::round::round_f16_f16c;
use super::round::{pairwise_squares, round_f16};
use super::sum::{Bf16, F16, F32, F64, I8, f16c, mean, mean_quantized};
#[cfg(target_arch = "x86_64")]
use super::sum::{mean_f16c, mean_quantized_avx2};
use super::{Quantized, TOKENS_PER_TASK, Table};

pub(in crate::cpu) struct Static {
    table: Arc<Table>,
    /// The I8 table a FASTEST session sums instead of the stored one.
    quantized: Option<Arc<Quantized>>,
    isa: Isa,
    /// The processor converts F16 to F32 eight at a time (x86_64's F16C,
    /// with AVX2). Read only on x86_64.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    f16c: bool,
    /// Every row's live ids, one row after another.
    ids: Vec<u32>,
    /// Row r's ids are ids[starts[r]..starts[r + 1]].
    starts: Vec<usize>,
    /// Each row's first column, live or not, for CLS pooling.
    first: Vec<u32>,
    /// Written with ranges, [batch, seq] with no stride: for each column,
    /// the place in ids of its row's first live id at that column or
    /// after it. Not kept by `write`, which has no ranges to place.
    live: Vec<u32>,
    /// Each range's live ids, as ids[lo..hi], its capacity the session's
    /// max_ranges. Empty after `write`: the run then pools each row.
    spans: Vec<(usize, usize)>,
    /// The live ids the spans cover, counted again where ranges overlap.
    span_tokens: usize,
    batch: usize,
    pooling: u32,
    normalize: u32,
    output_dim: usize,
}

impl Static {
    /// A session's state for rows of up to `max_seq` tokens, `max_batch`
    /// of them, and up to `max_ranges` ranges. Err is the bytes that could
    /// not be allocated.
    pub(in crate::cpu) fn new(
        table: Arc<Table>,
        quantized: Option<Arc<Quantized>>,
        isa: Isa,
        max_batch: usize,
        max_seq: usize,
        max_ranges: usize,
    ) -> Result<Static, usize> {
        let tokens = max_batch.checked_mul(max_seq).ok_or(usize::MAX)?;
        let zeroed = |n: usize| -> Result<Vec<u32>, usize> {
            let mut v = Vec::new();
            v.try_reserve_exact(n).map_err(|_| n.saturating_mul(4))?;
            v.resize(n, 0);
            Ok(v)
        };
        let (ids, live) = (zeroed(tokens)?, zeroed(tokens)?);
        let mut spans = Vec::new();
        spans.try_reserve_exact(max_ranges).map_err(|_| max_ranges.saturating_mul(size_of::<(usize, usize)>()))?;
        Ok(Static {
            table,
            quantized,
            isa,
            f16c: f16c(isa),
            ids,
            starts: vec![0; max_batch + 1],
            first: vec![0; max_batch],
            live,
            spans,
            span_tokens: 0,
            batch: 0,
            pooling: 0,
            normalize: 0,
            output_dim: 0,
        })
    }

    pub(in crate::cpu) fn normalize(&self) -> u32 {
        self.normalize
    }

    /// Keep each row's live ids, in order, dropping the rest; the run
    /// pools each row.
    pub(in crate::cpu) fn write(&mut self, r: &turbo_backend_embed_rows, ids: &[i32], mask: &[i32]) {
        self.keep::<false>(r, ids, mask);
        self.spans.clear();
    }

    /// Keep each row's live ids and the span of them each range covers;
    /// the run pools each range. The core has checked every range, and
    /// that there are at most the session's max_ranges, so the spans fit
    /// in the memory `new` gave them.
    pub(in crate::cpu) fn write_ranges(
        &mut self,
        r: &turbo_backend_embed_rows,
        ids: &[i32],
        mask: &[i32],
        ranges: &[turbo_token_range],
    ) {
        debug_assert!(!ranges.is_empty() && ranges.len() <= self.spans.capacity());
        self.keep::<true>(r, ids, mask);
        let s = r.seq as usize;
        self.spans.clear();
        self.span_tokens = 0;
        for range in ranges {
            let (row, start, end) = (range.row as usize, range.start as usize, range.end as usize);
            let lo = self.live[row * s + start] as usize;
            // The column past the row's last is the next row's first.
            let hi = if end == s { self.starts[row + 1] } else { self.live[row * s + end] as usize };
            debug_assert!(lo < hi, "range ({row}, {start}, {end}) holds no live token");
            self.spans.push((lo, hi));
            self.span_tokens += hi - lo;
        }
    }

    /// Each row's live ids, in order, and with `PLACES` each column's
    /// place among them (`live`).
    fn keep<const PLACES: bool>(&mut self, r: &turbo_backend_embed_rows, ids: &[i32], mask: &[i32]) {
        let (b, s, stride) = (r.batch as usize, r.seq as usize, r.row_stride as usize);
        let mut n = 0;
        for row in 0..b {
            let at = row * stride;
            self.starts[row] = n;
            self.first[row] = ids[at] as u32;
            for p in 0..s {
                if PLACES {
                    self.live[row * s + p] = n as u32;
                }
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

    /// Run the written rows into `out`, [batch, output_dim] packed, or
    /// [ranges, output_dim] after `write_ranges`, on `pool`.
    pub(in crate::cpu) fn run(&mut self, pool: &mut Pool, out: &mut [f32]) {
        let (count, tokens) = match self.spans.len() {
            0 => (self.batch, self.starts[self.batch]),
            n => (n, self.span_tokens),
        };
        debug_assert!(out.len() >= count * self.output_dim);
        let per_task = if pool.threads() == 1 || tokens < TOKENS_PER_TASK {
            count
        } else {
            // About four tasks a thread, each of at least one vector.
            count.div_ceil(pool.threads() * 4).max(1)
        };
        let tasks = count.div_ceil(per_task);
        let job = Job { s: self, out: Out(out.as_mut_ptr()), count, per_task };
        let entry = entry(self.isa);
        // SAFETY: entry is the kernel of an instruction set this processor
        // has (Isa::detect); each task writes only its own vectors of out,
        // which holds count x output_dim values.
        pool.run(tasks, &|t, _| unsafe { entry(&job, t) });
    }

    /// Vector `i` of the run into `dst`, output_dim values: row i's, or
    /// range i's after `write_ranges`.
    #[inline(always)]
    fn vector(&self, i: usize, dst: &mut [f32]) {
        match self.spans.get(i) {
            Some(&(lo, hi)) => self.pooled(&self.ids[lo..hi], dst),
            None => self.row(i, dst),
        }
    }

    /// Row `r` into `dst`, output_dim values.
    #[inline(always)]
    fn row(&self, r: usize, dst: &mut [f32]) {
        let ids = &self.ids[self.starts[r]..self.starts[r + 1]];
        if ids.is_empty() {
            dst.fill(0.0);
            return;
        }
        let first = [self.first[r]];
        let ids = match self.pooling {
            TURBO_POOLING_CLS => &first[..],
            TURBO_POOLING_LAST => &ids[ids.len() - 1..],
            _ => ids,
        };
        self.pooled(ids, dst)
    }

    /// The mean of the table rows of `ids`, one or more, into `dst`,
    /// output_dim values: rounded as the table's dtype says, and
    /// normalized as the write asked. CLS and LAST pooling are the mean of
    /// their one token.
    #[inline(always)]
    fn pooled(&self, ids: &[u32], dst: &mut [f32]) {
        let t = &*self.table;
        if let Some(q) = &self.quantized {
            match self.isa {
                #[cfg(target_arch = "x86_64")]
                // SAFETY: Isa::Avx2 and Avx512 have avx2 and fma.
                Isa::Avx2 | Isa::Avx512 => unsafe { mean_quantized_avx2(t, q, ids, dst) },
                _ => mean_quantized(t, q, ids, dst),
            }
            if self.normalize == TURBO_NORMALIZE_L2 {
                let norm = pairwise_squares(dst).sqrt() + 1e-32;
                for d in dst.iter_mut() {
                    *d /= norm;
                }
            }
            return;
        }
        match t.dtype {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: f16c and avx2 are present (Static::new). The same
            // bits as mean::<F16>.
            TURBO_DTYPE_F16 if self.f16c && t.weights.is_none() => unsafe { mean_f16c(t, ids, dst) },
            TURBO_DTYPE_F16 => mean::<F16>(t, ids, dst),
            TURBO_DTYPE_BF16 => mean::<Bf16>(t, ids, dst),
            TURBO_DTYPE_F64 => mean::<F64>(t, ids, dst),
            TURBO_DTYPE_I8 => mean::<I8>(t, ids, dst),
            _ => mean::<F32>(t, ids, dst),
        }
        // numpy stores an F16 table's mean as F16.
        let half = t.dtype == TURBO_DTYPE_F16;
        if half {
            self.round_f16(dst);
        }
        if self.normalize == TURBO_NORMALIZE_L2 {
            // x / (norm + 1e-32) in F32, so the zero vector
            // stays zero, the norm's squares summed as numpy sums them.
            let norm = pairwise_squares(dst).sqrt() + 1e-32;
            for d in dst.iter_mut() {
                *d /= norm;
            }
            if half {
                self.round_f16(dst);
            }
        }
    }

    /// Each value rounded to F16 and back, eight at a time where the
    /// processor converts them (the same rounding to nearest even).
    #[inline(always)]
    fn round_f16(&self, x: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if self.f16c {
            // SAFETY: f16c and avx2 are present (Static::new).
            return unsafe { round_f16_f16c(x) };
        }
        round_f16(x)
    }
}

/// What a task reads and where it writes.
pub(super) struct Job<'a> {
    s: &'a Static,
    out: Out,
    /// The run's vectors: rows, or ranges.
    count: usize,
    per_task: usize,
}

/// The run's output; each task writes only its own vectors.
pub(super) struct Out(*mut f32);
unsafe impl Sync for Out {}

impl Job<'_> {
    #[inline(always)]
    fn task(&self, t: usize) {
        let s = self.s;
        let od = s.output_dim;
        for i in t * self.per_task..((t + 1) * self.per_task).min(self.count) {
            // SAFETY: vector i is this task's alone, inside the output.
            let dst = unsafe { std::slice::from_raw_parts_mut(self.out.0.add(i * od), od) };
            s.vector(i, dst);
        }
    }
}

pub(super) type Entry = unsafe fn(&Job, usize);

pub(super) fn entry(isa: Isa) -> Entry {
    match isa {
        Isa::Portable => entry_portable,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => entry_avx2,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => entry_avx512,
        // The table's arithmetic is plain Rust, which NEON, the aarch64
        // baseline, already compiles for.
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => entry_portable,
    }
}

pub(super) unsafe fn entry_portable(j: &Job, t: usize) {
    j.task(t)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn entry_avx2(j: &Job, t: usize) {
    j.task(t)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx2,fma,f16c")]
pub(super) unsafe fn entry_avx512(j: &Job, t: usize) {
    j.task(t)
}
