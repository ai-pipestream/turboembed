//! A static model on the host: the family TURBO_FAMILY_STATIC names.
//!
//! There is no encoder. A row's vector is computed by the rules of
//! docs/static.md, in numpy's order and dtypes, so a correct numpy
//! implementation gives the same bits: each
//! live token's table row (the row the token mapping names, when the
//! model has one) times the token's weight (no product without weights),
//! summed in token order and divided by the count of live tokens; the
//! zero vector when there are none. The sum runs in F32, or in F64 where
//! numpy promotes to it (an I8 or F64 table, or F64 weights), and the
//! mean is rounded to F32, and to F16 for an F16 table, as numpy stores
//! it. Then the row is cut to output_dim and, when asked, divided by its
//! L2 norm plus 1e-32, the norm summed pairwise as numpy sums it, the
//! quotient rounded to F16 again for an F16 table. CLS pooling takes the
//! row's first column and LAST its last live token instead of the mean;
//! a row with no live token is the zero vector under every pooling.
//!
//! A range of a row's columns (`write_ranges`) is pooled the way a row is
//! under mean pooling, over the live tokens in its columns alone: their
//! table rows times their weights, summed in token order, divided by
//! their count, then rounded, cut and normalized as a row's mean is. The
//! core refuses a range with no live token, so none reaches here.
//!
//! `write` keeps only what the run reads: each row's live ids, in order,
//! one after another, and for ranges, the span of those ids each range
//! covers. A run splits the rows, or the ranges, over the session's
//! threads; each vector is summed by one task, token by token in the
//! row's order, so a run gives the same bits for the same rows on any
//! number of threads.
//! The sum runs over the width in blocks of BLOCK values held in
//! registers, every token's block before the next block, so the table is
//! read once per token and each output is written once.
//!
//! Every buffer is allocated by `new`, for the session's largest batch and
//! its longest rows; `write` and `run` allocate nothing.

use std::sync::Arc;

use crate::backend::{
    TURBO_STATIC_EMBEDDINGS, TURBO_STATIC_MAPPING, TURBO_STATIC_WEIGHTS, turbo_backend_embed_rows, turbo_backend_tensor,
};
use crate::bundle::AlignedBytes;
use crate::turbo_token_range;
use crate::{
    TURBO_DTYPE_BF16, TURBO_DTYPE_F16, TURBO_DTYPE_F64, TURBO_DTYPE_I8, TURBO_DTYPE_I32, TURBO_NORMALIZE_L2,
    TURBO_POOLING_CLS, TURBO_POOLING_LAST,
};

use super::kernels::Isa;
use super::pool::Pool;

/// Values of a vector summed at once, in registers: eight AVX2 registers,
/// four AVX-512 ones.
const BLOCK: usize = 64;

/// Live tokens a task should sum, at least, before the run splits rows or
/// ranges over threads: below it a task is shorter than handing it to a
/// thread.
const TOKENS_PER_TASK: usize = 2048;

/// The table as the model holds it: the bytes the core verified, in their
/// stored dtype, read in place; the weights and the mapping, which are a
/// value per token, widened once when the model is loaded.
pub(super) struct Table {
    rows: *const u8,
    /// TURBO_DTYPE_F32, F16, BF16, F64 or I8.
    dtype: u32,
    pub(super) dim: usize,
    /// Rows.
    height: usize,
    /// Each token's weight, exact: an F32 or F16 weight is an F64 too.
    weights: Option<Box<[f64]>>,
    mapping: Option<Box<[u32]>>,
    /// The sum runs in F64: numpy promotes an I8 or F64 table, or F64
    /// weights, to it.
    wide: bool,
    /// The copy of the table in huge pages that `rows` points into, when
    /// the model's sessions read one (docs/cpu.md).
    _copy: Option<AlignedBytes>,
}

// The core's verified bytes, unchanged until model_release.
unsafe impl Send for Table {}
unsafe impl Sync for Table {}

impl Table {
    /// The tensors model_load was handed, in TURBO_STATIC_* order; the
    /// core has checked their shapes, dtypes and every mapping value.
    /// `huge`: read a copy of the table in huge pages, where the host
    /// gives them.
    pub(super) fn new(tensors: &[turbo_backend_tensor], huge: bool) -> Table {
        let e = &tensors[TURBO_STATIC_EMBEDDINGS as usize];
        let (w, m) = (&tensors[TURBO_STATIC_WEIGHTS as usize], &tensors[TURBO_STATIC_MAPPING as usize]);
        let weights = (!w.data.is_null()).then(|| {
            let n = w.shape[0] as usize;
            (0..n)
                .map(|i| unsafe {
                    match w.dtype {
                        TURBO_DTYPE_F16 => f16_to_f32(*(w.data as *const u16).add(i)) as f64,
                        TURBO_DTYPE_F64 => *(w.data as *const f64).add(i),
                        _ => *(w.data as *const f32).add(i) as f64,
                    }
                })
                .collect()
        });
        let mapping = (!m.data.is_null()).then(|| {
            let n = m.shape[0] as usize;
            (0..n)
                .map(|i| unsafe {
                    match m.dtype {
                        TURBO_DTYPE_I32 => *(m.data as *const i32).add(i) as u32,
                        _ => *(m.data as *const i64).add(i) as u32,
                    }
                })
                .collect()
        });
        let wide =
            matches!(e.dtype, TURBO_DTYPE_I8 | TURBO_DTYPE_F64) || (weights.is_some() && w.dtype == TURBO_DTYPE_F64);
        let bytes = e.shape[0] as usize * e.shape[1] as usize * dtype_bytes(e.dtype);
        let copy = huge.then(|| AlignedBytes::zeroed_huge(bytes)).flatten().map(|mut c| {
            // SAFETY: the core verified `bytes` bytes of the table at e.data.
            c.copy_from_slice(unsafe { std::slice::from_raw_parts(e.data as *const u8, bytes) });
            c
        });
        Table {
            rows: copy.as_ref().map_or(e.data as *const u8, |c| c.as_ptr()),
            dtype: e.dtype,
            dim: e.shape[1] as usize,
            height: e.shape[0] as usize,
            weights,
            mapping,
            wide,
            _copy: copy,
        }
    }

    /// The table row token `id` reads.
    #[inline(always)]
    fn row_of(&self, id: usize) -> usize {
        self.mapping.as_ref().map_or(id, |m| m[id] as usize)
    }

    /// Value j of row `row` as F64, whatever the stored dtype.
    fn value(&self, row: usize, j: usize) -> f64 {
        match self.dtype {
            TURBO_DTYPE_F16 => F16::at64(self, row, j),
            TURBO_DTYPE_BF16 => Bf16::at64(self, row, j),
            TURBO_DTYPE_F64 => F64::at64(self, row, j),
            TURBO_DTYPE_I8 => I8::at64(self, row, j),
            _ => F32::at64(self, row, j),
        }
    }
}

/// The table for TURBO_PRECISION_FASTEST: each row as I8 values over the
/// row's own F32 scale (its largest magnitude over 127, the values
/// rounded to the nearest step), a quarter of an F32 table's bytes. The
/// model makes it once, for its first FASTEST session; the token weights
/// and the mapping stay the table's.
/// Bytes a value of a table dtype.
fn dtype_bytes(dtype: u32) -> usize {
    match dtype {
        TURBO_DTYPE_F16 | TURBO_DTYPE_BF16 => 2,
        TURBO_DTYPE_F64 => 8,
        TURBO_DTYPE_I8 => 1,
        _ => 4,
    }
}

pub(super) struct Quantized {
    values: I8s,
    scales: Box<[f32]>,
}

/// The I8 table's values: on the heap, or in huge pages.
enum I8s {
    Heap(Box<[i8]>),
    Huge(AlignedBytes),
}

impl std::ops::Deref for I8s {
    type Target = [i8];
    fn deref(&self) -> &[i8] {
        match self {
            I8s::Heap(v) => v,
            // SAFETY: i8 has u8's size and alignment.
            I8s::Huge(b) => unsafe { std::slice::from_raw_parts(b.as_ptr() as *const i8, b.len()) },
        }
    }
}

impl std::ops::DerefMut for I8s {
    fn deref_mut(&mut self) -> &mut [i8] {
        match self {
            I8s::Heap(v) => v,
            // SAFETY: i8 has u8's size and alignment.
            I8s::Huge(b) => unsafe { std::slice::from_raw_parts_mut(b.as_mut_ptr() as *mut i8, b.len()) },
        }
    }
}

impl Quantized {
    /// `huge`: in huge pages, where the host gives them. Err says which
    /// value of the table the I8 copy cannot hold: one that is not
    /// finite, or a row whose scale is past F32's range.
    pub(super) fn new(t: &Table, huge: bool) -> Result<Quantized, String> {
        let n = t.height * t.dim;
        let mut values = match huge.then(|| AlignedBytes::zeroed_huge(n)).flatten() {
            Some(b) => I8s::Huge(b),
            None => I8s::Heap(vec![0i8; n].into_boxed_slice()),
        };
        let mut scales = vec![0f32; t.height].into_boxed_slice();
        for (row, (q, scale)) in values.chunks_exact_mut(t.dim).zip(scales.iter_mut()).enumerate() {
            let mut most = 0f64;
            for j in 0..t.dim {
                let v = t.value(row, j);
                if !v.is_finite() {
                    return Err(format!("row {row} of the table holds {v} at column {j}"));
                }
                most = most.max(v.abs());
            }
            if most == 0.0 {
                continue;
            }
            *scale = (most / 127.0) as f32;
            if !scale.is_finite() {
                return Err(format!("row {row} of the table has a largest magnitude of {most}, past an F32 scale"));
            }
            for (j, v) in q.iter_mut().enumerate() {
                *v = (t.value(row, j) / *scale as f64).round().clamp(-127.0, 127.0) as i8;
            }
        }
        Ok(Quantized { values, scales })
    }
}

pub(super) struct Static {
    table: Arc<Table>,
    /// The I8 table a FASTEST session sums instead of the stored one.
    quantized: Option<Arc<Quantized>>,
    isa: Isa,
    /// The processor converts F16 to F32 eight at a time (x86_64's F16C,
    /// with AVX2).
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
    pub(super) fn new(
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

    pub(super) fn normalize(&self) -> u32 {
        self.normalize
    }

    /// Keep each row's live ids, in order, dropping the rest; the run
    /// pools each row.
    pub(super) fn write(&mut self, r: &turbo_backend_embed_rows, ids: &[i32], mask: &[i32]) {
        self.keep::<false>(r, ids, mask);
        self.spans.clear();
    }

    /// Keep each row's live ids and the span of them each range covers;
    /// the run pools each range. The core has checked every range, and
    /// that there are at most the session's max_ranges, so the spans fit
    /// in the memory `new` gave them.
    pub(super) fn write_ranges(
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
    pub(super) fn run(&mut self, pool: &mut Pool, out: &mut [f32]) {
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

/// A stored dtype of the table.
trait Stored {
    /// Value j of table row `row`, as F32: exact, except an F64 value.
    fn at(t: &Table, row: usize, j: usize) -> f32;
    /// Value j of table row `row`, as F64, exact.
    #[inline(always)]
    fn at64(t: &Table, row: usize, j: usize) -> f64 {
        Self::at(t, row, j) as f64
    }
}

struct F32;
struct F16;
struct Bf16;
struct F64;
/// An I8 value is the number it holds.
struct I8;

impl Stored for F32 {
    #[inline(always)]
    fn at(t: &Table, row: usize, j: usize) -> f32 {
        unsafe { *(t.rows as *const f32).add(row * t.dim + j) }
    }
}

impl Stored for F16 {
    #[inline(always)]
    fn at(t: &Table, row: usize, j: usize) -> f32 {
        f16_to_f32(unsafe { *(t.rows as *const u16).add(row * t.dim + j) })
    }
}

impl Stored for Bf16 {
    #[inline(always)]
    fn at(t: &Table, row: usize, j: usize) -> f32 {
        f32::from_bits((unsafe { *(t.rows as *const u16).add(row * t.dim + j) } as u32) << 16)
    }
}

impl Stored for F64 {
    #[inline(always)]
    fn at(t: &Table, row: usize, j: usize) -> f32 {
        Self::at64(t, row, j) as f32
    }
    #[inline(always)]
    fn at64(t: &Table, row: usize, j: usize) -> f64 {
        unsafe { *(t.rows as *const f64).add(row * t.dim + j) }
    }
}

impl Stored for I8 {
    #[inline(always)]
    fn at(t: &Table, row: usize, j: usize) -> f32 {
        unsafe { *(t.rows as *const i8).add(row * t.dim + j) as f32 }
    }
}

/// The mean of the rows of `ids` into `dst`, its first dst.len() values,
/// as numpy's mean over the token axis computes it: summed in token
/// order, in F64 when the table is wide, else in F32, each value times
/// its token's weight first when there are weights; then divided by the
/// count and rounded to F32. Block by block of the width, each block's
/// sum in registers over every token.
#[inline(always)]
fn mean<S: Stored>(t: &Table, ids: &[u32], dst: &mut [f32]) {
    let od = dst.len();
    let n = ids.len();
    let mut from = 0;
    while from < od {
        let w = BLOCK.min(od - from);
        if t.wide {
            let mut acc = [0f64; BLOCK];
            for &id in ids {
                let row = t.row_of(id as usize);
                match &t.weights {
                    Some(ws) => {
                        let x = ws[id as usize];
                        for (j, a) in acc[..w].iter_mut().enumerate() {
                            *a += S::at64(t, row, from + j) * x;
                        }
                    }
                    None => {
                        for (j, a) in acc[..w].iter_mut().enumerate() {
                            *a += S::at64(t, row, from + j);
                        }
                    }
                }
            }
            for (d, a) in dst[from..from + w].iter_mut().zip(&acc[..w]) {
                *d = (a / n as f64) as f32;
            }
        } else {
            let mut acc = [0f32; BLOCK];
            match &t.weights {
                Some(ws) => {
                    for &id in ids {
                        let (row, x) = (t.row_of(id as usize), ws[id as usize] as f32);
                        // A product, then a sum: two roundings, as numpy
                        // makes them, never fused.
                        for (j, a) in acc[..w].iter_mut().enumerate() {
                            *a += S::at(t, row, from + j) * x;
                        }
                    }
                }
                None if w == BLOCK => {
                    for &id in ids {
                        let row = t.row_of(id as usize);
                        for (j, a) in acc.iter_mut().enumerate() {
                            *a += S::at(t, row, from + j);
                        }
                    }
                }
                None => {
                    for &id in ids {
                        let row = t.row_of(id as usize);
                        for (j, a) in acc[..w].iter_mut().enumerate() {
                            *a += S::at(t, row, from + j);
                        }
                    }
                }
            }
            for (d, a) in dst[from..from + w].iter_mut().zip(&acc[..w]) {
                *d = a / n as f32;
            }
        }
        from += w;
    }
}

/// The mean of the rows of `ids` in the quantized table: each token's I8
/// values times its row's scale and its weight, summed in F32 a block of
/// the width at a time, then divided by the count.
#[inline(always)]
fn mean_quantized(t: &Table, q: &Quantized, ids: &[u32], dst: &mut [f32]) {
    mean_quantized_from(t, q, ids, dst, 0)
}

/// `mean_quantized` from value `from` of the width on.
#[inline(always)]
fn mean_quantized_from(t: &Table, q: &Quantized, ids: &[u32], dst: &mut [f32], mut from: usize) {
    let (od, dim) = (dst.len(), t.dim);
    let n = ids.len() as f32;
    while from < od {
        let w = BLOCK.min(od - from);
        let mut acc = [0f32; BLOCK];
        for &id in ids {
            let row = t.row_of(id as usize);
            let x = match &t.weights {
                Some(ws) => q.scales[row] * ws[id as usize] as f32,
                None => q.scales[row],
            };
            let v = &q.values[row * dim + from..row * dim + from + w];
            for (a, &v) in acc[..w].iter_mut().zip(v) {
                *a += v as f32 * x;
            }
        }
        for (d, a) in dst[from..from + w].iter_mut().zip(&acc[..w]) {
            *d = a / n;
        }
        from += w;
    }
}

/// `mean_quantized` eight values at a time: per token, each I8 value
/// widened to F32 and multiplied into its sum with the token's scale.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mean_quantized_avx2(t: &Table, q: &Quantized, ids: &[u32], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let (od, dim) = (dst.len(), t.dim);
    let base = q.values.as_ptr();
    let n = _mm256_set1_ps(ids.len() as f32);
    let scale = |id: u32, row: usize| match &t.weights {
        Some(ws) => q.scales[row] * ws[id as usize] as f32,
        None => q.scales[row],
    };
    let mut from = 0;
    while from + 64 <= od {
        let mut acc = [_mm256_setzero_ps(); 8];
        for &id in ids {
            let row = t.row_of(id as usize);
            let x = _mm256_set1_ps(scale(id, row));
            let p = unsafe { base.add(row * dim + from) };
            for (k, a) in acc.iter_mut().enumerate() {
                let v = unsafe { _mm_loadl_epi64(p.add(8 * k) as *const __m128i) };
                *a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v)), x, *a);
            }
        }
        for (k, a) in acc.iter().enumerate() {
            unsafe { _mm256_storeu_ps(dst.as_mut_ptr().add(from + 8 * k), _mm256_div_ps(*a, n)) };
        }
        from += 64;
    }
    while from + 8 <= od {
        let mut acc = _mm256_setzero_ps();
        for &id in ids {
            let row = t.row_of(id as usize);
            let v = unsafe { _mm_loadl_epi64(base.add(row * dim + from) as *const __m128i) };
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v)), _mm256_set1_ps(scale(id, row)), acc);
        }
        unsafe { _mm256_storeu_ps(dst.as_mut_ptr().add(from), _mm256_div_ps(acc, n)) };
        from += 8;
    }
    if from < od {
        mean_quantized_from(t, q, ids, dst, from);
    }
}

/// Whether `isa` comes with F16C, which every AVX2 processor made has but
/// which is its own feature bit.
fn f16c(isa: Isa) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        isa != Isa::Portable && std::arch::is_x86_feature_detected!("f16c")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = isa;
        false
    }
}

/// `mean::<F16>` for a table without weights, the F16 values converted
/// eight at a time: the same sums in the same order, so the same bits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn mean_f16c(t: &Table, ids: &[u32], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let od = dst.len();
    let base = t.rows as *const u16;
    let n = _mm256_set1_ps(ids.len() as f32);
    let mut from = 0;
    // Eight registers of eight sums: 64 values a pass, as BLOCK.
    while from + 64 <= od {
        let mut acc = [_mm256_setzero_ps(); 8];
        for &id in ids {
            let p = unsafe { base.add(t.row_of(id as usize) * t.dim + from) };
            for (k, a) in acc.iter_mut().enumerate() {
                let h = unsafe { _mm_loadu_si128(p.add(8 * k) as *const __m128i) };
                *a = _mm256_add_ps(*a, _mm256_cvtph_ps(h));
            }
        }
        for (k, a) in acc.iter().enumerate() {
            unsafe { _mm256_storeu_ps(dst.as_mut_ptr().add(from + 8 * k), _mm256_div_ps(*a, n)) };
        }
        from += 64;
    }
    while from + 8 <= od {
        let mut acc = _mm256_setzero_ps();
        for &id in ids {
            let p = unsafe { base.add(t.row_of(id as usize) * t.dim + from) };
            acc = _mm256_add_ps(acc, _mm256_cvtph_ps(unsafe { _mm_loadu_si128(p as *const __m128i) }));
        }
        unsafe { _mm256_storeu_ps(dst.as_mut_ptr().add(from), _mm256_div_ps(acc, n)) };
        from += 8;
    }
    if from < od {
        let tail = od - from;
        let mut sums = [0f32; 8];
        for &id in ids {
            let row = t.row_of(id as usize);
            for (j, s) in sums[..tail].iter_mut().enumerate() {
                *s += F16::at(t, row, from + j);
            }
        }
        for (d, s) in dst[from..].iter_mut().zip(&sums[..tail]) {
            *d = s / ids.len() as f32;
        }
    }
}

/// The sum of the squares of `x`, each square rounded to F32, summed as
/// numpy's pairwise_sum sums a contiguous F32 axis: under 8 values one by
/// one; up to 128 in eight running sums, combined as a tree, then the
/// rest one by one; above, the two halves (the first a multiple of 8)
/// apart.
fn pairwise_squares(x: &[f32]) -> f32 {
    let n = x.len();
    if n < 8 {
        let mut res = 0f32;
        for &v in x {
            res += v * v;
        }
        res
    } else if n <= 128 {
        let (chunks, rest) = x.as_chunks::<8>();
        let mut r = chunks[0].map(|v| v * v);
        for c in &chunks[1..] {
            for k in 0..8 {
                r[k] += c[k] * c[k];
            }
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in rest {
            res += v * v;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_squares(&x[..n2]) + pairwise_squares(&x[n2..])
    }
}

/// Each value rounded to the nearest F16, ties to even, and back.
/// `round_f16` with F16C: VCVTPS2PH rounds to nearest even, as numpy's
/// conversion does, subnormals and overflow included.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn round_f16_f16c(x: &mut [f32]) {
    use std::arch::x86_64::*;
    let (chunks, rest) = x.as_chunks_mut::<8>();
    for c in chunks {
        unsafe {
            let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(c.as_ptr()));
            _mm256_storeu_ps(c.as_mut_ptr(), _mm256_cvtph_ps(h));
        }
    }
    round_f16(rest);
}

fn round_f16(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = f16_to_f32(f32_to_f16(*v));
    }
}

/// An F32 to the nearest IEEE half, ties to even: infinity past the
/// largest, subnormals below the smallest normal.
fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // Subnormal: the implicit bit made explicit, shifted into place.
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rest = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let up = rest > mid || (rest == mid && half & 1 == 1);
        return sign | (half + up as u32) as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    let up = rest > 0x1000 || (rest == 0x1000 && half & 1 == 1);
    // A carry out of the mantissa raises the exponent, to infinity at most.
    sign | (half + up as u32) as u16
}

/// What a task reads and where it writes.
struct Job<'a> {
    s: &'a Static,
    out: Out,
    /// The run's vectors: rows, or ranges.
    count: usize,
    per_task: usize,
}

/// The run's output; each task writes only its own vectors.
struct Out(*mut f32);
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
    j.task(t)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn entry_avx2(j: &Job, t: usize) {
    j.task(t)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx2,fma,f16c")]
unsafe fn entry_avx512(j: &Job, t: usize) {
    j.task(t)
}

/// An IEEE half to the single it names exactly.
#[inline(always)]
fn f16_to_f32(h: u16) -> f32 {
    super::f16_to_f32(h)
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    /// numpy's pairwise sum of squares as its loops read, one running sum
    /// at a time.
    fn numpy_squares(x: &[f32]) -> f32 {
        let n = x.len();
        if n < 8 {
            x.iter().fold(0f32, |res, &v| res + v * v)
        } else if n <= 128 {
            let mut r = [0f32; 8];
            for k in 0..8 {
                r[k] = x[k] * x[k];
            }
            let mut i = 8;
            while i < n - n % 8 {
                for k in 0..8 {
                    r[k] += x[i + k] * x[i + k];
                }
                i += 8;
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for &v in &x[i..] {
                res += v * v;
            }
            res
        } else {
            let n2 = n / 2 - n / 2 % 8;
            numpy_squares(&x[..n2]) + numpy_squares(&x[n2..])
        }
    }

    /// The norm's sum of squares is numpy's, bit for bit, at every length
    /// up to well past a split.
    #[test]
    fn the_sum_of_squares_is_numpy_s() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let x: Vec<f32> = (0..700)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect();
        for n in 0..x.len() {
            assert_eq!(pairwise_squares(&x[..n]).to_bits(), numpy_squares(&x[..n]).to_bits(), "{n} values");
        }
    }

    /// F16C's rounding to F16 against the portable one, bit for bit, over
    /// values of every exponent F16 has and beyond.
    #[test]
    fn the_f16c_rounding_is_the_portable_rounding() {
        if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
            return;
        }
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut values: Vec<f32> = (0..100_003)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                // Exponents from 2^-30 to 2^20, either sign, any mantissa.
                let bits = (seed as u32 & 0x807f_ffff) | ((97 + (seed >> 32) % 51) as u32) << 23;
                f32::from_bits(bits)
            })
            .collect();
        values.extend([0.0, -0.0, 65504.0, 65519.0, 65520.0, 1e-8, 5.96e-8, 2.98e-8, 2.99e-8, f32::INFINITY]);
        let mut want = values.clone();
        round_f16(&mut want);
        unsafe { round_f16_f16c(&mut values) };
        for (g, w) in values.iter().zip(&want) {
            assert_eq!(g.to_bits(), w.to_bits(), "{g} vs {w}");
        }
    }

    /// The AVX2 I8 mean against the portable one, with and without
    /// weights, on widths that take each of its loops. They differ only by
    /// the fused multiply-add's roundings.
    #[test]
    fn the_avx2_quantized_mean_is_the_portable_one() {
        if !Isa::Avx2.available() {
            return;
        }
        for dim in [1, 7, 8, 63, 64, 77, 136, 256, 300] {
            let vocab = 40;
            let values: Vec<f32> = (0..vocab * dim).map(|i| ((i * 7919 % 1013) as f32 - 506.0) / 97.0).collect();
            for weights in [None, Some((0..vocab).map(|i| 0.25 + i as f64 / 16.0).collect::<Box<[f64]>>())] {
                let t = Table {
                    rows: values.as_ptr() as *const u8,
                    dtype: crate::TURBO_DTYPE_F32,
                    dim,
                    height: vocab,
                    weights,
                    mapping: None,
                    wide: false,
                    _copy: None,
                };
                let q = Quantized::new(&t, false).unwrap();
                let ids: Vec<u32> = (0..23).map(|i| (i * 17 % vocab) as u32).collect();
                let (mut want, mut got) = (vec![0f32; dim], vec![0f32; dim]);
                mean_quantized(&t, &q, &ids, &mut want);
                unsafe { mean_quantized_avx2(&t, &q, &ids, &mut got) };
                for (g, w) in got.iter().zip(&want) {
                    assert!((g - w).abs() <= 1e-5 * (1.0 + w.abs()), "dim {dim}: {g} vs {w}");
                }
            }
        }
    }

    /// The F16C mean against the portable one, on widths that take each of
    /// its loops, over halves of every exponent: the same bits.
    #[test]
    fn the_f16c_mean_is_the_portable_mean() {
        if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
            return;
        }
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for dim in [1, 7, 8, 63, 64, 77, 136, 256, 300] {
            let vocab = 50;
            // Finite halves only: exponent 31 is Inf and NaN.
            let halves: Vec<u16> = (0..vocab * dim)
                .map(|_| {
                    let h = next() as u16;
                    if (h >> 10) & 0x1f == 0x1f { h & !0x4000 } else { h }
                })
                .collect();
            let t = Table {
                rows: halves.as_ptr() as *const u8,
                dtype: TURBO_DTYPE_F16,
                dim,
                height: vocab,
                weights: None,
                mapping: None,
                wide: false,
                _copy: None,
            };
            for len in [1, 2, 5, 33] {
                let ids: Vec<u32> = (0..len).map(|_| (next() % vocab as u64) as u32).collect();
                let (mut want, mut got) = (vec![0f32; dim], vec![0f32; dim]);
                mean::<F16>(&t, &ids, &mut want);
                unsafe { mean_f16c(&t, &ids, &mut got) };
                let (w, g): (Vec<u32>, Vec<u32>) =
                    (want.iter().map(|v| v.to_bits()).collect(), got.iter().map(|v| v.to_bits()).collect());
                assert_eq!(w, g, "dim {dim}, {len} ids");
            }
        }
    }
}

#[cfg(test)]
mod range_tests {
    use super::*;
    use crate::{TURBO_DTYPE_F32, TURBO_NORMALIZE_NONE, TURBO_POOLING_MEAN};

    const VOCAB: usize = 300;
    const DIM: usize = 77;

    /// A table's stored values, kept alive while the table reads them.
    enum Values {
        F32(Vec<f32>),
        F16(Vec<u16>),
        F64(Vec<f64>),
    }

    /// A synthetic table of `dtype` (F32, F16 or F64), with token weights
    /// when asked, its values kept alongside.
    fn table(dtype: u32, weighted: bool) -> (Arc<Table>, Values) {
        let value = |i: usize| ((i * 7919 % 2003) as f32 - 1001.0) / 997.0;
        let values = match dtype {
            TURBO_DTYPE_F16 => Values::F16((0..VOCAB * DIM).map(|i| f32_to_f16(value(i))).collect()),
            TURBO_DTYPE_F64 => Values::F64((0..VOCAB * DIM).map(|i| value(i) as f64 / 3.0).collect()),
            _ => Values::F32((0..VOCAB * DIM).map(value).collect()),
        };
        let rows = match &values {
            Values::F32(v) => v.as_ptr() as *const u8,
            Values::F16(v) => v.as_ptr() as *const u8,
            Values::F64(v) => v.as_ptr() as *const u8,
        };
        let weights = weighted.then(|| (0..VOCAB).map(|i| 0.5 + (i % 13) as f64 / 8.0).collect::<Box<[f64]>>());
        let t = Table {
            rows,
            dtype,
            dim: DIM,
            height: VOCAB,
            weights,
            mapping: None,
            wide: dtype == TURBO_DTYPE_F64,
            _copy: None,
        };
        (Arc::new(t), values)
    }

    /// Every kind of table the sum has a path for: F32 with and without
    /// weights, F16 (rounded to F16 as numpy stores it), F64 (summed in
    /// F64), and the I8 copy.
    fn sessions(max_batch: usize, max_seq: usize, max_ranges: usize) -> Vec<(String, Static, Values)> {
        let mut all = Vec::new();
        for (dtype, weighted, quantized) in [
            (TURBO_DTYPE_F32, false, false),
            (TURBO_DTYPE_F32, true, false),
            (TURBO_DTYPE_F16, false, false),
            (TURBO_DTYPE_F16, true, false),
            (TURBO_DTYPE_F64, false, false),
            (TURBO_DTYPE_F32, true, true),
        ] {
            let (t, values) = table(dtype, weighted);
            let q = quantized.then(|| Arc::new(Quantized::new(&t, false).unwrap()));
            let s = Static::new(t, q, Isa::detect(), max_batch, max_seq, max_ranges).unwrap();
            all.push((format!("dtype {dtype}, weights {weighted}, i8 {quantized}"), s, values));
        }
        all
    }

    fn rows(
        ids: &[i32],
        mask: &[i32],
        batch: usize,
        seq: usize,
        normalize: u32,
        od: usize,
    ) -> turbo_backend_embed_rows {
        turbo_backend_embed_rows {
            struct_size: size_of::<turbo_backend_embed_rows>() as u32,
            batch: batch as u32,
            seq: seq as u32,
            row_stride: seq as u32,
            ids: ids.as_ptr(),
            mask: mask.as_ptr(),
            types: std::ptr::null(),
            pooling: TURBO_POOLING_MEAN,
            normalize,
            output_dim: od as u32,
            reserved: 0,
        }
    }

    /// Rows of assorted lengths with padding, one with masked tokens
    /// between live ones.
    fn batch() -> (Vec<i32>, Vec<i32>, usize, usize) {
        let (batch, seq) = (5, 40);
        let lens = [40, 1, 23, 9, 31];
        let (mut ids, mut mask) = (vec![0; batch * seq], vec![0; batch * seq]);
        for (r, &n) in lens.iter().enumerate() {
            for p in 0..n {
                ids[r * seq + p] = ((r * 131 + p * 17) % VOCAB) as i32;
                mask[r * seq + p] = 1;
            }
        }
        for p in [0, 4, 5, 22] {
            mask[2 * seq + p] = 0;
        }
        (ids, mask, batch, seq)
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|f| f.to_bits()).collect()
    }

    fn range(row: usize, start: usize, end: usize) -> turbo_token_range {
        turbo_token_range { row: row as u32, start: start as u32, end: end as u32 }
    }

    /// Each range's vector is the vector of a row holding the range's
    /// columns alone, bit for bit: the mean of the same live tokens in
    /// the same order, rounded and normalized the same way. Ranges over
    /// a whole row, single tokens, and ranges into padding or over masked
    /// tokens, cut to the full width and to less.
    #[test]
    fn a_range_is_the_mean_of_its_live_tokens() {
        let (ids, mask, b, s) = batch();
        let ranges = [
            range(0, 0, 40),
            range(4, 0, 40),
            range(1, 0, 40),
            range(0, 39, 40),
            range(0, 7, 8),
            range(2, 0, 2),
            range(2, 3, 7),
            range(2, 20, 40),
            range(3, 8, 9),
            range(4, 10, 30),
            range(0, 7, 8),
        ];
        // The rows each range alone would make, as a batch.
        let n = ranges.len();
        let (mut sub_ids, mut sub_mask) = (vec![0; n * s], vec![0; n * s]);
        for (k, r) in ranges.iter().enumerate() {
            for (j, p) in (r.start as usize..r.end as usize).enumerate() {
                sub_ids[k * s + j] = ids[r.row as usize * s + p];
                sub_mask[k * s + j] = mask[r.row as usize * s + p];
            }
        }
        for (normalize, od) in [(TURBO_NORMALIZE_L2, DIM), (TURBO_NORMALIZE_NONE, DIM), (TURBO_NORMALIZE_L2, 64)] {
            for (name, mut st, _values) in sessions(n, s, n) {
                let mut pool = Pool::new(2);
                let mut want = vec![f32::NAN; n * od];
                st.write(&rows(&sub_ids, &sub_mask, n, s, normalize, od), &sub_ids, &sub_mask);
                st.run(&mut pool, &mut want);
                let mut got = vec![f32::NAN; n * od];
                st.write_ranges(&rows(&ids, &mask, b, s, normalize, od), &ids, &mask, &ranges);
                st.run(&mut pool, &mut got);
                assert!(want.iter().all(|v| v.is_finite()), "{name}");
                for k in 0..n {
                    let (g, w) = (&got[k * od..][..od], &want[k * od..][..od]);
                    assert!(bits(g) == bits(w), "{name}, {od} values: range {k} {:?}", ranges[k]);
                }
            }
        }
    }

    /// A range over a whole row, padding included, is that row's mean.
    #[test]
    fn a_range_over_a_whole_row_is_the_row_s_mean() {
        let (ids, mask, b, s) = batch();
        let ranges: Vec<_> = (0..b).rev().map(|r| range(r, 0, s)).collect();
        for (name, mut st, _values) in sessions(b, s, b) {
            let mut pool = Pool::new(1);
            let mut want = vec![f32::NAN; b * DIM];
            st.write(&rows(&ids, &mask, b, s, TURBO_NORMALIZE_L2, DIM), &ids, &mask);
            st.run(&mut pool, &mut want);
            let mut got = vec![f32::NAN; b * DIM];
            st.write_ranges(&rows(&ids, &mask, b, s, TURBO_NORMALIZE_L2, DIM), &ids, &mask, &ranges);
            st.run(&mut pool, &mut got);
            for (k, r) in ranges.iter().enumerate() {
                let row = r.row as usize;
                assert!(bits(&got[k * DIM..][..DIM]) == bits(&want[row * DIM..][..DIM]), "{name}: row {row}");
            }
        }
    }

    /// Enough ranges, and tokens in them, that the run splits them over
    /// threads: the same bits on 1, 3 and every processor's thread.
    #[test]
    fn ranges_are_the_same_bits_on_any_number_of_threads() {
        let (b, s) = (8, 300);
        let mut ids = vec![0; b * s];
        let mut mask = vec![1; b * s];
        for (i, id) in ids.iter_mut().enumerate() {
            *id = ((i * 37) % VOCAB) as i32;
        }
        // Masked columns are even, and every range starts on an odd one,
        // so each holds a live token.
        for p in (2..s).step_by(10) {
            mask[3 * s + p] = 0;
        }
        let mut ranges = Vec::new();
        for r in 0..b {
            for start in (1..s).step_by(2) {
                ranges.push(range(r, start, (start + 1 + start % 23).min(s)));
            }
        }
        let tokens: usize = ranges.iter().map(|r| (r.end - r.start) as usize).sum();
        assert!(tokens > 4 * TOKENS_PER_TASK);
        let most = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        for (name, mut st, _values) in sessions(b, s, ranges.len()) {
            let mut want = Vec::new();
            for threads in [1, 3, most] {
                let mut pool = Pool::with_spin(threads, super::super::pool::SPIN_BATCH);
                let mut got = vec![f32::NAN; ranges.len() * DIM];
                st.write_ranges(&rows(&ids, &mask, b, s, TURBO_NORMALIZE_L2, DIM), &ids, &mask, &ranges);
                st.run(&mut pool, &mut got);
                if want.is_empty() {
                    want = got;
                } else {
                    assert!(bits(&got) == bits(&want), "{name}: {threads} threads differ from 1");
                }
            }
        }
    }
}
