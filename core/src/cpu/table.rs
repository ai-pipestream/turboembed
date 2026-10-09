//! A static model on the host: the family TURBO_FAMILY_STATIC names.
//!
//! There is no encoder. A row's vector is computed as model2vec's
//! StaticModel computes it, to the bit where numpy's order allows: each
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

use std::sync::Arc;

use crate::backend::{
    TURBO_STATIC_EMBEDDINGS, TURBO_STATIC_MAPPING, TURBO_STATIC_WEIGHTS, turbo_backend_embed_rows, turbo_backend_tensor,
};
use crate::{
    TURBO_DTYPE_BF16, TURBO_DTYPE_F16, TURBO_DTYPE_F64, TURBO_DTYPE_I8, TURBO_DTYPE_I32, TURBO_NORMALIZE_L2,
    TURBO_POOLING_CLS, TURBO_POOLING_LAST,
};

use super::kernels::Isa;
use super::pool::Pool;

/// Values of a vector summed at once, in registers: eight AVX2 registers,
/// four AVX-512 ones.
const BLOCK: usize = 64;

/// Live tokens a task should sum, at least, before the run splits rows
/// over threads: below it a task is shorter than handing it to a thread.
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
}

// The core's verified bytes, unchanged until model_release.
unsafe impl Send for Table {}
unsafe impl Sync for Table {}

impl Table {
    /// The tensors model_load was handed, in TURBO_STATIC_* order; the
    /// core has checked their shapes, dtypes and every mapping value.
    pub(super) fn new(tensors: &[turbo_backend_tensor]) -> Table {
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
        Table {
            rows: e.data as *const u8,
            dtype: e.dtype,
            dim: e.shape[1] as usize,
            height: e.shape[0] as usize,
            weights,
            mapping,
            wide,
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
pub(super) struct Quantized {
    values: Box<[i8]>,
    scales: Box<[f32]>,
}

impl Quantized {
    pub(super) fn new(t: &Table) -> Quantized {
        let mut values = vec![0i8; t.height * t.dim].into_boxed_slice();
        let mut scales = vec![0f32; t.height].into_boxed_slice();
        for (row, (q, scale)) in values.chunks_exact_mut(t.dim).zip(scales.iter_mut()).enumerate() {
            let most = (0..t.dim).map(|j| t.value(row, j).abs()).fold(0f64, f64::max);
            if most == 0.0 || !most.is_finite() {
                continue;
            }
            *scale = (most / 127.0) as f32;
            for (j, v) in q.iter_mut().enumerate() {
                *v = (t.value(row, j) / *scale as f64).round().clamp(-127.0, 127.0) as i8;
            }
        }
        Quantized { values, scales }
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
    batch: usize,
    pooling: u32,
    normalize: u32,
    output_dim: usize,
}

impl Static {
    /// A session's state for rows of up to `max_seq` tokens, `max_batch`
    /// of them. Err is the bytes that could not be allocated.
    pub(super) fn new(
        table: Arc<Table>,
        quantized: Option<Arc<Quantized>>,
        isa: Isa,
        max_batch: usize,
        max_seq: usize,
    ) -> Result<Static, usize> {
        let tokens = max_batch.checked_mul(max_seq).ok_or(usize::MAX)?;
        let mut ids = Vec::new();
        ids.try_reserve_exact(tokens).map_err(|_| tokens.saturating_mul(4))?;
        ids.resize(tokens, 0);
        Ok(Static {
            table,
            quantized,
            isa,
            f16c: f16c(isa),
            ids,
            starts: vec![0; max_batch + 1],
            first: vec![0; max_batch],
            batch: 0,
            pooling: 0,
            normalize: 0,
            output_dim: 0,
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

    /// Row `r` into `dst`, output_dim values.
    #[inline(always)]
    fn row(&self, r: usize, dst: &mut [f32]) {
        let ids = &self.ids[self.starts[r]..self.starts[r + 1]];
        let t = &*self.table;
        if ids.is_empty() {
            dst.fill(0.0);
            return;
        }
        if let Some(q) = &self.quantized {
            let first = [self.first[r]];
            let ids = match self.pooling {
                TURBO_POOLING_CLS => &first[..],
                TURBO_POOLING_LAST => &ids[ids.len() - 1..],
                _ => ids,
            };
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
        match self.pooling {
            TURBO_POOLING_CLS => self.one(self.first[r] as usize, dst),
            TURBO_POOLING_LAST => self.one(*ids.last().expect("not empty") as usize, dst),
            _ => match t.dtype {
                #[cfg(target_arch = "x86_64")]
                // SAFETY: f16c and avx2 are present (Static::new).
                TURBO_DTYPE_F16 if self.f16c && t.weights.is_none() => unsafe { mean_f16c(t, ids, dst) },
                TURBO_DTYPE_F16 => mean::<F16>(t, ids, dst),
                TURBO_DTYPE_BF16 => mean::<Bf16>(t, ids, dst),
                TURBO_DTYPE_F64 => mean::<F64>(t, ids, dst),
                TURBO_DTYPE_I8 => mean::<I8>(t, ids, dst),
                _ => mean::<F32>(t, ids, dst),
            },
        }
        // numpy stores an F16 table's mean as F16.
        let half = t.dtype == TURBO_DTYPE_F16;
        if half {
            self.round_f16(dst);
        }
        if self.normalize == TURBO_NORMALIZE_L2 {
            // As StaticModel: x / (norm + 1e-32) in F32, so the zero vector
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

    /// One token's row times its weight, as one row's mean.
    fn one(&self, id: usize, dst: &mut [f32]) {
        let t = &*self.table;
        match t.dtype {
            TURBO_DTYPE_F16 => mean::<F16>(t, &[id as u32], dst),
            TURBO_DTYPE_BF16 => mean::<Bf16>(t, &[id as u32], dst),
            TURBO_DTYPE_F64 => mean::<F64>(t, &[id as u32], dst),
            TURBO_DTYPE_I8 => mean::<I8>(t, &[id as u32], dst),
            _ => mean::<F32>(t, &[id as u32], dst),
        }
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
/// An I8 value is the number it holds, as StaticModel reads it.
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
        let mut r = [0f32; 8];
        for (k, v) in r.iter_mut().enumerate() {
            *v = x[k] * x[k];
        }
        let mut i = 8;
        while i < n - n % 8 {
            for (k, v) in r.iter_mut().enumerate() {
                *v += x[i + k] * x[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += x[i] * x[i];
            i += 1;
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
    rows_per_task: usize,
}

/// The run's output; each task writes only its own rows.
struct Out(*mut f32);
unsafe impl Sync for Out {}

impl Job<'_> {
    #[inline(always)]
    fn task(&self, t: usize) {
        let s = self.s;
        let od = s.output_dim;
        let rows = t * self.rows_per_task..((t + 1) * self.rows_per_task).min(s.batch);
        for r in rows {
            // SAFETY: row r is this task's alone, inside the output.
            let dst = unsafe { std::slice::from_raw_parts_mut(self.out.0.add(r * od), od) };
            s.row(r, dst);
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
                };
                let q = Quantized::new(&t);
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
