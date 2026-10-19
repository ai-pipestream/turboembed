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

use crate::backend::{TURBO_STATIC_EMBEDDINGS, TURBO_STATIC_MAPPING, TURBO_STATIC_WEIGHTS, turbo_backend_tensor};
use crate::bundle::AlignedBytes;
use crate::{TURBO_DTYPE_BF16, TURBO_DTYPE_F16, TURBO_DTYPE_F64, TURBO_DTYPE_I8, TURBO_DTYPE_I32};

mod round;
mod session;
mod sum;

#[cfg(test)]
mod range_tests;
#[cfg(all(test, target_arch = "x86_64"))]
mod tests;

pub(super) use session::Static;
use sum::{Bf16, F16, F32, F64, I8, Stored};

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

/// Bytes a value of a table dtype.
fn dtype_bytes(dtype: u32) -> usize {
    match dtype {
        TURBO_DTYPE_F16 | TURBO_DTYPE_BF16 => 2,
        TURBO_DTYPE_F64 => 8,
        TURBO_DTYPE_I8 => 1,
        _ => 4,
    }
}

/// The table for TURBO_PRECISION_FASTEST: each row as I8 values over the
/// row's own F32 scale (its largest magnitude over 127, the values
/// rounded to the nearest step), a quarter of an F32 table's bytes. The
/// model makes it once, for its first FASTEST session; the token weights
/// and the mapping stay the table's.
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

/// An IEEE half to the single it names exactly.
#[inline(always)]
fn f16_to_f32(h: u16) -> f32 {
    super::model::f16_to_f32(h)
}
