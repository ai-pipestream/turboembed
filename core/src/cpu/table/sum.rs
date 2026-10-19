//! The sums: the mean of a set of table rows in each stored dtype, and
//! in the I8 copy.

use super::super::kernels::Isa;
use super::{BLOCK, Quantized, Table, f16_to_f32};

/// A stored dtype of the table.
pub(super) trait Stored {
    /// Value j of table row `row`, as F32: exact, except an F64 value.
    fn at(t: &Table, row: usize, j: usize) -> f32;
    /// Value j of table row `row`, as F64, exact.
    #[inline(always)]
    fn at64(t: &Table, row: usize, j: usize) -> f64 {
        Self::at(t, row, j) as f64
    }
}

pub(super) struct F32;
pub(super) struct F16;
pub(super) struct Bf16;
pub(super) struct F64;
/// An I8 value is the number it holds.
pub(super) struct I8;

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
pub(super) fn mean<S: Stored>(t: &Table, ids: &[u32], dst: &mut [f32]) {
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
pub(super) fn mean_quantized(t: &Table, q: &Quantized, ids: &[u32], dst: &mut [f32]) {
    mean_quantized_from(t, q, ids, dst, 0)
}

/// `mean_quantized` from value `from` of the width on.
#[inline(always)]
pub(super) fn mean_quantized_from(t: &Table, q: &Quantized, ids: &[u32], dst: &mut [f32], mut from: usize) {
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
pub(super) unsafe fn mean_quantized_avx2(t: &Table, q: &Quantized, ids: &[u32], dst: &mut [f32]) {
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
pub(super) fn f16c(isa: Isa) -> bool {
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
pub(super) unsafe fn mean_f16c(t: &Table, ids: &[u32], dst: &mut [f32]) {
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
