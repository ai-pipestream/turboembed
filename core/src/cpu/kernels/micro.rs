//! The micro-kernels of the matrix product: a register tile of rows by
//! one panel of columns, per instruction set.

#[cfg(target_arch = "aarch64")]
use super::simd::N16;
#[cfg(target_arch = "x86_64")]
use super::simd::{Y16, Z16};

use super::simd::{P16, Simd};

/// The most accumulators a micro-kernel tile holds: MR x NR.
pub(super) const ACC: usize = 8 * 32;

/// A register tile of the product: `rows` (at most MR) input rows by one
/// panel of NR columns, each a chain of FMAs over k from 0.
pub(in crate::cpu) trait Micro {
    const MR: usize;
    const NR: usize;
    /// Whether madd fuses.
    const FMA: bool;
    /// Sixteen lanes of the same instruction set, for attention.
    type V: Simd;

    /// acc[i * NR + j] = sum over kk of x[i * ldx + kk] * p[kk * NR + j],
    /// for i under rows.
    ///
    /// # Safety
    /// x holds rows rows of k floats at stride ldx, p holds k * NR floats,
    /// and the processor has the kernel's instruction set.
    unsafe fn tile(rows: usize, x: *const f32, ldx: usize, p: *const f32, k: usize, acc: &mut [f32; ACC]);
}

#[inline(always)]
pub(super) fn madd(fma: bool, a: f32, b: f32, c: f32) -> f32 {
    if fma { a.mul_add(b, c) } else { a * b + c }
}

pub(in crate::cpu) struct Portable;

impl Micro for Portable {
    const MR: usize = 4;
    const NR: usize = 8;
    const FMA: bool = cfg!(target_arch = "aarch64");
    type V = P16;

    #[inline(always)]
    unsafe fn tile(rows: usize, x: *const f32, ldx: usize, p: *const f32, k: usize, acc: &mut [f32; ACC]) {
        // SAFETY: as the trait says, for each arm.
        unsafe {
            match rows {
                1 => portable_tile::<1>(x, ldx, p, k, acc),
                2 => portable_tile::<2>(x, ldx, p, k, acc),
                3 => portable_tile::<3>(x, ldx, p, k, acc),
                _ => portable_tile::<4>(x, ldx, p, k, acc),
            }
        }
    }
}

#[inline(always)]
pub(super) unsafe fn portable_tile<const R: usize>(
    x: *const f32,
    ldx: usize,
    p: *const f32,
    k: usize,
    acc: &mut [f32; ACC],
) {
    const NR: usize = Portable::NR;
    let mut c = [[0.0f32; NR]; R];
    for kk in 0..k {
        // SAFETY: kk < k, so the panel has these NR floats, and each row
        // under R has float kk.
        let b: [f32; NR] = unsafe { *(p.add(kk * NR) as *const [f32; NR]) };
        for (i, ci) in c.iter_mut().enumerate() {
            let a = unsafe { *x.add(i * ldx + kk) };
            for j in 0..NR {
                ci[j] = madd(Portable::FMA, a, b[j], ci[j]);
            }
        }
    }
    for (i, ci) in c.iter().enumerate() {
        acc[i * NR..][..NR].copy_from_slice(ci);
    }
}

#[cfg(target_arch = "x86_64")]
pub(in crate::cpu) struct Avx2;

#[cfg(target_arch = "x86_64")]
impl Micro for Avx2 {
    const MR: usize = 6;
    const NR: usize = 16;
    const FMA: bool = true;
    type V = Y16;

    #[inline(always)]
    unsafe fn tile(rows: usize, x: *const f32, ldx: usize, p: *const f32, k: usize, acc: &mut [f32; ACC]) {
        // SAFETY: as the trait says, for each arm.
        unsafe {
            match rows {
                1 => avx2_tile::<1>(x, ldx, p, k, acc),
                2 => avx2_tile::<2>(x, ldx, p, k, acc),
                3 => avx2_tile::<3>(x, ldx, p, k, acc),
                4 => avx2_tile::<4>(x, ldx, p, k, acc),
                5 => avx2_tile::<5>(x, ldx, p, k, acc),
                _ => avx2_tile::<6>(x, ldx, p, k, acc),
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn avx2_tile<const R: usize>(
    x: *const f32,
    ldx: usize,
    p: *const f32,
    k: usize,
    acc: &mut [f32; ACC],
) {
    use std::arch::x86_64::*;
    let mut c0 = [_mm256_setzero_ps(); R];
    let mut c1 = [_mm256_setzero_ps(); R];
    for kk in 0..k {
        // SAFETY: kk < k: the panel has 16 floats at kk * 16, and each
        // row under R has float kk.
        let (b0, b1) = unsafe { (_mm256_loadu_ps(p.add(kk * 16)), _mm256_loadu_ps(p.add(kk * 16 + 8))) };
        for i in 0..R {
            let a = _mm256_set1_ps(unsafe { *x.add(i * ldx + kk) });
            c0[i] = _mm256_fmadd_ps(a, b0, c0[i]);
            c1[i] = _mm256_fmadd_ps(a, b1, c1[i]);
        }
    }
    for i in 0..R {
        // SAFETY: R * 16 <= ACC.
        unsafe {
            _mm256_storeu_ps(acc.as_mut_ptr().add(i * 16), c0[i]);
            _mm256_storeu_ps(acc.as_mut_ptr().add(i * 16 + 8), c1[i]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(in crate::cpu) struct Avx512;

#[cfg(target_arch = "x86_64")]
impl Micro for Avx512 {
    const MR: usize = 8;
    const NR: usize = 32;
    const FMA: bool = true;
    type V = Z16;

    #[inline(always)]
    unsafe fn tile(rows: usize, x: *const f32, ldx: usize, p: *const f32, k: usize, acc: &mut [f32; ACC]) {
        // SAFETY: as the trait says, for each arm.
        unsafe {
            match rows {
                1 => avx512_tile::<1>(x, ldx, p, k, acc),
                2 => avx512_tile::<2>(x, ldx, p, k, acc),
                3 => avx512_tile::<3>(x, ldx, p, k, acc),
                4 => avx512_tile::<4>(x, ldx, p, k, acc),
                5 => avx512_tile::<5>(x, ldx, p, k, acc),
                6 => avx512_tile::<6>(x, ldx, p, k, acc),
                7 => avx512_tile::<7>(x, ldx, p, k, acc),
                _ => avx512_tile::<8>(x, ldx, p, k, acc),
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn avx512_tile<const R: usize>(
    x: *const f32,
    ldx: usize,
    p: *const f32,
    k: usize,
    acc: &mut [f32; ACC],
) {
    use std::arch::x86_64::*;
    let mut c0 = [_mm512_setzero_ps(); R];
    let mut c1 = [_mm512_setzero_ps(); R];
    for kk in 0..k {
        // SAFETY: kk < k: the panel has 32 floats at kk * 32, and each
        // row under R has float kk.
        let (b0, b1) = unsafe { (_mm512_loadu_ps(p.add(kk * 32)), _mm512_loadu_ps(p.add(kk * 32 + 16))) };
        for i in 0..R {
            let a = _mm512_set1_ps(unsafe { *x.add(i * ldx + kk) });
            c0[i] = _mm512_fmadd_ps(a, b0, c0[i]);
            c1[i] = _mm512_fmadd_ps(a, b1, c1[i]);
        }
    }
    for i in 0..R {
        // SAFETY: R * 32 <= ACC.
        unsafe {
            _mm512_storeu_ps(acc.as_mut_ptr().add(i * 32), c0[i]);
            _mm512_storeu_ps(acc.as_mut_ptr().add(i * 32 + 16), c1[i]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(in crate::cpu) struct Neon;

#[cfg(target_arch = "aarch64")]
impl Micro for Neon {
    // 6 rows by 4 vectors of accumulators, the panel's 4 vectors and the
    // broadcast row value: 29 of the 32 vector registers.
    const MR: usize = 6;
    const NR: usize = 16;
    const FMA: bool = true;
    type V = N16;

    #[inline(always)]
    unsafe fn tile(rows: usize, x: *const f32, ldx: usize, p: *const f32, k: usize, acc: &mut [f32; ACC]) {
        // SAFETY: as the trait says, for each arm.
        unsafe {
            match rows {
                1 => neon_tile::<1>(x, ldx, p, k, acc),
                2 => neon_tile::<2>(x, ldx, p, k, acc),
                3 => neon_tile::<3>(x, ldx, p, k, acc),
                4 => neon_tile::<4>(x, ldx, p, k, acc),
                5 => neon_tile::<5>(x, ldx, p, k, acc),
                _ => neon_tile::<6>(x, ldx, p, k, acc),
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(super) unsafe fn neon_tile<const R: usize>(
    x: *const f32,
    ldx: usize,
    p: *const f32,
    k: usize,
    acc: &mut [f32; ACC],
) {
    use std::arch::aarch64::*;
    let mut c = [[vdupq_n_f32(0.0); 4]; R];
    for kk in 0..k {
        // SAFETY: kk < k: the panel has 16 floats at kk * 16, and each
        // row under R has float kk.
        let b = unsafe {
            let q = p.add(kk * 16);
            [vld1q_f32(q), vld1q_f32(q.add(4)), vld1q_f32(q.add(8)), vld1q_f32(q.add(12))]
        };
        for (i, ci) in c.iter_mut().enumerate() {
            let a = vdupq_n_f32(unsafe { *x.add(i * ldx + kk) });
            for v in 0..4 {
                ci[v] = vfmaq_f32(ci[v], a, b[v]);
            }
        }
    }
    for (i, ci) in c.iter().enumerate() {
        for (v, cv) in ci.iter().enumerate() {
            // SAFETY: R * 16 <= ACC.
            unsafe { vst1q_f32(acc.as_mut_ptr().add(i * 16 + v * 4), *cv) }
        }
    }
}
