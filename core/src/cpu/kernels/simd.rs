//! Sixteen F32 lanes in the registers of each instruction set.

use super::micro::{Micro, Portable, madd};

/// Sixteen F32 lanes, in the registers of one instruction set: what the
/// attention kernel is written in, so the compiler need not find the
/// vectors itself. Every operation is lane by lane, and madd fuses exactly
/// where the instruction set's matrix kernel does.
///
/// # Safety
/// Every method may be called only on a processor with the instruction
/// set, and load and store only where 16 floats may be read or written.
pub(in crate::cpu) trait Simd: Copy {
    /// Queries attention takes at once: as many as keep its sums in the
    /// instruction set's registers.
    const GROUP: usize;

    unsafe fn zero() -> Self;
    unsafe fn splat(x: f32) -> Self;
    unsafe fn load(p: *const f32) -> Self;
    unsafe fn store(self, p: *mut f32);
    /// self * b + c
    unsafe fn madd(self, b: Self, c: Self) -> Self;
    unsafe fn add(self, b: Self) -> Self;
    unsafe fn mul(self, b: Self) -> Self;
    /// The larger, lane by lane, of values that are not NaN.
    unsafe fn max(self, b: Self) -> Self;
}

#[derive(Clone, Copy)]
pub(in crate::cpu) struct P16([f32; 16]);

impl Simd for P16 {
    const GROUP: usize = 4;
    #[inline(always)]
    unsafe fn zero() -> Self {
        P16([0.0; 16])
    }
    #[inline(always)]
    unsafe fn splat(x: f32) -> Self {
        P16([x; 16])
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        // SAFETY: the trait's contract.
        P16(unsafe { *(p as *const [f32; 16]) })
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        // SAFETY: the trait's contract.
        unsafe { *(p as *mut [f32; 16]) = self.0 }
    }
    #[inline(always)]
    unsafe fn madd(self, b: Self, c: Self) -> Self {
        P16(std::array::from_fn(|l| madd(Portable::FMA, self.0[l], b.0[l], c.0[l])))
    }
    #[inline(always)]
    unsafe fn add(self, b: Self) -> Self {
        P16(std::array::from_fn(|l| self.0[l] + b.0[l]))
    }
    #[inline(always)]
    unsafe fn mul(self, b: Self) -> Self {
        P16(std::array::from_fn(|l| self.0[l] * b.0[l]))
    }
    #[inline(always)]
    unsafe fn max(self, b: Self) -> Self {
        P16(std::array::from_fn(|l| if b.0[l] > self.0[l] { b.0[l] } else { self.0[l] }))
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub(in crate::cpu) struct Y16(std::arch::x86_64::__m256, std::arch::x86_64::__m256);

// SAFETY, for every method: the trait's contract; the intrinsics need AVX2
// and FMA, which the processor has.
#[cfg(target_arch = "x86_64")]
impl Simd for Y16 {
    const GROUP: usize = 2;
    #[inline(always)]
    unsafe fn zero() -> Self {
        use std::arch::x86_64::*;
        unsafe { Y16(_mm256_setzero_ps(), _mm256_setzero_ps()) }
    }
    #[inline(always)]
    unsafe fn splat(x: f32) -> Self {
        use std::arch::x86_64::*;
        unsafe { Y16(_mm256_set1_ps(x), _mm256_set1_ps(x)) }
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        use std::arch::x86_64::*;
        unsafe { Y16(_mm256_loadu_ps(p), _mm256_loadu_ps(p.add(8))) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        use std::arch::x86_64::*;
        unsafe {
            _mm256_storeu_ps(p, self.0);
            _mm256_storeu_ps(p.add(8), self.1);
        }
    }
    #[inline(always)]
    unsafe fn madd(self, b: Self, c: Self) -> Self {
        use std::arch::x86_64::*;
        unsafe { Y16(_mm256_fmadd_ps(self.0, b.0, c.0), _mm256_fmadd_ps(self.1, b.1, c.1)) }
    }
    #[inline(always)]
    unsafe fn add(self, b: Self) -> Self {
        use std::arch::x86_64::*;
        unsafe { Y16(_mm256_add_ps(self.0, b.0), _mm256_add_ps(self.1, b.1)) }
    }
    #[inline(always)]
    unsafe fn mul(self, b: Self) -> Self {
        use std::arch::x86_64::*;
        unsafe { Y16(_mm256_mul_ps(self.0, b.0), _mm256_mul_ps(self.1, b.1)) }
    }
    #[inline(always)]
    unsafe fn max(self, b: Self) -> Self {
        use std::arch::x86_64::*;
        unsafe { Y16(_mm256_max_ps(self.0, b.0), _mm256_max_ps(self.1, b.1)) }
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub(in crate::cpu) struct Z16(std::arch::x86_64::__m512);

// SAFETY, for every method: the trait's contract; the intrinsics need
// AVX-512F, which the processor has.
#[cfg(target_arch = "x86_64")]
impl Simd for Z16 {
    const GROUP: usize = 4;
    #[inline(always)]
    unsafe fn zero() -> Self {
        use std::arch::x86_64::*;
        unsafe { Z16(_mm512_setzero_ps()) }
    }
    #[inline(always)]
    unsafe fn splat(x: f32) -> Self {
        use std::arch::x86_64::*;
        unsafe { Z16(_mm512_set1_ps(x)) }
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        use std::arch::x86_64::*;
        unsafe { Z16(_mm512_loadu_ps(p)) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        use std::arch::x86_64::*;
        unsafe { _mm512_storeu_ps(p, self.0) }
    }
    #[inline(always)]
    unsafe fn madd(self, b: Self, c: Self) -> Self {
        use std::arch::x86_64::*;
        unsafe { Z16(_mm512_fmadd_ps(self.0, b.0, c.0)) }
    }
    #[inline(always)]
    unsafe fn add(self, b: Self) -> Self {
        use std::arch::x86_64::*;
        unsafe { Z16(_mm512_add_ps(self.0, b.0)) }
    }
    #[inline(always)]
    unsafe fn mul(self, b: Self) -> Self {
        use std::arch::x86_64::*;
        unsafe { Z16(_mm512_mul_ps(self.0, b.0)) }
    }
    #[inline(always)]
    unsafe fn max(self, b: Self) -> Self {
        use std::arch::x86_64::*;
        unsafe { Z16(_mm512_max_ps(self.0, b.0)) }
    }
}

#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub(in crate::cpu) struct N16([std::arch::aarch64::float32x4_t; 4]);

// SAFETY, for every method: the trait's contract; the intrinsics need
// NEON, which every aarch64 processor has.
#[cfg(target_arch = "aarch64")]
impl Simd for N16 {
    const GROUP: usize = 4;
    #[inline(always)]
    unsafe fn zero() -> Self {
        use std::arch::aarch64::*;
        unsafe { N16([vdupq_n_f32(0.0); 4]) }
    }
    #[inline(always)]
    unsafe fn splat(x: f32) -> Self {
        use std::arch::aarch64::*;
        unsafe { N16([vdupq_n_f32(x); 4]) }
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        use std::arch::aarch64::*;
        unsafe { N16([vld1q_f32(p), vld1q_f32(p.add(4)), vld1q_f32(p.add(8)), vld1q_f32(p.add(12))]) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        use std::arch::aarch64::*;
        unsafe {
            for (v, x) in self.0.iter().enumerate() {
                vst1q_f32(p.add(4 * v), *x);
            }
        }
    }
    #[inline(always)]
    unsafe fn madd(self, b: Self, c: Self) -> Self {
        use std::arch::aarch64::*;
        unsafe { N16(std::array::from_fn(|v| vfmaq_f32(c.0[v], self.0[v], b.0[v]))) }
    }
    #[inline(always)]
    unsafe fn add(self, b: Self) -> Self {
        use std::arch::aarch64::*;
        unsafe { N16(std::array::from_fn(|v| vaddq_f32(self.0[v], b.0[v]))) }
    }
    #[inline(always)]
    unsafe fn mul(self, b: Self) -> Self {
        use std::arch::aarch64::*;
        unsafe { N16(std::array::from_fn(|v| vmulq_f32(self.0[v], b.0[v]))) }
    }
    #[inline(always)]
    unsafe fn max(self, b: Self) -> Self {
        use std::arch::aarch64::*;
        // b where it is the larger, else self: P16's choice, so a tie of
        // 0.0 and -0.0 keeps the same zero.
        unsafe { N16(std::array::from_fn(|v| vbslq_f32(vcgtq_f32(b.0[v], self.0[v]), b.0[v], self.0[v]))) }
    }
}
