//! The norm's sum of squares and the rounding to F16, as numpy makes
//! them.

use super::f16_to_f32;

/// The sum of the squares of `x`, each square rounded to F32, summed as
/// numpy's pairwise_sum sums a contiguous F32 axis: under 8 values one by
/// one; up to 128 in eight running sums, combined as a tree, then the
/// rest one by one; above, the two halves (the first a multiple of 8)
/// apart.
pub(super) fn pairwise_squares(x: &[f32]) -> f32 {
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
pub(super) unsafe fn round_f16_f16c(x: &mut [f32]) {
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

pub(super) fn round_f16(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = f16_to_f32(f32_to_f16(*v));
    }
}

/// An F32 to the nearest IEEE half, ties to even: infinity past the
/// largest, subnormals below the smallest normal.
pub(super) fn f32_to_f16(v: f32) -> u16 {
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
