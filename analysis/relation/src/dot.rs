//! Dot reductions for relation scores; projections use the shared Matrix.

/// A dot product: with AVX2 and FMA where the processor has them, as the
/// encoder's kernels detect, else in eight independent lanes the compiler
/// can keep in one vector register.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot product of unlike lengths");
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: the processor has AVX2 and FMA.
        return unsafe { dot_avx2(a, b) };
    }
    dot_lanes(a, b)
}

#[inline(always)]
fn dot_lanes(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (ac, ar) = a.as_chunks::<8>();
    let (bc, br) = b.as_chunks::<8>();
    for (x, y) in ac.iter().zip(bc) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let tail: f32 = ar.iter().zip(br).map(|(x, y)| x * y).sum();
    acc.iter().sum::<f32>() + tail
}

/// [`dot`] with 256-bit FMA, four accumulators so the FMAs pipeline.
///
/// # Safety
/// The processor has AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 32 <= n {
        for (k, c) in acc.iter_mut().enumerate() {
            // SAFETY: i + 32 <= n, so both slices have these 8 floats.
            let (x, y) =
                unsafe { (_mm256_loadu_ps(a.as_ptr().add(i + 8 * k)), _mm256_loadu_ps(b.as_ptr().add(i + 8 * k))) };
            *c = _mm256_fmadd_ps(x, y, *c);
        }
        i += 32;
    }
    let s = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
    let mut lanes = [0.0f32; 8];
    // SAFETY: lanes holds 8 floats.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), s) };
    lanes.iter().sum::<f32>() + dot_lanes(&a[i..n], &b[i..n])
}
