//! The SPLADE decoder's inner product over the transposed weight:
//! `o[i] += sum over j of x[j] * wt[j * v + i0 + i]` for the `o.len()`
//! columns from `i0`. Every kernel sums each column in the order
//! `Linear::apply` does, from its bias over j = 0, 1, .. in turn. The SIMD
//! kernels (AVX2 on x86_64, NEON on aarch64) fuse each multiply-add, in
//! the vector loop and in the tail alike, so they give the same bits as
//! `fused` on the same input. The portable loop rounds the product and
//! the sum separately, as `Linear::apply` compiles without FMA, so it
//! differs from them by that rounding alone.

/// The kernel this processor runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Kernel {
    Portable,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl Kernel {
    pub(super) fn detect() -> Kernel {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return Kernel::Avx2;
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Kernel::Neon;
        }
        Kernel::Portable
    }

    /// # Panics
    /// If `wt` does not hold `x.len()` rows of `v` floats or the columns
    /// pass `v`.
    pub(super) fn run(self, x: &[f32], wt: &[f32], o: &mut [f32], v: usize, i0: usize) {
        assert!(i0 + o.len() <= v && wt.len() >= x.len() * v, "decoder product shape");
        match self {
            Kernel::Portable => portable(x, wt, o, v, i0),
            // SAFETY: detect chose it, so the processor has AVX2 and FMA;
            // the shapes are checked above.
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => unsafe { avx2(x, wt, o, v, i0) },
            // SAFETY: detect chose it, so the processor has NEON; the
            // shapes are checked above.
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => unsafe { neon(x, wt, o, v, i0) },
        }
    }
}

/// Plain Rust, a rounded product then a rounded sum per step.
pub(super) fn portable(x: &[f32], wt: &[f32], o: &mut [f32], v: usize, i0: usize) {
    let w = o.len();
    for (j, &xj) in x.iter().enumerate() {
        let row = &wt[j * v + i0..j * v + i0 + w];
        for (o, &r) in o.iter_mut().zip(row) {
            *o += xj * r;
        }
    }
}

/// The SIMD kernels' arithmetic in plain Rust: one fused multiply-add per
/// step, in the same order. Their reference in the tests.
#[cfg(test)]
pub(super) fn fused(x: &[f32], wt: &[f32], o: &mut [f32], v: usize, i0: usize) {
    for (i, o) in o.iter_mut().enumerate() {
        for (j, &xj) in x.iter().enumerate() {
            *o = xj.mul_add(wt[j * v + i0 + i], *o);
        }
    }
}

/// # Safety
/// The processor has AVX2 and FMA, and `wt` holds `x.len()` rows of `v`
/// floats with `i0 + o.len() <= v`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2(x: &[f32], wt: &[f32], o: &mut [f32], v: usize, i0: usize) {
    use std::arch::x86_64::*;
    let w = o.len();
    let (op, wp) = (o.as_mut_ptr(), wt.as_ptr());
    let mut i = 0;
    while i + 32 <= w {
        // Four 8-lane accumulators cover 32 columns a step, so the loads
        // and FMAs pipeline.
        // SAFETY: i + 32 <= w, and row j has columns i0..i0 + w (contract).
        unsafe {
            let mut c = [0; 4].map(|_| _mm256_setzero_ps());
            for (q, c) in c.iter_mut().enumerate() {
                *c = _mm256_loadu_ps(op.add(i + 8 * q));
            }
            for (j, &xj) in x.iter().enumerate() {
                let a = _mm256_set1_ps(xj);
                let row = wp.add(j * v + i0 + i);
                for (q, c) in c.iter_mut().enumerate() {
                    *c = _mm256_fmadd_ps(a, _mm256_loadu_ps(row.add(8 * q)), *c);
                }
            }
            for (q, c) in c.iter().enumerate() {
                _mm256_storeu_ps(op.add(i + 8 * q), *c);
            }
        }
        i += 32;
    }
    tail(x, wt, &mut o[i..], v, i0 + i);
}

/// # Safety
/// The processor has NEON, and `wt` holds `x.len()` rows of `v` floats
/// with `i0 + o.len() <= v`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon(x: &[f32], wt: &[f32], o: &mut [f32], v: usize, i0: usize) {
    use std::arch::aarch64::*;
    let w = o.len();
    let (op, wp) = (o.as_mut_ptr(), wt.as_ptr());
    let mut i = 0;
    while i + 16 <= w {
        // Four 4-lane accumulators cover 16 columns a step.
        // SAFETY: i + 16 <= w, and row j has columns i0..i0 + w (contract).
        unsafe {
            let mut c = [0; 4].map(|_| vdupq_n_f32(0.0));
            for (q, c) in c.iter_mut().enumerate() {
                *c = vld1q_f32(op.add(i + 4 * q));
            }
            for (j, &xj) in x.iter().enumerate() {
                let a = vdupq_n_f32(xj);
                let row = wp.add(j * v + i0 + i);
                for (q, c) in c.iter_mut().enumerate() {
                    // c + a * row, one rounding.
                    *c = vfmaq_f32(*c, a, vld1q_f32(row.add(4 * q)));
                }
            }
            for (q, c) in c.iter().enumerate() {
                vst1q_f32(op.add(i + 4 * q), *c);
            }
        }
        i += 16;
    }
    while i + 4 <= w {
        // SAFETY: as above, for 4 columns.
        unsafe {
            let mut c = vld1q_f32(op.add(i));
            for (j, &xj) in x.iter().enumerate() {
                c = vfmaq_f32(c, vdupq_n_f32(xj), vld1q_f32(wp.add(j * v + i0 + i)));
            }
            vst1q_f32(op.add(i), c);
        }
        i += 4;
    }
    tail(x, wt, &mut o[i..], v, i0 + i);
}

/// The columns under one vector step, fused as the vector loop is.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn tail(x: &[f32], wt: &[f32], o: &mut [f32], v: usize, i0: usize) {
    for (i, o) in o.iter_mut().enumerate() {
        let mut s = *o;
        for (j, &xj) in x.iter().enumerate() {
            s = xj.mul_add(wt[j * v + i0 + i], s);
        }
        *o = s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761).max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 8) as f32 / 8_388_608.0 - 1.0
            })
            .collect()
    }

    /// The kernel this processor runs against the plain loops, over
    /// column counts with and without a vector tail and offsets into
    /// the row: bit for bit against the fused order, within rounding
    /// against the portable loop.
    #[test]
    fn the_detected_kernel_matches_the_portable_loop() {
        let kernel = Kernel::detect();
        for (h, v) in [(1, 1), (3, 5), (17, 37), (64, 100), (768, 301)] {
            let x = values(h, 1);
            let wt = values(h * v, 2);
            let bias = values(v, 3);
            for (i0, w) in [(0, v), (v / 3, v - v / 3), (v - v.min(19), v.min(19))] {
                let mut got = bias[i0..i0 + w].to_vec();
                kernel.run(&x, &wt, &mut got, v, i0);
                let mut plain = bias[i0..i0 + w].to_vec();
                portable(&x, &wt, &mut plain, v, i0);
                let mut fma = bias[i0..i0 + w].to_vec();
                fused(&x, &wt, &mut fma, v, i0);
                if kernel == Kernel::Portable {
                    assert_eq!(got, plain);
                } else {
                    assert_eq!(got, fma, "{kernel:?} {h}x{v} from {i0}");
                }
                for (i, (g, p)) in got.iter().zip(&plain).enumerate() {
                    let scale: f32 = x.iter().enumerate().map(|(j, xj)| (xj * wt[j * v + i0 + i]).abs()).sum();
                    assert!((g - p).abs() <= 1e-6 * (1.0 + scale), "{kernel:?} {h}x{v} [{i}]: {g} vs {p}");
                }
            }
        }
    }

    /// The decoder product at a base model's shape, detected kernel
    /// against the portable loop. Run in release:
    /// `cargo test --release --locked -p turbo --lib decode -- --ignored --nocapture`
    #[test]
    #[ignore = "timing; run in release"]
    fn decoder_kernel_timing() {
        let (h, v, tokens) = (768, 30_522, 8);
        let x = values(tokens * h, 4);
        let wt = values(h * v, 5);
        let kernel = Kernel::detect();
        for (name, k) in [("portable", Kernel::Portable), ("detected", kernel)] {
            let mut o = vec![0.0f32; 2048];
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                for row in x.chunks_exact(h) {
                    for i0 in (0..v).step_by(2048) {
                        let w = 2048.min(v - i0);
                        k.run(row, &wt, &mut o[..w], v, i0);
                    }
                }
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            println!("decoder {name} ({k:?}): {best:.1} ms for {tokens} tokens of {h}x{v}");
        }
    }
}
