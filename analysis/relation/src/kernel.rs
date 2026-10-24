//! The products the GLiNER2 and GLiREL heads run on. One row at a time, a dot
//! product with AVX2 and FMA where the processor has them, eight portable
//! lanes otherwise. Many rows at once (the span projections of every
//! word), the shared packed GEMM in `turbo::cpu::Matrix`.

use turbo::cpu::{Matrix, Mode};
use turbo::head::Linear;

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

/// `l` on one row, through [`dot`].
pub(crate) fn apply(l: &Linear, x: &[f32], out: &mut [f32]) {
    assert!(x.len() == l.in_dim && out.len() == l.out_dim, "a row of the layer's widths");
    for ((o, w), &b) in out.iter_mut().zip(l.weight().chunks_exact(l.in_dim)).zip(l.bias()) {
        *o = dot(w, x) + b;
    }
}

/// A linear layer packed once for [`Packed::rows`], on the shared
/// packed GEMM in `turbo::cpu::Matrix`: the same micro-kernels the
/// encoder and the other model heads run, tail rows handled in place
/// with no padded copy of the input.
pub(crate) struct Packed {
    pub in_dim: usize,
    pub out_dim: usize,
    w: Matrix,
    bias: Vec<f32>,
}

impl Packed {
    /// Packs `l`'s row-major `[out][in]` weights. The error is the
    /// allocation the packing could not make, in bytes.
    pub fn new(l: &Linear) -> Result<Packed, String> {
        let w = Matrix::from_rows(&[l.weight()], l.in_dim)
            .map_err(|bytes| format!("packing a {}x{} layer needs {bytes} bytes", l.out_dim, l.in_dim))?;
        Ok(Packed { in_dim: l.in_dim, out_dim: l.out_dim, w, bias: l.bias().to_vec() })
    }

    /// The layer on each of the `n` rows of `x`, `[n][out]`.
    ///
    /// # Panics
    /// When `x` is not `n` rows of the input width.
    pub fn rows(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut out = vec![0.0; n * self.out_dim];
        self.rows_into(x, n, &mut out);
        out
    }

    /// [`Packed::rows`] into a caller-supplied `[n][out]` buffer.
    ///
    /// # Panics
    /// When `x` is not `n` rows of the input width or `out` is not
    /// `n` rows of the output width.
    fn rows_into(&self, x: &[f32], n: usize, out: &mut [f32]) {
        assert_eq!(Some(x.len()), n.checked_mul(self.in_dim), "{n} rows of width {}", self.in_dim);
        assert_eq!(Some(out.len()), n.checked_mul(self.out_dim), "{n} rows of width {}", self.out_dim);
        self.w.apply(x, self.in_dim, n, Some(&self.bias), out, self.out_dim, Mode::Plain);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The packed product agrees with one row at a time, on widths and
    /// row counts that leave partial tiles and blocks.
    #[test]
    fn packed_rows_agree_with_one_row_at_a_time() {
        let mut state = 9u64;
        let mut next = || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        for (inp, out_dim, n) in [(48, 64, 19), (40, 96, 8), (33, 32, 3), (16, 40, 11), (24, 48, 7), (8, 8, 0)] {
            let w: Vec<f32> = (0..inp * out_dim).map(|_| next()).collect();
            let b: Vec<f32> = (0..out_dim).map(|_| next()).collect();
            let l = Linear::new(inp, out_dim, w, b).unwrap();
            let x: Vec<f32> = (0..n * inp).map(|_| next()).collect();
            let p = Packed::new(&l).unwrap();
            let fast = p.rows(&x, n);
            let mut slow = vec![0.0; n * out_dim];
            p.rows_into(&x, n, &mut slow);
            assert_eq!(fast, slow);
            for i in 0..n {
                let mut one = vec![0.0; out_dim];
                apply(&l, &x[i * inp..(i + 1) * inp], &mut one);
                for (o, &want) in one.iter().enumerate() {
                    let got = fast[i * out_dim + o];
                    assert!((got - want).abs() < 1e-4, "{inp}x{out_dim} row {i} out {o}: {got} against {want}");
                }
            }
        }
    }
}
