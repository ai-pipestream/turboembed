//! The products the GLiNER2 heads run on. One row at a time, a dot
//! product with AVX2 and FMA where the processor has them, eight portable
//! lanes otherwise. Many rows at once (the span projections of every
//! word), a layer with its weights transposed, tiled in vector registers:
//! 32 outputs by 8 rows with AVX-512, 16 by 6 with AVX2 and FMA, plain
//! loops otherwise.

use turbo::head::Linear;

/// A dot product: with AVX2 and FMA where the processor has them, as the
/// encoder's kernels detect, else in eight independent lanes the compiler
/// can keep in one vector register.
pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
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
pub(super) fn apply(l: &Linear, x: &[f32], out: &mut [f32]) {
    for ((o, w), &b) in out.iter_mut().zip(l.weight.chunks_exact(l.in_dim)).zip(&l.bias) {
        *o = dot(w, x) + b;
    }
}

/// A linear layer with its weights transposed, `[in][out]`, for
/// [`Packed::rows`]: the outputs of one input value are contiguous, so a
/// tile of outputs accumulates in vector registers with no reduction.
pub(super) struct Packed {
    pub in_dim: usize,
    pub out_dim: usize,
    wt: Vec<f32>,
    bias: Vec<f32>,
}

/// Outputs per tile and rows per block of [`Packed::rows`].
const TILE: usize = 32;
const ROWS: usize = 8;

impl Packed {
    pub fn new(l: &Linear) -> Packed {
        let mut wt = vec![0.0; l.in_dim * l.out_dim];
        for (o, w) in l.weight.chunks_exact(l.in_dim).enumerate() {
            for (k, &v) in w.iter().enumerate() {
                wt[k * l.out_dim + o] = v;
            }
        }
        Packed { in_dim: l.in_dim, out_dim: l.out_dim, wt, bias: l.bias.clone() }
    }

    /// The layer on each of the `n` rows of `x`, `[n][out]`.
    pub fn rows(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut out = vec![0.0; n * self.out_dim];
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            // SAFETY: the processor has AVX-512F and FMA.
            unsafe { self.rows_avx512(x, n, &mut out) };
            return out;
        }
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: the processor has AVX2 and FMA.
            unsafe { self.rows_avx2(x, n, &mut out) };
            return out;
        }
        self.rows_into(x, n, &mut out);
        out
    }

    /// Full tiles of 32 outputs (two 512-bit registers) by blocks of eight
    /// rows: sixteen accumulators, two weight loads and eight broadcasts
    /// per input value. A partial tile or block goes through
    /// [`Packed::rows_into`].
    ///
    /// # Safety
    /// The processor has AVX-512F and FMA.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,fma")]
    unsafe fn rows_avx512(&self, x: &[f32], n: usize, out: &mut [f32]) {
        use std::arch::x86_64::*;
        let (m, kd) = (self.out_dim, self.in_dim);
        if !m.is_multiple_of(TILE) {
            return self.rows_into(x, n, out);
        }
        let full = n - n % ROWS;
        for o0 in (0..m).step_by(TILE) {
            // SAFETY: o0 + 32 <= m, so the bias and each weight row hold
            // these 32 floats; every row index is below n.
            unsafe {
                let b = [_mm512_loadu_ps(self.bias.as_ptr().add(o0)), _mm512_loadu_ps(self.bias.as_ptr().add(o0 + 16))];
                for i0 in (0..full).step_by(ROWS) {
                    let mut acc = [[b[0], b[1]]; ROWS];
                    for k in 0..kd {
                        let w = self.wt.as_ptr().add(k * m + o0);
                        let (w0, w1) = (_mm512_loadu_ps(w), _mm512_loadu_ps(w.add(16)));
                        for (r, a) in acc.iter_mut().enumerate() {
                            let xv = _mm512_set1_ps(*x.get_unchecked((i0 + r) * kd + k));
                            a[0] = _mm512_fmadd_ps(xv, w0, a[0]);
                            a[1] = _mm512_fmadd_ps(xv, w1, a[1]);
                        }
                    }
                    for (r, a) in acc.iter().enumerate() {
                        let o = out.as_mut_ptr().add((i0 + r) * m + o0);
                        _mm512_storeu_ps(o, a[0]);
                        _mm512_storeu_ps(o.add(16), a[1]);
                    }
                }
            }
        }
        if full < n {
            let mut tail = vec![0.0; (n - full) * m];
            self.rows_into(&x[full * kd..n * kd], n - full, &mut tail);
            out[full * m..n * m].copy_from_slice(&tail);
        }
    }

    /// Tiles of 16 outputs (two 256-bit registers) by blocks of six rows:
    /// twelve accumulators. A partial tile or block goes through
    /// [`Packed::rows_into`].
    ///
    /// # Safety
    /// The processor has AVX2 and FMA.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn rows_avx2(&self, x: &[f32], n: usize, out: &mut [f32]) {
        use std::arch::x86_64::*;
        const T: usize = 16;
        const R: usize = 6;
        let (m, kd) = (self.out_dim, self.in_dim);
        if !m.is_multiple_of(T) {
            return self.rows_into(x, n, out);
        }
        let full = n - n % R;
        for o0 in (0..m).step_by(T) {
            // SAFETY: o0 + 16 <= m, so the bias and each weight row hold
            // these 16 floats; every row index is below n.
            unsafe {
                let b = [_mm256_loadu_ps(self.bias.as_ptr().add(o0)), _mm256_loadu_ps(self.bias.as_ptr().add(o0 + 8))];
                for i0 in (0..full).step_by(R) {
                    let mut acc = [[b[0], b[1]]; R];
                    for k in 0..kd {
                        let w = self.wt.as_ptr().add(k * m + o0);
                        let (w0, w1) = (_mm256_loadu_ps(w), _mm256_loadu_ps(w.add(8)));
                        for (r, a) in acc.iter_mut().enumerate() {
                            let xv = _mm256_set1_ps(*x.get_unchecked((i0 + r) * kd + k));
                            a[0] = _mm256_fmadd_ps(xv, w0, a[0]);
                            a[1] = _mm256_fmadd_ps(xv, w1, a[1]);
                        }
                    }
                    for (r, a) in acc.iter().enumerate() {
                        let o = out.as_mut_ptr().add((i0 + r) * m + o0);
                        _mm256_storeu_ps(o, a[0]);
                        _mm256_storeu_ps(o.add(8), a[1]);
                    }
                }
            }
        }
        if full < n {
            let mut tail = vec![0.0; (n - full) * m];
            self.rows_into(&x[full * kd..n * kd], n - full, &mut tail);
            out[full * m..n * m].copy_from_slice(&tail);
        }
    }

    /// Tiles of `TILE` outputs by blocks of `ROWS` rows, accumulated in
    /// fixed arrays the compiler keeps in vector registers.
    #[inline(always)]
    fn rows_into(&self, x: &[f32], n: usize, out: &mut [f32]) {
        let (m, k_dim) = (self.out_dim, self.in_dim);
        for o0 in (0..m).step_by(TILE) {
            let width = TILE.min(m - o0);
            for i0 in (0..n).step_by(ROWS) {
                let rows = ROWS.min(n - i0);
                let mut acc = [[0.0f32; TILE]; ROWS];
                for (r, a) in acc.iter_mut().enumerate().take(rows) {
                    let _ = r;
                    a[..width].copy_from_slice(&self.bias[o0..o0 + width]);
                }
                for k in 0..k_dim {
                    let w = &self.wt[k * m + o0..k * m + o0 + width];
                    for (r, a) in acc.iter_mut().enumerate().take(rows) {
                        let xv = x[(i0 + r) * k_dim + k];
                        for (acc, &wv) in a[..width].iter_mut().zip(w) {
                            *acc = xv.mul_add(wv, *acc);
                        }
                    }
                }
                for (r, a) in acc.iter().enumerate().take(rows) {
                    out[(i0 + r) * m + o0..(i0 + r) * m + o0 + width].copy_from_slice(&a[..width]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tiled AVX-512 and AVX2 paths, where the processor runs them, and
    /// the portable loops agree, on widths and row counts that leave partial
    /// tiles and blocks.
    #[test]
    fn packed_rows_agree_with_one_row_at_a_time() {
        let mut state = 9u64;
        let mut next = || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        for (inp, out_dim, n) in [(48, 64, 19), (40, 96, 8), (33, 32, 3), (16, 40, 11)] {
            let w: Vec<f32> = (0..inp * out_dim).map(|_| next()).collect();
            let b: Vec<f32> = (0..out_dim).map(|_| next()).collect();
            let l = Linear::new(inp, out_dim, w, b).unwrap();
            let x: Vec<f32> = (0..n * inp).map(|_| next()).collect();
            let p = Packed::new(&l);
            let fast = p.rows(&x, n);
            let mut slow = vec![0.0; n * out_dim];
            p.rows_into(&x, n, &mut slow);
            let mut avx2 = slow.clone();
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                // SAFETY: the processor has AVX2 and FMA.
                unsafe { p.rows_avx2(&x, n, &mut avx2) };
            }
            for i in 0..n {
                let mut one = vec![0.0; out_dim];
                apply(&l, &x[i * inp..(i + 1) * inp], &mut one);
                for o in 0..out_dim {
                    let (f, s, e) = (fast[i * out_dim + o], slow[i * out_dim + o], one[o]);
                    assert!(
                        (f - e).abs() < 1e-4 && (s - e).abs() < 1e-4,
                        "{inp}x{out_dim} row {i} out {o}: {f} {s} {e}"
                    );
                }
            }
        }
    }
}
