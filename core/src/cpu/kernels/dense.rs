//! The matrix product for callers outside the BERT encoder (the native
//! DeBERTa encoder in crate::deberta): a right-hand matrix packed in the
//! micro-kernel's panels once, then applied to any number of rows on the
//! calling thread. As everywhere in kernels/, each output is one chain of
//! multiply-adds over k in order, so a row's result does not depend on
//! how many other rows are in the call. `Matrix` and `Mode` are public
//! (turbo::cpu) so that model heads outside this crate (turbo-gliner) use
//! the same packed kernels.

use super::gemm::{Epilogue, Gemm};
#[cfg(target_arch = "x86_64")]
use super::{Avx2, Avx512};
use super::{Isa, Micro, Panels, Portable, rowwise::exp};

/// What a product does with each output once its sum is complete.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// y = sum + b
    Plain,
    /// y = gelu(sum + b), GELU by erf.
    Gelu,
    /// y = y + (sum + b)
    Residual,
}

/// W [n, k] (y = x W^T) packed for the processor's widest kernel.
pub struct Matrix {
    panels: Panels,
    isa: Isa,
    /// n zeros, the bias of a product that has none.
    zero: Vec<f32>,
}

impl Matrix {
    /// No columns yet: `pack_rows` or `pack_cols` fills it, reusing its
    /// memory from one fill to the next.
    pub fn empty() -> Matrix {
        let isa = Isa::detect();
        let (mr, nr) = isa.tile();
        Matrix { panels: Panels { buf: Vec::new(), off: 0, n: 0, k: 0, mr, nr }, isa, zero: Vec::new() }
    }

    /// The rows of `parts`, each [n_i, k] row-major, one after another.
    /// Err is the bytes it could not allocate.
    ///
    /// # Panics
    /// If k is 0 or a part's length is not a multiple of k.
    pub fn from_rows(parts: &[&[f32]], k: usize) -> Result<Matrix, usize> {
        assert!(k > 0 && parts.iter().all(|p| p.len() % k == 0), "dense product: parts are not rows of k");
        let isa = Isa::detect();
        let panels = Panels::pack(parts, k, isa)?;
        let mut zero = Vec::new();
        zero.try_reserve_exact(panels.n).map_err(|_| panels.n.saturating_mul(4))?;
        zero.resize(panels.n, 0.0);
        Ok(Matrix { panels, isa, zero })
    }

    /// W[o][kk] = src[o * ld + kk] for o under n and kk under k.
    pub fn pack_rows(&mut self, src: &[f32], ld: usize, n: usize, k: usize) -> Result<(), usize> {
        if n > 0 && extent(n, ld, k).is_none_or(|e| src.len() < e) {
            return Err(usize::MAX);
        }
        self.fill(n, k, |o, kk| src[o * ld + kk])
    }

    /// W[o][kk] = src[kk * ld + o] for o under n and kk under k: the
    /// transpose of a row-major [k, n] block.
    pub fn pack_cols(&mut self, src: &[f32], ld: usize, n: usize, k: usize) -> Result<(), usize> {
        if k > 0 && extent(k, ld, n).is_none_or(|e| src.len() < e) {
            return Err(usize::MAX);
        }
        self.fill(n, k, |o, kk| src[kk * ld + o])
    }

    fn fill(&mut self, n: usize, k: usize, at: impl Fn(usize, usize) -> f32) -> Result<(), usize> {
        let nr = self.panels.nr;
        let panels = n.div_ceil(nr);
        let floats = panels.checked_mul(k).and_then(|f| f.checked_mul(nr)).ok_or(usize::MAX)?;
        for (v, len) in [(&mut self.panels.buf, floats), (&mut self.zero, n)] {
            if v.len() < len {
                v.try_reserve_exact(len - v.len()).map_err(|_| len.saturating_mul(4))?;
                v.resize(len, 0.0);
            }
        }
        let buf = &mut self.panels.buf;
        for p in 0..panels {
            let cols = (n - p * nr).min(nr);
            for kk in 0..k {
                let row = &mut buf[(p * k + kk) * nr..][..nr];
                for (j, w) in row.iter_mut().enumerate() {
                    *w = if j < cols { at(p * nr + j, kk) } else { 0.0 };
                }
            }
        }
        self.panels.off = 0;
        self.panels.n = n;
        self.panels.k = k;
        Ok(())
    }

    /// y[r * ldy + o] = mode(sum over kk of x[r * ldx + kk] W[o][kk],
    /// bias[o] or 0) for r under rows and o under n.
    ///
    /// # Panics
    /// If ldx < k, ldy < n, or x, y or bias is shorter than the shapes
    /// need (an extent that overflows usize is too long for any slice).
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &self,
        x: &[f32],
        ldx: usize,
        rows: usize,
        bias: Option<&[f32]>,
        y: &mut [f32],
        ldy: usize,
        mode: Mode,
    ) {
        let (n, k) = (self.panels.n, self.panels.k);
        if rows == 0 || n == 0 {
            return;
        }
        assert!(ldx >= k, "dense product: ldx {ldx} under k {k}");
        assert!(ldy >= n, "dense product: ldy {ldy} under n {n}");
        assert!(extent(rows, ldx, k).is_some_and(|e| x.len() >= e), "dense product: x too short");
        assert!(extent(rows, ldy, n).is_some_and(|e| y.len() >= e), "dense product: y too short");
        let bias = bias.unwrap_or(&self.zero);
        assert!(bias.len() >= n, "dense product: bias too short");
        let epilogue = match mode {
            // A split over no columns is the plain sum and bias.
            Mode::Plain => {
                Epilogue::Split { lo: 0, hi: 0, t_out: std::ptr::null_mut(), stride: 0, slot: std::ptr::null() }
            }
            Mode::Gelu => Epilogue::Gelu,
            Mode::Residual => Epilogue::Residual,
        };
        let g = Gemm { x: x.as_ptr(), ldx, rows, w: &self.panels, bias, y: y.as_mut_ptr(), ldy, epilogue };
        // SAFETY: the shapes are checked above, the panels were packed for
        // self.isa, which this processor has (Isa::detect), and the one
        // thread runs every task in turn.
        unsafe {
            match self.isa {
                Isa::Portable => all::<Portable>(&g),
                #[cfg(target_arch = "x86_64")]
                Isa::Avx2 => avx2(&g),
                #[cfg(target_arch = "x86_64")]
                Isa::Avx512 => avx512(&g),
            }
        }
    }
}

/// Floats from the first of `rows` rows `ld` apart to the end of the last
/// one's `width`, or None if that overflows. rows is at least 1.
fn extent(rows: usize, ld: usize, width: usize) -> Option<usize> {
    (rows - 1).checked_mul(ld)?.checked_add(width)
}

#[inline(always)]
unsafe fn all<M: Micro>(g: &Gemm) {
    for t in 0..g.tasks() {
        // SAFETY: as apply.
        unsafe { g.task::<M>(t) }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2(g: &Gemm) {
    // SAFETY: as apply.
    unsafe { all::<Avx2>(g) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx2,fma")]
unsafe fn avx512(g: &Gemm) {
    // SAFETY: as apply.
    unsafe { all::<Avx512>(g) }
}

/// The row's softmax in place, in F32: the max, then e^(v - max) with the
/// kernels' vectorizing exp, summed in 16 lanes added in a fixed tree.
pub(crate) fn softmax(row: &mut [f32]) {
    let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| if v > m { v } else { m });
    let mut lanes = [0.0f32; 16];
    for chunk in row.chunks_mut(16) {
        for (l, v) in lanes.iter_mut().zip(chunk) {
            *v = exp(*v - max);
            *l += *v;
        }
    }
    let inv = 1.0 / super::attend::tree(lanes);
    for v in row {
        *v *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_products_match_the_plain_sum_and_ignore_other_rows() {
        let (rows, n, k) = (11, 37, 23);
        let x = super::super::values(rows * k, 1, 1.0);
        let w = super::super::values(n * k, 2, 1.0);
        let b = super::super::values(n, 3, 1.0);
        let by_rows = Matrix::from_rows(&[&w], k).unwrap();
        let mut by_cols = Matrix::empty();
        let wt: Vec<f32> = (0..k * n).map(|i| w[(i % n) * k + i / n]).collect();
        by_cols.pack_cols(&wt, n, n, k).unwrap();
        for m in [&by_rows, &by_cols] {
            let mut y = vec![0.0; rows * n];
            m.apply(&x, k, rows, Some(&b), &mut y, n, Mode::Plain);
            for r in 0..rows {
                for o in 0..n {
                    let want: f64 =
                        b[o] as f64 + (0..k).map(|i| x[r * k + i] as f64 * w[o * k + i] as f64).sum::<f64>();
                    assert!((y[r * n + o] as f64 - want).abs() < 1e-4, "[{r}, {o}]");
                }
            }
            // One row alone gives the same bits as inside the batch.
            let mut one = vec![0.0; n];
            m.apply(&x[5 * k..], k, 1, Some(&b), &mut one, n, Mode::Plain);
            assert_eq!(one, y[5 * n..6 * n]);
        }
        let mut p = vec![1.0, 2.0, f32::NEG_INFINITY, 3.0];
        softmax(&mut p);
        assert_eq!(p[2], 0.0);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn no_rows_is_a_no_op() {
        let m = Matrix::from_rows(&[&[1.0]], 1).unwrap();
        let mut y = [7.0; 2];
        m.apply(&[], usize::MAX, 0, None, &mut y, usize::MAX, Mode::Plain);
        assert_eq!(y, [7.0; 2]);
    }

    #[test]
    #[should_panic(expected = "x too short")]
    fn an_input_extent_that_overflows_panics() {
        let m = Matrix::from_rows(&[&[1.0]], 1).unwrap();
        let mut y = [0.0; 2];
        m.apply(&[], usize::MAX, 2, None, &mut y, 1, Mode::Plain);
    }

    #[test]
    #[should_panic(expected = "y too short")]
    fn an_output_extent_that_overflows_panics() {
        let m = Matrix::from_rows(&[&[1.0]], 1).unwrap();
        m.apply(&[1.0, 2.0], 1, 2, None, &mut [], usize::MAX, Mode::Plain);
    }

    #[test]
    #[should_panic(expected = "ldx")]
    fn a_short_input_stride_panics() {
        let m = Matrix::from_rows(&[&[1.0, 2.0]], 2).unwrap();
        m.apply(&[1.0; 4], 1, 2, None, &mut [0.0; 2], 1, Mode::Plain);
    }

    #[test]
    #[should_panic(expected = "not rows of k")]
    fn ragged_parts_panic() {
        let _ = Matrix::from_rows(&[&[1.0, 2.0, 3.0]], 2);
    }

    #[test]
    fn packing_extents_that_overflow_are_refused() {
        let mut m = Matrix::empty();
        assert!(m.pack_rows(&[], usize::MAX, 2, 1).is_err());
        assert!(m.pack_cols(&[], usize::MAX, 1, 2).is_err());
        let mut y = [0.0; 2];
        m.pack_rows(&[1.0, 2.0], 1, 2, 1).unwrap();
        m.apply(&[3.0], 1, 1, None, &mut y, 2, Mode::Plain);
        assert_eq!(y, [3.0, 6.0]);
    }
}
