//! LayerNorm over the rows of [rows, width], forward and back, with the
//! core's numerics: the mean and the biased variance of each row summed
//! in f64, y = (x - mean) / sqrt(var + eps) * w + b. A norm without a
//! bias (ModernBERT's) passes an empty `b`.

use crate::real::Real;

/// Each row's mean and 1 / sqrt(var + eps), kept for backward.
#[derive(Debug, Default, Clone)]
pub struct Cache {
    pub mean: Vec<f64>,
    pub rstd: Vec<f64>,
}

/// y [rows, width] from x, over `w` (the width) and `b` (the width, or
/// empty for no bias).
pub fn forward<R: Real>(x: &[R], w: &[R], b: &[R], eps: f64, y: &mut [R], c: &mut Cache) {
    let width = w.len();
    debug_assert!(b.is_empty() || b.len() == width);
    debug_assert_eq!(x.len(), y.len());
    let rows = x.len() / width;
    c.mean.clear();
    c.rstd.clear();
    for (xr, yr) in x.chunks_exact(width).zip(y.chunks_exact_mut(width)) {
        let mean = xr.iter().map(|v| v.to_f64()).sum::<f64>() / width as f64;
        let var = xr.iter().map(|v| (v.to_f64() - mean).powi(2)).sum::<f64>() / width as f64;
        let rstd = 1.0 / (var + eps).sqrt();
        c.mean.push(mean);
        c.rstd.push(rstd);
        for (i, (y, &x)) in yr.iter_mut().zip(xr).enumerate() {
            let xhat = R::from_f64((x.to_f64() - mean) * rstd);
            *y = xhat * w[i] + if b.is_empty() { R::ZERO } else { b[i] };
        }
    }
    debug_assert_eq!(c.mean.len(), rows);
}

/// Adds the gradients of `forward` from `dy`: to `dx`, to `dw` and `db`
/// when given (`db` is ignored for a norm without a bias).
pub fn backward<R: Real>(
    x: &[R],
    w: &[R],
    c: &Cache,
    dy: &[R],
    dx: &mut [R],
    mut dw: Option<&mut [R]>,
    mut db: Option<&mut [R]>,
) {
    let width = w.len();
    let n = width as f64;
    for (r, ((xr, dyr), dxr)) in
        x.chunks_exact(width).zip(dy.chunks_exact(width)).zip(dx.chunks_exact_mut(width)).enumerate()
    {
        let (mean, rstd) = (c.mean[r], c.rstd[r]);
        let xhat = |i: usize| (xr[i].to_f64() - mean) * rstd;
        // g is the gradient at the normalized row; its mean and its
        // projection on the row take the gradient through the statistics.
        let g = |i: usize| dyr[i].to_f64() * w[i].to_f64();
        let (mut m1, mut m2) = (0.0, 0.0);
        for i in 0..width {
            m1 += g(i);
            m2 += g(i) * xhat(i);
        }
        (m1, m2) = (m1 / n, m2 / n);
        for (i, d) in dxr.iter_mut().enumerate() {
            *d += R::from_f64(rstd * (g(i) - m1 - xhat(i) * m2));
        }
        if let Some(dw) = dw.as_deref_mut() {
            for (i, d) in dw.iter_mut().enumerate() {
                *d += R::from_f64(dyr[i].to_f64() * xhat(i));
            }
        }
        if let Some(db) = db.as_deref_mut() {
            for (d, &g) in db.iter_mut().zip(dyr) {
                *d += g;
            }
        }
    }
}
