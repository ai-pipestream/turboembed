//! `L2R_LR`, L2-regularized logistic regression in the primal: liblinear's
//! Newton method with a preconditioned conjugate gradient and a line
//! search (`Newton`), on the objective of `L2R_LrFunction` and
//! `L2R_ErmFunction`, with the vector routines of `Blas` it calls.
//!
//! The add-on never turns off the regularization of the bias
//! (`regularize_bias` stays true) and gives no initial solution, so those
//! branches are left out.

use std::cmp::Ordering;

use super::problem::{Rows, axpy, dot, java_min};
use super::{exp, log};

/// `Blas.dnrm2_`: the Euclidean norm, scaled to avoid overflow, summed
/// from the last element to the first as the Fortran routine does.
pub(crate) fn dnrm2(x: &[f64]) -> f64 {
    match x.len() {
        0 => 0.0,
        1 => x[0].abs(),
        _ => {
            let mut scale = 0.0f64;
            let mut ssq = 1.0f64;
            for &xi in x.iter().rev() {
                if xi != 0.0 {
                    let absxi = xi.abs();
                    if scale < absxi {
                        let temp = scale / absxi;
                        ssq = ssq * (temp * temp) + 1.0;
                        scale = absxi;
                    } else {
                        let temp = absxi / scale;
                        ssq += temp * temp;
                    }
                }
            }
            scale * ssq.sqrt()
        }
    }
}

/// `Blas.ddot_` with unit strides: five products at a time, added left to
/// right and then to the running sum, then the rest one by one.
pub(crate) fn ddot(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let y = &y[..n];
    let mut stemp = 0.0;
    let (fives, rest) = x.as_chunks::<5>();
    let (y5, yrest) = y.split_at(fives.len() * 5);
    let (y5, _) = y5.as_chunks::<5>();
    for (a, b) in fives.iter().zip(y5) {
        stemp += a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] + a[4] * b[4];
    }
    for (a, b) in rest.iter().zip(yrest) {
        stemp += a * b;
    }
    stemp
}

/// `Blas.daxpy_`: `y += a * x`, nothing when `a` is zero.
fn daxpy(a: f64, x: &[f64], y: &mut [f64]) {
    if a != 0.0 {
        for (yi, &xi) in y.iter_mut().zip(x) {
            *yi += a * xi;
        }
    }
}

/// `L2R_LrFunction`: `w'w / 2 + sum C_i log(1 + exp(-y_i w'x_i))`, with
/// the per instance state `L2R_ErmFunction` keeps between calls: `wx`
/// (the products of the last point evaluated), `tmp` and `D`.
pub(crate) struct LrFunction<'p> {
    x: &'p Rows,
    y: &'p [f64],
    c: &'p [f64],
    w_size: usize,
    wx: Vec<f64>,
    tmp: Vec<f64>,
    d: Vec<f64>,
    w_tw: f64,
}

impl<'p> LrFunction<'p> {
    pub(crate) fn new(x: &'p Rows, y: &'p [f64], c: &'p [f64], w_size: usize) -> LrFunction<'p> {
        let l = x.len();
        LrFunction { x, y, c, w_size, wx: vec![0.0; l], tmp: vec![0.0; l], d: vec![0.0; l], w_tw: 0.0 }
    }

    /// `C_times_loss`, in the form that does not overflow for either sign.
    #[inline]
    fn c_times_loss(&self, i: usize, wx_i: f64) -> f64 {
        let ywx_i = wx_i * self.y[i];
        if ywx_i >= 0.0 { self.c[i] * log(1.0 + exp(-ywx_i)) } else { self.c[i] * (-ywx_i + log(1.0 + exp(ywx_i))) }
    }

    /// `fun`: the objective at `w`, keeping `w'x_i` and `w'w`.
    pub(crate) fn fun(&mut self, w: &[f64]) -> f64 {
        for i in 0..self.x.len() {
            self.wx[i] = dot(w, self.x.row(i));
        }
        let mut w_tw = 0.0;
        for &wi in &w[..self.w_size] {
            w_tw += wi * wi;
        }
        self.w_tw = w_tw;
        let mut f = 0.0;
        for i in 0..self.x.len() {
            f += self.c_times_loss(i, self.wx[i]);
        }
        f + 0.5 * self.w_tw
    }

    /// `grad`: the gradient at the point `fun` or the line search last
    /// saw, keeping `D`, the diagonal of the Hessian's middle factor.
    pub(crate) fn grad(&mut self, w: &[f64], g: &mut [f64]) {
        for i in 0..self.x.len() {
            let t = 1.0 / (1.0 + exp(-self.y[i] * self.wx[i]));
            self.d[i] = t * (1.0 - t);
            self.tmp[i] = self.c[i] * (t - 1.0) * self.y[i];
        }
        g[..self.w_size].fill(0.0);
        for i in 0..self.x.len() {
            axpy(self.tmp[i], self.x.row(i), g);
        }
        for i in 0..self.w_size {
            g[i] += w[i];
        }
    }

    /// `get_diag_preconditioner`: one plus the diagonal of `X' C D X`.
    fn diag_preconditioner(&self, m: &mut [f64]) {
        m[..self.w_size].fill(1.0);
        for i in 0..self.x.len() {
            let (index, value) = self.x.row(i);
            for (&j, &v) in index.iter().zip(value) {
                m[j as usize] += v * v * self.c[i] * self.d[i];
            }
        }
    }

    /// `Hv`: the Hessian times `s`.
    fn hv(&self, s: &[f64], hs: &mut [f64]) {
        hs[..self.w_size].fill(0.0);
        for i in 0..self.x.len() {
            let row = self.x.row(i);
            let x_ts = self.c[i] * self.d[i] * dot(s, row);
            axpy(x_ts, row, hs);
        }
        for i in 0..self.w_size {
            hs[i] += s[i];
        }
    }

    /// `linesearch_and_update`: halves the step from `alpha` until the
    /// objective decreases enough, at most twenty times, then moves `w`.
    /// `f` holds the objective at `w` on entry and at the new point on
    /// exit. Returns the step taken, zero when none was found (`w` and `f`
    /// are then left as they were).
    fn linesearch_and_update(&mut self, w: &mut [f64], s: &[f64], f: &mut f64, g: &[f64], mut alpha: f64) -> f64 {
        const ETA: f64 = 0.01;
        const MAX_NUM_LINESEARCH: usize = 20;
        let fold = *f;
        for i in 0..self.x.len() {
            self.tmp[i] = dot(s, self.x.row(i));
        }
        let (mut s_ts, mut w_ts, mut g_ts) = (0.0, 0.0, 0.0);
        for i in 0..self.w_size {
            s_ts += s[i] * s[i];
            w_ts += s[i] * w[i];
            g_ts += s[i] * g[i];
        }
        let mut found = false;
        for _ in 0..MAX_NUM_LINESEARCH {
            let mut loss = 0.0;
            for i in 0..self.x.len() {
                let inner_product = self.tmp[i] * alpha + self.wx[i];
                loss += self.c_times_loss(i, inner_product);
            }
            *f = loss + (alpha * alpha * s_ts + self.w_tw) / 2.0 + alpha * w_ts;
            if *f - fold <= ETA * alpha * g_ts {
                for i in 0..self.x.len() {
                    self.wx[i] += alpha * self.tmp[i];
                }
                found = true;
                break;
            }
            alpha *= 0.5;
        }
        if !found {
            *f = fold;
            return 0.0;
        }
        for i in 0..self.w_size {
            w[i] += alpha * s[i];
        }
        self.w_tw += alpha * alpha * s_ts + 2.0 * alpha * w_ts;
        alpha
    }
}

/// `Newton.newton`: minimizes the function from `w`, in place. Stops when
/// the gradient norm falls under `eps` times its norm at zero, after
/// `max_iter` iterations, when the line search fails, or when the
/// objective stops changing.
pub(crate) fn newton(fun: &mut LrFunction<'_>, w: &mut [f64], eps: f64, max_iter: usize) {
    const ALPHA_PCG: f64 = 0.01;
    const EPS_CG: f64 = 0.5;
    let n = fun.w_size;
    let mut s = vec![0.0; n];
    let mut r = vec![0.0; n];
    let mut g = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut scratch = Pcg::new(n);

    // The gradient norm at w = 0, for the stopping condition.
    let w0 = vec![0.0; n];
    fun.fun(&w0);
    fun.grad(&w0, &mut g);
    let gnorm0 = dnrm2(&g);

    let mut f = fun.fun(w);
    fun.grad(w, &mut g);
    let gnorm = dnrm2(&g);
    // liblinear clears `search` when gnorm <= eps * gnorm0, which a NaN
    // norm does not satisfy, so a NaN norm searches.
    let search = !matches!(gnorm.partial_cmp(&(eps * gnorm0)), Some(Ordering::Less | Ordering::Equal));
    let mut iter = 1;
    while iter <= max_iter && search {
        fun.diag_preconditioner(&mut m);
        for mi in &mut m {
            *mi = (1.0 - ALPHA_PCG) + ALPHA_PCG * *mi;
        }
        scratch.solve(fun, &g, &m, &mut s, &mut r, EPS_CG);
        let fold = f;
        let step = fun.linesearch_and_update(w, &s, &mut f, &g, 1.0);
        if step == 0.0 {
            // The line search failed.
            break;
        }
        fun.grad(w, &mut g);
        let gnorm = dnrm2(&g);
        if gnorm <= eps * gnorm0 || f < -1.0e+32 {
            break;
        }
        let actred = fold - f;
        if actred.abs() <= 1.0e-12 * f.abs() {
            break;
        }
        iter += 1;
    }
}

/// The work arrays of `Newton.pcg`, allocated once per Newton run.
struct Pcg {
    d: Vec<f64>,
    hd: Vec<f64>,
    z: Vec<f64>,
}

impl Pcg {
    fn new(n: usize) -> Pcg {
        Pcg { d: vec![0.0; n], hd: vec![0.0; n], z: vec![0.0; n] }
    }

    /// `Newton.pcg`: the preconditioned conjugate gradient for `H s = -g`,
    /// stopped by the quadratic approximation's decrease, `max(n, 5)`
    /// steps, or a direction of no curvature.
    fn solve(&mut self, fun: &LrFunction<'_>, g: &[f64], m: &[f64], s: &mut [f64], r: &mut [f64], eps_cg: f64) {
        let n = g.len();
        let Pcg { d, hd, z } = self;
        for i in 0..n {
            s[i] = 0.0;
            r[i] = -g[i];
            z[i] = r[i] / m[i];
            d[i] = z[i];
        }
        let mut z_tr = ddot(z, r);
        let g_minv_norm = z_tr.sqrt();
        let cgtol = java_min(eps_cg, g_minv_norm.sqrt());
        let max_cg_iter = n.max(5);
        let mut q = 0.0;
        let mut cg_iter = 0;
        while cg_iter < max_cg_iter {
            cg_iter += 1;
            fun.hv(d, hd);
            let d_hd = ddot(d, hd);
            // Avoids 0/0 in alpha.
            if d_hd <= 1.0e-16 {
                break;
            }
            let alpha = z_tr / d_hd;
            daxpy(alpha, d, s);
            daxpy(-alpha, hd, r);
            // The quadratic approximation as the stopping criterion.
            let new_q = -0.5 * (ddot(s, r) - ddot(s, g));
            let q_diff = new_q - q;
            if new_q <= 0.0 && q_diff <= 0.0 {
                if cg_iter as f64 * q_diff >= cgtol * new_q {
                    break;
                }
            } else {
                break;
            }
            q = new_q;
            for i in 0..n {
                z[i] = r[i] / m[i];
            }
            let znew_trnew = ddot(z, r);
            let beta = znew_trnew / z_tr;
            for di in d.iter_mut() {
                *di *= beta;
            }
            daxpy(1.0, z, d);
            z_tr = znew_trnew;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ddot_adds_five_products_before_the_running_sum() {
        // With the products 1e16, 1, -1e16, 1, 1 added left to right the
        // ones are lost, then the last three join the sum one by one.
        let x = [1e16, 1.0, -1e16, 1.0, 1.0, 1.0, 1.0, 1.0];
        let y = [1.0; 8];
        let mut naive = 0.0;
        for (a, b) in x.iter().zip(&y) {
            naive += a * b;
        }
        assert_eq!(ddot(&x, &y), (((1e16 + 1.0) + -1e16) + 1.0 + 1.0) + 1.0 + 1.0 + 1.0);
        assert_eq!(ddot(&x, &y), naive);
        assert_eq!(ddot(&[], &[]), 0.0);
    }

    #[test]
    fn dnrm2_scales_from_the_last_element() {
        assert_eq!(dnrm2(&[3.0, 4.0]), 5.0);
        assert_eq!(dnrm2(&[-2.0]), 2.0);
        assert_eq!(dnrm2(&[0.0, 0.0]), 0.0);
        assert_eq!(dnrm2(&[1e300, 1e300]), 1e300 * 2f64.sqrt());
    }

    #[test]
    fn daxpy_with_zero_leaves_nan_alone() {
        let mut y = [f64::NAN, 1.0];
        daxpy(0.0, &[f64::INFINITY, 1.0], &mut y);
        assert!(y[0].is_nan());
        daxpy(2.0, &[1.0, 1.0], &mut y);
        assert_eq!(y[1], 3.0);
    }
}
