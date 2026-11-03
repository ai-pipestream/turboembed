//! The three products of a projection without a bias, y = x W^T with W
//! [out, in] as the core stores every projection, and its two gradients.
//! Every sum runs in a fixed order, so a result never depends on how the
//! work was split.

use crate::real::{Real, dot};

/// y [rows, out] = x [rows, inp] W^T.
pub fn forward<R: Real>(x: &[R], w: &[R], inp: usize, out: usize, y: &mut [R]) {
    let rows = x.len() / inp;
    debug_assert_eq!((w.len(), y.len()), (out * inp, rows * out));
    for (xr, yr) in x.chunks_exact(inp).zip(y.chunks_exact_mut(out)) {
        for (wo, yo) in w.chunks_exact(inp).zip(yr) {
            *yo = dot(wo, xr);
        }
    }
    debug_assert_eq!(rows * inp, x.len());
}

/// dw [out, inp] += dy [rows, out]^T x [rows, inp], rows in order.
pub fn grad_w<R: Real>(x: &[R], dy: &[R], inp: usize, out: usize, dw: &mut [R]) {
    debug_assert_eq!(dw.len(), out * inp);
    for (xr, dyr) in x.chunks_exact(inp).zip(dy.chunks_exact(out)) {
        for (&g, dwo) in dyr.iter().zip(dw.chunks_exact_mut(inp)) {
            for (d, &xv) in dwo.iter_mut().zip(xr) {
                *d += g * xv;
            }
        }
    }
}

/// dx [rows, inp] += dy [rows, out] W.
pub fn grad_x<R: Real>(w: &[R], dy: &[R], inp: usize, out: usize, dx: &mut [R]) {
    debug_assert_eq!(w.len(), out * inp);
    for (dyr, dxr) in dy.chunks_exact(out).zip(dx.chunks_exact_mut(inp)) {
        for (&g, wo) in dyr.iter().zip(w.chunks_exact(inp)) {
            for (d, &wv) in dxr.iter_mut().zip(wo) {
                *d += g * wv;
            }
        }
    }
}
