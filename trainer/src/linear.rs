//! A linear layer over rows: y = x W^T + b, W [out, in] row-major as the
//! core's `Linear` holds it.

use crate::real::{Real, dot};

/// y [rows, out] from x [rows, inp].
pub fn forward<R: Real>(w: &[R], b: &[R], x: &[R], inp: usize, out: usize, y: &mut [R]) {
    let rows = x.len() / inp;
    debug_assert_eq!((w.len(), b.len(), y.len()), (out * inp, out, rows * out));
    for r in 0..rows {
        let xr = &x[r * inp..][..inp];
        for (o, yo) in y[r * out..][..out].iter_mut().enumerate() {
            *yo = b[o] + dot(&w[o * inp..][..inp], xr);
        }
    }
}

/// Adds the gradients of `forward` to `dw` and `db`, and to `dx` when
/// given, from `dy` [rows, out]. Rows are summed in order.
#[allow(clippy::too_many_arguments)]
pub fn backward<R: Real>(
    w: &[R],
    x: &[R],
    dy: &[R],
    inp: usize,
    out: usize,
    dw: &mut [R],
    db: &mut [R],
    mut dx: Option<&mut [R]>,
) {
    let rows = x.len() / inp;
    for r in 0..rows {
        let xr = &x[r * inp..][..inp];
        for o in 0..out {
            let g = dy[r * out + o];
            db[o] += g;
            for (d, xv) in dw[o * inp..][..inp].iter_mut().zip(xr) {
                *d += g * *xv;
            }
            if let Some(dx) = dx.as_deref_mut() {
                for (d, wv) in dx[r * inp..][..inp].iter_mut().zip(&w[o * inp..][..inp]) {
                    *d += g * *wv;
                }
            }
        }
    }
}
