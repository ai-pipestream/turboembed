//! Low-rank adapters (Hu et al. 2021, arXiv 2106.09685) on a frozen
//! projection: y = x W^T + s (x A^T) B^T, with A [rank, in] seeded and B
//! [out, rank] zero, so a fresh adapter changes nothing, and s = alpha /
//! rank. Only A and B take gradients.

use crate::real::Real;
use crate::rng::Rng;

use super::matmul;

/// A seeded A, uniform in +-1 / sqrt(in) as the base weights are
/// (Kaiming's bound), and a zero B.
pub fn init<R: Real>(inp: usize, out: usize, rank: usize, rng: &mut Rng) -> (Vec<R>, Vec<R>) {
    let limit = 1.0 / (inp as f64).sqrt();
    ((0..rank * inp).map(|_| R::from_f64(rng.uniform(limit))).collect(), vec![R::ZERO; out * rank])
}

/// y [rows, out] += s (x A^T) B^T; h [rows, rank] keeps x A^T for
/// backward, rounded to bfloat16 before B^T when `round` is set.
#[allow(clippy::too_many_arguments)]
pub fn forward<R: Real>(
    x: &[R],
    a: &[R],
    b: &[R],
    inp: usize,
    rank: usize,
    s: R,
    round: bool,
    y: &mut [R],
    h: &mut Vec<R>,
) {
    let rows = x.len() / inp;
    let out = y.len() / rows.max(1);
    h.clear();
    h.resize(rows * rank, R::ZERO);
    matmul::forward(x, a, inp, rank, h);
    if round {
        h.iter_mut().for_each(|v| *v = v.bf16());
    }
    let mut low = vec![R::ZERO; rows * out];
    matmul::forward(h, b, rank, out, &mut low);
    for (y, l) in y.iter_mut().zip(&low) {
        *y += s * *l;
    }
}

/// The gradients of `forward` from dy [rows, out], added to da, db and,
/// when given, dx.
#[allow(clippy::too_many_arguments)]
pub fn backward<R: Real>(
    x: &[R],
    a: &[R],
    b: &[R],
    h: &[R],
    inp: usize,
    rank: usize,
    s: R,
    dy: &[R],
    da: &mut [R],
    db: &mut [R],
    dx: Option<&mut [R]>,
) {
    let rows = x.len() / inp;
    let out = dy.len() / rows.max(1);
    // dh = s dy B, [rows, rank].
    let mut dh = vec![R::ZERO; rows * rank];
    matmul::grad_x(b, dy, rank, out, &mut dh);
    for v in dh.iter_mut() {
        *v *= s;
    }
    // dB += s dy^T h: the s is in dh's place, so scale h's product instead.
    let mut dbs = vec![R::ZERO; out * rank];
    matmul::grad_w(h, dy, rank, out, &mut dbs);
    for (d, v) in db.iter_mut().zip(dbs) {
        *d += s * v;
    }
    matmul::grad_w(x, &dh, inp, rank, da);
    if let Some(dx) = dx {
        matmul::grad_x(a, &dh, inp, rank, dx);
    }
}
