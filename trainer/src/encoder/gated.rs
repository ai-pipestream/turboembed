//! The gated MLP's middle (GeGLU): from the first projection's output
//! [rows, 2 inter], the first `inter` columns through GELU times the last
//! `inter`, as the core's `TURBO_MODERNBERT_FFN_IN_WEIGHT` orders them
//! (the activated outputs first).

use super::math::{gelu, gelu_grad};
use crate::real::Real;

/// act [rows, inter] = gelu(pre[.., ..inter]) * pre[.., inter..].
pub fn forward<R: Real>(pre: &[R], inter: usize, act: &mut [R]) {
    for (p, a) in pre.chunks_exact(2 * inter).zip(act.chunks_exact_mut(inter)) {
        let (gate, up) = p.split_at(inter);
        for ((a, &g), &u) in a.iter_mut().zip(gate).zip(up) {
            *a = gelu(g) * u;
        }
    }
}

/// dpre [rows, 2 inter] += the gradients of `forward` from dact.
pub fn backward<R: Real>(pre: &[R], inter: usize, dact: &[R], dpre: &mut [R]) {
    for ((p, da), dp) in pre.chunks_exact(2 * inter).zip(dact.chunks_exact(inter)).zip(dpre.chunks_exact_mut(2 * inter))
    {
        let (gate, up) = p.split_at(inter);
        let (dgate, dup) = dp.split_at_mut(inter);
        for i in 0..inter {
            dgate[i] += da[i] * up[i] * gelu_grad(gate[i]);
            dup[i] += da[i] * gelu(gate[i]);
        }
    }
}
