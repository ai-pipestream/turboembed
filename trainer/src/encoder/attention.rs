//! Softmax attention of one sequence's heads over its own tokens, forward
//! and backward. Query i reads the keys j with |i - j| <= window
//! (usize::MAX for the whole sequence), as the core's sliding-window
//! layers do, and scores are scaled by 1 / sqrt(d). qkv is [n, 3 hidden]:
//! Q, then K, then V, each head-major, with Q and K already rotated.
//!
//! Nothing is kept between forward and backward: backward recomputes
//! each query's probabilities from Q and K by the same operations in the
//! same order, so they are the bits forward used, and memory does not
//! grow with heads x n^2 a layer (the checkpointing of #1519 for the one
//! activation that would).

use crate::real::{Real, dot};

/// The shape of the last forward, and its row of probabilities.
#[derive(Debug, Default, Clone)]
pub struct Cache<R> {
    row: Vec<R>,
    n: usize,
}

/// The keys query `i` of `n` reads.
fn keys(i: usize, n: usize, window: usize) -> std::ops::Range<usize> {
    i.saturating_sub(window)..i.saturating_add(window).saturating_add(1).min(n)
}

/// The columns of head `h` in one part of a qkv row.
fn cols(part: usize, hidden: usize, h: usize, d: usize) -> std::ops::Range<usize> {
    part * hidden + h * d..part * hidden + (h + 1) * d
}

/// Query `i`'s probabilities over `span` into `row` (zero elsewhere),
/// from q and the keys in `kc` of each qkv row; `round` rounds them to
/// bfloat16, as a bfloat16 kernel hands them to the value product.
#[allow(clippy::too_many_arguments)]
fn probabilities<R: Real>(
    qkv: &[R],
    hidden: usize,
    q: &[R],
    kc: std::ops::Range<usize>,
    span: std::ops::Range<usize>,
    scale: R,
    round: bool,
    row: &mut [R],
) {
    row.fill(R::ZERO);
    let mut max = R::from_f64(f64::NEG_INFINITY);
    for j in span.clone() {
        let s = dot(q, &qkv[j * 3 * hidden..][kc.clone()]) * scale;
        row[j] = s;
        if s > max {
            max = s;
        }
    }
    let mut sum = R::ZERO;
    for j in span.clone() {
        row[j] = (row[j] - max).exp();
        sum += row[j];
    }
    let inv = R::ONE / sum;
    for j in span {
        row[j] = if round { (row[j] * inv).bf16() } else { row[j] * inv };
    }
}

/// ctx [n, hidden] from qkv [n, 3 hidden]; `round` rounds the
/// probabilities to bfloat16 before they weight the values.
#[allow(clippy::too_many_arguments)]
pub fn forward<R: Real>(
    qkv: &[R],
    hidden: usize,
    heads: usize,
    window: usize,
    round: bool,
    ctx: &mut [R],
    c: &mut Cache<R>,
) {
    let (n, d) = (qkv.len() / (3 * hidden), hidden / heads);
    let scale = R::from_f64(1.0 / (d as f64).sqrt());
    c.n = n;
    c.row.resize(n, R::ZERO);
    ctx.fill(R::ZERO);
    for h in 0..heads {
        let (qc, kc, vc) = (cols(0, hidden, h, d), cols(1, hidden, h, d), cols(2, hidden, h, d));
        for i in 0..n {
            let q = &qkv[i * 3 * hidden..][qc.clone()];
            let span = keys(i, n, window);
            probabilities(qkv, hidden, q, kc.clone(), span.clone(), scale, round, &mut c.row);
            let out = &mut ctx[i * hidden..][h * d..(h + 1) * d];
            for j in span {
                let p = c.row[j];
                for (o, &v) in out.iter_mut().zip(&qkv[j * 3 * hidden..][vc.clone()]) {
                    *o += p * v;
                }
            }
        }
    }
}

/// dqkv [n, 3 hidden] += the gradients of `forward` from dctx [n, hidden],
/// `round` as forward's.
#[allow(clippy::too_many_arguments)]
pub fn backward<R: Real>(
    qkv: &[R],
    hidden: usize,
    heads: usize,
    window: usize,
    round: bool,
    c: &Cache<R>,
    dctx: &[R],
    dqkv: &mut [R],
) {
    let (n, d) = (c.n, hidden / heads);
    debug_assert_eq!(qkv.len(), n * 3 * hidden);
    let scale = R::from_f64(1.0 / (d as f64).sqrt());
    let (mut row, mut dp, mut dq) = (vec![R::ZERO; n], vec![R::ZERO; n], vec![R::ZERO; d]);
    for h in 0..heads {
        let (qc, kc, vc) = (cols(0, hidden, h, d), cols(1, hidden, h, d), cols(2, hidden, h, d));
        for i in 0..n {
            let q = &qkv[i * 3 * hidden..][qc.clone()];
            let dout = &dctx[i * hidden..][h * d..(h + 1) * d];
            let span = keys(i, n, window);
            probabilities(qkv, hidden, q, kc.clone(), span.clone(), scale, round, &mut row);
            // dp_j = dctx_i . v_j, and dv_j += p_ij dctx_i.
            let mut s = R::ZERO;
            for j in span.clone() {
                dp[j] = dot(dout, &qkv[j * 3 * hidden..][vc.clone()]);
                s += row[j] * dp[j];
                for (dv, &g) in dqkv[j * 3 * hidden..][vc.clone()].iter_mut().zip(dout) {
                    *dv += row[j] * g;
                }
            }
            // ds_j = p_ij (dp_j - sum_j' p_ij' dp_j'), through the scale
            // into q_i and k_j.
            dq.fill(R::ZERO);
            for j in span {
                let ds = row[j] * (dp[j] - s) * scale;
                for (dq, &k) in dq.iter_mut().zip(&qkv[j * 3 * hidden..][kc.clone()]) {
                    *dq += ds * k;
                }
                for (dk, &q) in dqkv[j * 3 * hidden..][kc.clone()].iter_mut().zip(q) {
                    *dk += ds * q;
                }
            }
            for (d, &g) in dqkv[i * 3 * hidden..][qc.clone()].iter_mut().zip(&dq) {
                *d += g;
            }
        }
    }
}
