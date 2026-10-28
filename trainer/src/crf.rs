//! The negative log-likelihood of a first-order CRF over the lattice a
//! scheme allows (`turbo::head::crf::allowed`, the same lattice decode
//! uses), and its gradients, by forward-backward in log space.
//!
//! Sums run in f64 whatever the op's number type, so f32 training loses
//! nothing in the partition function.

use turbo::head::crf::Allowed;
use turbo::status::{Result, invalid};

use crate::real::Real;

/// The CRF's parameters: transitions [from, to], start and end [labels].
#[derive(Debug, Clone, Copy)]
pub struct Params<'a, R> {
    pub transitions: &'a [R],
    pub start: &'a [R],
    pub end: &'a [R],
}

/// Gradients of the CRF parameters, added to.
pub struct Grad<'a, R> {
    pub transitions: &'a mut [R],
    pub start: &'a mut [R],
    pub end: &'a mut [R],
}

fn logsumexp(v: impl Iterator<Item = f64> + Clone) -> f64 {
    let m = v.clone().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return m;
    }
    m + v.map(|x| (x - m).exp()).sum::<f64>().ln()
}

/// exp(v) normalized to sum to one, worked from v's maximum, so no large
/// common score rounds the normalizer away; all zero when v is all -inf.
fn softmax(v: &mut [f64]) {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        v.fill(0.0);
        return;
    }
    v.iter_mut().for_each(|x| *x = (*x - m).exp());
    let z: f64 = v.iter().sum();
    v.iter_mut().for_each(|x| *x /= z);
}

/// The loss -log p(gold | emit) for one sequence, emit [n, labels], and
/// its gradients added to `demit` [n, labels] and `grad`. A gold
/// sequence the lattice forbids is refused: the scheme would never decode
/// it, so it cannot be a training target. So are a lattice, parameters,
/// scores or gradients of another label count or length, nonfinite
/// scores or parameters, and a loss or gradient that overflows. A
/// refusal leaves `demit` and `grad` as they were.
pub fn nll<R: Real>(
    p: Params<R>,
    allowed: &Allowed,
    emit: &[R],
    gold: &[u32],
    labels: usize,
    demit: &mut [R],
    grad: Grad<R>,
) -> Result<f64> {
    let n = gold.len();
    let cells = n.checked_mul(labels);
    let shaped = allowed.labels() == labels
        && labels > 0
        && cells == Some(emit.len())
        && cells == Some(demit.len())
        && labels.checked_mul(labels).is_some_and(|l2| p.transitions.len() == l2 && grad.transitions.len() == l2)
        && [p.start.len(), p.end.len(), grad.start.len(), grad.end.len()].iter().all(|&l| l == labels);
    if !shaped {
        return Err(invalid(format!("crf: the lattice, parameters, scores or gradients are not over {labels} labels")));
    }
    let finite = |v: &[R]| v.iter().all(|x| x.to_f64().is_finite());
    if !finite(emit) || !finite(p.transitions) || !finite(p.start) || !finite(p.end) {
        return Err(invalid("crf: a score or parameter is not finite"));
    }
    if n == 0 {
        return Ok(0.0);
    }
    let y: Vec<usize> = gold.iter().map(|&g| g as usize).collect();
    if y.iter().any(|&g| g >= labels) {
        return Err(invalid(format!("crf: a gold label is past the {labels} labels")));
    }
    // Every path takes one start, one emission a word, one transition a
    // step and one end, so each is measured from the gold path's own at
    // that place: each path's score is then its score less the gold's,
    // the gold path scores exactly 0, and the loss is log Z alone. No
    // large score the gold path shares, nor one only a forbidden path
    // could take, swallows the normaliser.
    let at = |t: usize, j: usize| emit[t * labels + j].to_f64();
    let e = |t: usize, j: usize| at(t, j) - at(t, y[t]);
    let start = |j: usize| p.start[j].to_f64() - p.start[y[0]].to_f64();
    let end = |j: usize| p.end[j].to_f64() - p.end[y[n - 1]].to_f64();
    let raw = |a: usize, b: usize| p.transitions[a * labels + b].to_f64();
    // Into word t, from 1.
    let tr = |t: usize, a: usize, b: usize| {
        if allowed.may_follow(a, b) { raw(a, b) - raw(y[t - 1], y[t]) } else { f64::NEG_INFINITY }
    };
    let legal =
        allowed.may_start(y[0]) && allowed.may_end(y[n - 1]) && y.windows(2).all(|w| allowed.may_follow(w[0], w[1]));
    if !legal {
        return Err(invalid("crf: the gold sequence is one the scheme forbids"));
    }
    let mut alpha = vec![f64::NEG_INFINITY; n * labels];
    for (j, a) in alpha[..labels].iter_mut().enumerate() {
        if allowed.may_start(j) {
            *a = start(j) + e(0, j);
        }
    }
    for t in 1..n {
        for b in 0..labels {
            let prev = &alpha[(t - 1) * labels..][..labels];
            let s = logsumexp((0..labels).map(|a| prev[a] + tr(t, a, b)));
            alpha[t * labels + b] = s + e(t, b);
        }
    }
    let mut beta = vec![f64::NEG_INFINITY; n * labels];
    for j in 0..labels {
        if allowed.may_end(j) {
            beta[(n - 1) * labels + j] = end(j);
        }
    }
    for t in (0..n - 1).rev() {
        for a in 0..labels {
            let next = &beta[(t + 1) * labels..][..labels];
            beta[t * labels + a] = logsumexp((0..labels).map(|b| tr(t + 1, a, b) + e(t + 1, b) + next[b]));
        }
    }
    let log_z = logsumexp((0..labels).map(|j| alpha[(n - 1) * labels + j] + beta[(n - 1) * labels + j]));
    // The gold path's score is 0.
    let loss = log_z;
    if !loss.is_finite() {
        return Err(invalid("crf: the loss is not finite"));
    }
    // Every gradient is worked out before any is written, so a refusal
    // leaves the caller's sums whole.
    // Each place's marginals are normalized over that place on its own:
    // every place's alpha + beta sums to Z, but taking log Z from one
    // place and subtracting it at another loses the normalizer when the
    // scores are large (codex: [0, 1e20, 1e20] gave marginals [0, 1, 1]).
    let mut marginals: Vec<f64> = alpha.iter().zip(&beta).map(|(a, b)| a + b).collect();
    marginals.chunks_mut(labels).for_each(softmax);
    let marginal = |t: usize, j: usize| marginals[t * labels + j];
    let mut d_emit = vec![0.0f64; n * labels];
    for t in 0..n {
        for j in 0..labels {
            let gold_here = if y[t] == j { 1.0 } else { 0.0 };
            d_emit[t * labels + j] = marginal(t, j) - gold_here;
        }
    }
    let d_start: Vec<f64> = (0..labels).map(|j| marginal(0, j) - if y[0] == j { 1.0 } else { 0.0 }).collect();
    let d_end: Vec<f64> = (0..labels).map(|j| marginal(n - 1, j) - if y[n - 1] == j { 1.0 } else { 0.0 }).collect();
    let mut d_tr = vec![0.0f64; labels * labels];
    let mut pair = vec![f64::NEG_INFINITY; labels * labels];
    for t in 1..n {
        for a in 0..labels {
            for b in 0..labels {
                pair[a * labels + b] = alpha[(t - 1) * labels + a] + tr(t, a, b) + e(t, b) + beta[t * labels + b];
            }
        }
        // As the nodes: normalized over this step's pairs on their own.
        softmax(&mut pair);
        d_tr.iter_mut().zip(&pair).for_each(|(d, &p)| *d += p);
        d_tr[y[t - 1] * labels + y[t]] -= 1.0;
    }
    if ![&d_emit, &d_start, &d_end, &d_tr].iter().all(|v| v.iter().all(|x| x.is_finite())) {
        return Err(invalid("crf: a gradient is not finite"));
    }
    let add = |dst: &mut [R], src: &[f64]| dst.iter_mut().zip(src).for_each(|(d, &x)| *d += R::from_f64(x));
    add(demit, &d_emit);
    add(grad.start, &d_start);
    add(grad.end, &d_end);
    add(grad.transitions, &d_tr);
    Ok(loss)
}
