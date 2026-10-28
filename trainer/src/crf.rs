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

/// The loss -log p(gold | emit) for one sequence, emit [n, labels], and
/// its gradients added to `demit` [n, labels] and `grad`. A gold
/// sequence the lattice forbids is refused: the scheme would never decode
/// it, so it cannot be a training target.
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
    if n == 0 {
        return Ok(0.0);
    }
    let e = |t: usize, j: usize| emit[t * labels + j].to_f64();
    let tr = |a: usize, b: usize| {
        if allowed.may_follow(a, b) { p.transitions[a * labels + b].to_f64() } else { f64::NEG_INFINITY }
    };
    let y: Vec<usize> = gold.iter().map(|&g| g as usize).collect();
    if y.iter().any(|&g| g >= labels) {
        return Err(invalid(format!("crf: a gold label is past the {labels} labels")));
    }
    let legal =
        allowed.may_start(y[0]) && allowed.may_end(y[n - 1]) && y.windows(2).all(|w| allowed.may_follow(w[0], w[1]));
    if !legal {
        return Err(invalid("crf: the gold sequence is one the scheme forbids"));
    }
    let mut gold_score = p.start[y[0]].to_f64() + p.end[y[n - 1]].to_f64();
    for t in 0..n {
        gold_score += e(t, y[t]);
        if t > 0 {
            gold_score += p.transitions[y[t - 1] * labels + y[t]].to_f64();
        }
    }
    let mut alpha = vec![f64::NEG_INFINITY; n * labels];
    for (j, a) in alpha[..labels].iter_mut().enumerate() {
        if allowed.may_start(j) {
            *a = p.start[j].to_f64() + e(0, j);
        }
    }
    for t in 1..n {
        for b in 0..labels {
            let prev = &alpha[(t - 1) * labels..][..labels];
            let s = logsumexp((0..labels).map(|a| prev[a] + tr(a, b)));
            alpha[t * labels + b] = s + e(t, b);
        }
    }
    let mut beta = vec![f64::NEG_INFINITY; n * labels];
    for j in 0..labels {
        if allowed.may_end(j) {
            beta[(n - 1) * labels + j] = p.end[j].to_f64();
        }
    }
    for t in (0..n - 1).rev() {
        for a in 0..labels {
            let next = &beta[(t + 1) * labels..][..labels];
            beta[t * labels + a] = logsumexp((0..labels).map(|b| tr(a, b) + e(t + 1, b) + next[b]));
        }
    }
    let log_z = logsumexp((0..labels).map(|j| alpha[(n - 1) * labels + j] + beta[(n - 1) * labels + j]));
    let marginal = |t: usize, j: usize| (alpha[t * labels + j] + beta[t * labels + j] - log_z).exp();
    for t in 0..n {
        for j in 0..labels {
            let gold_here = if y[t] == j { 1.0 } else { 0.0 };
            demit[t * labels + j] += R::from_f64(marginal(t, j) - gold_here);
        }
    }
    for j in 0..labels {
        grad.start[j] += R::from_f64(marginal(0, j) - if y[0] == j { 1.0 } else { 0.0 });
        grad.end[j] += R::from_f64(marginal(n - 1, j) - if y[n - 1] == j { 1.0 } else { 0.0 });
    }
    for t in 1..n {
        for a in 0..labels {
            let from = alpha[(t - 1) * labels + a];
            if from == f64::NEG_INFINITY {
                continue;
            }
            for b in 0..labels {
                let s = from + tr(a, b) + e(t, b) + beta[t * labels + b] - log_z;
                if s > f64::NEG_INFINITY {
                    grad.transitions[a * labels + b] += R::from_f64(s.exp());
                }
            }
        }
        grad.transitions[y[t - 1] * labels + y[t]] -= R::ONE;
    }
    Ok(log_z - gold_score)
}
