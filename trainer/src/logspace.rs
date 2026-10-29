//! The one place the trainer sums exponentials: every sum works from its
//! own maximum, so a large score shared by every term never rounds the
//! normalizer away, and a term at -inf (a forbidden label) counts as 0.

/// log sum exp(v), from v's maximum; -inf when every term is -inf.
pub fn logsumexp(v: impl Iterator<Item = f64> + Clone) -> f64 {
    let m = v.clone().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return m;
    }
    m + v.map(|x| (x - m).exp()).sum::<f64>().ln()
}

/// exp(v) normalized to sum to one in place, worked from v's maximum m;
/// returns m and ln of the sum after subtracting it, so the log of the
/// normalizer is their sum and `v[i] - m - ln` is the log-probability of
/// term i without the rounding of `v[i] - (m + ln)`. All zero, and
/// (-inf, 0), when every term is -inf.
pub fn softmax(v: &mut [f64]) -> (f64, f64) {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        v.fill(0.0);
        return (m, 0.0);
    }
    v.iter_mut().for_each(|x| *x = (*x - m).exp());
    let z: f64 = v.iter().sum();
    v.iter_mut().for_each(|x| *x /= z);
    (m, z.ln())
}
