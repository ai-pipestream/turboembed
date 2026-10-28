//! Softmax cross-entropy of each word's label scores against its gold
//! label, for heads decoded by argmax.

use turbo::status::{Result, invalid};

use crate::real::Real;

/// The summed loss over the words of scores [n, labels], and its
/// gradients added to `dscores`. Sums run in f64. Refuses scores or
/// gradients not shaped [gold, labels], a gold label out of range,
/// nonfinite scores, and a loss or gradient that overflows; a refusal
/// leaves `dscores` as it was.
pub fn loss<R: Real>(scores: &[R], gold: &[u32], labels: usize, dscores: &mut [R]) -> Result<f64> {
    let cells = gold.len().checked_mul(labels);
    if labels == 0 || cells != Some(scores.len()) || cells != Some(dscores.len()) {
        return Err(invalid(format!("xent: scores and gradients are not [{}, {labels}]", gold.len())));
    }
    if scores.iter().any(|v| !v.to_f64().is_finite()) {
        return Err(invalid("xent: a score is not finite"));
    }
    if gold.iter().any(|&g| g as usize >= labels) {
        return Err(invalid(format!("xent: a gold label is past the {labels} labels")));
    }
    let mut total = 0.0;
    let mut delta = vec![0.0f64; scores.len()];
    for (t, &g) in gold.iter().enumerate() {
        let g = g as usize;
        let row = &scores[t * labels..][..labels];
        let m = row.iter().map(|v| v.to_f64()).fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = row.iter().map(|v| (v.to_f64() - m).exp()).sum();
        total += m + z.ln() - row[g].to_f64();
        for (j, d) in delta[t * labels..][..labels].iter_mut().enumerate() {
            let p = (row[j].to_f64() - m).exp() / z;
            *d = p - if j == g { 1.0 } else { 0.0 };
        }
    }
    if !total.is_finite() || !delta.iter().all(|d| d.is_finite()) {
        return Err(invalid("xent: the loss or a gradient is not finite"));
    }
    for (d, x) in dscores.iter_mut().zip(delta) {
        *d += R::from_f64(x);
    }
    Ok(total)
}
