//! Softmax cross-entropy of each word's label scores against its gold
//! label, for heads decoded by argmax.

use turbo::status::{Result, invalid};

use crate::logspace::softmax;
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
    let mut delta: Vec<f64> = scores.iter().map(|v| v.to_f64()).collect();
    for (t, &g) in gold.iter().enumerate() {
        let g = g as usize;
        let row = &mut delta[t * labels..][..labels];
        let gold_score = row[g];
        // Centred on the row's largest score (see `logspace`), so a large
        // common offset does not swallow the normaliser.
        let (m, ln_z) = softmax(row);
        total += ln_z - (gold_score - m);
        row[g] -= 1.0;
    }
    if !total.is_finite() || !delta.iter().all(|d| d.is_finite()) {
        return Err(invalid("xent: the loss or a gradient is not finite"));
    }
    for (d, x) in dscores.iter_mut().zip(delta) {
        *d += R::from_f64(x);
    }
    Ok(total)
}
