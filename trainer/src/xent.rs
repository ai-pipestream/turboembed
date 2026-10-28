//! Softmax cross-entropy of each word's label scores against its gold
//! label, for heads decoded by argmax.

use turbo::status::{Result, invalid};

use crate::real::Real;

/// The summed loss over the words of scores [n, labels], and its
/// gradients added to `dscores`. Sums run in f64.
pub fn loss<R: Real>(scores: &[R], gold: &[u32], labels: usize, dscores: &mut [R]) -> Result<f64> {
    let mut total = 0.0;
    for (t, &g) in gold.iter().enumerate() {
        let g = g as usize;
        if g >= labels {
            return Err(invalid(format!("xent: a gold label is past the {labels} labels")));
        }
        let row = &scores[t * labels..][..labels];
        let m = row.iter().map(|v| v.to_f64()).fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = row.iter().map(|v| (v.to_f64() - m).exp()).sum();
        total += m + z.ln() - row[g].to_f64();
        for (j, d) in dscores[t * labels..][..labels].iter_mut().enumerate() {
            let p = (row[j].to_f64() - m).exp() / z;
            *d += R::from_f64(p - if j == g { 1.0 } else { 0.0 });
        }
    }
    Ok(total)
}
