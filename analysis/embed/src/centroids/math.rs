use super::*;
pub(crate) fn softmax_in_place(logits: &mut [f64]) {
    if logits.is_empty() {
        return;
    }
    let max_val = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0f64;
    for z in logits.iter_mut() {
        *z = (*z - max_val).exp();
        sum += *z;
    }
    if sum > 0.0 && sum.is_finite() {
        let inv = 1.0 / sum;
        for z in logits.iter_mut() {
            *z *= inv;
        }
    } else {
        let u = 1.0 / logits.len() as f64;
        logits.fill(u);
    }
}

pub(crate) fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

pub(crate) fn argmax_tag(scores: &[f64]) -> (Tag, f64) {
    let mut best_idx = 0usize;
    let mut best_score = scores[0];
    for (i, &s) in scores.iter().enumerate().skip(1) {
        if s > best_score {
            best_score = s;
            best_idx = i;
        }
    }
    (Tag(best_idx as u16), best_score)
}

pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    turbo_analysis::fnv1a(bytes)
}
