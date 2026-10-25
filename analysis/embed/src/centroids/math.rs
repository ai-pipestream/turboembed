use super::*;
pub(crate) use turbo::head::softmax_f64_in_place as softmax_in_place;

pub(crate) use turbo::head::sigmoid_f64 as sigmoid;

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
