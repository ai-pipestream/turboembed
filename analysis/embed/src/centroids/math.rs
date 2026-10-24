use super::*;
pub(crate) use turbo::head::softmax_f64_in_place as softmax_in_place;

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
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}
