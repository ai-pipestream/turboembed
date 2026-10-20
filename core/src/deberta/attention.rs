use super::{
    Config,
    math::{bucket, finite, linear},
    weights::Layer,
};
use crate::status::Result;

pub(super) fn attention(c: &Config, layer: &Layer, input: &[f32], relative: &[f32], mask: &[bool]) -> Result<Vec<f32>> {
    let (h, n, d) = (c.hidden, mask.len(), c.hidden / c.heads);
    let q = linear(input, &layer.q)?;
    let k = linear(input, &layer.k)?;
    let v = linear(input, &layer.v)?;
    let pq = linear(relative, &layer.q)?;
    let pk = linear(relative, &layer.k)?;
    let scale = (3.0 * d as f64).sqrt();
    let mut output = vec![0.0; input.len()];
    let mut scores = vec![0.0f64; n];
    for head in 0..c.heads {
        let offset = head * d;
        for i in 0..n {
            for (j, score) in scores.iter_mut().enumerate() {
                if !mask[i] || !mask[j] {
                    // HF uses finite minimum then softmax: a masked query
                    // has a uniform row, not an all-zero attention row.
                    *score = f64::from(f32::MIN);
                    continue;
                }
                let r = bucket(i as isize - j as isize, c.position_buckets, c.max_relative_positions);
                let mut sum = 0.0;
                for x in 0..d {
                    let qi = f64::from(q[i * h + offset + x]);
                    let kj = f64::from(k[j * h + offset + x]);
                    // p2c's gather is transposed in HF, so the final index
                    // uses i-j, just like c2p (not j-i).
                    sum += qi * kj + qi * f64::from(pk[r * h + offset + x]) + kj * f64::from(pq[r * h + offset + x]);
                }
                *score = sum / scale;
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0.0;
            for s in &mut scores {
                *s = (*s - max).exp();
                total += *s;
            }
            for x in 0..d {
                let sum: f64 =
                    scores.iter().enumerate().map(|(j, &s)| s / total * f64::from(v[j * h + offset + x])).sum();
                output[i * h + offset + x] = sum as f32;
            }
        }
    }
    finite(&output)?;
    linear(&output, &layer.attention)
}
