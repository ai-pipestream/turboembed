#![allow(dead_code)]

use turbo_analysis::ModelRef;
use turbo_vector_index::{Hit, IndexBuilder, Metric, Normalization, Space, VectorIndex};

pub fn model(id: &str) -> ModelRef {
    ModelRef { id: id.into(), revision: "1".into(), sha256: format!("{id}-manifest") }
}

pub fn space(dim: u32) -> Space {
    Space::new(model("e5-small"), dim, Normalization::None)
}

/// A fixed sequence of values in `[-1, 1)`, from a 64-bit LCG.
pub fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

pub fn index_of(space: &Space, metric: Metric, rows: &[Vec<f32>]) -> VectorIndex<usize> {
    let mut b = IndexBuilder::new(space.clone(), metric).unwrap();
    for (i, r) in rows.iter().enumerate() {
        b.push(r, i).unwrap();
    }
    b.build()
}

/// Every row scored, sorted best first and rows in order where scores are
/// equal, the first `k` kept.
pub fn brute_force(index: &VectorIndex<usize>, space: &Space, query: &[f32], k: usize) -> Vec<Hit> {
    let mut all: Vec<Hit> =
        (0..index.len() as u32).map(|row| Hit { row, score: index.score(space, query, row).unwrap() }).collect();
    // A stable sort on the score alone keeps rows in order where scores
    // are equal (`==`, so 0.0 and -0.0 too).
    all.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    all.truncate(k);
    all
}

/// The score in `f64` from the definition of the metric.
pub fn reference(metric: Metric, q: &[f32], r: &[f32]) -> f64 {
    let dot: f64 = q.iter().zip(r).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
    match metric {
        Metric::Dot => dot,
        Metric::Cosine => {
            let n = |v: &[f32]| v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
            dot / (n(q) * n(r))
        }
    }
}
