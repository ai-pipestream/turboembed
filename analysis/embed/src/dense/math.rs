use super::OutputKind;
use crate::heads::{DenseLayer, HiddenActivation};

pub(super) fn validate_targets(y: &[Vec<f64>], n: usize, k: usize, output: OutputKind) -> Result<(), String> {
    if n == 0 || y.len() != n {
        return Err("nonempty training rows and matching targets required".into());
    }
    for row in y {
        if row.len() != k || row.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)) {
            return Err("targets require one finite probability per label".into());
        }
        if output == OutputKind::Softmax && (row.iter().sum::<f64>() - 1.).abs() > 1e-9 {
            return Err("softmax targets must sum to one".into());
        }
    }
    Ok(())
}
pub(super) fn probabilities(z: &[f64], output: OutputKind, t: f64) -> Result<Vec<f64>, String> {
    if z.iter().any(|v| !v.is_finite()) {
        return Err("non-finite classifier logits".into());
    }
    let mut p = z.to_vec();
    match output {
        OutputKind::Softmax => {
            let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for v in &mut p {
                *v = ((*v - max) / t).exp();
            }
            let sum = p.iter().sum::<f64>();
            for v in &mut p {
                *v /= sum;
            }
        }
        OutputKind::Sigmoid => {
            for v in &mut p {
                *v = crate::centroids::sigmoid(*v / t);
            }
        }
    }
    Ok(p)
}
pub(super) fn loss(z: &[f64], y: &[f64], output: OutputKind, t: f64) -> Result<f64, String> {
    if z.iter().any(|v| !v.is_finite()) {
        return Err("non-finite loss logits".into());
    }
    let value = match output {
        OutputKind::Softmax => {
            let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let logsum = z.iter().map(|v| ((*v - max) / t).exp()).sum::<f64>().ln();
            z.iter().zip(y).filter(|(_, y)| **y > 0.).map(|(v, y)| y * (logsum + (max - v) / t)).sum()
        }
        OutputKind::Sigmoid => {
            z.iter()
                .zip(y)
                .map(|(v, y)| {
                    let v = v / t;
                    let positive = if *y == 1. { 0. } else { (1. - y) * v.max(0.) };
                    let negative = if *y == 0. { 0. } else { y * (-v).max(0.) };
                    positive + negative + (-v.abs()).exp().ln_1p()
                })
                .sum::<f64>()
                / y.len() as f64
        }
    };
    if !value.is_finite() {
        return Err("non-finite classifier loss".into());
    }
    Ok(value)
}
pub(super) fn forward(l: &DenseLayer, x: &[f64]) -> Result<Vec<f64>, String> {
    let z: Vec<_> = l
        .weights
        .chunks_exact(l.in_dim as usize)
        .zip(&l.bias)
        .map(|(w, b)| b + w.iter().zip(x).map(|(w, x)| w * x).sum::<f64>())
        .collect();
    if z.iter().any(|v| !v.is_finite()) {
        return Err("non-finite classifier activation".into());
    }
    Ok(z)
}
pub(super) fn derivative(a: HiddenActivation, x: f64) -> f64 {
    match a {
        HiddenActivation::Relu => f64::from(x > 0.),
        HiddenActivation::Gelu => {
            let cdf = if x == 0. { 0.5 } else { a.apply(x) / x };
            cdf + x * (-0.5 * x * x).exp() / (2. * std::f64::consts::PI).sqrt()
        }
        _ => 1.,
    }
}
/// SplitMix64: local RNG, so no process-global seed or scheduling dependency.
pub(super) struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64) / ((1u64 << 53) as f64)
    }
}
