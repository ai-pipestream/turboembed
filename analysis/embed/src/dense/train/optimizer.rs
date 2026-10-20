use super::AdamParameters;
use crate::{
    dense::{OutputKind, math},
    heads::{DenseLayer, HiddenActivation},
};

pub(super) fn zeros(layers: &[DenseLayer]) -> Vec<Vec<f64>> {
    layers.iter().map(|l| vec![0.; l.weights.len() + l.bias.len()]).collect()
}
pub(super) fn activations(
    layers: &[DenseLayer],
    activation: HiddenActivation,
    x: &[f32],
) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
    let mut a = vec![x.iter().map(|&v| f64::from(v)).collect::<Vec<_>>()];
    let mut hidden = Vec::new();
    for (i, l) in layers.iter().enumerate() {
        let mut z = math::forward(l, a.last().unwrap())?;
        if i + 1 < layers.len() {
            hidden = z.clone();
            for v in &mut z {
                *v = activation.apply(*v);
            }
        }
        a.push(z);
    }
    Ok((a, hidden))
}
pub(super) fn gradient(
    layers: &[DenseLayer],
    activation: HiddenActivation,
    output: OutputKind,
    x: &[f32],
    y: &[f64],
    grads: &mut [Vec<f64>],
) -> Result<(), String> {
    let (a, hidden) = activations(layers, activation, x)?;
    let mut delta = math::probabilities(a.last().unwrap(), output, 1.)?;
    let scale = if output == OutputKind::Sigmoid { 1. / y.len() as f64 } else { 1. };
    for (d, y) in delta.iter_mut().zip(y) {
        *d = (*d - y) * scale;
    }
    for i in (0..layers.len()).rev() {
        let layer = &layers[i];
        let n = layer.in_dim as usize;
        for (j, &d) in delta.iter().enumerate() {
            for (k, &x) in a[i].iter().enumerate() {
                grads[i][j * n + k] += d * x;
            }
            grads[i][layer.weights.len() + j] += d;
        }
        if i > 0 {
            let mut prev = vec![0.; n];
            for (j, &d) in delta.iter().enumerate() {
                for (k, p) in prev.iter_mut().enumerate() {
                    *p += layer.weights[j * n + k] * d;
                }
            }
            for (d, &z) in prev.iter_mut().zip(&hidden) {
                *d *= math::derivative(activation, z);
            }
            delta = prev;
        }
    }
    Ok(())
}
pub(super) struct Adam {
    first: Vec<Vec<f64>>,
    second: Vec<Vec<f64>>,
    step: u64,
}
impl Adam {
    pub fn new(layers: &[DenseLayer]) -> Self {
        Self { first: zeros(layers), second: zeros(layers), step: 0 }
    }
    pub fn step(
        &mut self,
        layers: &mut [DenseLayer],
        grads: &[Vec<f64>],
        batch: usize,
        p: &AdamParameters,
    ) -> Result<(), String> {
        self.step = self.step.checked_add(1).ok_or("Adam step overflow")?;
        let c1 = 1. - 0.9f64.powf(self.step as f64);
        let c2 = 1. - 0.999f64.powf(self.step as f64);
        for (i, l) in layers.iter_mut().enumerate() {
            let weights = l.weights.len();
            for (j, w) in l.weights.iter_mut().chain(&mut l.bias).enumerate() {
                let g = grads[i][j] / batch as f64 + if j < weights { p.weight_decay * *w } else { 0. };
                let v = g * g;
                if !g.is_finite() || !v.is_finite() {
                    return Err("non-finite Adam gradient".into());
                }
                self.first[i][j] = 0.9 * self.first[i][j] + 0.1 * g;
                self.second[i][j] = 0.999 * self.second[i][j] + 0.001 * v;
                *w -= p.learning_rate * (self.first[i][j] / c1) / ((self.second[i][j] / c2).sqrt() + 1e-8);
                if !w.is_finite() {
                    return Err("non-finite Adam weight".into());
                }
            }
        }
        Ok(())
    }
}
