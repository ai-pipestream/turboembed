//! Fitting an [`EmotionVectorHead`] to labelled vectors: logistic
//! regression per label, by full-batch gradient descent on the mean log
//! loss with an L2 penalty, from zero weights. Every step is in `f64` and
//! in a fixed order, so the same data and options give the same head.

use turbo_analysis_layers::{EmotionDimension, EmotionScores};

use super::{EmotionHeadError, EmotionVectorHead, N};

/// How [`EmotionVectorHead::fit`] runs.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct EmotionFitOptions {
    /// Full passes over the data.
    pub epochs: u32,
    /// The step size.
    pub learning_rate: f64,
    /// The L2 penalty on the weights (not the biases).
    pub l2: f64,
}

impl Default for EmotionFitOptions {
    fn default() -> Self {
        EmotionFitOptions { epochs: 500, learning_rate: 0.5, l2: 1e-4 }
    }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

impl EmotionVectorHead {
    /// A head fitted to `vectors` and their `targets` (each label's wanted
    /// probability, `0.0..=1.0`).
    pub fn fit(vectors: &[&[f32]], targets: &[EmotionScores], o: &EmotionFitOptions) -> Result<Self, EmotionHeadError> {
        let value = |m: String| Err(EmotionHeadError::Value(m));
        let Some(first) = vectors.first() else { return value("no vectors to fit".into()) };
        let dim = first.len();
        if targets.len() != vectors.len() {
            return value(format!("{} vectors, {} targets", vectors.len(), targets.len()));
        }
        for v in vectors {
            if v.len() != dim || v.iter().any(|x| !x.is_finite()) {
                return value(format!("a vector of {} values, or not finite, among vectors of {dim}", v.len()));
            }
        }
        if targets.iter().any(|t| !t.is_valid()) {
            return value("a target outside 0..=1".into());
        }
        if o.epochs == 0 {
            return value("0 epochs: the head would not be fitted at all".into());
        }
        if !(o.learning_rate.is_finite() && o.learning_rate > 0.0 && o.l2.is_finite() && o.l2 >= 0.0) {
            return value(format!("learning rate {} and L2 {}", o.learning_rate, o.l2));
        }
        let n = vectors.len() as f64;
        let mut w = vec![0f64; N * dim];
        let mut b = [0f64; N];
        for _ in 0..o.epochs {
            let mut gw = vec![0f64; w.len()];
            let mut gb = [0f64; N];
            for (i, v) in vectors.iter().enumerate() {
                for (l, label) in EmotionDimension::ALL.into_iter().enumerate() {
                    let row = l * dim..(l + 1) * dim;
                    let z = b[l] + v.iter().zip(&w[row.clone()]).map(|(&x, &y)| f64::from(x) * y).sum::<f64>();
                    let err = sigmoid(z) - targets[i].get(label);
                    gb[l] += err;
                    for (g, &x) in gw[row].iter_mut().zip(v.iter()) {
                        *g += err * f64::from(x);
                    }
                }
            }
            for (x, g) in w.iter_mut().zip(&gw) {
                *x -= o.learning_rate * (g / n + o.l2 * *x);
            }
            for (x, g) in b.iter_mut().zip(gb) {
                *x -= o.learning_rate * g / n;
            }
        }
        let weights: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
        let bias: [f32; N] = std::array::from_fn(|l| b[l] as f32);
        EmotionVectorHead::new(dim, weights, bias)
    }
}
