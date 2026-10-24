//! Fitting an [`IncivilityVectorHead`] to labelled vectors: logistic
//! regression per label, by full-batch gradient descent on the mean log
//! loss with an L2 penalty, from zero weights. Every step is in `f64` and
//! in a fixed order, so the same data and options give the same head.

use turbo_analysis_layers::{IncivilityLabel, IncivilityScores};

use super::{IncivilityHeadError, IncivilityVectorHead, LABELS, sigmoid};

/// How [`IncivilityVectorHead::fit`] runs.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct IncivilityFitOptions {
    /// Full passes over the data.
    pub epochs: u32,
    /// The step size.
    pub learning_rate: f64,
    /// The L2 penalty on the weights (not the biases).
    pub l2: f64,
}

impl Default for IncivilityFitOptions {
    fn default() -> Self {
        IncivilityFitOptions { epochs: 500, learning_rate: 0.5, l2: 1e-4 }
    }
}

impl IncivilityVectorHead {
    /// A head fitted to `vectors` and their `targets` (each label's wanted
    /// probability, `0.0..=1.0`). With `contexts`, one per vector (`None`
    /// where a unit has none), the head also learns a context matrix.
    pub fn fit(
        vectors: &[&[f32]],
        contexts: Option<&[Option<&[f32]>]>,
        targets: &[IncivilityScores],
        o: &IncivilityFitOptions,
    ) -> Result<Self, IncivilityHeadError> {
        let value = |m: String| Err(IncivilityHeadError::Value(m));
        let Some(first) = vectors.first() else { return value("no vectors to fit".into()) };
        let dim = first.len();
        if targets.len() != vectors.len() || contexts.is_some_and(|c| c.len() != vectors.len()) {
            return value(format!(
                "{} vectors, {} targets and the contexts differ in count",
                vectors.len(),
                targets.len()
            ));
        }
        let rows = vectors.iter().copied().chain(contexts.into_iter().flatten().flatten().copied());
        for v in rows {
            if v.len() != dim || v.iter().any(|x| !x.is_finite()) {
                return value(format!("a vector of {} values, or not finite, among vectors of {dim}", v.len()));
            }
        }
        if targets.iter().any(|t| t.iter().into_iter().any(|(_, y)| !(0.0..=1.0).contains(&y))) {
            return value("a target outside 0..=1".into());
        }
        if o.epochs == 0 {
            return value("0 epochs: the head would not be fitted at all".into());
        }
        if !(o.learning_rate.is_finite() && o.learning_rate > 0.0 && o.l2.is_finite() && o.l2 >= 0.0) {
            return value(format!("learning rate {} and L2 {}", o.learning_rate, o.l2));
        }
        let n = vectors.len() as f64;
        let mut w = vec![0f64; LABELS * dim];
        let mut c = vec![0f64; if contexts.is_some() { LABELS * dim } else { 0 }];
        let mut b = [0f64; LABELS];
        let ctx = |i: usize| contexts.and_then(|cs| cs[i]);
        for _ in 0..o.epochs {
            let mut gw = vec![0f64; w.len()];
            let mut gc = vec![0f64; c.len()];
            let mut gb = [0f64; LABELS];
            for (i, v) in vectors.iter().enumerate() {
                for (l, label) in IncivilityLabel::ALL.into_iter().enumerate() {
                    let row = l * dim..(l + 1) * dim;
                    let mut z = b[l] + v.iter().zip(&w[row.clone()]).map(|(&x, &y)| f64::from(x) * y).sum::<f64>();
                    if let Some(p) = ctx(i) {
                        z += p.iter().zip(&c[row.clone()]).map(|(&x, &y)| f64::from(x) * y).sum::<f64>();
                    }
                    let err = sigmoid(z) - targets[i].get(label);
                    gb[l] += err;
                    for (g, &x) in gw[row.clone()].iter_mut().zip(v.iter()) {
                        *g += err * f64::from(x);
                    }
                    if let Some(p) = ctx(i) {
                        for (g, &x) in gc[row].iter_mut().zip(p.iter()) {
                            *g += err * f64::from(x);
                        }
                    }
                }
            }
            for (x, g) in w.iter_mut().zip(&gw) {
                *x -= o.learning_rate * (g / n + o.l2 * *x);
            }
            for (x, g) in c.iter_mut().zip(&gc) {
                *x -= o.learning_rate * (g / n + o.l2 * *x);
            }
            for (x, g) in b.iter_mut().zip(gb) {
                *x -= o.learning_rate * g / n;
            }
        }
        let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let bias: [f32; LABELS] = std::array::from_fn(|l| b[l] as f32);
        let head = IncivilityVectorHead::new(dim, f32s(&w), bias)?;
        if contexts.is_some() { head.with_context(f32s(&c)) } else { Ok(head) }
    }
}
