//! Fitting an [`EmotionVectorHead`] to labelled vectors with `turbo-trainer`
//! (`AdamW`, `linear::forward`, `linear::backward`, and `Real::sigmoid`):
//! L2-regularized logistic regression per label from zero weights.

use turbo_analysis_layers::{EmotionDimension, EmotionScores};
use turbo_trainer::Real;
use turbo_trainer::adamw::{AdamW, Settings};
use turbo_trainer::linear;

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

impl EmotionVectorHead {
    /// A head fitted to `vectors` and their `targets` (each label's wanted
    /// probability, `0.0..=1.0`) through `turbo-trainer`.
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
        let n_inv = 1.0 / (vectors.len() as f32);
        let l2 = o.l2 as f32;
        let lr = (o.learning_rate * 0.05).clamp(1e-4, 0.15);
        let settings = Settings {
            lr,
            weight_decay: 0.0,
            clip: 10.0,
            warmup: 0,
            total: u64::from(o.epochs),
            ..Settings::default()
        };
        let mut opt = AdamW::<f32>::new(settings, &[N * dim, N]).map_err(|e| EmotionHeadError::Value(e.to_string()))?;
        let mut params = vec![vec![0.0f32; N * dim], vec![0.0f32; N]];
        let mut grads = vec![vec![0.0f32; N * dim], vec![0.0f32; N]];
        let decay = [false, false];

        let chunk_rows = 256usize.min(vectors.len());
        let mut x_chunk = Vec::with_capacity(chunk_rows * dim);
        let mut logits = Vec::with_capacity(chunk_rows * N);
        let mut dy = Vec::with_capacity(chunk_rows * N);

        for _ in 0..o.epochs {
            let [gw, gb] = &mut grads[..] else { unreachable!() };
            gw.fill(0.0);
            gb.fill(0.0);

            for (v_chunk, t_chunk) in vectors.chunks(chunk_rows).zip(targets.chunks(chunk_rows)) {
                let b_len = v_chunk.len();
                x_chunk.clear();
                for v in v_chunk {
                    x_chunk.extend_from_slice(v);
                }
                logits.resize(b_len * N, 0.0);
                linear::forward(&params[0], &params[1], &x_chunk, dim, N, &mut logits);

                dy.clear();
                for (row_logits, t) in logits.as_chunks::<N>().0.iter().zip(t_chunk) {
                    for (l, label) in EmotionDimension::ALL.into_iter().enumerate() {
                        dy.push((row_logits[l].sigmoid() - t.get(label) as f32) * n_inv);
                    }
                }
                linear::backward(&params[0], &x_chunk, &dy, dim, N, gw, gb, None);
            }
            for (g, &w) in gw.iter_mut().zip(&params[0]) {
                *g += l2 * w;
            }
            opt.step(&mut params, &mut grads, &decay).map_err(|e| EmotionHeadError::Value(e.to_string()))?;
        }
        let mut it = params.into_iter();
        let weights = it.next().unwrap();
        let bias_vec = it.next().unwrap();
        let bias: [f32; N] = std::array::from_fn(|l| bias_vec[l]);
        EmotionVectorHead::new(dim, weights, bias)
    }
}
