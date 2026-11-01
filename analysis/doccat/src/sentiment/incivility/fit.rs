//! Fitting an [`IncivilityVectorHead`] to labelled vectors with `turbo-trainer`
//! (`AdamW`, `linear::forward`, `linear::backward`, and `Real::sigmoid`):
//! L2-regularized logistic regression per label from zero weights.

use turbo_analysis_layers::{IncivilityLabel, IncivilityScores};
use turbo_trainer::Real;
use turbo_trainer::adamw::{AdamW, Settings};
use turbo_trainer::linear;

use super::super::INCIVILITY_DIMENSIONS as N;
use super::{IncivilityHeadError, IncivilityVectorHead};

/// How [`IncivilityVectorHead::fit`] runs.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct IncivilityFitOptions {
    /// Full passes over the data.
    pub epochs: u32,
    /// The step size.
    pub learning_rate: f64,
    /// The L2 penalty on the weights and context weights (not the biases).
    pub l2: f64,
}

impl Default for IncivilityFitOptions {
    fn default() -> Self {
        IncivilityFitOptions { epochs: 500, learning_rate: 0.5, l2: 1e-4 }
    }
}

impl IncivilityVectorHead {
    /// A head fitted to `vectors`, optional `contexts` (each row's preceding
    /// or parent vector, `None` when it has none), and their `targets`
    /// (`0.0..=1.0` per label) through `turbo-trainer`.
    pub fn fit(
        vectors: &[&[f32]],
        contexts: Option<&[Option<&[f32]>]>,
        targets: &[IncivilityScores],
        o: &IncivilityFitOptions,
    ) -> Result<Self, IncivilityHeadError> {
        let value = |m: String| Err(IncivilityHeadError::Value(m));
        let Some(first) = vectors.first() else { return value("no vectors to fit".into()) };
        let dim = first.len();
        if targets.len() != vectors.len() {
            return value(format!("{} vectors, {} targets", vectors.len(), targets.len()));
        }
        if let Some(ctx) = contexts
            && ctx.len() != vectors.len()
        {
            return value(format!("{} vectors, {} contexts", vectors.len(), ctx.len()));
        }
        for v in vectors {
            if v.len() != dim || v.iter().any(|x| !x.is_finite()) {
                return value(format!("a vector of {} values, or not finite, among vectors of {dim}", v.len()));
            }
        }
        if let Some(ctx) = contexts {
            for c in ctx.iter().flatten() {
                if c.len() != dim || c.iter().any(|x| !x.is_finite()) {
                    return value(format!("a context vector of {} values, or not finite, for dim {dim}", c.len()));
                }
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
        let has_ctx = contexts.is_some();
        let sizes: &[usize] = if has_ctx { &[N * dim, N, N * dim] } else { &[N * dim, N] };
        let mut opt = AdamW::<f32>::new(settings, sizes).map_err(|e| IncivilityHeadError::Value(e.to_string()))?;
        let mut params = if has_ctx {
            vec![vec![0.0f32; N * dim], vec![0.0f32; N], vec![0.0f32; N * dim]]
        } else {
            vec![vec![0.0f32; N * dim], vec![0.0f32; N]]
        };
        let mut grads = params.clone();
        let decay = if has_ctx { vec![false, false, false] } else { vec![false, false] };
        let zero_bias = [0.0f32; N];
        let mut dummy_gb = [0.0f32; N];

        let chunk_rows = 256usize.min(vectors.len());
        let mut x_chunk = Vec::with_capacity(chunk_rows * dim);
        let mut logits = Vec::with_capacity(chunk_rows * N);
        let mut ctx_logits = [0.0f32; N];
        let mut dy = Vec::with_capacity(chunk_rows * N);

        for _ in 0..o.epochs {
            for g in &mut grads {
                g.fill(0.0);
            }
            for chunk_idx in 0..vectors.len().div_ceil(chunk_rows) {
                let start = chunk_idx * chunk_rows;
                let end = (start + chunk_rows).min(vectors.len());
                let v_chunk = &vectors[start..end];
                let t_chunk = &targets[start..end];
                let b_len = v_chunk.len();
                x_chunk.clear();
                for v in v_chunk {
                    x_chunk.extend_from_slice(v);
                }
                logits.resize(b_len * N, 0.0);
                linear::forward(&params[0], &params[1], &x_chunk, dim, N, &mut logits);
                if let Some(ctx) = contexts {
                    for (local_i, p) in ctx[start..end].iter().enumerate() {
                        if let Some(p_vec) = p {
                            linear::forward(&params[2], &zero_bias, p_vec, dim, N, &mut ctx_logits);
                            let row = &mut logits[local_i * N..(local_i + 1) * N];
                            for (dst, &add) in row.iter_mut().zip(&ctx_logits) {
                                *dst += add;
                            }
                        }
                    }
                }
                dy.clear();
                for (row_logits, t) in logits.as_chunks::<N>().0.iter().zip(t_chunk) {
                    for (l, label) in IncivilityLabel::ALL.into_iter().enumerate() {
                        dy.push((row_logits[l].sigmoid() - t.get(label) as f32) * n_inv);
                    }
                }
                {
                    let (first_two, rest) = grads.split_at_mut(2);
                    let [gw, gb] = first_two else { unreachable!() };
                    linear::backward(&params[0], &x_chunk, &dy, dim, N, gw, gb, None);
                    if let Some(ctx) = contexts {
                        let gc = &mut rest[0];
                        for (local_i, p) in ctx[start..end].iter().enumerate() {
                            if let Some(p_vec) = p {
                                let dy_row = &dy[local_i * N..(local_i + 1) * N];
                                linear::backward(&params[2], p_vec, dy_row, dim, N, gc, &mut dummy_gb, None);
                            }
                        }
                    }
                }
            }
            for (g, &w) in grads[0].iter_mut().zip(&params[0]) {
                *g += l2 * w;
            }
            if has_ctx {
                for (g, &c) in grads[2].iter_mut().zip(&params[2]) {
                    *g += l2 * c;
                }
            }
            opt.step(&mut params, &mut grads, &decay).map_err(|e| IncivilityHeadError::Value(e.to_string()))?;
        }

        let mut it = params.into_iter();
        let weights = it.next().unwrap();
        let bias_vec = it.next().unwrap();
        let bias: [f32; N] = std::array::from_fn(|l| bias_vec[l]);
        let head = IncivilityVectorHead::new(dim, weights, bias)?;
        if has_ctx { head.with_context(it.next().unwrap()) } else { Ok(head) }
    }
}
