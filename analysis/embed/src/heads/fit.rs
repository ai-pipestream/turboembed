//! Full-batch gradient descent fitting for [`super::ClassifierHead`] using
//! `turbo-trainer` (`linear::forward`, `linear::backward`, `xent::loss`, and
//! `Real::sigmoid`).

use std::sync::Arc;

use turbo_analysis::{ModelRef, Tag, TagSet};
use turbo_trainer::Real;
use turbo_trainer::{linear, xent};

use super::{ClassifierHead, HeadMode};

impl ClassifierHead {
    /// Fits a deterministic 1-layer single-label linear head over labeled
    /// `f32` embedding vectors by minimizing L2-regularized multinomial
    /// cross-entropy loss in `f64` via `turbo-trainer`.
    pub fn fit_single_label<V: AsRef<[f32]>>(
        model: ModelRef,
        in_dim: u32,
        tags: Arc<TagSet>,
        examples: &[(V, Tag)],
        l2_reg: f64,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<ClassifierHead, String> {
        if examples.is_empty() {
            return Err("fit_single_label requires at least one training example".into());
        }
        if in_dim == 0 {
            return Err("in_dim must be at least 1".into());
        }
        if tags.is_empty() {
            return Err("tags must not be empty".into());
        }
        if !l2_reg.is_finite() || l2_reg < 0.0 {
            return Err(format!("l2_reg {l2_reg} must be finite and non-negative"));
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(format!("learning_rate {learning_rate} must be finite and positive"));
        }
        let d = in_dim as usize;
        let k = tags.len();
        for (idx, (vec, tag)) in examples.iter().enumerate() {
            if vec.as_ref().len() != d || vec.as_ref().iter().any(|x| !x.is_finite()) {
                return Err(format!("example {idx} vector has length {}, expected {d}", vec.as_ref().len()));
            }
            if tag.0 as usize >= k {
                return Err(format!("example {idx} tag {} is outside TagSet of size {k}", tag.0));
            }
        }
        let inv_n = 1.0 / examples.len() as f64;
        let mean_sq_norm = examples
            .iter()
            .map(|(v, _)| v.as_ref().iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>())
            .sum::<f64>()
            * inv_n;
        let mean = if mean_sq_norm > 4.0 {
            let mut m = vec![0.0f64; d];
            for (v, _) in examples {
                for (dst, &src) in m.iter_mut().zip(v.as_ref()) {
                    *dst += f64::from(src) * inv_n;
                }
            }
            Some(m)
        } else {
            None
        };
        let centered_sq_norm = if let Some(m) = &mean {
            examples
                .iter()
                .map(|(v, _)| {
                    v.as_ref()
                        .iter()
                        .zip(m)
                        .map(|(&x, &mi)| {
                            let diff = f64::from(x) - mi;
                            diff * diff
                        })
                        .sum::<f64>()
                })
                .sum::<f64>()
                * inv_n
        } else {
            mean_sq_norm
        };
        let w_lr = if mean_sq_norm > 4.0 { learning_rate / centered_sq_norm.max(1.0) } else { learning_rate };
        let mut weights = vec![0.0f64; k * d];
        let mut bias = vec![0.0f64; k];
        let mut grad_w = vec![0.0f64; k * d];
        let mut grad_b = vec![0.0f64; k];
        let mut x_row = vec![0.0f64; d];
        let mut logits = vec![0.0f64; k];
        let mut dy = vec![0.0f64; k];

        for _ in 0..epochs {
            grad_w.fill(0.0);
            grad_b.fill(0.0);
            for (vec, gold) in examples {
                if let Some(m) = &mean {
                    for ((dst, &src), &mi) in x_row.iter_mut().zip(vec.as_ref()).zip(m) {
                        *dst = f64::from(src) - mi;
                    }
                } else {
                    for (dst, &src) in x_row.iter_mut().zip(vec.as_ref()) {
                        *dst = f64::from(src);
                    }
                }
                linear::forward(&weights, &bias, &x_row, d, k, &mut logits);
                dy.fill(0.0);
                xent::loss(&logits, &[u32::from(gold.0)], k, &mut dy).map_err(|e| e.to_string())?;
                linear::backward(&weights, &x_row, &dy, d, k, &mut grad_w, &mut grad_b, None);
            }
            for (w, &gw) in weights.iter_mut().zip(&grad_w) {
                let g = gw * inv_n + l2_reg * *w;
                *w -= w_lr * g;
            }
            for (b, &gb) in bias.iter_mut().zip(&grad_b) {
                *b -= learning_rate * (gb * inv_n);
            }
        }
        if let Some(m) = &mean {
            for c in 0..k {
                let row = &weights[c * d..(c + 1) * d];
                let dot: f64 = row.iter().zip(m).map(|(&wi, &mi)| wi * mi).sum();
                bias[c] -= dot;
            }
        }
        Self::linear(model, in_dim, tags, HeadMode::SingleLabel, weights, bias)
    }

    /// Fits a deterministic 1-layer multi-label linear head over labeled
    /// `f32` embedding vectors by minimizing L2-regularized binary
    /// cross-entropy loss per label in `f64` via `turbo-trainer`.
    pub fn fit_multi_label<V: AsRef<[f32]>, B: AsRef<[bool]>>(
        model: ModelRef,
        in_dim: u32,
        tags: Arc<TagSet>,
        examples: &[(V, B)],
        l2_reg: f64,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<ClassifierHead, String> {
        if examples.is_empty() {
            return Err("fit_multi_label requires at least one training example".into());
        }
        if in_dim == 0 {
            return Err("in_dim must be at least 1".into());
        }
        if tags.is_empty() {
            return Err("tags must not be empty".into());
        }
        if !l2_reg.is_finite() || l2_reg < 0.0 {
            return Err(format!("l2_reg {l2_reg} must be finite and non-negative"));
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(format!("learning_rate {learning_rate} must be finite and positive"));
        }
        let d = in_dim as usize;
        let k = tags.len();
        for (idx, (vec, y)) in examples.iter().enumerate() {
            if vec.as_ref().len() != d || vec.as_ref().iter().any(|x| !x.is_finite()) {
                return Err(format!("example {idx} vector has length {}, expected {d}", vec.as_ref().len()));
            }
            if y.as_ref().len() != k {
                return Err(format!("example {idx} target has {} labels, expected {k}", y.as_ref().len()));
            }
        }
        let inv_n = 1.0 / examples.len() as f64;
        let mean_sq_norm = examples
            .iter()
            .map(|(v, _)| v.as_ref().iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>())
            .sum::<f64>()
            * inv_n;
        let w_lr = if mean_sq_norm > 4.0 { learning_rate / mean_sq_norm } else { learning_rate };
        let mut weights = vec![0.0f64; k * d];
        let mut bias = vec![0.0f64; k];
        let mut grad_w = vec![0.0f64; k * d];
        let mut grad_b = vec![0.0f64; k];
        let mut x_row = vec![0.0f64; d];
        let mut logits = vec![0.0f64; k];
        let mut dy = vec![0.0f64; k];

        for _ in 0..epochs {
            grad_w.fill(0.0);
            grad_b.fill(0.0);
            for (vec, y) in examples {
                for (dst, &src) in x_row.iter_mut().zip(vec.as_ref()) {
                    *dst = f64::from(src);
                }
                linear::forward(&weights, &bias, &x_row, d, k, &mut logits);
                for ((slot, &z), &target) in dy.iter_mut().zip(&logits).zip(y.as_ref()) {
                    *slot = z.sigmoid() - if target { 1.0 } else { 0.0 };
                }
                linear::backward(&weights, &x_row, &dy, d, k, &mut grad_w, &mut grad_b, None);
            }
            for (w, &gw) in weights.iter_mut().zip(&grad_w) {
                let g = gw * inv_n + l2_reg * *w;
                *w -= w_lr * g;
            }
            for (b, &gb) in bias.iter_mut().zip(&grad_b) {
                *b -= learning_rate * (gb * inv_n);
            }
        }
        Self::linear(model, in_dim, tags, HeadMode::MultiLabel, weights, bias)
    }
}
