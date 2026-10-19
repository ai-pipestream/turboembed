//! Full-batch gradient descent fitting for [`super::ClassifierHead`].

use std::sync::Arc;

use turbo_analysis::{ModelRef, Tag, TagSet};

use super::{ClassifierHead, HeadMode};
use crate::centroids::{sigmoid, softmax_in_place};

impl ClassifierHead {
    /// Fits a deterministic 1-layer single-label linear head over labeled
    /// `f32` embedding vectors by minimizing L2-regularized multinomial
    /// cross-entropy loss in `f64` via full-batch gradient descent.
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
            if vec.as_ref().len() != d {
                return Err(format!("example {idx} vector has length {}, expected {d}", vec.as_ref().len()));
            }
            if tag.0 as usize >= k {
                return Err(format!("example {idx} tag {} is outside TagSet of size {k}", tag.0));
            }
        }
        let mut weights = vec![0.0f64; k * d];
        let mut bias = vec![0.0f64; k];
        let mut grad_w = vec![0.0f64; k * d];
        let mut grad_b = vec![0.0f64; k];
        let mut probs = vec![0.0f64; k];
        let inv_n = 1.0 / examples.len() as f64;

        for _ in 0..epochs {
            grad_w.fill(0.0);
            grad_b.fill(0.0);
            for (vec, gold) in examples {
                let x = vec.as_ref();
                for (c, (row, &b)) in weights.chunks_exact(d).zip(&bias).enumerate() {
                    let mut z = b;
                    for (&w, &xi) in row.iter().zip(x) {
                        z += w * f64::from(xi);
                    }
                    probs[c] = z;
                }
                softmax_in_place(&mut probs);
                probs[gold.0 as usize] -= 1.0;
                for (c, &err) in probs.iter().enumerate() {
                    grad_b[c] += err;
                    let gw_row = &mut grad_w[c * d..(c + 1) * d];
                    for (gw, &xi) in gw_row.iter_mut().zip(x) {
                        *gw += err * f64::from(xi);
                    }
                }
            }
            for (w, &gw) in weights.iter_mut().zip(&grad_w) {
                let g = gw * inv_n + l2_reg * *w;
                *w -= learning_rate * g;
            }
            for (b, &gb) in bias.iter_mut().zip(&grad_b) {
                *b -= learning_rate * (gb * inv_n);
            }
        }
        Self::linear(model, in_dim, tags, HeadMode::SingleLabel, weights, bias)
    }

    /// Fits a deterministic 1-layer multi-label linear head over labeled
    /// `f32` embedding vectors by minimizing L2-regularized binary
    /// cross-entropy loss per label in `f64` via full-batch gradient descent.
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
            if vec.as_ref().len() != d {
                return Err(format!("example {idx} vector has length {}, expected {d}", vec.as_ref().len()));
            }
            if y.as_ref().len() != k {
                return Err(format!("example {idx} target has {} labels, expected {k}", y.as_ref().len()));
            }
        }
        let mut weights = vec![0.0f64; k * d];
        let mut bias = vec![0.0f64; k];
        let mut grad_w = vec![0.0f64; k * d];
        let mut grad_b = vec![0.0f64; k];
        let inv_n = 1.0 / examples.len() as f64;

        for _ in 0..epochs {
            grad_w.fill(0.0);
            grad_b.fill(0.0);
            for (vec, y) in examples {
                let x = vec.as_ref();
                let targets = y.as_ref();
                for (c, ((row, &b), &target)) in weights.chunks_exact(d).zip(&bias).zip(targets).enumerate() {
                    let mut z = b;
                    for (&w, &xi) in row.iter().zip(x) {
                        z += w * f64::from(xi);
                    }
                    let pred = sigmoid(z);
                    let err = pred - if target { 1.0 } else { 0.0 };
                    grad_b[c] += err;
                    let gw_row = &mut grad_w[c * d..(c + 1) * d];
                    for (gw, &xi) in gw_row.iter_mut().zip(x) {
                        *gw += err * f64::from(xi);
                    }
                }
            }
            for (w, &gw) in weights.iter_mut().zip(&grad_w) {
                let g = gw * inv_n + l2_reg * *w;
                *w -= learning_rate * g;
            }
            for (b, &gb) in bias.iter_mut().zip(&grad_b) {
                *b -= learning_rate * (gb * inv_n);
            }
        }
        Self::linear(model, in_dim, tags, HeadMode::MultiLabel, weights, bias)
    }
}
