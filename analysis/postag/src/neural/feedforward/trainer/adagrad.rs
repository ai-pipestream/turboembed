//! Weight initialization and the mini-batch Adagrad updates (`FeedforwardPOSTrainer`).

use crate::neural::{JavaRng, NeuralError};

const ADAGRAD_EPS: f64 = 1e-6;

pub(super) fn uniform_f32_matrix(rng: &mut JavaRng, rows: usize, cols: usize, scale: f64) -> Vec<Vec<f32>> {
    let mut m = vec![vec![0.0f32; cols]; rows];
    for row in &mut m {
        for val in row {
            *val = ((rng.next_double() * 2.0 - 1.0) * scale) as f32;
        }
    }
    m
}

pub(super) fn adagrad_matrix(
    w: &mut [Vec<f32>],
    dw: &[Vec<f64>],
    g2: &mut [Vec<f64>],
    batch: f64,
    lr: f64,
    l2: f64,
) -> Result<(), NeuralError> {
    for r in 0..w.len() {
        let w_row = &mut w[r];
        let g_row = &dw[r];
        let a_row = &mut g2[r];
        for c in 0..w_row.len() {
            let g = g_row[c] / batch + l2 * f64::from(w_row[c]);
            a_row[c] += g * g;
            w_row[c] = update_weight(w_row[c], g, a_row[c], lr)?;
        }
    }
    Ok(())
}

pub(super) fn adagrad_vector(
    b: &mut [f32],
    db: &[f64],
    g2: &mut [f64],
    batch: f64,
    lr: f64,
) -> Result<(), NeuralError> {
    for i in 0..b.len() {
        let g = db[i] / batch;
        g2[i] += g * g;
        b[i] = update_weight(b[i], g, g2[i], lr)?;
    }
    Ok(())
}

#[inline]
pub(super) fn update_weight(
    weight: f32,
    gradient: f64,
    accumulator: f64,
    learning_rate: f64,
) -> Result<f32, NeuralError> {
    let updated = (f64::from(weight) - learning_rate * gradient / (accumulator.sqrt() + ADAGRAD_EPS)) as f32;
    if !gradient.is_finite() || !accumulator.is_finite() || !updated.is_finite() {
        return Err(NeuralError::NonFiniteScore(
            "training produced non-finite values; check learningRate, l2 and word vectors".into(),
        ));
    }
    Ok(updated)
}
