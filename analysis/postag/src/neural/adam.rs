//! Adam optimizer (`AdamOptimizer`) with overflow-safe global L2 norm and gradient clipping.

use crate::neural::NeuralError;

const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;

/// Stateful Adam optimizer over 2D parameter matrices and 1D vectors (wrapped as 1-row matrices).
#[derive(Clone, Debug, Default)]
pub struct AdamOptimizer {
    params: Vec<Vec<Vec<f64>>>,
    rate_multipliers: Vec<f64>,
    grads: Vec<Vec<Vec<f64>>>,
    m: Vec<Vec<Vec<f64>>>,
    v: Vec<Vec<Vec<f64>>>,
}

impl AdamOptimizer {
    /// Creates an empty Adam optimizer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a 2D parameter matrix with a learning-rate multiplier of `1.0`.
    pub fn register_matrix(&mut self, param: Vec<Vec<f64>>) -> usize {
        self.register_matrix_with_multiplier(param, 1.0).expect("1.0 is a valid multiplier")
    }

    /// Registers a 2D parameter matrix with a custom positive finite learning-rate multiplier.
    pub fn register_matrix_with_multiplier(
        &mut self,
        param: Vec<Vec<f64>>,
        rate_multiplier: f64,
    ) -> Result<usize, NeuralError> {
        if !rate_multiplier.is_finite() || rate_multiplier <= 0.0 {
            return Err(NeuralError::InvalidArgument(format!("rateMultiplier must be > 0, got {rate_multiplier}")));
        }
        let g: Vec<Vec<f64>> = param.iter().map(|row| vec![0.0; row.len()]).collect();
        let m_buf: Vec<Vec<f64>> = param.iter().map(|row| vec![0.0; row.len()]).collect();
        let v_buf: Vec<Vec<f64>> = param.iter().map(|row| vec![0.0; row.len()]).collect();
        let idx = self.params.len();
        self.params.push(param);
        self.rate_multipliers.push(rate_multiplier);
        self.grads.push(g);
        self.m.push(m_buf);
        self.v.push(v_buf);
        Ok(idx)
    }

    /// Registers a 1D parameter vector (stored as a single-row matrix) with multiplier `1.0`.
    pub fn register_vector(&mut self, param: Vec<f64>) -> usize {
        self.register_matrix(vec![param])
    }

    /// Returns the parameter matrix at `index`.
    pub fn params(&self, index: usize) -> &[Vec<f64>] {
        &self.params[index]
    }

    /// Returns a mutable reference to the parameter matrix at `index`.
    pub fn params_mut(&mut self, index: usize) -> &mut [Vec<f64>] {
        &mut self.params[index]
    }

    /// Returns the gradient buffer for the parameter matrix at `index`.
    pub fn gradient(&self, index: usize) -> &[Vec<f64>] {
        &self.grads[index]
    }

    /// Returns a mutable reference to the gradient buffer at `index`.
    pub fn gradient_mut(&mut self, index: usize) -> &mut [Vec<f64>] {
        &mut self.grads[index]
    }

    /// Allocates a fresh set of zeroed gradient buffers matching the shapes of all registered parameters.
    pub fn new_gradient_buffers(&self) -> Vec<Vec<Vec<f64>>> {
        self.params.iter().map(|p| p.iter().map(|row| vec![0.0; row.len()]).collect()).collect()
    }

    /// Zeros every element of a set of gradient buffers.
    pub fn zero_buffers(buffers: &mut [Vec<Vec<f64>>]) {
        for g in buffers {
            for row in g {
                row.fill(0.0);
            }
        }
    }

    /// Adds a worker's gradient buffers into the optimizer's own gradient accumulators.
    pub fn absorb(&mut self, other: &[Vec<Vec<f64>>]) {
        for (target, src) in self.grads.iter_mut().zip(other.iter()) {
            for (t_row, s_row) in target.iter_mut().zip(src.iter()) {
                for (t_val, &s_val) in t_row.iter_mut().zip(s_row.iter()) {
                    *t_val += s_val;
                }
            }
        }
    }

    /// Zeros all accumulated gradients in the optimizer.
    pub fn zero(&mut self) {
        Self::zero_buffers(&mut self.grads);
    }

    /// Scales all accumulated gradients by `factor`.
    pub fn scale_gradients(&mut self, factor: f64) {
        for g in &mut self.grads {
            for row in g {
                for val in row {
                    *val *= factor;
                }
            }
        }
    }

    /// Computes the global L2 norm across all registered gradient buffers without intermediate overflow/underflow.
    pub fn global_norm(&self) -> Result<f64, NeuralError> {
        let mut sum_squares = 0.0f64;
        let mut scale = 0.0f64;
        for g in &self.grads {
            for row in g {
                for &value in row {
                    if !value.is_finite() {
                        return Err(NeuralError::NonFiniteScore("gradient must be finite".into()));
                    }
                    sum_squares += value * value;
                    scale = scale.max(value.abs());
                }
            }
        }
        if scale > 0.0 && (!sum_squares.is_finite() || sum_squares < f64::MIN_POSITIVE) {
            return Ok(scale * self.normalized_norm(scale));
        }
        Ok(sum_squares.sqrt())
    }

    /// Clips all accumulated gradients in place so their global L2 norm does not exceed `limit`.
    pub fn clip_gradients(&mut self, limit: f64) -> Result<(), NeuralError> {
        let norm = self.global_norm()?;
        if norm > limit {
            let factor = limit / norm;
            if factor >= f64::MIN_POSITIVE {
                self.scale_gradients(factor);
                return Ok(());
            }
            let mut scale = 0.0f64;
            for g in &self.grads {
                for row in g {
                    for &value in row {
                        scale = scale.max(value.abs());
                    }
                }
            }
            let normalized_norm = self.normalized_norm(scale);
            let scaled_factor = (limit / scale) / normalized_norm;
            if scaled_factor >= f64::MIN_POSITIVE {
                self.scale_gradients(scaled_factor);
            } else {
                let magnitude = limit / normalized_norm;
                for g in &mut self.grads {
                    for row in g {
                        for val in row {
                            *val = (*val / scale) * magnitude;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn normalized_norm(&self, scale: f64) -> f64 {
        let mut sum_squares = 0.0f64;
        for g in &self.grads {
            for row in g {
                for &value in row {
                    let normalized = value / scale;
                    sum_squares += normalized * normalized;
                }
            }
        }
        sum_squares.sqrt()
    }

    /// Runs one bias-corrected Adam update step (`timestep` is 1-based).
    pub fn step(&mut self, learning_rate: f64, timestep: i32) -> Result<(), NeuralError> {
        if timestep <= 0 {
            return Err(NeuralError::InvalidArgument(format!("timestep must be positive, got {timestep}")));
        }
        if !learning_rate.is_finite() || learning_rate < 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "learningRate must be finite and nonnegative, got {learning_rate}"
            )));
        }
        let bias_correction = (1.0 - BETA2.powi(timestep)).sqrt() / (1.0 - BETA1.powi(timestep));

        for p in 0..self.params.len() {
            let scaled_rate = learning_rate * self.rate_multipliers[p];
            if !scaled_rate.is_finite() {
                return Err(NeuralError::NonFiniteScore("scaled learning rate must be finite".into()));
            }
            let weight = &mut self.params[p];
            let gradient = &self.grads[p];
            let first = &mut self.m[p];
            let second = &mut self.v[p];
            for r in 0..weight.len() {
                let p_row = &mut weight[r];
                let g_row = &gradient[r];
                let m_row = &mut first[r];
                let v_row = &mut second[r];
                for i in 0..p_row.len() {
                    let g = g_row[i];
                    let first_moment = BETA1 * m_row[i] + (1.0 - BETA1) * g;
                    let second_moment = BETA2 * v_row[i] + (1.0 - BETA2) * g * g;
                    let updated =
                        p_row[i] - scaled_rate * bias_correction * first_moment / (second_moment.sqrt() + EPS);
                    if !g.is_finite() || !first_moment.is_finite() || !second_moment.is_finite() || !updated.is_finite()
                    {
                        return Err(NeuralError::NonFiniteScore(format!(
                            "Adam update must be finite at parameter {p}, row {r}, component {i}"
                        )));
                    }
                    m_row[i] = first_moment;
                    v_row[i] = second_moment;
                    p_row[i] = updated;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn optimizer_of(values: &[f64]) -> AdamOptimizer {
        let mut opt = AdamOptimizer::new();
        let idx = opt.register_vector(vec![0.0; values.len()]);
        opt.gradient_mut(idx)[0].copy_from_slice(values);
        opt
    }

    #[test]
    fn norm_handles_extreme_scales_without_overflow_or_underflow() {
        let cases = [
            (&[0.0, -0.0][..], 0.0),
            (&[3.0, 4.0][..], 5.0),
            (&[1e200, -2e200][..], 1e200_f64.hypot(2e200)),
            (&[1e-200, -2e-200][..], 1e-200_f64.hypot(2e-200)),
            (&[1e-160, -2e-160][..], 1e-160_f64.hypot(2e-160)),
            (&[f64::from_bits(1)][..], f64::from_bits(1)),
            (&[f64::MAX][..], f64::MAX),
            (&[1e308, 1e308][..], 1e308_f64.hypot(1e308)),
            (&[f64::MAX, f64::MAX][..], f64::INFINITY),
        ];
        for (vals, expected) in cases {
            let actual = optimizer_of(vals).global_norm().unwrap();
            if expected.is_finite() && expected.abs() >= f64::MIN_POSITIVE {
                let tol = 4.0 * expected.abs() * f64::EPSILON;
                assert!((actual - expected).abs() <= tol, "expected {expected}, got {actual}");
            } else {
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn clipping_preserves_direction_at_all_scales() {
        let s2 = 2.0_f64.sqrt();
        let cases: &[(&[f64], f64, &[f64])] = &[
            (&[3.0, 4.0], 2.5, &[1.5, 2.0]),
            (&[3.0, 4.0], 5.0, &[3.0, 4.0]),
            (&[3e200, -4e200], 5.0, &[3.0, -4.0]),
            (&[3e-200, -4e-200], 2.5e-200, &[1.5e-200, -2e-200]),
            (&[1e308, -1e308], 1.0, &[1.0 / s2, -1.0 / s2]),
            (&[f64::MAX, -f64::MAX], 1.0, &[1.0 / s2, -1.0 / s2]),
            (&[f64::MAX], f64::from_bits(1), &[f64::from_bits(1)]),
            (&[f64::MAX], 1e-15, &[1e-15]),
            (&[f64::MAX, f64::MAX, 1e-100], f64::MAX, &[f64::MAX / s2, f64::MAX / s2, 1e-100 / s2]),
            (&[0.0, -0.0], f64::from_bits(1), &[0.0, -0.0]),
        ];
        for &(values, limit, expected) in cases {
            let mut opt = optimizer_of(values);
            opt.clip_gradients(limit).unwrap();
            let actual = &opt.gradient(0)[0];
            for i in 0..actual.len() {
                if expected[i].abs() < f64::MIN_POSITIVE {
                    assert_eq!(actual[i], expected[i]);
                } else {
                    let tol = 4.0 * expected[i].abs() * f64::EPSILON;
                    assert!((actual[i] - expected[i]).abs() <= tol, "expected {}, got {}", expected[i], actual[i]);
                }
            }
        }
    }

    #[test]
    fn clipped_step_and_overflow_rejection() {
        let mut opt = AdamOptimizer::new();
        let idx = opt.register_vector(vec![1.0]);
        opt.gradient_mut(idx)[0][0] = 1e200;
        opt.clip_gradients(1.0).unwrap();
        opt.step(0.1, 1).unwrap();
        let expected = 1.0 - 0.1 / (1.0 + 1e-8 / (1.0 - 0.999_f64).sqrt());
        assert!((opt.params(idx)[0][0] - expected).abs() <= 4.0 * f64::EPSILON);

        let mut bad = AdamOptimizer::new();
        let idx = bad.register_matrix_with_multiplier(vec![vec![0.0]], 1.0).unwrap();
        bad.gradient_mut(idx)[0][0] = 1e200;
        assert!(bad.step(0.01, 1).is_err());
    }
}
