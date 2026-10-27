//! Lazy, coherent packed linear projections.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::cpu::{Matrix, Mode};
use crate::status::{Error, INVALID_ARGUMENT, INVALID_SHAPE, Result};

/// A host-side linear layer `y = x * W^T + b`, with `weight` stored row-major
/// as `[out_dim, in_dim]` and `bias` as `[out_dim]`.
///
/// Products use the shared packed `turbo::cpu::Matrix` panels, made lazily
/// on first use. Direct readers such as the MLM decoder do not allocate a
/// packed copy. [`Linear::weight_mut`] and [`Linear::bias_mut`] invalidate
/// it before mutation. Warm products need neither a weight scan nor a lock.
/// If packing cannot allocate, the scalar loop uses the current weights.
pub struct Linear {
    pub in_dim: usize,
    pub out_dim: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
    matrix: OnceLock<Matrix>,
    initialize: Mutex<()>,
    /// The finiteness scan over weight and bias has passed for their
    /// current contents. Cleared by every mutation route, so warm
    /// `validate` calls cost only the shape checks instead of a pass
    /// over what can be tens of millions of weights (the MLM decoder)
    /// on every query. Shape is still checked on every call, since
    /// the dims are public fields.
    finite_checked: AtomicBool,
}

impl Clone for Linear {
    fn clone(&self) -> Self {
        Self {
            in_dim: self.in_dim,
            out_dim: self.out_dim,
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            matrix: OnceLock::new(),
            initialize: Mutex::new(()),
            finite_checked: AtomicBool::new(self.finite_checked.load(Ordering::Acquire)),
        }
    }
}

impl PartialEq for Linear {
    fn eq(&self, other: &Self) -> bool {
        // The packed form is derived from `weight`, so equal layers pack
        // to the same panels.
        self.in_dim == other.in_dim
            && self.out_dim == other.out_dim
            && self.weight == other.weight
            && self.bias == other.bias
    }
}

impl core::fmt::Debug for Linear {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Linear")
            .field("in_dim", &self.in_dim)
            .field("out_dim", &self.out_dim)
            .field("weight", &self.weight)
            .field("bias", &self.bias)
            .finish_non_exhaustive()
    }
}

fn check_linear_shape(what: &str, in_dim: usize, out_dim: usize, weight: &[f32], bias: &[f32]) -> Result<()> {
    if in_dim == 0 || out_dim == 0 {
        return Err(Error::new(INVALID_ARGUMENT, format!("{what} dimensions [{out_dim}, {in_dim}] must be positive")));
    }
    let Some(expected) = out_dim.checked_mul(in_dim) else {
        return Err(Error::new(INVALID_SHAPE, format!("{what} dimensions [{out_dim}, {in_dim}] overflow usize")));
    };
    if weight.len() != expected {
        return Err(Error::new(
            INVALID_SHAPE,
            format!("{what} weight has {} values, expected {expected} ({out_dim}x{in_dim})", weight.len()),
        ));
    }
    if bias.len() != out_dim {
        return Err(Error::new(INVALID_SHAPE, format!("{what} bias has {} values, expected {out_dim}", bias.len())));
    }
    Ok(())
}

fn validate_linear_parts(what: &str, in_dim: usize, out_dim: usize, weight: &[f32], bias: &[f32]) -> Result<()> {
    check_linear_shape(what, in_dim, out_dim, weight, bias)?;
    if !weight.iter().all(|v| v.is_finite()) || !bias.iter().all(|v| v.is_finite()) {
        return Err(Error::new(INVALID_ARGUMENT, format!("{what} contains non-finite weights or biases")));
    }
    Ok(())
}

impl Linear {
    /// Construct a linear layer from row-major `weight` (`[out_dim, in_dim]`)
    /// and `bias` (`[out_dim]`).
    pub fn new(in_dim: usize, out_dim: usize, weight: Vec<f32>, bias: Vec<f32>) -> Result<Self> {
        validate_linear_parts("linear layer", in_dim, out_dim, &weight, &bias)?;
        Ok(Self {
            in_dim,
            out_dim,
            weight,
            bias,
            matrix: OnceLock::new(),
            initialize: Mutex::new(()),
            finite_checked: AtomicBool::new(true),
        })
    }

    /// The plain row-major weights, without creating a packed copy.
    #[inline]
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    /// Changes the plain weights and invalidates the packed copy first.
    /// Call [`Linear::validate`] after editing to check shape and finiteness.
    pub fn weight_mut(&mut self) -> &mut Vec<f32> {
        self.matrix.take();
        self.finite_checked.store(false, Ordering::Release);
        &mut self.weight
    }

    /// The plain biases, without creating a packed copy.
    #[inline]
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// Changes the biases and invalidates the packed copy first.
    /// Call [`Linear::validate`] after editing to check shape and finiteness.
    pub fn bias_mut(&mut self) -> &mut Vec<f32> {
        self.matrix.take();
        self.finite_checked.store(false, Ordering::Release);
        &mut self.bias
    }

    fn with_matrix(&self, f: impl FnOnce(&Matrix)) -> bool {
        if self.matrix.get().is_none() {
            // Serialize only first-use packing. Products read the immutable
            // matrix after this guard is dropped, including on the first call.
            let _guard = self.initialize.lock().unwrap_or_else(|p| p.into_inner());
            if self.matrix.get().is_none() {
                let Ok(matrix) = Matrix::from_rows(&[&self.weight], self.in_dim) else { return false };
                let _ = self.matrix.set(matrix);
            }
        }
        f(self.matrix.get().expect("packed above"));
        true
    }

    fn apply_scalar(&self, x: &[f32], out: &mut [f32]) {
        for (dst, (row, &b)) in out.iter_mut().zip(self.weight.chunks_exact(self.in_dim).zip(&self.bias)) {
            let mut sum = b;
            for (&w, &xi) in row.iter().zip(x) {
                sum += w * xi;
            }
            *dst = sum;
        }
    }

    /// Validate that dimensions, `weight` cardinality, `bias` length, and
    /// finiteness hold on this layer. The finiteness scan runs once per
    /// contents (see `finite_checked`); the shape checks run every call.
    pub fn validate(&self, what: &str) -> Result<()> {
        check_linear_shape(what, self.in_dim, self.out_dim, &self.weight, &self.bias)?;
        if !self.finite_checked.load(Ordering::Acquire) {
            if !self.weight.iter().all(|v| v.is_finite()) || !self.bias.iter().all(|v| v.is_finite()) {
                return Err(Error::new(INVALID_ARGUMENT, format!("{what} contains non-finite weights or biases")));
            }
            self.finite_checked.store(true, Ordering::Release);
        }
        Ok(())
    }

    /// Compute `out = x * W^T + b` for a single vector `x` of length `in_dim`.
    pub fn apply(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.in_dim);
        assert_eq!(out.len(), self.out_dim);
        assert_eq!(self.bias.len(), self.out_dim);
        assert_eq!(self.weight.len(), self.out_dim.checked_mul(self.in_dim).expect("linear weight size overflows"));
        let packed = self.with_matrix(|m| m.apply(x, self.in_dim, 1, Some(&self.bias), out, self.out_dim, Mode::Plain));
        if !packed {
            self.apply_scalar(x, out);
        }
    }

    /// `out = states * W^T + b` for `rows` states of length `in_dim`
    /// (`out` is `[rows, out_dim]`), in one packed product. A row's
    /// result is the one [`Linear::apply`] computes for it alone.
    pub(super) fn apply_rows(&self, states: &[f32], rows: usize, out: &mut [f32]) {
        let packed =
            self.with_matrix(|m| m.apply(states, self.in_dim, rows, Some(&self.bias), out, self.out_dim, Mode::Plain));
        if !packed {
            for (state, dst) in states.chunks_exact(self.in_dim).zip(out.chunks_exact_mut(self.out_dim)) {
                self.apply_scalar(state, dst);
            }
        }
    }

    /// Compute `out = tanh(x * W^T + b)` for a single vector `x` of length `in_dim`.
    pub fn apply_tanh(&self, x: &[f32], out: &mut [f32]) {
        self.apply(x, out);
        for v in out.iter_mut() {
            *v = v.tanh();
        }
    }
}

#[cfg(test)]
mod tests;
