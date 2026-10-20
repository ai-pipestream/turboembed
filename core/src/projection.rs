//! A bundle's linear token-projection head on the host: the linear layer
//! (`dense`, with optional bias) named by `manifest.projection` and
//! `artifacts[].tensor_names` (`projection_weight` and optional
//! `projection_bias`), plus optional per-row L2 normalization.

use crate::bundle::Bundle;
use crate::head::{Linear, read_f32};
use crate::manifest::Format;
use crate::model::{place, projection_tensors};
use crate::status::{BUNDLE_NO_ARTIFACT, Error, INVALID_ARGUMENT, INVALID_SHAPE, Result, invalid};

/// A linear projection head `y = x * W^T (+ b)` loaded from a bundle's
/// raw-weights artifact or constructed in memory.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionHead {
    pub hidden: usize,
    pub dim: usize,
    pub has_bias: bool,
    pub linear: Linear,
}

impl ProjectionHead {
    /// Construct a projection head from row-major `weight` (`[dim, hidden]`)
    /// and optional `bias` (`[dim]`). When `bias` is `None`, a zero bias vector
    /// of length `dim` is used and `has_bias` is `false`.
    pub fn new(hidden: usize, dim: usize, weight: Vec<f32>, bias: Option<Vec<f32>>) -> Result<Self> {
        let has_bias = bias.is_some();
        let bias = bias.unwrap_or_else(|| vec![0.0f32; dim]);
        let linear = Linear::new(hidden, dim, weight, bias)?;
        Ok(Self { hidden, dim, has_bias, linear })
    }

    /// Validate that `hidden`, `dim`, and the inner `Linear` layer agree and
    /// contain only finite values.
    pub fn validate(&self) -> Result<()> {
        if self.hidden == 0 || self.dim == 0 {
            return Err(Error::new(
                INVALID_ARGUMENT,
                format!("projection dimensions [{}, {}] must be positive", self.dim, self.hidden),
            ));
        }
        if self.linear.in_dim != self.hidden || self.linear.out_dim != self.dim {
            return Err(Error::new(
                INVALID_SHAPE,
                format!(
                    "projection linear dimensions [{}, {}] do not match head [{}, {}]",
                    self.linear.out_dim, self.linear.in_dim, self.dim, self.hidden
                ),
            ));
        }
        self.linear.validate("projection")
    }

    /// Load the projection head from `bundle` if `manifest.projection` is
    /// present, or return `Ok(None)` when the bundle has no projection head.
    pub fn load_optional(bundle: &Bundle) -> Result<Option<Self>> {
        if bundle.manifest.projection.is_none() {
            return Ok(None);
        }
        Self::load(bundle).map(Some)
    }

    /// Load the projection head from the first `FORMAT_SAFETENSORS` artifact
    /// in `bundle`.
    pub fn load(bundle: &Bundle) -> Result<Self> {
        let index =
            bundle.manifest.artifacts.iter().position(|a| a.format == Format::Safetensors).ok_or_else(|| {
                Error::new(BUNDLE_NO_ARTIFACT, "no FORMAT_SAFETENSORS artifact carries the projection weights")
            })?;
        Self::load_artifact(bundle, index)
    }

    /// Load the projection head from artifact `index` of `bundle`, verifying
    /// its files and checking tensor shapes, dtypes and alignment.
    pub fn load_artifact(bundle: &Bundle, index: usize) -> Result<Self> {
        let m = &bundle.manifest;
        let art = m.artifacts.get(index).ok_or_else(|| {
            Error::new(INVALID_ARGUMENT, format!("artifact {index}: bundle has {} artifacts", m.artifacts.len()))
        })?;
        if art.format != Format::Safetensors {
            return Err(invalid(format!(
                "manifest.json: artifacts[{index}].format: projection weights require FORMAT_SAFETENSORS"
            )));
        }
        let p =
            m.projection.as_ref().ok_or_else(|| invalid("manifest.json: projection: bundle has no projection head"))?;
        let a = m
            .architecture
            .as_ref()
            .ok_or_else(|| invalid("manifest.json: architecture: required for a projection head"))?;

        let expected = projection_tensors(index, art, a, p)?;
        let files = art.files.iter().map(|f| bundle.read_verified_aligned(f)).collect::<Result<Vec<_>>>()?;
        let (placed, _) = place(art, &files, &expected)?;

        let weight = read_f32(&files, &placed[0]);
        let bias = placed.get(1).map(|b| read_f32(&files, b));
        Self::new(a.hidden as usize, p.dim as usize, weight, bias)
    }

    /// Project `n` hidden states (`[n, hidden]`) to `[n, dim]`, optionally
    /// L2-normalizing each output row in `f64` accumulation.
    pub fn project(&self, states: &[f32], l2: bool) -> Result<Vec<f32>> {
        let (_, total) = self.check_states(states)?;
        let mut out = vec![0.0f32; total];
        self.project_into(states, l2, &mut out)?;
        Ok(out)
    }

    /// Project `states` (`[n, hidden]`) into `out` (`[n, dim]`), optionally
    /// L2-normalizing each output row in `f64` accumulation.
    pub fn project_into(&self, states: &[f32], l2: bool, out: &mut [f32]) -> Result<()> {
        let (rows, total) = self.check_states(states)?;
        if out.len() != total {
            return Err(Error::new(
                INVALID_SHAPE,
                format!("projection output has {} values, expected {total} ({rows}x{})", out.len(), self.dim),
            ));
        }
        for (state, dst) in states.chunks_exact(self.hidden).zip(out.chunks_exact_mut(self.dim)) {
            if !state.iter().all(|v| v.is_finite()) {
                return Err(Error::new(INVALID_ARGUMENT, "hidden state contains non-finite value"));
            }
            self.linear.apply(state, dst);
            if l2 {
                l2_normalize_row(dst);
            }
        }
        Ok(())
    }

    fn check_states(&self, states: &[f32]) -> Result<(usize, usize)> {
        self.validate()?;
        if !states.len().is_multiple_of(self.hidden) {
            return Err(Error::new(
                INVALID_SHAPE,
                format!("states length {} is not a multiple of hidden width {}", states.len(), self.hidden),
            ));
        }
        let rows = states.len() / self.hidden;
        let total = rows.checked_mul(self.dim).ok_or_else(|| {
            Error::new(INVALID_SHAPE, format!("projection output dimensions [{rows}, {}] overflow usize", self.dim))
        })?;
        Ok((rows, total))
    }
}

fn l2_normalize_row(row: &mut [f32]) {
    let norm = row.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    if norm > 0.0 {
        let inv = (1.0 / norm) as f32;
        for x in row.iter_mut() {
            *x *= inv;
        }
    }
}
