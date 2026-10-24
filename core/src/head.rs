//! A bundle's classification head on the host: the optional pooler
//! (`dense` + `tanh`) and linear classifier (`dense`) named by
//! `manifest.classifier` and `artifacts[].tensor_names`, plus numerically
//! stable softmax over the head's logits.

use crate::bundle::{AlignedBytes, Bundle};
use crate::cpu::{Matrix, Mode};
use crate::manifest::{ClassifierKind, Format};
use crate::model::{Placed, classifier_tensors, place};
pub use crate::projection::ProjectionHead;
use crate::status::{BUNDLE_NO_ARTIFACT, Error, INVALID_ARGUMENT, INVALID_SHAPE, Result, invalid};
use crate::{TURBO_DTYPE_F16, TURBO_DTYPE_F32};

/// A host-side linear layer `y = x * W^T + b`, with `weight` stored row-major
/// as `[out_dim, in_dim]` and `bias` as `[out_dim]`.
///
/// Products go through the shared packed `turbo::cpu::Matrix` panels,
/// packed lazily on first use: a layer whose plain weights are read
/// directly (the MLM decoder) never pays for a packed copy. The cache
/// is keyed by a checksum of the current fields, so a caller that
/// mutates `weight` or `bias` sees its values used, exactly as before
/// packing existed; if packing cannot allocate, products fall back
/// to the scalar loop over the plain weights.
pub struct Linear {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    matrix: std::sync::Mutex<Option<(Matrix, u64)>>,
}

impl Clone for Linear {
    fn clone(&self) -> Self {
        Self {
            in_dim: self.in_dim,
            out_dim: self.out_dim,
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            matrix: std::sync::Mutex::new(None),
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

fn validate_linear_parts(what: &str, in_dim: usize, out_dim: usize, weight: &[f32], bias: &[f32]) -> Result<()> {
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
        Ok(Self { in_dim, out_dim, weight, bias, matrix: std::sync::Mutex::new(None) })
    }

    fn checksum(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for v in [self.in_dim as u64, self.out_dim as u64]
            .into_iter()
            .chain(self.weight.iter().chain(&self.bias).map(|v| v.to_bits() as u64))
        {
            h ^= v;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Runs `f` with the packed form of the current fields, packing
    /// or re-packing first when the cache is empty or stale. Returns
    /// false when packing cannot allocate; the caller then uses the
    /// scalar loop over the plain weights.
    fn with_matrix(&self, f: impl FnOnce(&Matrix)) -> bool {
        let sum = self.checksum();
        let mut cache = self.matrix.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if cache.as_ref().is_none_or(|(_, packed_sum)| *packed_sum != sum) {
            match Matrix::from_rows(&[&self.weight], self.in_dim) {
                Ok(matrix) => *cache = Some((matrix, sum)),
                Err(_) => return false,
            }
        }
        f(&cache.as_ref().expect("packed above").0);
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
    /// finiteness hold on this layer.
    pub fn validate(&self, what: &str) -> Result<()> {
        validate_linear_parts(what, self.in_dim, self.out_dim, &self.weight, &self.bias)
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
    fn apply_rows(&self, states: &[f32], rows: usize, out: &mut [f32]) {
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

/// A classification head loaded from a bundle's raw-weights artifact.
#[derive(Debug, Clone, PartialEq)]
pub struct Head {
    pub kind: ClassifierKind,
    pub hidden: usize,
    pub labels: Vec<String>,
    pub pooler: Option<Linear>,
    pub classifier: Linear,
}

impl Head {
    /// Load the classification head from the first `FORMAT_SAFETENSORS`
    /// artifact in `bundle`.
    pub fn load(bundle: &Bundle) -> Result<Self> {
        let index =
            bundle.manifest.artifacts.iter().position(|a| a.format == Format::Safetensors).ok_or_else(|| {
                Error::new(BUNDLE_NO_ARTIFACT, "no FORMAT_SAFETENSORS artifact carries the head weights")
            })?;
        Self::load_artifact(bundle, index)
    }

    /// Load the classification head from artifact `index` of `bundle`,
    /// verifying its files and checking tensor shapes, dtypes and alignment.
    pub fn load_artifact(bundle: &Bundle, index: usize) -> Result<Self> {
        let m = &bundle.manifest;
        let art = m.artifacts.get(index).ok_or_else(|| {
            Error::new(INVALID_ARGUMENT, format!("artifact {index}: bundle has {} artifacts", m.artifacts.len()))
        })?;
        if art.format != Format::Safetensors {
            return Err(invalid(format!(
                "manifest.json: artifacts[{index}].format: head weights require FORMAT_SAFETENSORS"
            )));
        }
        let c = m
            .classifier
            .as_ref()
            .ok_or_else(|| invalid("manifest.json: classifier: bundle has no classification head"))?;
        let a = m
            .architecture
            .as_ref()
            .ok_or_else(|| invalid("manifest.json: architecture: required for a classification head"))?;

        let expected = classifier_tensors(index, art, a, c)?;
        let files = art.files.iter().map(|f| bundle.read_verified_aligned(f)).collect::<Result<Vec<_>>>()?;
        let (placed, _) = place(art, &files, &expected)?;

        let hidden = a.hidden as usize;
        let num_labels = c.labels.len();
        let (pooler, cls_slice) = if placed.len() == 4 {
            let pw = read_f32(&files, &placed[0]);
            let pb = read_f32(&files, &placed[1]);
            (Some(Linear::new(hidden, hidden, pw, pb)?), &placed[2..4])
        } else {
            (None, &placed[0..2])
        };
        let cw = read_f32(&files, &cls_slice[0]);
        let cb = read_f32(&files, &cls_slice[1]);
        let classifier = Linear::new(hidden, num_labels, cw, cb)?;

        Ok(Self { kind: c.kind, hidden, labels: c.labels.clone(), pooler, classifier })
    }

    /// Number of output labels.
    pub fn num_labels(&self) -> usize {
        self.labels.len()
    }

    /// The label for `index`, if in `0..self.num_labels()`.
    pub fn label(&self, index: usize) -> Option<&str> {
        self.labels.get(index).map(String::as_str)
    }

    /// Compute logits (`[n, num_labels]`) for `n` hidden states (`[n, hidden]`).
    ///
    /// For a sequence classifier with a pooler (`BertPooler`), each `hidden`
    /// vector is passed through `tanh(x * W_pool^T + b_pool)` before the
    /// linear classifier; otherwise the linear classifier is applied directly.
    pub fn logits(&self, states: &[f32]) -> Result<Vec<f32>> {
        let (rows, total) = self.check_states(states)?;
        let _ = rows;
        let mut out = vec![0.0f32; total];
        self.logits_into(states, &mut out)?;
        Ok(out)
    }

    /// Compute logits for `states` (`[n, hidden]`) into `out` (`[n, num_labels]`).
    pub fn logits_into(&self, states: &[f32], out: &mut [f32]) -> Result<()> {
        let (rows, total) = self.check_states(states)?;
        let num_labels = self.labels.len();
        if out.len() != total {
            return Err(Error::new(
                INVALID_SHAPE,
                format!("logits output has {} values, expected {total} ({rows}x{num_labels})", out.len()),
            ));
        }
        self.check_finite(states)?;
        self.logits_batched(states, rows, out);
        Ok(())
    }

    /// Every state finite, checked before any product runs so a refused
    /// call leaves the caller's output untouched.
    fn check_finite(&self, states: &[f32]) -> Result<()> {
        if states.iter().all(|v| v.is_finite()) {
            Ok(())
        } else {
            Err(Error::new(INVALID_ARGUMENT, "hidden state contains non-finite value"))
        }
    }

    /// The pooler (when the head has one) and the classifier over all
    /// `rows` states at once, one packed product each.
    fn logits_batched(&self, states: &[f32], rows: usize, out: &mut [f32]) {
        if let Some(pooler) = &self.pooler {
            let mut pooled = vec![0.0f32; rows * self.hidden];
            pooler.apply_rows(states, rows, &mut pooled);
            for v in &mut pooled {
                *v = v.tanh();
            }
            self.classifier.apply_rows(&pooled, rows, out);
        } else {
            self.classifier.apply_rows(states, rows, out);
        }
    }

    /// Compute softmax probabilities (`[n, num_labels]`) in `f64` for `n`
    /// hidden states (`[n, hidden]`).
    pub fn probabilities(&self, states: &[f32]) -> Result<Vec<f64>> {
        let (rows, total) = self.check_states(states)?;
        let _ = rows;
        let mut out = vec![0.0f64; total];
        self.probabilities_into(states, &mut out)?;
        Ok(out)
    }

    /// Compute softmax probabilities for `states` (`[n, hidden]`) into `out`
    /// (`[n, num_labels]`).
    pub fn probabilities_into(&self, states: &[f32], out: &mut [f64]) -> Result<()> {
        let (rows, total) = self.check_states(states)?;
        let num_labels = self.labels.len();
        if out.len() != total {
            return Err(Error::new(
                INVALID_SHAPE,
                format!("probabilities output has {} values, expected {total} ({rows}x{num_labels})", out.len()),
            ));
        }
        self.check_finite(states)?;
        let mut logits = vec![0.0f32; total];
        self.logits_batched(states, rows, &mut logits);
        for (row, dst) in logits.chunks_exact(num_labels).zip(out.chunks_exact_mut(num_labels)) {
            softmax(row, dst)?;
        }
        Ok(())
    }

    fn check_states(&self, states: &[f32]) -> Result<(usize, usize)> {
        if self.hidden == 0 {
            return Err(Error::new(INVALID_ARGUMENT, "head hidden width must be positive"));
        }
        if self.labels.is_empty() {
            return Err(Error::new(INVALID_SHAPE, "head labels must not be empty"));
        }
        if self.classifier.in_dim != self.hidden || self.classifier.out_dim != self.labels.len() {
            return Err(Error::new(
                INVALID_SHAPE,
                format!(
                    "classifier dimensions [{}, {}] do not match head [{}, {}]",
                    self.classifier.out_dim,
                    self.classifier.in_dim,
                    self.labels.len(),
                    self.hidden
                ),
            ));
        }
        self.classifier.validate("classifier")?;
        if let Some(pooler) = &self.pooler {
            if pooler.in_dim != self.hidden || pooler.out_dim != self.hidden {
                return Err(Error::new(
                    INVALID_SHAPE,
                    format!(
                        "pooler dimensions [{}, {}] do not match head hidden width {}",
                        pooler.out_dim, pooler.in_dim, self.hidden
                    ),
                ));
            }
            pooler.validate("pooler")?;
        }
        if !states.len().is_multiple_of(self.hidden) {
            return Err(Error::new(
                INVALID_SHAPE,
                format!("states length {} is not a multiple of hidden width {}", states.len(), self.hidden),
            ));
        }
        let rows = states.len() / self.hidden;
        let total = rows.checked_mul(self.labels.len()).ok_or_else(|| {
            Error::new(INVALID_SHAPE, format!("head output dimensions [{rows}, {}] overflow usize", self.labels.len()))
        })?;
        Ok((rows, total))
    }
}

/// The logistic sigmoid, numerically stable at both tails: the
/// branch keeps `exp` away from overflow for large negative
/// inputs, where the one-branch form yields `1.0 / (1.0 + inf)`
/// through an intermediate infinity (and `NaN` in further
/// arithmetic).
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// [`sigmoid`] in `f64`.
pub fn sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Numerically stable softmax over a single row of `logits`, written as `f64`
/// probabilities into `out`.
pub fn softmax(logits: &[f32], out: &mut [f64]) -> Result<()> {
    if logits.is_empty() {
        return Err(Error::new(INVALID_SHAPE, "softmax logits slice is empty"));
    }
    if logits.len() != out.len() {
        return Err(Error::new(
            INVALID_SHAPE,
            format!("softmax output length {} does not match logits length {}", out.len(), logits.len()),
        ));
    }
    let mut max = f64::NEG_INFINITY;
    for &v in logits {
        if !v.is_finite() {
            return Err(Error::new(INVALID_ARGUMENT, "logit is not finite"));
        }
        let v = v as f64;
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f64;
    for (&v, dst) in logits.iter().zip(out.iter_mut()) {
        let e = (v as f64 - max).exp();
        *dst = e;
        sum += e;
    }
    if sum <= 0.0 || !sum.is_finite() {
        let uniform = 1.0 / (out.len() as f64);
        out.fill(uniform);
    } else {
        for dst in out.iter_mut() {
            *dst /= sum;
        }
    }
    Ok(())
}

pub(crate) fn read_f32(files: &[AlignedBytes], p: &Placed) -> Vec<f32> {
    let bytes = &files[p.file][p.range.clone()];
    if p.dtype == TURBO_DTYPE_F32 {
        return bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    }
    let widen = if p.dtype == TURBO_DTYPE_F16 { f16_to_f32 } else { bf16_to_f32 };
    bytes.as_chunks::<2>().0.iter().map(|b| widen(u16::from_le_bytes(*b))).collect()
}

/// An IEEE half to the single it names exactly.
fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let man = (h & 0x3ff) as u32;
    match exp {
        0 => sign * man as f32 * f32::from_bits(0x3380_0000),
        0x1f => f32::from_bits(((h as u32 & 0x8000) << 16) | 0x7f80_0000 | (man << 13)),
        e => f32::from_bits(((h as u32 & 0x8000) << 16) | ((e as u32 + 112) << 23) | (man << 13)),
    }
}

/// A bfloat16 is the top half of the single it names.
fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[cfg(test)]
#[path = "head/tests.rs"]
mod tests;
