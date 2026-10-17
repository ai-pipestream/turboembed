//! A bundle's classification head on the host: the optional pooler
//! (`dense` + `tanh`) and linear classifier (`dense`) named by
//! `manifest.classifier` and `artifacts[].tensor_names`, plus numerically
//! stable softmax over the head's logits.

use crate::bundle::{AlignedBytes, Bundle};
use crate::manifest::{ClassifierKind, Format};
use crate::model::{Placed, classifier_tensors, place};
use crate::status::{BUNDLE_NO_ARTIFACT, Error, INVALID_ARGUMENT, INVALID_SHAPE, Result, invalid};
use crate::{TURBO_DTYPE_F16, TURBO_DTYPE_F32};

/// A host-side linear layer `y = x * W^T + b`, with `weight` stored row-major
/// as `[out_dim, in_dim]` and `bias` as `[out_dim]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Linear {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
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
        Ok(Self { in_dim, out_dim, weight, bias })
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
        for (dst, (row, &b)) in out.iter_mut().zip(self.weight.chunks_exact(self.in_dim).zip(&self.bias)) {
            let mut sum = b;
            for (&w, &xi) in row.iter().zip(x) {
                sum += w * xi;
            }
            *dst = sum;
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
        let mut pooled = if self.pooler.is_some() { vec![0.0f32; self.hidden] } else { Vec::new() };
        for (state, dst) in states.chunks_exact(self.hidden).zip(out.chunks_exact_mut(num_labels)) {
            if !state.iter().all(|v| v.is_finite()) {
                return Err(Error::new(INVALID_ARGUMENT, "hidden state contains non-finite value"));
            }
            if let Some(pooler) = &self.pooler {
                pooler.apply_tanh(state, &mut pooled);
                self.classifier.apply(&pooled, dst);
            } else {
                self.classifier.apply(state, dst);
            }
        }
        Ok(())
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
        let mut row_logits = vec![0.0f32; num_labels];
        let mut pooled = if self.pooler.is_some() { vec![0.0f32; self.hidden] } else { Vec::new() };
        for (state, dst) in states.chunks_exact(self.hidden).zip(out.chunks_exact_mut(num_labels)) {
            if !state.iter().all(|v| v.is_finite()) {
                return Err(Error::new(INVALID_ARGUMENT, "hidden state contains non-finite value"));
            }
            if let Some(pooler) = &self.pooler {
                pooler.apply_tanh(state, &mut pooled);
                self.classifier.apply(&pooled, &mut row_logits);
            } else {
                self.classifier.apply(state, &mut row_logits);
            }
            softmax(&row_logits, dst)?;
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

fn read_f32(files: &[AlignedBytes], p: &Placed) -> Vec<f32> {
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
mod tests {
    use super::*;

    #[test]
    fn linear_and_head_reject_overflowing_misaligned_and_non_finite_inputs() {
        let err = Linear::new(usize::MAX, 2, vec![], vec![0.0, 0.0]).unwrap_err();
        assert_eq!(err.code, INVALID_SHAPE);

        let cls = Linear::new(2, 2, vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0]).unwrap();
        let head = Head {
            kind: ClassifierKind::Sequence,
            hidden: 2,
            labels: vec!["neg".into(), "pos".into()],
            pooler: Some(cls.clone()),
            classifier: cls,
        };
        assert!(head.logits(&[1.0, 2.0]).is_ok());
        assert_eq!(head.logits(&[f32::NAN, 1.0]).unwrap_err().code, INVALID_ARGUMENT);

        let zero_hidden = Head { hidden: 0, ..head.clone() };
        assert_eq!(zero_hidden.logits(&[]).unwrap_err().code, INVALID_ARGUMENT);

        let bad_cls = Head { hidden: 3, ..head.clone() };
        assert_eq!(bad_cls.logits(&[1.0, 2.0, 3.0]).unwrap_err().code, INVALID_SHAPE);

        // Short and long mutated classifier/pooler weight and bias storage are rejected
        // across logits, logits_into, probabilities, and probabilities_into.
        let mut short_cls = head.clone();
        short_cls.classifier.weight.clear();
        let mut out_f32 = [0.0f32; 2];
        let mut out_f64 = [0.0f64; 2];
        assert_eq!(short_cls.logits(&[1.0, 2.0]).unwrap_err().code, INVALID_SHAPE);
        assert_eq!(short_cls.logits_into(&[1.0, 2.0], &mut out_f32).unwrap_err().code, INVALID_SHAPE);
        assert_eq!(short_cls.probabilities(&[1.0, 2.0]).unwrap_err().code, INVALID_SHAPE);
        assert_eq!(short_cls.probabilities_into(&[1.0, 2.0], &mut out_f64).unwrap_err().code, INVALID_SHAPE);

        let mut long_cls = head.clone();
        long_cls.classifier.weight.extend([1.0, 1.0]);
        long_cls.classifier.bias.push(0.0);
        assert_eq!(long_cls.logits(&[1.0, 2.0]).unwrap_err().code, INVALID_SHAPE);
        assert_eq!(long_cls.logits_into(&[1.0, 2.0], &mut out_f32).unwrap_err().code, INVALID_SHAPE);
        assert_eq!(long_cls.probabilities(&[1.0, 2.0]).unwrap_err().code, INVALID_SHAPE);
        assert_eq!(long_cls.probabilities_into(&[1.0, 2.0], &mut out_f64).unwrap_err().code, INVALID_SHAPE);

        let mut bad_pooler = head;
        bad_pooler.pooler.as_mut().unwrap().bias.pop();
        assert_eq!(bad_pooler.logits(&[1.0, 2.0]).unwrap_err().code, INVALID_SHAPE);
        assert_eq!(bad_pooler.probabilities(&[1.0, 2.0]).unwrap_err().code, INVALID_SHAPE);
    }
}
