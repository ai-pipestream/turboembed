//! Trainable dense classification over a checked encoder space.
//! Adam fits linear or one-hidden-layer heads; temperature calibration uses
//! held-out rows. Model bytes include encoder identity, weights and temperature.
mod annotate;
mod math;
mod train;
pub use annotate::DenseClassifierAnnotator;
pub use train::{AdamParameters, Architecture, DenseClassifierTrainer, TrainingResult};

use crate::heads::{ClassifierHead, HeadMode, HiddenActivation};
use turbo_analysis::{Layer, ModelRef, Tag};
use turbo_analysis_layers::Vectors;

/// Independent binary probabilities or mutually exclusive classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputKind {
    Softmax,
    Sigmoid,
}
impl OutputKind {
    pub(super) fn mode(self) -> HeadMode {
        match self {
            Self::Softmax => HeadMode::SingleLabel,
            Self::Sigmoid => HeadMode::MultiLabel,
        }
    }
}

/// An immutable classifier with a positive temperature, bound to one encoder.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseClassifierModel {
    head: ClassifierHead,
    temperature: f64,
}
impl DenseClassifierModel {
    pub fn new(head: ClassifierHead) -> Result<Self, String> {
        let k = head.tags().len();
        if k == 0 || k > u16::MAX as usize {
            return Err("classifier label count must fit u16".into());
        }
        let mut seen = std::collections::HashSet::new();
        for i in 0..k {
            let label = head.tags().label(Tag(i as u16));
            if label.is_empty() || !seen.insert(label) {
                return Err("classifier labels must be unique and nonempty".into());
            }
        }
        if !matches!(head.hidden_activation(), HiddenActivation::None | HiddenActivation::Relu | HiddenActivation::Gelu)
        {
            return Err("dense classifier supports ReLU or GeLU hidden layers".into());
        }
        Ok(Self { head, temperature: 1. })
    }
    pub fn head(&self) -> &ClassifierHead {
        &self.head
    }
    pub fn encoder(&self) -> &ModelRef {
        self.head.model()
    }
    pub fn temperature(&self) -> f64 {
        self.temperature
    }
    pub fn output(&self) -> OutputKind {
        match self.head.mode() {
            HeadMode::SingleLabel => OutputKind::Softmax,
            HeadMode::MultiLabel => OutputKind::Sigmoid,
        }
    }
    pub fn with_temperature(mut self, temperature: f64) -> Result<Self, String> {
        if !temperature.is_finite() || temperature <= 0. {
            return Err("temperature must be finite and positive".into());
        }
        self.temperature = temperature;
        Ok(self)
    }
    pub fn probabilities(&self, vector: &[f32]) -> Result<Vec<f64>, String> {
        if vector.iter().any(|x| !x.is_finite()) {
            return Err("non-finite classifier input".into());
        }
        let mut logits = vec![0.; self.head.tags().len()];
        self.head.logits(vector, &mut logits)?;
        math::probabilities(&logits, self.output(), self.temperature)
    }
    pub(super) fn check_vectors<L: Layer>(&self, vectors: &Vectors<L>) -> Result<(), String> {
        vectors.check()?;
        if vectors.model != *self.encoder() || vectors.dim != self.head.in_dim() {
            return Err("classifier encoder identity or input dimension mismatch".into());
        }
        if !vectors.empty.is_empty() || vectors.data.iter().any(|v| !v.is_finite()) {
            return Err("classifier requires finite, present vectors".into());
        }
        Ok(())
    }
    /// Search 81 log-spaced temperatures in exp(-4)..=exp(4), including 1.
    /// Uses validation loss only; weights and training data are not changed.
    pub fn calibrate_temperature<L: Layer>(&self, vectors: &Vectors<L>, targets: &[Vec<f64>]) -> Result<Self, String> {
        self.check_vectors(vectors)?;
        math::validate_targets(targets, vectors.len(), self.head.tags().len(), self.output())?;
        let mut rows = Vec::with_capacity(vectors.len());
        for x in vectors.data.chunks_exact(vectors.dim as usize) {
            let mut logits = vec![0.; self.head.tags().len()];
            self.head.logits(x, &mut logits)?;
            rows.push(logits);
        }
        let loss = |t| -> Result<f64, String> {
            let mut sum = 0.;
            for (z, y) in rows.iter().zip(targets) {
                sum += math::loss(z, y, self.output(), t)? / rows.len() as f64;
            }
            if !sum.is_finite() {
                return Err("non-finite calibration loss".into());
            }
            Ok(sum)
        };
        let mut best = (self.temperature, loss(self.temperature)?);
        for i in -40..=40 {
            let t = (f64::from(i) / 10.).exp();
            let l = loss(t)?;
            if l < best.1 {
                best = (t, l);
            }
        }
        self.clone().with_temperature(best.0)
    }
    /// Versioned wrapper around THEAD001. No host paths or executable payloads.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = b"TDENSE01".to_vec();
        bytes.extend(self.temperature.to_le_bytes());
        bytes.extend(self.head.to_bytes());
        bytes
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 16 || &bytes[..8] != b"TDENSE01" {
            return Err("invalid dense classifier header".into());
        }
        let t = f64::from_le_bytes(bytes[8..16].try_into().unwrap());
        Self::new(ClassifierHead::from_bytes(&bytes[16..])?)?.with_temperature(t)
    }
}
