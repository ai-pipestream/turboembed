//! Trainable dense classification over a checked encoder space.
//! Adam fits linear or one-hidden-layer heads; temperature calibration uses
//! held-out rows, and sigmoid heads calibrate per-label F_beta decision
//! thresholds via [`ThresholdCalibrator`] (`turbo_doccat::multilabel`).
//! Model bytes include encoder identity, weights, temperature and any
//! calibrated per-label thresholds.
mod annotate;
mod math;
mod train;
pub use annotate::DenseClassifierAnnotator;
pub use train::{AdamParameters, Architecture, DenseClassifierTrainer, TrainingResult};
pub use turbo_doccat::multilabel::{MultiLabelReport, Prf, ThresholdCalibrator, evaluate as evaluate_multilabel_sets};

use std::collections::BTreeSet;

use crate::heads::{ClassifierHead, HeadMode, HiddenActivation};
use turbo_analysis::{Layer, ModelRef, Tag};
use turbo_analysis_layers::Vectors;

/// Default decision threshold (`0.5`) for each label of an
/// [`OutputKind::Sigmoid`] classifier before [`ThresholdCalibrator`] tuning.
pub const DEFAULT_THRESHOLD: f64 = turbo_doccat::multilabel::model::DEFAULT_THRESHOLD;

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

/// An immutable classifier with a positive temperature and (for
/// [`OutputKind::Sigmoid`]) per-label decision thresholds in `(0.0, 1.0)`,
/// bound to one encoder.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseClassifierModel {
    head: ClassifierHead,
    temperature: f64,
    thresholds: Vec<f64>,
}
impl DenseClassifierModel {
    pub fn new(head: ClassifierHead) -> Result<Self, String> {
        let k = head.tags().len();
        if k == 0 || k >= Tag::NONE.0 as usize {
            return Err("classifier label count must fit below Tag::NONE".into());
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
        let thresholds = match head.mode() {
            HeadMode::SingleLabel => Vec::new(),
            HeadMode::MultiLabel => vec![DEFAULT_THRESHOLD; k],
        };
        Ok(Self { head, temperature: 1., thresholds })
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
    /// Per-label decision thresholds in `(0.0, 1.0)` for [`OutputKind::Sigmoid`],
    /// or an empty slice for [`OutputKind::Softmax`].
    pub fn thresholds(&self) -> &[f64] {
        &self.thresholds
    }
    pub fn with_temperature(mut self, temperature: f64) -> Result<Self, String> {
        if !temperature.is_finite() || temperature <= 0. {
            return Err("temperature must be finite and positive".into());
        }
        self.temperature = temperature;
        Ok(self)
    }
    /// Replaces the per-label decision thresholds of a sigmoid classifier.
    /// Every threshold must be finite and lie in the open interval `(0.0, 1.0)`.
    pub fn with_thresholds(mut self, thresholds: Vec<f64>) -> Result<Self, String> {
        if self.output() != OutputKind::Sigmoid {
            return Err("per-label thresholds apply only to sigmoid classifiers".into());
        }
        let k = self.head.tags().len();
        if thresholds.len() != k {
            return Err(format!("expected {k} thresholds, got {}", thresholds.len()));
        }
        if thresholds.iter().any(|t| !t.is_finite() || *t <= 0.0 || *t >= 1.0) {
            return Err("thresholds must be finite and in the open interval (0.0, 1.0)".into());
        }
        self.thresholds = thresholds;
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
    /// Selects `(best, selected)` from a probability row:
    /// - For [`OutputKind::Softmax`]: the single argmax tag (ties broken toward
    ///   lower index).
    /// - For [`OutputKind::Sigmoid`]: every label `i` whose score reaches
    ///   `self.thresholds()[i]` in label order, with `best` set to the highest
    ///   scoring selected tag (or [`Tag::NONE`] when no label clears its
    ///   threshold).
    pub fn select_tags(&self, probabilities: &[f64]) -> Result<(Tag, Vec<Tag>), String> {
        let k = self.head.tags().len();
        if probabilities.len() != k {
            return Err(format!("expected {k} probabilities, got {}", probabilities.len()));
        }
        if probabilities.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p)) {
            return Err("probabilities must be finite in 0.0..=1.0".into());
        }
        match self.output() {
            OutputKind::Softmax => {
                let best =
                    probabilities.iter().enumerate().fold(0, |b, (i, v)| if *v > probabilities[b] { i } else { b });
                let tag = Tag(best as u16);
                Ok((tag, vec![tag]))
            }
            OutputKind::Sigmoid => {
                let mut selected = Vec::new();
                let mut best = Tag::NONE;
                let mut best_score = 0.0;
                for (i, (&p, &t)) in probabilities.iter().zip(&self.thresholds).enumerate() {
                    if p >= t {
                        let tag = Tag(i as u16);
                        selected.push(tag);
                        if best == Tag::NONE || p > best_score {
                            best = tag;
                            best_score = p;
                        }
                    }
                }
                Ok((best, selected))
            }
        }
    }
    /// Scores `vector` and returns the emitted [`Tag`]s in label order.
    pub fn predicted_tags(&self, vector: &[f32]) -> Result<Vec<Tag>, String> {
        let p = self.probabilities(vector)?;
        Ok(self.select_tags(&p)?.1)
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
    /// Fits per-label F_beta decision thresholds on a held-out validation set
    /// using [`ThresholdCalibrator`] (`turbo_doccat::multilabel`).
    pub fn calibrate_thresholds<L: Layer>(
        &self,
        vectors: &Vectors<L>,
        truth: &[BTreeSet<usize>],
        calibrator: &ThresholdCalibrator,
    ) -> Result<Self, String> {
        if self.output() != OutputKind::Sigmoid {
            return Err("per-label thresholds apply only to sigmoid classifiers".into());
        }
        self.check_vectors(vectors)?;
        if truth.len() != vectors.len() {
            return Err(format!("{} validation vectors for {} truth sets", vectors.len(), truth.len()));
        }
        let mut scores = Vec::with_capacity(vectors.len());
        for x in vectors.data.chunks_exact(vectors.dim as usize) {
            scores.push(self.probabilities(x)?);
        }
        let fitted = calibrator.fit(&scores, truth).map_err(|e| e.to_string())?;
        self.clone().with_thresholds(fitted)
    }
    /// Fits per-label F_beta decision thresholds from target probability rows
    /// in `[0.0, 1.0]` (treating `target >= 0.5` as positive).
    pub fn calibrate_thresholds_from_targets<L: Layer>(
        &self,
        vectors: &Vectors<L>,
        targets: &[Vec<f64>],
        calibrator: &ThresholdCalibrator,
    ) -> Result<Self, String> {
        if self.output() != OutputKind::Sigmoid {
            return Err("per-label thresholds apply only to sigmoid classifiers".into());
        }
        self.check_vectors(vectors)?;
        math::validate_targets(targets, vectors.len(), self.head.tags().len(), OutputKind::Sigmoid)?;
        let truth: Vec<BTreeSet<usize>> = targets
            .iter()
            .map(|row| row.iter().enumerate().filter_map(|(i, &y)| (y >= DEFAULT_THRESHOLD).then_some(i)).collect())
            .collect();
        self.calibrate_thresholds(vectors, &truth, calibrator)
    }
    /// Evaluates a sigmoid classifier on `vectors` against `gold` positive
    /// label sets using its per-label thresholds, returning micro, macro, and
    /// sample precision/recall/F1 and exact match ([`MultiLabelReport`]).
    pub fn evaluate_multilabel<L: Layer>(
        &self,
        vectors: &Vectors<L>,
        gold: &[BTreeSet<usize>],
    ) -> Result<MultiLabelReport, String> {
        if self.output() != OutputKind::Sigmoid {
            return Err("multi-label evaluation applies only to sigmoid classifiers".into());
        }
        self.check_vectors(vectors)?;
        if gold.len() != vectors.len() {
            return Err(format!("{} evaluation vectors for {} gold sets", vectors.len(), gold.len()));
        }
        let mut predicted = Vec::with_capacity(vectors.len());
        for x in vectors.data.chunks_exact(vectors.dim as usize) {
            let p = self.probabilities(x)?;
            let (_, tags) = self.select_tags(&p)?;
            predicted.push(tags.into_iter().map(|t| t.0 as usize).collect());
        }
        evaluate_multilabel_sets(&predicted, gold, self.head.tags().len()).map_err(|e| e.to_string())
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
    /// Versioned wrapper around THEAD001. Uses `TDENSE01` when thresholds are
    /// at their default (`0.5`) or unused (`Softmax`), and `TDENSE02` when
    /// sigmoid per-label thresholds have been calibrated.
    pub fn to_bytes(&self) -> Vec<u8> {
        let custom_thresholds =
            self.output() == OutputKind::Sigmoid && self.thresholds.iter().any(|&t| t != DEFAULT_THRESHOLD);
        if !custom_thresholds {
            let mut bytes = b"TDENSE01".to_vec();
            bytes.extend(self.temperature.to_le_bytes());
            bytes.extend(self.head.to_bytes());
            bytes
        } else {
            let mut bytes = b"TDENSE02".to_vec();
            bytes.extend(self.temperature.to_le_bytes());
            bytes.extend((self.thresholds.len() as u32).to_le_bytes());
            for &t in &self.thresholds {
                bytes.extend(t.to_le_bytes());
            }
            bytes.extend(self.head.to_bytes());
            bytes
        }
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 16 {
            return Err("invalid dense classifier header".into());
        }
        let t = f64::from_le_bytes(bytes[8..16].try_into().unwrap());
        match &bytes[..8] {
            b"TDENSE01" => Self::new(ClassifierHead::from_bytes(&bytes[16..])?)?.with_temperature(t),
            b"TDENSE02" => {
                if bytes.len() < 20 {
                    return Err("truncated TDENSE02 threshold count".into());
                }
                let k = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
                let thresh_bytes = k.checked_mul(8).ok_or("TDENSE02 threshold count overflow")?;
                let head_offset = 20usize.checked_add(thresh_bytes).ok_or("TDENSE02 offset overflow")?;
                if bytes.len() < head_offset {
                    return Err("truncated TDENSE02 thresholds".into());
                }
                let mut thresholds = Vec::with_capacity(k);
                for &chunk in bytes[20..head_offset].as_chunks::<8>().0 {
                    thresholds.push(f64::from_le_bytes(chunk));
                }
                Self::new(ClassifierHead::from_bytes(&bytes[head_offset..])?)?
                    .with_temperature(t)?
                    .with_thresholds(thresholds)
            }
            _ => Err("invalid dense classifier header".into()),
        }
    }
}
