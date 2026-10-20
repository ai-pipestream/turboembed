//! Per-label threshold calibration: sweep candidate cut points over a
//! held-out validation set's scores and keep, for each label, the
//! threshold that maximizes the label's F_beta (beta configurable,
//! default 1.0), optionally under a minimum-precision floor. The fitted
//! thresholds replace the model's defaults via
//! [`MultiLabelDoccatModel::with_thresholds`] and serialize with it.

use std::collections::BTreeSet;

use super::error::MultiLabelError;
use super::model::{DEFAULT_THRESHOLD, MultiLabelDoccatModel};

/// Fits per-label decision thresholds on validation scores. The
/// predictions a candidate threshold `t` makes for a label are "every
/// score `>= t`": sweeping the scores themselves as the candidates
/// covers every distinct prediction the threshold can make.
///
/// The F_beta conventions: predicting nothing when the label has no
/// positive is precision 1.0, a label with no positives in the set is
/// recall 1.0, and precision and recall both 0 is an F_beta of 0.0. A
/// label with no positive in the validation set keeps the default
/// threshold, calibration cannot see it.
#[derive(Clone, Debug, PartialEq)]
pub struct ThresholdCalibrator {
    beta: f64,
    min_precision: Option<f64>,
}

impl Default for ThresholdCalibrator {
    /// F_1, no precision floor.
    fn default() -> Self {
        ThresholdCalibrator { beta: 1.0, min_precision: None }
    }
}

impl ThresholdCalibrator {
    /// An F_1 calibrator, no precision floor.
    pub fn new() -> ThresholdCalibrator {
        ThresholdCalibrator::default()
    }

    /// The same calibrator scoring F_beta: recall weighs beta times
    /// precision. `beta` must be positive and finite.
    pub fn with_beta(mut self, beta: f64) -> Result<ThresholdCalibrator, MultiLabelError> {
        if !beta.is_finite() || beta <= 0.0 {
            return Err(MultiLabelError::Invalid(format!("beta {beta}, must be positive and finite")));
        }
        self.beta = beta;
        Ok(self)
    }

    /// The same calibrator restricting candidates to those whose
    /// precision reaches `min_precision` (in the half-open interval
    /// `(0.0, 1.0]`). When no candidate reaches the floor, the fallback
    /// is the candidate of the highest precision, ties toward the
    /// highest threshold.
    pub fn with_min_precision(mut self, min_precision: f64) -> Result<ThresholdCalibrator, MultiLabelError> {
        if !min_precision.is_finite() || min_precision <= 0.0 || min_precision > 1.0 {
            return Err(MultiLabelError::Invalid(format!(
                "min_precision {min_precision}, outside the half-open 0.0..=1.0"
            )));
        }
        self.min_precision = Some(min_precision);
        Ok(self)
    }

    /// The beta F_beta weighs recall by.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// The precision floor, when one is set.
    pub fn min_precision(&self) -> Option<f64> {
        self.min_precision
    }

    /// Fits one threshold per label: `scores[i]` the validation
    /// document scores (every label's, in label order) and `truth[i]`
    /// the labels document `i` truly holds, by index.
    ///
    /// Returns [`MultiLabelError::Invalid`] when the rows are ragged or a
    /// truth index names no label.
    pub fn fit(&self, scores: &[Vec<f64>], truth: &[BTreeSet<usize>]) -> Result<Vec<f64>, MultiLabelError> {
        let k = scores.first().map_or(0, Vec::len);
        if scores.iter().any(|row| row.len() != k) {
            return Err(MultiLabelError::Invalid("validation score rows of differing widths".into()));
        }
        if scores.len() != truth.len() {
            return Err(MultiLabelError::Invalid(format!(
                "{} validation rows of scores for {} of truth",
                scores.len(),
                truth.len()
            )));
        }
        if truth.iter().flatten().any(|&t| t >= k) {
            return Err(MultiLabelError::Invalid("a validation truth index outside the labels".into()));
        }
        Ok((0..k).map(|label| self.fit_label(scores, truth, label)).collect())
    }

    /// The threshold for one label: its validation scores and which rows
    /// hold it, swept as the candidate cut points.
    fn fit_label(&self, scores: &[Vec<f64>], truth: &[BTreeSet<usize>], label: usize) -> f64 {
        let col: Vec<(f64, bool)> = scores.iter().zip(truth).map(|(row, t)| (row[label], t.contains(&label))).collect();
        if !col.iter().any(|&(_, positive)| positive) {
            // No positive in the validation set: no candidate is visible
            // to F_beta, so the default threshold stays, documented in
            // the type's doc.
            return DEFAULT_THRESHOLD;
        }
        let mut candidates: Vec<f64> = col.iter().map(|&(s, _)| s).collect();
        candidates.sort_by(f64::total_cmp);
        candidates.dedup();

        let mut best: Option<(f64, f64)> = None;
        for &t in &candidates {
            let (tp, fp, fn_) = counts_at(&col, t);
            let (precision, _, f) = f_beta(tp, fp, fn_, self.beta);
            if self.min_precision.is_some_and(|floor| precision < floor) {
                continue;
            }
            // The argmax over F_beta; the `>=` keeps the later candidate,
            // so ties break toward the highest threshold.
            if best.is_none_or(|(_, bf)| f >= bf) {
                best = Some((t, f));
            }
        }
        best.map_or_else(|| self.floor_fallback(&col, &candidates), |(t, _)| t)
    }

    /// No candidate met the precision floor: the candidate of the
    /// highest precision, ties toward the highest threshold.
    fn floor_fallback(&self, col: &[(f64, bool)], candidates: &[f64]) -> f64 {
        let mut best: Option<(f64, f64)> = None;
        for &t in candidates {
            let (tp, fp, _) = counts_at(col, t);
            let precision = precision_recall(tp, fp, 0).0;
            if best.is_none_or(|(_, bp)| precision >= bp) {
                best = Some((t, precision));
            }
        }
        best.expect("candidates is not empty: the label has a positive").0
    }

    /// The same model with the fitted thresholds, validated as in
    /// [`MultiLabelDoccatModel::with_thresholds`].
    pub fn calibrate(
        &self,
        model: &MultiLabelDoccatModel,
        scores: &[Vec<f64>],
        truth: &[BTreeSet<usize>],
    ) -> Result<MultiLabelDoccatModel, MultiLabelError> {
        model.clone().with_thresholds(self.fit(scores, truth)?)
    }
}

/// The predicted counts for one label at threshold `t` over its
/// validation column: true positives, false positives, and false
/// negatives, "positive" every score `>= t`.
fn counts_at(col: &[(f64, bool)], t: f64) -> (usize, usize, usize) {
    let mut counts = (0, 0, 0);
    for &(s, positive) in col {
        let predicted = s >= t;
        match (predicted, positive) {
            (true, true) => counts.0 += 1,
            (true, false) => counts.1 += 1,
            (false, true) => counts.2 += 1,
            (false, false) => {}
        }
    }
    counts
}

/// Precision and recall from the counts: precision 1.0 when nothing was
/// predicted, recall 1.0 when the label had no positive.
pub fn precision_recall(tp: usize, fp: usize, fn_: usize) -> (f64, f64) {
    let precision = if tp + fp == 0 { 1.0 } else { tp as f64 / (tp + fp) as f64 };
    let recall = if tp + fn_ == 0 { 1.0 } else { tp as f64 / (tp + fn_) as f64 };
    (precision, recall)
}

/// Precision, recall, and their F_beta (`beta` positive; recall weighs
/// `beta` times precision). Precision and recall both 0 is an F_beta of
/// 0.0.
pub fn f_beta(tp: usize, fp: usize, fn_: usize, beta: f64) -> (f64, f64, f64) {
    debug_assert!(beta.is_finite() && beta > 0.0);
    let (precision, recall) = precision_recall(tp, fp, fn_);
    let b2 = beta * beta;
    let f = if precision + recall == 0.0 { 0.0 } else { (1.0 + b2) * precision * recall / (b2 * precision + recall) };
    (precision, recall, f)
}

/// The macro F1 of thresholding `scores` at `thresholds`: every label's
/// F1 over the set, averaged. A label with no positive and no prediction
/// scores F1 1.0, the conventions of [`f_beta`].
pub fn macro_f1(scores: &[Vec<f64>], truth: &[BTreeSet<usize>], thresholds: &[f64]) -> f64 {
    if scores.is_empty() || thresholds.is_empty() {
        return 0.0;
    }
    let k = thresholds.len();
    let mut sum = 0.0;
    for (label, &threshold) in thresholds.iter().enumerate() {
        let col: Vec<(f64, bool)> = scores
            .iter()
            .zip(truth)
            .map(|(row, t)| (row.get(label).copied().unwrap_or(0.0), t.contains(&label)))
            .collect();
        let (tp, fp, fn_) = counts_at(&col, threshold);
        sum += f_beta(tp, fp, fn_, 1.0).2;
    }
    sum / k as f64
}
