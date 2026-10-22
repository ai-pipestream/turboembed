//! Per-chunk softmax and multi-chunk score aggregation for
//! `DocumentCategorizerDL` (`opennlp.dl.doccat.scoring.*` and
//! `opennlp.dl.AbstractDL.softmax`).

use super::DlDoccatError;

/// Strategy for combining per-chunk classification probability distributions
/// into a single document-level or span-level score distribution, ported from
/// `opennlp.dl.doccat.scoring.ClassificationScoringStrategy`.
pub trait ClassificationScoringStrategy: Send + Sync {
    /// Calculates the final classification scores from the scores of each
    /// individual document chunk (`scores` has shape `[num_chunks][num_categories]`).
    fn score(&self, scores: &[Vec<f64>]) -> Result<Vec<f64>, DlDoccatError>;

    /// Stable identifier of this strategy for provenance config hashing.
    fn strategy_id(&self) -> &'static str {
        "custom"
    }

    /// Strategy-specific parameter fingerprint folded into [`super::DocumentCategorizerDL`]'s
    /// provenance `config_hash`. Custom strategies with configurable parameters
    /// should override this to distinguish instances with different settings.
    fn strategy_hash(&self) -> u64 {
        0
    }
}

/// Computes the arithmetic mean across all chunks for each category, ported
/// from `opennlp.dl.doccat.scoring.AverageClassificationScoringStrategy`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AverageClassificationScoringStrategy;

impl AverageClassificationScoringStrategy {
    /// Creates an `AverageClassificationScoringStrategy`.
    pub fn new() -> Self {
        Self
    }
}

impl ClassificationScoringStrategy for AverageClassificationScoringStrategy {
    fn score(&self, scores: &[Vec<f64>]) -> Result<Vec<f64>, DlDoccatError> {
        if scores.is_empty() {
            return Err(DlDoccatError::InvalidScores("Scores list must not be null or empty".into()));
        }
        let first_row = &scores[0];
        if first_row.is_empty() {
            return Err(DlDoccatError::InvalidScores("Score rows must contain at least one category".into()));
        }
        let num_categories = first_row.len();
        for score in scores {
            if score.len() != num_categories {
                return Err(DlDoccatError::InvalidScores("All score rows must have the same length".into()));
            }
            for &value in score {
                if !value.is_finite() {
                    return Err(DlDoccatError::InvalidScores(format!("Classification score is not finite: {value}")));
                }
            }
        }
        let count = scores.len() as f64;
        let mut avg = Vec::with_capacity(num_categories);
        for i in 0..num_categories {
            let mut sum = 0.0f64;
            for score in scores {
                sum += score[i];
            }
            avg.push(sum / count);
        }
        Ok(avg)
    }

    fn strategy_id(&self) -> &'static str {
        "average"
    }
}

/// Applies numerically stable softmax to a slice of `f32` logits in `f64`
/// arithmetic, matching `opennlp.dl.AbstractDL.softmax(float[])`.
///
/// Rejects empty slices and any non-finite (`NaN`, `+Infinity`, `-Infinity`)
/// logit value.
pub fn softmax(input: &[f32]) -> Result<Vec<f64>, DlDoccatError> {
    if input.is_empty() {
        return Err(DlDoccatError::InvalidLogits("Logit array must not be null or empty".into()));
    }
    let mut max = f64::NEG_INFINITY;
    for &value in input {
        if !value.is_finite() {
            return Err(DlDoccatError::InvalidLogits(format!("Model produced non-finite logit: {value}")));
        }
        let v = f64::from(value);
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f64;
    let mut z = Vec::with_capacity(input.len());
    for &x in input {
        let val = (f64::from(x) - max).exp();
        sum += val;
        z.push(val);
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(DlDoccatError::InvalidLogits("Softmax normalization sum is not finite and positive".into()));
    }
    for v in &mut z {
        *v /= sum;
    }
    Ok(z)
}
