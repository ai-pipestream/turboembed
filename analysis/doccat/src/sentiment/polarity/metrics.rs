//! Evaluation metrics and majority-class baseline for three-class polarity
//! classification (`positive`, `negative`, `neutral`).

use turbo_analysis_layers::Polarity;

use crate::scorers::PolarityVectorScorer;

use super::fit::PolarityExample;
use super::head::{POLARITY_COUNT, index_polarity, polarity_index};

/// Per-class precision, recall, F1, and gold support for one [`Polarity`] class.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct PolarityClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

/// Aggregate and per-class evaluation metrics across `positive`, `negative`,
/// and `neutral`.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct PolarityMetrics {
    pub accuracy: f64,
    pub macro_f1: f64,
    pub per_class: [PolarityClassMetrics; POLARITY_COUNT],
}

impl PolarityMetrics {
    /// Returns the metrics for a specific [`Polarity`] class.
    pub const fn class(&self, polarity: Polarity) -> PolarityClassMetrics {
        self.per_class[polarity_index(polarity)]
    }
}

/// Evaluates `scorer` on labeled `examples`, returning overall accuracy,
/// three-class macro-F1, and per-class precision/recall/F1.
pub fn evaluate_polarity(
    scorer: &dyn PolarityVectorScorer,
    examples: &[PolarityExample],
) -> Result<PolarityMetrics, String> {
    if examples.is_empty() {
        return Err("cannot evaluate polarity scorer on empty example slice".into());
    }
    let mut gold = Vec::with_capacity(examples.len());
    let mut predicted = Vec::with_capacity(examples.len());
    for ex in examples {
        let scores = scorer.score_polarity(&ex.vector)?;
        gold.push(ex.polarity);
        predicted.push(scores.label());
    }
    evaluate_predictions(&gold, &predicted)
}

/// Computes [`PolarityMetrics`] from aligned `gold` and `predicted` labels.
pub fn evaluate_predictions(gold: &[Polarity], predicted: &[Polarity]) -> Result<PolarityMetrics, String> {
    if gold.is_empty() {
        return Err("cannot evaluate empty polarity predictions".into());
    }
    if gold.len() != predicted.len() {
        return Err(format!("gold length {} != predicted length {}", gold.len(), predicted.len()));
    }
    let mut tp = [0usize; POLARITY_COUNT];
    let mut fp = [0usize; POLARITY_COUNT];
    let mut fn_count = [0usize; POLARITY_COUNT];
    let mut support = [0usize; POLARITY_COUNT];
    let mut correct = 0usize;

    for (&g, &p) in gold.iter().zip(predicted.iter()) {
        let gi = polarity_index(g);
        let pi = polarity_index(p);
        support[gi] += 1;
        if gi == pi {
            tp[gi] += 1;
            correct += 1;
        } else {
            fp[pi] += 1;
            fn_count[gi] += 1;
        }
    }

    let mut per_class = [PolarityClassMetrics::default(); POLARITY_COUNT];
    let mut f1_sum = 0.0;
    let mut active_classes = 0usize;
    for c in 0..POLARITY_COUNT {
        let prec = if tp[c] + fp[c] > 0 { tp[c] as f64 / (tp[c] + fp[c]) as f64 } else { 0.0 };
        let rec = if tp[c] + fn_count[c] > 0 { tp[c] as f64 / (tp[c] + fn_count[c]) as f64 } else { 0.0 };
        let f1 = if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 };
        per_class[c] = PolarityClassMetrics { precision: prec, recall: rec, f1, support: support[c] };
        if support[c] > 0 || tp[c] + fp[c] > 0 {
            f1_sum += f1;
            active_classes += 1;
        }
    }

    let macro_f1 = if active_classes > 0 { f1_sum / active_classes as f64 } else { 0.0 };
    let accuracy = correct as f64 / gold.len() as f64;
    Ok(PolarityMetrics { accuracy, macro_f1, per_class })
}

/// Computes the majority-class baseline on `eval_gold`, choosing the most
/// frequent class in `train_gold` (ties broken in canonical `Positive`,
/// `Negative`, `Neutral` order).
pub fn majority_baseline(
    train_gold: &[Polarity],
    eval_gold: &[Polarity],
) -> Result<(Polarity, PolarityMetrics), String> {
    if train_gold.is_empty() || eval_gold.is_empty() {
        return Err("majority baseline requires non-empty train and eval slices".into());
    }
    let mut counts = [0usize; POLARITY_COUNT];
    for &g in train_gold {
        counts[polarity_index(g)] += 1;
    }
    let mut best_idx = 0usize;
    for idx in 1..POLARITY_COUNT {
        if counts[idx] > counts[best_idx] {
            best_idx = idx;
        }
    }
    let majority = index_polarity(best_idx).unwrap_or(Polarity::Neutral);
    let predicted = vec![majority; eval_gold.len()];
    let metrics = evaluate_predictions(eval_gold, &predicted)?;
    Ok((majority, metrics))
}
