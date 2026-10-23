//! Per-label precision, recall and F1 of an [`EmotionLayer`] against gold
//! labels, with macro and micro averages and the share of active emotions
//! that carry evidence.

use turbo_analysis::Ix;
use turbo_analysis_layers::{EMOTION_DIMENSIONS, EmotionDimension, EmotionLayer};

use crate::sentiment::incivility::LabelMetrics;

/// The gold labels of one unit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoldEmotionUnit {
    pub labels: Vec<EmotionDimension>,
}

/// See the module comment.
#[derive(Clone, Debug, PartialEq)]
pub struct EmotionMetrics {
    pub per_label: [(EmotionDimension, LabelMetrics); EMOTION_DIMENSIONS],
    pub macro_precision: f64,
    pub macro_recall: f64,
    pub macro_f1: f64,
    pub micro_precision: f64,
    pub micro_recall: f64,
    pub micro_f1: f64,
    /// Active emotions (neutral aside) with at least one evidence span.
    pub active_with_evidence: usize,
    /// Active emotions, neutral aside.
    pub active: usize,
}

fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 { 1.0 } else { a as f64 / b as f64 }
}

fn f1(p: f64, r: f64) -> f64 {
    if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
}

/// Scores `predicted` against `gold`, one unit each, a label active at or
/// above `threshold`. An empty label has precision and recall 1.
pub fn evaluate_emotion_corpus(
    gold: &[GoldEmotionUnit],
    predicted: &EmotionLayer,
    threshold: f64,
) -> Result<EmotionMetrics, String> {
    if gold.len() != predicted.target.len() {
        return Err(format!("{} gold units, {} predicted", gold.len(), predicted.target.len()));
    }
    if !threshold.is_finite() {
        return Err(format!("threshold {threshold}"));
    }
    let mut per_label = EmotionDimension::ALL.map(|d| (d, LabelMetrics::default()));
    let (mut active, mut active_with_evidence) = (0usize, 0usize);
    for (i, g) in gold.iter().enumerate() {
        let ix = Ix::<EmotionLayer>::new(i as u32);
        let scores = predicted.scores_of(ix);
        for (d, m) in per_label.iter_mut() {
            let (want, got) = (g.labels.contains(d), scores.get(*d) >= threshold);
            if got && *d != EmotionDimension::Neutral {
                active += 1;
                active_with_evidence += usize::from(!predicted.evidence_for_label(ix, *d).is_empty());
            }
            match (want, got) {
                (true, true) => m.tp += 1,
                (false, true) => m.fp += 1,
                (true, false) => m.fn_count += 1,
                (false, false) => m.tn += 1,
            }
        }
    }
    let (mut sp, mut sr, mut sf, mut tp, mut fp, mut fnc) = (0.0, 0.0, 0.0, 0, 0, 0);
    for (_, m) in per_label.iter_mut() {
        m.precision = ratio(m.tp, m.tp + m.fp);
        m.recall = ratio(m.tp, m.tp + m.fn_count);
        m.f1 = f1(m.precision, m.recall);
        (sp, sr, sf) = (sp + m.precision, sr + m.recall, sf + m.f1);
        (tp, fp, fnc) = (tp + m.tp, fp + m.fp, fnc + m.fn_count);
    }
    let n = EMOTION_DIMENSIONS as f64;
    let (micro_precision, micro_recall) = (ratio(tp, tp + fp), ratio(tp, tp + fnc));
    Ok(EmotionMetrics {
        per_label,
        macro_precision: sp / n,
        macro_recall: sr / n,
        macro_f1: sf / n,
        micro_precision,
        micro_recall,
        micro_f1: f1(micro_precision, micro_recall),
        active_with_evidence,
        active,
    })
}
