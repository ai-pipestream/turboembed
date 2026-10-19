use super::*;

/// Per-label evaluation metrics (`precision`, `recall`, `f1`) on an
/// [`Incivility`] evaluation set.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct LabelMetrics {
    pub tp: usize,
    pub fp: usize,
    pub fn_count: usize,
    pub tn: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

/// Multi-label evaluation summary across all six [`IncivilityLabel`]
/// dimensions.
#[derive(Clone, Debug, PartialEq)]
pub struct IncivilityMetrics {
    pub per_label: [(IncivilityLabel, LabelMetrics); 6],
    pub macro_precision: f64,
    pub macro_recall: f64,
    pub macro_f1: f64,
    pub micro_precision: f64,
    pub micro_recall: f64,
    pub micro_f1: f64,
    /// Number of positive predictions (`score >= threshold`) that carried at
    /// least one non-empty [`IxRange<Tokens>`] evidence span.
    pub positive_with_evidence: usize,
    /// Total number of positive predictions (`score >= threshold`).
    pub total_positive_predictions: usize,
}

/// Gold multi-label annotation for one unit in [`evaluate_incivility_corpus`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoldIncivilityUnit {
    pub positive_labels: Vec<IncivilityLabel>,
}

/// Evaluates predicted [`Incivility`] items against `gold` at `threshold`,
/// computing per-label, macro, and micro precision/recall/F1 and checking
/// evidence span coverage on all positive predictions.
pub fn evaluate_incivility_corpus(
    gold: &[GoldIncivilityUnit],
    predicted: &Incivility,
    threshold: f64,
) -> IncivilityMetrics {
    assert_eq!(gold.len(), predicted.len(), "gold and predicted unit counts must match");
    let mut per_label = IncivilityLabel::ALL.map(|l| (l, LabelMetrics::default()));
    let mut positive_with_evidence = 0usize;
    let mut total_positive_predictions = 0usize;

    for (u_idx, g) in gold.iter().enumerate() {
        let ix = Ix::<Incivility>::new(u_idx as u32);
        let scores = predicted.scores(ix);
        let ev: Vec<(IncivilityLabel, IxRange<Tokens>)> = predicted.evidence_of(ix).collect();

        for (l_idx, &label) in IncivilityLabel::ALL.iter().enumerate() {
            let want = g.positive_labels.contains(&label);
            let got = scores.get(label) >= threshold;
            if got {
                total_positive_predictions += 1;
                if ev.iter().any(|&(el, r)| el == label && !r.is_empty()) {
                    positive_with_evidence += 1;
                }
            }
            let m = &mut per_label[l_idx].1;
            match (want, got) {
                (true, true) => m.tp += 1,
                (false, true) => m.fp += 1,
                (true, false) => m.fn_count += 1,
                (false, false) => m.tn += 1,
            }
        }
    }

    let mut sum_p = 0.0;
    let mut sum_r = 0.0;
    let mut sum_f1 = 0.0;
    let mut tot_tp = 0usize;
    let mut tot_fp = 0usize;
    let mut tot_fn = 0usize;

    for (_, m) in &mut per_label {
        m.precision = if m.tp + m.fp == 0 { 1.0 } else { (m.tp as f64) / ((m.tp + m.fp) as f64) };
        m.recall = if m.tp + m.fn_count == 0 { 1.0 } else { (m.tp as f64) / ((m.tp + m.fn_count) as f64) };
        m.f1 =
            if m.precision + m.recall == 0.0 { 0.0 } else { 2.0 * m.precision * m.recall / (m.precision + m.recall) };
        sum_p += m.precision;
        sum_r += m.recall;
        sum_f1 += m.f1;
        tot_tp += m.tp;
        tot_fp += m.fp;
        tot_fn += m.fn_count;
    }

    let micro_precision = if tot_tp + tot_fp == 0 { 1.0 } else { (tot_tp as f64) / ((tot_tp + tot_fp) as f64) };
    let micro_recall = if tot_tp + tot_fn == 0 { 1.0 } else { (tot_tp as f64) / ((tot_tp + tot_fn) as f64) };
    let micro_f1 = if micro_precision + micro_recall == 0.0 {
        0.0
    } else {
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
    };

    IncivilityMetrics {
        per_label,
        macro_precision: sum_p / 6.0,
        macro_recall: sum_r / 6.0,
        macro_f1: sum_f1 / 6.0,
        micro_precision,
        micro_recall,
        micro_f1,
        positive_with_evidence,
        total_positive_predictions,
    }
}
