//! Select thresholds on development using the deployed scoring configuration.
use super::*;

pub(super) fn tune_thresholds(
    annotator: &IncivilityAnnotator,
    p: &Prepared<'_>,
    vectors: &[f32],
    parent_lookup: Option<&HashMap<String, &[f32]>>,
    dim: usize,
) -> Result<(f64, f64, [f64; INCIVILITY_DIMENSIONS], f64, f64), String> {
    let mut best_uniform = (0.5, f64::NEG_INFINITY);
    let mut best_per_label = [0.5; INCIVILITY_DIMENSIONS];
    let mut best_label_f1 = [f64::NEG_INFINITY; INCIVILITY_DIMENSIONS];
    let severe = IncivilityLabel::SevereToxicity as usize;
    for t in THRESHOLDS {
        let candidate = annotator.clone().with_threshold(t);
        let layer = score_split(&candidate, p, vectors, parent_lookup, dim)?;
        let m = evaluate_incivility_corpus(&p.units, &layer, t);
        if m.micro_f1 > best_uniform.1 {
            best_uniform = (t, m.micro_f1);
        }
        // These scores do not depend on other labels' thresholds. SevereToxicity
        // does, so select it only after the other five thresholds are fixed.
        for (idx, &(_, lm)) in m.per_label.iter().enumerate() {
            if idx != severe && lm.f1 > best_label_f1[idx] {
                best_label_f1[idx] = lm.f1;
                best_per_label[idx] = t;
            }
        }
    }
    for t in THRESHOLDS {
        let mut thresholds = best_per_label;
        thresholds[severe] = t;
        let candidate = annotator.clone().with_thresholds(thresholds);
        let layer = score_split(&candidate, p, vectors, parent_lookup, dim)?;
        let m = evaluate_incivility_corpus_with_thresholds(&p.units, &layer, &thresholds);
        if m.per_label[severe].1.f1 > best_label_f1[severe] {
            best_label_f1[severe] = m.per_label[severe].1.f1;
            best_per_label[severe] = t;
        }
    }
    let selected = annotator.clone().with_thresholds(best_per_label);
    let layer = score_split(&selected, p, vectors, parent_lookup, dim)?;
    let m = evaluate_incivility_corpus_with_thresholds(&p.units, &layer, &best_per_label);
    Ok((best_uniform.0, best_uniform.1, best_per_label, m.macro_f1, m.micro_f1))
}
