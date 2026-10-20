//! Multi-label evaluation: micro-, macro-, and sample-averaged
//! precision, recall, and F1, plus Exact Match (subset accuracy), of
//! predicted label sets against gold label sets.
//!
//! The division-by-zero conventions are the crate's calibrator
//! conventions ([`f_beta`]): predicting nothing is precision 1.0 (no
//! false calls were made), a label or sample with no gold positive is
//! recall 1.0 (nothing to find), and precision and recall both 0 is an
//! F of 0.0. So a run whose predictions are all empty over nonempty gold
//! scores precision 1.0 and recall 0.0, and a run over documents that
//! hold no label at all, predicting none, scores 1.0 everywhere.

use std::collections::BTreeSet;

use super::calibrate::{f_beta, precision_recall};
use super::error::MultiLabelError;

/// Precision, recall, and F1 together.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prf {
    /// The fraction of predicted labels that are gold, 1.0 when nothing
    /// was predicted.
    pub precision: f64,
    /// The fraction of gold labels that were predicted, 1.0 when there
    /// was nothing to find.
    pub recall: f64,
    /// The harmonic mean of the two, 0.0 when both are 0.
    pub f1: f64,
}

/// One multi-label evaluation: the micro average (every label's counts
/// summed), the macro average (every label's measure averaged, the
/// labels with no predicted and no gold contribution included at their
/// convention values), the sample average (each document's measure
/// averaged), and exact match, the fraction of documents whose
/// prediction equals their gold set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiLabelReport {
    /// The counts summed over every label before measuring.
    pub micro: Prf,
    /// Every label's measure, averaged over `0..num_labels`.
    pub r#macro: Prf,
    /// Every document's measure, averaged over the documents.
    pub samples: Prf,
    /// The fraction of documents whose predicted set equals the gold set.
    pub exact_match: f64,
}

/// Evaluates `predicted` against `gold`, one label set per document, over
/// the `num_labels`-label space.
///
/// Returns [`MultiLabelError::Invalid`] when the two slices differ in
/// length, when either is empty (there is nothing to average over), when
/// `num_labels` is 0, or when a set holds an index outside the label
/// space.
pub fn evaluate(
    predicted: &[BTreeSet<usize>],
    gold: &[BTreeSet<usize>],
    num_labels: usize,
) -> Result<MultiLabelReport, MultiLabelError> {
    if predicted.len() != gold.len() {
        return Err(MultiLabelError::Invalid(format!("{} predictions for {} gold sets", predicted.len(), gold.len())));
    }
    if predicted.is_empty() {
        return Err(MultiLabelError::Invalid("no documents to evaluate".into()));
    }
    if num_labels == 0 {
        return Err(MultiLabelError::Invalid("no labels to evaluate over".into()));
    }
    let check = |set: &BTreeSet<usize>, what: &str| {
        if set.iter().any(|&i| i >= num_labels) {
            Err(MultiLabelError::Invalid(format!("{what} holds a label outside {num_labels}")))
        } else {
            Ok(())
        }
    };
    for (i, (p, g)) in predicted.iter().zip(gold).enumerate() {
        check(p, &format!("prediction {i}"))?;
        check(g, &format!("gold {i}"))?;
    }

    let mut tp = vec![0usize; num_labels];
    let mut fp = vec![0usize; num_labels];
    let mut fn_ = vec![0usize; num_labels];
    let (mut micro_tp, mut micro_fp, mut micro_fn) = (0, 0, 0);
    let (mut sample_p, mut sample_r, mut sample_f) = (0.0, 0.0, 0.0);
    let mut exact = 0usize;
    for (p, g) in predicted.iter().zip(gold) {
        for &label in p {
            if g.contains(&label) {
                tp[label] += 1;
            } else {
                fp[label] += 1;
            }
        }
        for &label in g {
            if !p.contains(&label) {
                fn_[label] += 1;
            }
        }
        micro_tp += p.intersection(g).count();
        micro_fp += p.difference(g).count();
        micro_fn += g.difference(p).count();

        let sample_prf = sample_prf(p, g);
        sample_p += sample_prf.precision;
        sample_r += sample_prf.recall;
        sample_f += sample_prf.f1;
        exact += usize::from(p == g);
    }

    let n = predicted.len() as f64;
    let k = num_labels as f64;
    let (micro_p, micro_r, micro_f) = f_beta(micro_tp, micro_fp, micro_fn, 1.0);
    let mut macro_p = 0.0;
    let mut macro_r = 0.0;
    let mut macro_f = 0.0;
    for label in 0..num_labels {
        let (p, r, f) = f_beta(tp[label], fp[label], fn_[label], 1.0);
        macro_p += p;
        macro_r += r;
        macro_f += f;
    }
    Ok(MultiLabelReport {
        micro: Prf { precision: micro_p, recall: micro_r, f1: micro_f },
        r#macro: Prf { precision: macro_p / k, recall: macro_r / k, f1: macro_f / k },
        samples: Prf { precision: sample_p / n, recall: sample_r / n, f1: sample_f / n },
        exact_match: exact as f64 / n,
    })
}

/// One document's precision, recall, and F1: its prediction against its
/// gold set, the [`precision_recall`] conventions, F1 0.0 when both are
/// 0 (a nonempty prediction over an empty gold, or none over nonempty).
fn sample_prf(predicted: &BTreeSet<usize>, gold: &BTreeSet<usize>) -> Prf {
    let tp = predicted.intersection(gold).count();
    let (precision, recall) = precision_recall(tp, predicted.len() - tp, gold.len() - tp);
    let f1 = if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) };
    Prf { precision, recall, f1 }
}
