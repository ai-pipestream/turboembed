use std::collections::BTreeMap;

use serde::Serialize;
use turbo_analysis::{Layer, Span};
use turbo_analysis_layers::Entities;

use crate::PrecisionCandidates;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FalsePositive {
    pub start: u32,
    pub end: u32,
    pub label: String,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PrecisionMetrics {
    pub candidates: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub true_positives: usize,
    pub false_negatives: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub false_positives: Vec<FalsePositive>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PrecisionEvaluation {
    pub model_only: PrecisionMetrics,
    pub dictionary_only: PrecisionMetrics,
    pub hybrid: PrecisionMetrics,
}

impl PrecisionMetrics {
    pub fn new(predicted: &Entities, gold: &Entities, candidates: usize) -> Result<Self, String> {
        predicted.check()?;
        gold.check()?;
        if predicted.spans.view() != gold.spans.view() {
            return Err("predicted and gold entities must share one view".into());
        }
        if candidates < predicted.spans.len() {
            return Err("candidate count cannot be below the accepted count".into());
        }
        let mut unmatched = BTreeMap::<(u32, u32, &str), usize>::new();
        for (i, span) in gold.spans.iter() {
            *unmatched.entry(key(span, gold, i.get())).or_default() += 1;
        }
        let mut true_positives = 0;
        let mut false_positives = Vec::new();
        for (i, span) in predicted.spans.iter() {
            let label = predicted.kinds.label(predicted.kind[i.get()]);
            let key = (span.start, span.end, label);
            if let Some(count) = unmatched.get_mut(&key)
                && *count > 0
            {
                *count -= 1;
                true_positives += 1;
            } else {
                false_positives.push(FalsePositive {
                    start: span.start,
                    end: span.end,
                    label: label.to_owned(),
                    score: predicted.score[i.get()],
                });
            }
        }
        let accepted = predicted.spans.len();
        let false_negatives = unmatched.into_values().sum();
        let precision = ratio(true_positives, accepted);
        let recall = ratio(true_positives, true_positives + false_negatives);
        let f1 = if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) };
        Ok(Self {
            candidates,
            accepted,
            rejected: candidates - accepted,
            true_positives,
            false_negatives,
            precision,
            recall,
            f1,
            false_positives,
        })
    }
}

impl PrecisionEvaluation {
    pub fn new(
        model: &Entities,
        dictionary: &Entities,
        hybrid: &PrecisionCandidates,
        gold: &Entities,
    ) -> Result<Self, String> {
        Ok(Self {
            model_only: PrecisionMetrics::new(model, gold, model.spans.len())?,
            dictionary_only: PrecisionMetrics::new(dictionary, gold, dictionary.spans.len())?,
            hybrid: PrecisionMetrics::new(&hybrid.accepted_entities(), gold, hybrid.spans.len())?,
        })
    }
}

fn key(span: Span, entities: &Entities, i: usize) -> (u32, u32, &str) {
    (span.start, span.end, entities.kinds.label(entities.kind[i]))
}

fn ratio(part: usize, whole: usize) -> f64 {
    if whole == 0 { 1.0 } else { part as f64 / whole as f64 }
}
