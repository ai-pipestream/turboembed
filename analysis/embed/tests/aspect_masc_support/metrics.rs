use std::collections::BTreeMap;

use turbo_analysis::{Ix, Layer};
use turbo_analysis_layers::{AspectTarget, Aspects, Polarity};

use super::corpus::CorpusDoc;

#[derive(Debug)]
pub struct Metrics {
    pub aspect_tp: usize,
    pub aspect_fp: usize,
    pub aspect_fn: usize,
    pub polarity_correct: usize,
    pub polarity_total: usize,
}

impl Metrics {
    pub fn f1(&self) -> f64 {
        let denominator = 2 * self.aspect_tp + self.aspect_fp + self.aspect_fn;
        if denominator == 0 { 0.0 } else { 2.0 * self.aspect_tp as f64 / denominator as f64 }
    }

    pub fn polarity_accuracy(&self) -> f64 {
        if self.polarity_total == 0 { 0.0 } else { self.polarity_correct as f64 / self.polarity_total as f64 }
    }

    pub fn add(&mut self, other: &Metrics) {
        self.aspect_tp += other.aspect_tp;
        self.aspect_fp += other.aspect_fp;
        self.aspect_fn += other.aspect_fn;
        self.polarity_correct += other.polarity_correct;
        self.polarity_total += other.polarity_total;
    }
}

pub fn empty() -> Metrics {
    Metrics { aspect_tp: 0, aspect_fp: 0, aspect_fn: 0, polarity_correct: 0, polarity_total: 0 }
}

fn gold(doc: &CorpusDoc) -> BTreeMap<(usize, usize), Polarity> {
    doc.aspects
        .iter()
        .map(|(range, _, label)| {
            let polarity = match *label {
                "positive" => Polarity::Positive,
                "negative" => Polarity::Negative,
                _ => Polarity::Neutral,
            };
            ((range.start.get(), range.end.get()), polarity)
        })
        .collect()
}

fn predicted(aspects: &Aspects) -> BTreeMap<(usize, usize), Polarity> {
    (0..aspects.len() as u32)
        .filter_map(|i| {
            let ix = Ix::new(i);
            let AspectTarget::Term { tokens, .. } = aspects.target(ix) else { return None };
            Some(((tokens.start.get(), tokens.end.get()), aspects.polarity(ix).label()))
        })
        .collect()
}

pub fn score(doc: &CorpusDoc, predicted_aspects: &Aspects) -> Metrics {
    let gold = gold(doc);
    let predicted = predicted(predicted_aspects);
    let mut out = empty();
    out.aspect_tp = predicted.keys().filter(|range| gold.contains_key(*range)).count();
    out.aspect_fp = predicted.len() - out.aspect_tp;
    out.aspect_fn = gold.keys().filter(|range| !predicted.contains_key(*range)).count();
    out.polarity_total = out.aspect_tp;
    out.polarity_correct = predicted.iter().filter(|(range, label)| gold.get(*range) == Some(*label)).count();
    out
}

pub fn assert_aligned(doc: &CorpusDoc, aspects: &Aspects) {
    assert_eq!(aspects.tokens_key.id(), turbo_analysis::Key::<turbo_analysis_layers::Tokens>::DEFAULT.id());
    assert_eq!(aspects.sentences_key.id(), turbo_analysis::Key::<turbo_analysis_layers::Sentences>::DEFAULT.id());
    for (range, _, _) in &doc.aspects {
        assert!(range.end.get() <= doc.tokens.len());
    }
}
