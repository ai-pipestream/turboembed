//! Exact-span NER comparisons with documents as the independent units.
//! Type mappings must be applied explicitly before calling this module.
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use serde::{Deserialize, Serialize};
use turbo_trainer::rng::Rng;

use crate::corpus::Sentence;
use crate::eval::{Comparison, SpanScore, spans};
use crate::{Error, Result};

pub struct Request<'a> {
    pub gold: &'a [Sentence],
    pub candidate: &'a [Vec<String>],
    pub peer: &'a [Vec<String>],
    /// Nonempty, contiguous, exhaustive sentence ranges in source document order.
    /// The caller retains the source identities and their digests with the report.
    pub documents: &'a [Range<usize>],
    /// The complete schema, including types absent from held-out gold.
    pub types: &'a [String],
    pub resamples: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scores {
    pub all: SpanScore,
    pub per_type: BTreeMap<String, SpanScore>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    pub statistical_unit: String,
    pub document_ranges: Vec<[usize; 2]>,
    pub candidate: Scores,
    pub peer: Scores,
    pub all: Comparison,
    pub per_type: BTreeMap<String, Comparison>,
}

#[derive(Clone, Copy, Default)]
struct Tally {
    gold: usize,
    found: usize,
    correct: usize,
}

impl Tally {
    fn add(&mut self, other: Self) {
        self.gold += other.gold;
        self.found += other.found;
        self.correct += other.correct;
    }

    fn score(self) -> SpanScore {
        let ratio = |a, b| if b == 0 { 0.0 } else { a as f64 / b as f64 };
        SpanScore {
            gold: self.gold,
            found: self.found,
            correct: self.correct,
            precision: ratio(self.correct, self.found),
            recall: ratio(self.correct, self.gold),
            f1: ratio(2 * self.correct, self.gold + self.found),
        }
    }
}

/// Pool exact-span counts, then resample whole documents with replacement.
/// Every draw uses the same documents for both systems and every type.
/// Sentences remain separate span domains, including at document boundaries.
/// No quality threshold or minimum-support gate is inferred from the interval.
pub fn compare(request: Request<'_>) -> Result<Report> {
    validate(&request)?;
    let Request { gold, candidate, peer, documents, types, resamples, seed } = request;
    let indices: BTreeMap<_, _> = types.iter().enumerate().map(|(i, kind)| (kind.as_str(), i + 1)).collect();
    let width = types.len() + 1;
    let rows = |predictions: &[Vec<String>]| {
        documents
            .iter()
            .map(|group| {
                let mut counts = vec![Tally::default(); width];
                for i in group.clone() {
                    let want: BTreeSet<_> = spans(&gold[i].labels).into_iter().collect();
                    let got: BTreeSet<_> = spans(&predictions[i]).into_iter().collect();
                    for span in &want {
                        let tally = Tally { gold: 1, correct: usize::from(got.contains(span)), found: 0 };
                        counts[0].add(tally);
                        counts[indices[span.2.as_str()]].add(tally);
                    }
                    for span in &got {
                        let tally = Tally { found: 1, ..Tally::default() };
                        counts[0].add(tally);
                        counts[indices[span.2.as_str()]].add(tally);
                    }
                }
                counts
            })
            .collect::<Vec<_>>()
    };
    let (model, reference) = (rows(candidate), rows(peer));
    let pool = |rows: &[Vec<Tally>], pick: &[usize]| {
        let mut sum = vec![Tally::default(); width];
        for &i in pick {
            for (total, value) in sum.iter_mut().zip(&rows[i]) {
                total.add(*value);
            }
        }
        sum.into_iter().map(Tally::score).collect::<Vec<_>>()
    };
    let original: Vec<_> = (0..documents.len()).collect();
    let (model_scores, reference_scores) = (pool(&model, &original), pool(&reference, &original));
    let mut deltas = vec![Vec::with_capacity(resamples); width];
    let mut rng = Rng::new(seed);
    for _ in 0..resamples {
        let pick: Vec<_> = (0..documents.len()).map(|_| rng.below(documents.len())).collect();
        let (a, b) = (pool(&model, &pick), pool(&reference, &pick));
        for ((samples, a), b) in deltas.iter_mut().zip(a).zip(b) {
            samples.push(a.f1 - b.f1);
        }
    }
    for samples in &mut deltas {
        samples.sort_by(f64::total_cmp);
    }
    let comparison = |i: usize| {
        let at = |q: f64| deltas[i][((resamples - 1) as f64 * q).round() as usize];
        let (model, baseline) = (model_scores[i].f1, reference_scores[i].f1);
        Comparison { model, baseline, delta: model - baseline, low: at(0.025), high: at(0.975), resamples, seed }
    };
    let scores = |v: &[SpanScore]| Scores {
        all: v[0],
        per_type: types.iter().enumerate().map(|(i, kind)| (kind.clone(), v[i + 1])).collect(),
    };
    Ok(Report {
        statistical_unit: "document".into(),
        document_ranges: documents.iter().map(|r| [r.start, r.end]).collect(),
        candidate: scores(&model_scores),
        peer: scores(&reference_scores),
        all: comparison(0),
        per_type: types.iter().enumerate().map(|(i, kind)| (kind.clone(), comparison(i + 1))).collect(),
    })
}

fn validate(request: &Request<'_>) -> Result<()> {
    if request.gold.is_empty() || request.documents.is_empty() || request.resamples == 0 {
        return Err(Error("document comparison needs gold, document groups and positive resamples".into()));
    }
    let mut end = 0;
    for range in request.documents {
        if range.start != end || range.end <= range.start || range.end > request.gold.len() {
            return Err(Error("document groups must be nonempty, contiguous and exhaustive".into()));
        }
        end = range.end;
    }
    if end != request.gold.len() {
        return Err(Error("document groups do not cover every sentence".into()));
    }
    let types: BTreeSet<_> = request.types.iter().map(String::as_str).collect();
    if types.is_empty()
        || types.len() != request.types.len()
        || types.iter().any(|t| t.is_empty() || *t == "O" || t.chars().any(char::is_whitespace))
    {
        return Err(Error("entity schema needs nonempty, unique type names".into()));
    }
    if request.candidate.len() != request.gold.len() || request.peer.len() != request.gold.len() {
        return Err(Error("prediction sentence count differs from gold".into()));
    }
    for (i, sentence) in request.gold.iter().enumerate() {
        if sentence.words.is_empty() || sentence.words.len() != sentence.labels.len() {
            return Err(Error(format!("gold sentence {i} needs one label per word")));
        }
        for (name, labels) in
            [("gold", &sentence.labels), ("candidate", &request.candidate[i]), ("peer", &request.peer[i])]
        {
            if labels.len() != sentence.words.len() {
                return Err(Error(format!("{name} sentence {i} token count differs from gold")));
            }
            let mut previous = None;
            for label in labels {
                if label == "O" {
                    previous = None;
                    continue;
                }
                let Some((prefix, kind)) = label.split_once('-') else {
                    return Err(Error(format!("{name} sentence {i} contains a non-BIO label")));
                };
                if !types.contains(kind) || !matches!(prefix, "B" | "I") || (prefix == "I" && previous != Some(kind)) {
                    return Err(Error(format!("{name} sentence {i} contains invalid BIO or an unmapped type")));
                }
                previous = Some(kind);
            }
        }
    }
    Ok(())
}
