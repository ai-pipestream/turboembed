//! Per-label GLiNER thresholds fitted on exact-match dev spans.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use turbo_gliner::{Candidate, Decode, ExtractionSchema};

use crate::corpus::Sentence;
use crate::{Error, Result};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Calibration {
    pub version: u32,
    pub schema: String,
    pub default: f32,
    pub thresholds: BTreeMap<String, f32>,
}

impl Calibration {
    pub fn uniform(labels: &[String], schema: String, threshold: f32) -> Self {
        Self {
            version: 1,
            schema,
            default: threshold,
            thresholds: labels.iter().map(|x| (x.clone(), threshold)).collect(),
        }
    }

    pub fn read(bytes: &[u8], labels: &[String], schema: &str) -> Result<Self> {
        let value: Self =
            serde_json::from_slice(bytes).map_err(|e| Error(format!("invalid GLiNER calibration: {e}")))?;
        if value.version != 1 || value.schema != schema || value.thresholds.len() != labels.len() {
            return Err(Error("GLiNER calibration version, schema, or labels do not match".into()));
        }
        if !value.default.is_finite()
            || !(0.0..=1.0).contains(&value.default)
            || labels.iter().any(|x| !value.thresholds.contains_key(x))
            || value.thresholds.values().any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
        {
            return Err(Error("GLiNER calibration must contain one finite threshold in [0,1] per schema label".into()));
        }
        Ok(value)
    }

    pub fn values(&self, labels: &[String]) -> Result<Vec<f32>> {
        labels
            .iter()
            .map(|x| self.thresholds.get(x).copied().ok_or_else(|| Error(format!("no threshold for {x}"))))
            .collect()
    }

    pub fn fit(
        sentences: &[Sentence],
        labels: &[String],
        candidates: &[Vec<Candidate>],
        default: f32,
        schema: String,
    ) -> Result<Self> {
        if sentences.len() != candidates.len() || labels.is_empty() {
            return Err(Error("GLiNER calibration predictions do not match dev sentences and labels".into()));
        }
        let gold: Vec<BTreeMap<usize, BTreeSet<(usize, usize)>>> =
            sentences.iter().map(|s| gold_spans(s, labels)).collect::<Result<_>>()?;
        let decode = ExtractionSchema::new(vec!["entity".into()], default, Decode::Flat).map_err(Error)?;
        let mut values = vec![default; labels.len()];
        let mut current = measure(candidates, &gold, &values, &decode)?;
        for _ in 0..2 {
            let mut changed = false;
            for label in 0..labels.len() {
                let mut best = (values[label], current.per_label[label], current.overall);
                for step in 0..=20 {
                    let threshold = step as f32 / 20.0;
                    values[label] = threshold;
                    let scored = measure(candidates, &gold, &values, &decode)?;
                    let candidate = (threshold, scored.per_label[label], scored.overall);
                    let better = candidate.1 > best.1 + 1e-12
                        || ((candidate.1 - best.1).abs() <= 1e-12
                            && (candidate.2 > best.2 + 1e-12
                                || ((candidate.2 - best.2).abs() <= 1e-12
                                    && ((threshold - default).abs(), -threshold)
                                        < ((best.0 - default).abs(), -best.0))));
                    if better {
                        best = candidate;
                    }
                }
                values[label] = best.0;
                if best.1 > current.per_label[label] + 1e-12 && best.2 + 1e-12 >= current.overall {
                    current = measure(candidates, &gold, &values, &decode)?;
                    changed = true;
                } else {
                    values[label] = current.values[label];
                }
            }
            if !changed {
                break;
            }
        }
        let thresholds = labels.iter().cloned().zip(values).collect();
        Ok(Self { version: 1, schema, default, thresholds })
    }

    pub fn bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec_pretty(self)
            .map(|mut x| {
                x.push(b'\n');
                x
            })
            .map_err(|e| Error(e.to_string()))
    }
}

struct Measurement {
    overall: f64,
    per_label: Vec<f64>,
    values: Vec<f32>,
}

fn measure(
    candidates: &[Vec<Candidate>],
    gold: &[BTreeMap<usize, BTreeSet<(usize, usize)>>],
    values: &[f32],
    decode: &ExtractionSchema,
) -> Result<Measurement> {
    let mut found = vec![0usize; values.len()];
    let mut correct = vec![0usize; values.len()];
    let mut total = vec![0usize; values.len()];
    for (raw, expected) in candidates.iter().zip(gold) {
        let selected = turbo_gliner::decode_with_thresholds(raw.clone(), decode, values).map_err(Error)?;
        for candidate in selected {
            if candidate.label >= values.len() {
                return Err(Error("GLiNER calibration candidate label is outside its schema".into()));
            }
            found[candidate.label] += 1;
            if expected.get(&candidate.label).is_some_and(|x| x.contains(&(candidate.start, candidate.end))) {
                correct[candidate.label] += 1;
            }
        }
        for (label, spans) in expected {
            total[*label] += spans.len();
        }
    }
    let f1 = |c, f, g| if f + g == 0 { 1.0 } else { 2.0 * c as f64 / (f + g) as f64 };
    let per_label = (0..values.len()).map(|i| f1(correct[i], found[i], total[i])).collect();
    let overall = f1(correct.iter().sum::<usize>(), found.iter().sum::<usize>(), total.iter().sum::<usize>());
    Ok(Measurement { overall, per_label, values: values.to_vec() })
}

fn gold_spans(sentence: &Sentence, labels: &[String]) -> Result<BTreeMap<usize, BTreeSet<(usize, usize)>>> {
    if sentence.words.len() != sentence.labels.len() {
        return Err(Error("GLiNER calibration sentence has inconsistent words and labels".into()));
    }
    let mut spans = BTreeMap::new();
    let mut open: Option<(usize, usize)> = None;
    for (at, tag) in sentence.labels.iter().map(String::as_str).chain(["O"]).enumerate() {
        let next = tag.strip_prefix("B-").or_else(|| tag.strip_prefix("I-"));
        let label = next.and_then(|x| labels.iter().position(|y| y == x));
        if tag.starts_with("B-") || next.is_none() || label != open.map(|x| x.0) {
            if let Some((kind, start)) = open.take() {
                spans.entry(kind).or_insert_with(BTreeSet::new).insert((start, at));
            }
            if let (Some(kind), true) = (label, tag.starts_with("B-")) {
                open = Some((kind, at));
            }
        }
    }
    Ok(spans)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fitting_uses_exact_spans_and_binds_the_schema() {
        let sentences =
            vec![Sentence { words: vec!["a".into(), "b".into()], labels: vec!["B-person".into(), "B-place".into()] }];
        let raw = vec![vec![
            Candidate { start: 0, end: 1, label: 0, logit: 0.0, score: 0.5 },
            Candidate { start: 0, end: 2, label: 0, logit: 4.0, score: 0.95 },
            Candidate { start: 1, end: 2, label: 1, logit: 2.0, score: 0.8 },
        ]];
        let fit =
            Calibration::fit(&sentences, &["person".into(), "place".into()], &raw, 0.5, "schema-a".into()).unwrap();
        assert_eq!(fit.thresholds["person"], 0.5);
        assert_eq!(fit.thresholds["place"], 0.5);
        let bytes = fit.bytes().unwrap();
        assert!(Calibration::read(&bytes, &["person".into(), "place".into()], "schema-a").is_ok());
        assert!(Calibration::read(&bytes, &["person".into(), "place".into()], "schema-b").is_err());
    }

    #[test]
    fn fitting_never_reduces_overall_dev_f1() {
        let sentences = vec![Sentence { words: vec!["a".into()], labels: vec!["B-person".into()] }];
        let raw = vec![vec![
            Candidate { start: 0, end: 1, label: 0, logit: 1.0, score: 0.75 },
            Candidate { start: 0, end: 1, label: 1, logit: 2.0, score: 0.8 },
        ]];
        let fit = Calibration::fit(&sentences, &["person".into(), "place".into()], &raw, 0.8, "schema".into()).unwrap();
        assert_eq!(fit.thresholds["person"], 0.7);
        assert_eq!(fit.thresholds["place"], 0.8);
    }
}
