//! Pretrained native GLiNER predictions on the open-span recipe's BIO corpora.

use std::collections::BTreeMap;

use turbo_gliner::{Candidate, Decode, ExtractionSchema, MAX_LABELS, Model};

use crate::corpus::Sentence;
use crate::{Error, Result};

fn prompts(labels: &[String], guidelines: Option<&BTreeMap<String, String>>) -> Vec<String> {
    labels.iter().map(|label| guidelines.and_then(|g| g.get(label)).unwrap_or(label).clone()).collect()
}

pub fn schema(
    labels: &[String],
    threshold: f32,
    guidelines: Option<&BTreeMap<String, String>>,
) -> Result<ExtractionSchema> {
    ExtractionSchema::with_prompts(labels.to_vec(), prompts(labels, guidelines), threshold, Decode::Flat)
        .map_err(|error| Error(format!("schema over {} labels: {error}", labels.len())))
}

pub fn digest(labels: &[String], threshold: f32, guidelines: Option<&BTreeMap<String, String>>) -> String {
    ExtractionSchema::digest_parts(labels, &prompts(labels, guidelines), threshold, Decode::Flat)
}

/// Predicts one flat BIO sequence per sentence. Schemas larger than GLiNER's
/// prompt limit are scored in label batches, then decoded together so candidates
/// from different batches still compete on equal footing.
pub fn predict(
    model: &Model,
    sentences: &[Sentence],
    labels: &[String],
    threshold: f32,
    guidelines: Option<&BTreeMap<String, String>>,
) -> Result<Vec<Vec<String>>> {
    let thresholds = vec![threshold; labels.len()];
    decode_candidates(sentences, labels, &candidates(model, sentences, labels, threshold, guidelines)?, &thresholds)
}

pub fn candidates(
    model: &Model,
    sentences: &[Sentence],
    labels: &[String],
    threshold: f32,
    guidelines: Option<&BTreeMap<String, String>>,
) -> Result<Vec<Vec<Candidate>>> {
    if labels.is_empty() {
        return Err(Error("GLiNER evaluation needs at least one entity label".into()));
    }
    let mut out = Vec::with_capacity(sentences.len());
    for sentence in sentences {
        let words: Vec<&str> = sentence.words.iter().map(String::as_str).collect();
        if words.is_empty() {
            out.push(Vec::new());
            continue;
        }
        let batches =
            fit_batches(labels, threshold, guidelines, |schema| model.pack(&words, schema).map(|_| ()).map_err(Error))?;
        let mut found = Vec::new();
        let mut offset = 0;
        for chunk in batches {
            let schema = schema(&chunk, threshold, guidelines)?;
            let packed = model.pack(&words, &schema).map_err(Error)?;
            for mut candidate in model.score_packed(&packed, &schema).map_err(Error)? {
                if candidate.start >= candidate.end || candidate.end > words.len() || candidate.label >= chunk.len() {
                    return Err(Error("GLiNER returned an invalid candidate span or label".into()));
                }
                candidate.label += offset;
                found.push(candidate);
            }
            offset += chunk.len();
        }
        out.push(found);
    }
    Ok(out)
}

pub fn decode_candidates(
    sentences: &[Sentence],
    labels: &[String],
    candidates: &[Vec<Candidate>],
    thresholds: &[f32],
) -> Result<Vec<Vec<String>>> {
    if candidates.len() != sentences.len() || thresholds.len() != labels.len() {
        return Err(Error("GLiNER candidate or threshold count does not match its schema".into()));
    }
    let decode_schema = ExtractionSchema::new(vec!["entity".into()], 0.5, Decode::Flat).map_err(Error)?;
    let mut out = Vec::with_capacity(sentences.len());
    for (sentence, found) in sentences.iter().zip(candidates) {
        let selected =
            turbo_gliner::decode_with_thresholds(found.clone(), &decode_schema, thresholds).map_err(Error)?;
        let mut bio = vec!["O".to_owned(); sentence.words.len()];
        for candidate in selected {
            if candidate.start >= candidate.end || candidate.end > sentence.words.len() {
                return Err(Error("GLiNER candidate span is outside its sentence".into()));
            }
            let kind = labels.get(candidate.label).ok_or_else(|| Error("GLiNER label index is out of range".into()))?;
            bio[candidate.start] = format!("B-{kind}");
            for tag in &mut bio[candidate.start + 1..candidate.end] {
                *tag = format!("I-{kind}");
            }
        }
        out.push(bio);
    }
    Ok(out)
}

fn fit_batches(
    labels: &[String],
    threshold: f32,
    guidelines: Option<&BTreeMap<String, String>>,
    mut check: impl FnMut(&ExtractionSchema) -> Result<()>,
) -> Result<Vec<Vec<String>>> {
    let mut batches = Vec::new();
    let mut current = Vec::new();
    for label in labels {
        current.push(label.clone());
        if current.len() > MAX_LABELS {
            current.pop();
            batches.push(current.clone());
            current = vec![label.clone()];
        }
        let candidate = schema(&current, threshold, guidelines)?;
        if check(&candidate).is_err() {
            if current.len() == 1 {
                return Err(Error("a single label and its guideline exceed the encoder context".into()));
            }
            current.pop();
            batches.push(current.clone());
            current = vec![label.clone()];
            check(&schema(&current, threshold, guidelines)?)?;
        }
    }
    if !current.is_empty() {
        batches.push(current);
    }
    Ok(batches)
}

#[cfg(test)]
fn predict_with(
    sentences: &[Sentence],
    labels: &[String],
    decode_schema: &ExtractionSchema,
    batches: impl Fn(&[&str]) -> Result<Vec<Vec<String>>>,
    mut score: impl FnMut(&[&str], &[String]) -> Result<Vec<Candidate>>,
) -> Result<Vec<Vec<String>>> {
    let mut out = Vec::with_capacity(sentences.len());
    for sentence in sentences {
        let words: Vec<&str> = sentence.words.iter().map(String::as_str).collect();
        if words.is_empty() {
            out.push(Vec::new());
            continue;
        }
        let batches = batches(&words)?;
        let mut candidates = Vec::new();
        let mut offset = 0;
        for chunk in batches {
            for mut candidate in score(&words, &chunk)? {
                if candidate.start >= candidate.end || candidate.end > words.len() || candidate.label >= chunk.len() {
                    return Err(Error("GLiNER returned an invalid candidate span or label".into()));
                }
                candidate.label += offset;
                candidates.push(candidate);
            }
            offset += chunk.len();
        }
        let selected = turbo_gliner::decode(candidates, decode_schema);
        let mut bio = vec!["O".to_owned(); words.len()];
        for candidate in selected {
            let kind = labels.get(candidate.label).ok_or_else(|| Error("GLiNER label index is out of range".into()))?;
            bio[candidate.start] = format!("B-{kind}");
            for tag in &mut bio[candidate.start + 1..candidate.end] {
                *tag = format!("I-{kind}");
            }
        }
        out.push(bio);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(start: usize, end: usize, label: usize, logit: f32) -> Candidate {
        Candidate { start, end, label, logit, score: logit }
    }

    #[test]
    fn label_batches_are_offset_and_decoded_together() {
        let sentences = vec![Sentence { words: vec!["a".into(), "b".into(), "c".into()], labels: vec!["O".into(); 3] }];
        let labels: Vec<String> = (0..35).map(|i| format!("kind{i}")).collect();
        let schema = ExtractionSchema::new(vec!["kind0".into()], 0.5, Decode::Flat).unwrap();
        let batch_list: Vec<Vec<String>> = labels.chunks(MAX_LABELS).map(|chunk| chunk.to_vec()).collect();
        let mut batches = Vec::new();
        let predicted = predict_with(
            &sentences,
            &labels,
            &schema,
            |_| Ok(batch_list.clone()),
            |_, chunk| {
                batches.push(chunk.len());
                let mut found = Vec::new();
                if chunk[0] == "kind0" {
                    found.push(candidate(0, 1, 0, 0.8));
                } else {
                    found.push(candidate(0, 1, 1, 0.9));
                    found.push(candidate(1, 3, 2, 0.7));
                }
                Ok(found)
            },
        )
        .unwrap();
        assert_eq!(batches, [32, 3]);
        assert_eq!(predicted[0], ["B-kind33", "B-kind34", "I-kind34"]);
    }

    #[test]
    fn malformed_candidates_and_empty_sentences_are_handled() {
        let labels = vec!["person".to_owned()];
        let schema = ExtractionSchema::new(labels.clone(), 0.5, Decode::Flat).unwrap();
        let empty = vec![Sentence { words: Vec::new(), labels: Vec::new() }];
        let predicted = predict_with(
            &empty,
            &labels,
            &schema,
            |_| Ok(vec![labels.clone()]),
            |_, _| Err(Error("an empty sentence must not reach the model".into())),
        )
        .unwrap();
        assert_eq!(predicted, [Vec::<String>::new()]);

        let sentence = vec![Sentence { words: vec!["a".into()], labels: vec!["O".into()] }];
        let error = predict_with(
            &sentence,
            &labels,
            &schema,
            |_| Ok(vec![labels.clone()]),
            |_, _| Ok(vec![candidate(0, 2, 0, 0.9)]),
        )
        .unwrap_err();
        assert!(error.0.contains("invalid candidate"));
        let error = decode_candidates(&sentence, &labels, &[vec![candidate(0, 2, 0, 0.9)]], &[0.5]).unwrap_err();
        assert!(error.0.contains("outside its sentence"));
    }

    #[test]
    fn prompt_batches_fit_the_encoder_context() {
        let labels = vec!["aaaa".into(), "bbbb".into(), "cccc".into(), "dddd".into()];
        let batches = fit_batches(&labels, 0.8, None, |schema| {
            let bytes = schema.prompts().iter().map(|prompt| prompt.len() + 1).sum::<usize>();
            if bytes > 10 { Err(Error("too long".into())) } else { Ok(()) }
        })
        .unwrap();
        assert_eq!(batches.iter().map(|batch| batch.join(",")).collect::<Vec<_>>(), ["aaaa,bbbb", "cccc,dddd"]);
        assert!(fit_batches(&labels, 0.8, None, |_| Err(Error("too long".into()))).is_err());
        let many: Vec<String> = (0..39).map(|i| format!("kind{i}")).collect();
        let batches = fit_batches(&many, 0.8, None, |_| Ok(())).unwrap();
        assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), [32, 7]);
    }

    #[test]
    fn full_schema_digest_exceeds_the_model_batch_limit() {
        let labels: Vec<String> = (0..39).map(|i| format!("kind{i}")).collect();
        let guidelines = BTreeMap::from([("kind0".into(), "visual artwork".into())]);
        let plain = digest(&labels, 0.8, None);
        let guided = digest(&labels, 0.8, Some(&guidelines));
        assert_ne!(plain, guided);
        assert_eq!(schema(&labels[..1], 0.8, Some(&guidelines)).unwrap().labels(), ["kind0"]);
    }
}
