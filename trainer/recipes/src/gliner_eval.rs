//! Pretrained native GLiNER predictions on the open-span recipe's BIO corpora.

use turbo_gliner::{Candidate, Decode, ExtractionSchema, MAX_LABELS, Model};

use crate::corpus::Sentence;
use crate::{Error, Result};

/// Predicts one flat BIO sequence per sentence. Schemas larger than GLiNER's
/// prompt limit are scored in label batches, then decoded together so candidates
/// from different batches still compete on equal footing.
pub fn predict(model: &Model, sentences: &[Sentence], labels: &[String], threshold: f32) -> Result<Vec<Vec<String>>> {
    if labels.is_empty() {
        return Err(Error("GLiNER evaluation needs at least one entity label".into()));
    }
    let decode_schema = ExtractionSchema::new(vec![labels[0].clone()], threshold, Decode::Flat).map_err(Error)?;
    predict_with(sentences, labels, &decode_schema, |words, chunk| {
        let schema = ExtractionSchema::new(chunk.to_vec(), threshold, Decode::Flat).map_err(Error)?;
        let packed = model.pack(words, &schema).map_err(Error)?;
        model.score_packed(&packed, &schema).map_err(Error)
    })
}

fn predict_with(
    sentences: &[Sentence],
    labels: &[String],
    decode_schema: &ExtractionSchema,
    mut score: impl FnMut(&[&str], &[String]) -> Result<Vec<Candidate>>,
) -> Result<Vec<Vec<String>>> {
    let mut out = Vec::with_capacity(sentences.len());
    for sentence in sentences {
        let words: Vec<&str> = sentence.words.iter().map(String::as_str).collect();
        if words.is_empty() {
            out.push(Vec::new());
            continue;
        }
        let mut candidates = Vec::new();
        for (batch, chunk) in labels.chunks(MAX_LABELS).enumerate() {
            let offset = batch * MAX_LABELS;
            for mut candidate in score(&words, chunk)? {
                if candidate.start >= candidate.end || candidate.end > words.len() || candidate.label >= chunk.len() {
                    return Err(Error("GLiNER returned an invalid candidate span or label".into()));
                }
                candidate.label += offset;
                candidates.push(candidate);
            }
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
        let mut batches = Vec::new();
        let predicted = predict_with(&sentences, &labels, &schema, |_, chunk| {
            batches.push(chunk.len());
            let mut found = Vec::new();
            if chunk[0] == "kind0" {
                found.push(candidate(0, 1, 0, 0.8));
            } else {
                found.push(candidate(0, 1, 1, 0.9));
                found.push(candidate(1, 3, 2, 0.7));
            }
            Ok(found)
        })
        .unwrap();
        assert_eq!(batches, [32, 3]);
        assert_eq!(predicted[0], ["B-kind33", "B-kind34", "I-kind34"]);
    }

    #[test]
    fn malformed_candidates_and_empty_sentences_are_handled() {
        let labels = vec!["person".to_owned()];
        let schema = ExtractionSchema::new(labels.clone(), 0.5, Decode::Flat).unwrap();
        let empty = vec![Sentence { words: Vec::new(), labels: Vec::new() }];
        let predicted = predict_with(&empty, &labels, &schema, |_, _| {
            Err(Error("an empty sentence must not reach the model".into()))
        })
        .unwrap();
        assert_eq!(predicted, [Vec::<String>::new()]);

        let sentence = vec![Sentence { words: vec!["a".into()], labels: vec!["O".into()] }];
        let error = predict_with(&sentence, &labels, &schema, |_, _| Ok(vec![candidate(0, 2, 0, 0.9)])).unwrap_err();
        assert!(error.0.contains("invalid candidate"));
    }
}
