//! Native GLiNER2 candidates in the shared open-span evaluation format.

use std::collections::BTreeMap;

use turbo_gliner::Candidate;
use turbo_relation::gliner2::words::{Word, lowercase};
use turbo_schema_extract::model::{Gliner2Extractor, Scores};
use turbo_schema_extract::prompt::Prompt;
use turbo_schema_extract::schema::{Cardinality, EntityType, ExtractionSchema};

use crate::corpus::Sentence;
use crate::{Error, Result};

fn prompt<'a>(
    model: &Gliner2Extractor,
    text: &'a str,
    sentence: &'a Sentence,
    schema: &ExtractionSchema,
) -> std::result::Result<Prompt<'a>, turbo_schema_extract::model::ExtractError> {
    let mut offset = 0;
    let words = sentence
        .words
        .iter()
        .map(|word| {
            let start = offset;
            let end = start + word.len();
            offset = end + 1;
            Word { text: lowercase(word), start, end, subword_len: 0 }
        })
        .collect();
    model.build_words(text, schema, words)
}

fn score_batch(
    model: &Gliner2Extractor,
    text: &str,
    sentence: &Sentence,
    labels: &[String],
    offset: usize,
    guidelines: Option<&BTreeMap<String, String>>,
) -> std::result::Result<Vec<Candidate>, turbo_schema_extract::model::ExtractError> {
    let schema = ExtractionSchema {
        entities: labels
            .iter()
            .map(|name| EntityType {
                name: name.clone(),
                description: guidelines.and_then(|g| g.get(name).cloned()),
                threshold: None,
                cardinality: Cardinality::Repeated,
                checks: Vec::new(),
            })
            .collect(),
        ..ExtractionSchema::default()
    };
    let built = prompt(model, text, sentence, &schema)?;
    model.check(&built)?;
    let pass = model.pass_prompt(built)?;
    let [Scores::Spans { instances, logits, .. }] = pass.scores() else {
        return Err(turbo_schema_extract::model::ExtractError::Model(
            "GLiNER2 returned an unexpected entity task layout".into(),
        ));
    };
    let (instances, logits) = (*instances, logits);
    if instances > 1 || logits.len() != instances * labels.len() * pass.prompt().token_first.len() * pass.width() {
        return Err(turbo_schema_extract::model::ExtractError::Model(
            "GLiNER2 returned invalid entity score dimensions".into(),
        ));
    }
    if instances == 0 {
        return Ok(Vec::new());
    }
    let (tokens, width) = (pass.prompt().token_first.len(), pass.width());
    if pass.prompt().words.len() < sentence.words.len() {
        return Err(turbo_schema_extract::model::ExtractError::Model("GLiNER2 dropped a sentence word".into()));
    }
    let mut found = Vec::new();
    for label in 0..labels.len() {
        for start in 0..sentence.words.len() {
            for span_width in 0..width.min(sentence.words.len() - start) {
                let logit = logits[(label * tokens + start) * width + span_width];
                if !logit.is_finite() {
                    return Err(turbo_schema_extract::model::ExtractError::Model(
                        "GLiNER2 returned a non-finite in-range span logit".into(),
                    ));
                }
                found.push(Candidate {
                    start,
                    end: start + span_width + 1,
                    label: label + offset,
                    logit,
                    score: turbo::head::sigmoid(logit),
                });
            }
        }
    }
    Ok(found)
}

pub fn candidates(
    model: &Gliner2Extractor,
    sentences: &[Sentence],
    labels: &[String],
    guidelines: Option<&BTreeMap<String, String>>,
) -> Result<Vec<Vec<Candidate>>> {
    if labels.is_empty() {
        return Err(Error("GLiNER2 evaluation needs at least one entity label".into()));
    }
    let mut out = Vec::with_capacity(sentences.len());
    for sentence in sentences {
        let mut found = Vec::new();
        if sentence.words.is_empty() {
            out.push(found);
            continue;
        }
        let text = sentence.words.join(" ");
        let mut start = 0;
        while start < labels.len() {
            let mut end = (start + turbo_gliner::MAX_LABELS).min(labels.len());
            loop {
                match score_batch(model, &text, sentence, &labels[start..end], start, guidelines) {
                    Ok(part) => {
                        found.extend(part);
                        start = end;
                        break;
                    }
                    Err(turbo_schema_extract::model::ExtractError::OverLimit { .. }) if end - start > 1 => {
                        end = start + (end - start) / 2;
                    }
                    Err(e) => return Err(Error(e.to_string())),
                }
            }
        }
        out.push(found);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_schema_extract::decode::{Corroborate, decode};
    use turbo_schema_extract::schema::ExtractionSchema;

    struct Nothing;
    impl Corroborate for Nothing {
        fn supporting(&self, _: turbo_schema_extract::schema::Source, _: usize, _: usize) -> Vec<u32> {
            Vec::new()
        }
    }

    #[test]
    #[ignore = "loads the local-only GLiNER2 checkpoint"]
    fn raw_candidates_match_the_native_schema_decoder() {
        let root = std::env::var_os("GLINER2_PACKAGE").expect("GLINER2_PACKAGE");
        let root = turbo_pins::require(&root);
        let model = Gliner2Extractor::load(
            &root.join("encoder"),
            &root,
            "fastino/gliner2-base-v1",
            "f9634218e53580c56edf0de97ca1a7d3f1c2354e",
        )
        .unwrap();
        let sentence = Sentence {
            words: vec![
                "Alice".into(),
                "works".into(),
                "at".into(),
                "Microsoft".into(),
                "in".into(),
                "Paris".into(),
                ".".into(),
            ],
            labels: vec!["O".into(); 7],
        };
        let labels = vec!["person".into(), "organization".into(), "location".into()];
        let raw = candidates(&model, std::slice::from_ref(&sentence), &labels, None).unwrap().remove(0);
        let decode_schema =
            turbo_gliner::ExtractionSchema::new(labels.clone(), 0.5, turbo_gliner::Decode::Flat).unwrap();
        let values = turbo_gliner::decode_with_thresholds(raw.clone(), &decode_schema, &[0.5; 3]).unwrap();
        let text = sentence.words.join(" ");
        let pass = model
            .pass_prompt(
                prompt(
                    &model,
                    &text,
                    &sentence,
                    &ExtractionSchema {
                        entities: labels
                            .iter()
                            .map(|name| EntityType {
                                name: name.clone(),
                                description: None,
                                threshold: None,
                                cardinality: Cardinality::Repeated,
                                checks: Vec::new(),
                            })
                            .collect(),
                        ..ExtractionSchema::default()
                    },
                )
                .unwrap(),
            )
            .unwrap();
        let expected = decode(
            &pass,
            &ExtractionSchema {
                entities: labels
                    .iter()
                    .map(|name| EntityType {
                        name: name.clone(),
                        description: None,
                        threshold: None,
                        cardinality: Cardinality::Repeated,
                        checks: Vec::new(),
                    })
                    .collect(),
                ..ExtractionSchema::default()
            },
            0.5,
            &Nothing,
        )
        .unwrap();
        let expected: Vec<(usize, usize, usize)> = expected
            .entities
            .into_iter()
            .enumerate()
            .flat_map(|(label, spans)| spans.into_iter().map(move |span| (span.start, span.end, label)))
            .collect();
        let mut actual: Vec<(usize, usize, usize)> = values.into_iter().map(|c| (c.start, c.end, c.label)).collect();
        let mut expected: Vec<(usize, usize, usize)> = expected
            .into_iter()
            .map(|(start, end, label)| {
                let a = pass.prompt().words.iter().position(|w| w.start == start).unwrap();
                let b = pass.prompt().words.iter().rposition(|w| w.end == end).unwrap();
                (a, b + 1, label)
            })
            .collect();
        actual.sort();
        expected.sort();
        assert_eq!(actual, expected);

        let many: Vec<String> = (0..33).map(|i| format!("entity {i}")).collect();
        let found = candidates(&model, std::slice::from_ref(&sentence), &many, None).unwrap().remove(0);
        assert!(found.iter().all(|c| c.label < many.len()));
        assert!(found.iter().any(|c| c.label == 32));
        let descriptions: BTreeMap<String, String> = many
            .iter()
            .map(|name| {
                (
                    name.clone(),
                    "a detailed annotation guideline with examples and boundary rules that deliberately expands the schema beyond the prompt budget".into(),
                )
            })
            .collect();
        let guided = candidates(&model, std::slice::from_ref(&sentence), &many, Some(&descriptions)).unwrap().remove(0);
        assert!(guided.iter().all(|c| c.label < many.len()));
        assert!(guided.iter().any(|c| c.label == 32));
    }
}
