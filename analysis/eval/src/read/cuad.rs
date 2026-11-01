//! CUAD v1 test: one corpus-wide query per exact question template.
//! All contracts with an answer are relevant to that template. Impossible
//! questions contribute no positive judgement. This is evidence retrieval,
//! not the original contract-scoped extraction task. No contract-question
//! copies are counted as independent queries.

use super::{SetDir, json, offsets::CodePoints};
use crate::{
    Result,
    metrics::{Qrel, Qrels},
    set::{Doc, LoadedSet, Query},
    spans::{Span, SpanRel, SpanRels},
};
use serde::Deserialize;
use std::collections::BTreeMap;

pub const DIGEST: &str = "007b6a40b0c65247f881627375c3d2e9b6aeeb5dfa89957494feed0765a1a073";

#[derive(Deserialize)]
struct Dataset {
    data: Vec<Contract>,
}
#[derive(Deserialize)]
struct Contract {
    title: String,
    paragraphs: Vec<Paragraph>,
}
#[derive(Deserialize)]
struct Paragraph {
    context: String,
    qas: Vec<Question>,
}
#[derive(Deserialize)]
struct Question {
    id: String,
    question: String,
    is_impossible: bool,
    answers: Vec<Answer>,
}
#[derive(Deserialize)]
struct Answer {
    text: String,
    answer_start: usize,
}

pub fn read_in(name: &str, mut files: SetDir) -> Result<LoadedSet> {
    let text = files.text("test.json")?;
    super::contract_nli::check_digest(&text, DIGEST)?;
    read_json(name, "test.json", &text, files.sha256())
}

pub fn read_json(name: &str, file: &str, text: &str, sha256: String) -> Result<LoadedSet> {
    let data: Dataset = json(file, text)?;
    let mut docs = Vec::new();
    let mut queries = BTreeMap::new();
    let mut qrels = Vec::new();
    let mut spans = Vec::new();
    for c in data.data {
        if c.paragraphs.len() != 1 {
            return Err(format!("{file}: {} must have one full-contract paragraph", c.title));
        }
        let p = c.paragraphs.into_iter().next().unwrap();
        let cp = CodePoints::new(&p.context);
        let mut seen = std::collections::BTreeSet::new();
        for qa in p.qas {
            let prefix = format!("{}__", c.title);
            let id = qa
                .id
                .strip_prefix(&prefix)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| format!("{file}: question {} does not name its contract and template", qa.id))?
                .to_owned();
            if !seen.insert(id.clone()) {
                return Err(format!("{file}: duplicate template {id} in {}", c.title));
            }
            if let Some(old) = queries.insert(id.clone(), qa.question.clone())
                && old != qa.question
            {
                return Err(format!("{file}: template {id} has different question text"));
            }
            if qa.is_impossible != qa.answers.is_empty() {
                return Err(format!("{file}: inconsistent answers for {}", qa.id));
            }
            if !qa.answers.is_empty() {
                qrels.push(Qrel { query: id.clone(), doc: c.title.clone(), grade: 1 });
            }
            for a in qa.answers {
                let end = a.answer_start.checked_add(a.text.chars().count()).ok_or("CUAD offset overflow")?;
                let start = cp.byte(a.answer_start).ok_or("CUAD answer start outside contract")?;
                let end = cp.byte(end).ok_or("CUAD answer end outside contract")?;
                if p.context.get(start..end) != Some(a.text.as_str()) {
                    return Err(format!("{file}: answer text mismatch for {}", qa.id));
                }
                spans.push(SpanRel { query: id.clone(), doc: c.title.clone(), span: Span::new(start, end)?, grade: 1 });
            }
        }
        docs.push(Doc { id: c.title, title: None, text: p.context });
    }
    let set = LoadedSet {
        name: name.into(),
        sha256,
        docs,
        queries: queries.into_iter().map(|(id, text)| Query { id, text }).collect(),
        qrels: Qrels::new(qrels)?,
        spans: Some(SpanRels::new(spans)?),
        unplaced: vec![],
        absent: vec![],
    };
    set.check()?;
    Ok(set)
}
