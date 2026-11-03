//! ContractNLI test evidence search, with one query per hypothesis.
//! Both entailment and contradiction annotations are relevant evidence;
//! NotMentioned has no evidence. Judgements retain their source contract.
//! This is corpus-wide retrieval, not NLI classification or contract-scoped
//! extraction. The seventeen hypotheses are not 2,091 independent queries.

use super::{SetDir, json, offsets::CodePoints};
use crate::{
    Result,
    metrics::{Qrel, Qrels},
    set::{Doc, LoadedSet, Query},
    spans::{Span, SpanRel, SpanRels},
};
use serde::Deserialize;
use std::collections::BTreeMap;

pub const DIGEST: &str = "460267b56052a2dc5aead98eb35eadef9e6734d5723d37b4a9790e410f812387";

#[derive(Deserialize)]
struct Dataset {
    documents: Vec<Contract>,
    labels: BTreeMap<String, Label>,
}
#[derive(Deserialize)]
struct Label {
    hypothesis: String,
}
#[derive(Deserialize)]
struct Contract {
    id: u64,
    text: String,
    spans: Vec<[usize; 2]>,
    annotation_sets: Vec<Annotations>,
}
#[derive(Deserialize)]
struct Annotations {
    annotations: BTreeMap<String, Annotation>,
}
#[derive(Deserialize)]
struct Annotation {
    choice: Choice,
    spans: Vec<usize>,
}
#[derive(Deserialize)]
enum Choice {
    Entailment,
    Contradiction,
    NotMentioned,
}

pub(super) fn check_digest(text: &str, expected: &str) -> Result<()> {
    let actual = crate::sha::to_hex(&crate::sha::sha256(text.as_bytes()));
    if actual != expected {
        return Err(format!("test.json: SHA-256 is {actual}, expected {expected}"));
    }
    Ok(())
}

pub fn read_in(name: &str, mut files: SetDir) -> Result<LoadedSet> {
    let text = files.text("test.json")?;
    check_digest(&text, DIGEST)?;
    read_json(name, "test.json", &text, files.sha256())
}

pub fn read_json(name: &str, file: &str, text: &str, sha256: String) -> Result<LoadedSet> {
    let data: Dataset = json(file, text)?;
    let queries = data.labels.iter().map(|(id, l)| Query { id: id.clone(), text: l.hypothesis.clone() }).collect();
    let mut docs = Vec::new();
    let mut qrels = Vec::new();
    let mut spans = Vec::new();
    for c in data.documents {
        let id = c.id.to_string();
        if c.annotation_sets.len() != 1 {
            return Err(format!("{file}: contract {id} must have one annotation set"));
        }
        let cp = CodePoints::new(&c.text);
        let ranges = c
            .spans
            .iter()
            .map(|&[s, e]| {
                let start = cp.byte(s).ok_or("ContractNLI span start outside contract")?;
                let end = cp.byte(e).ok_or("ContractNLI span end outside contract")?;
                Span::new(start, end)
            })
            .collect::<Result<Vec<_>>>()?;
        let annotations = c.annotation_sets.into_iter().next().unwrap().annotations;
        if annotations.keys().ne(data.labels.keys()) {
            return Err(format!("{file}: contract {id} has different hypothesis keys"));
        }
        for (q, a) in annotations {
            if matches!(a.choice, Choice::NotMentioned) {
                if !a.spans.is_empty() {
                    return Err(format!("{file}: NotMentioned has evidence in contract {id}"));
                }
                continue;
            }
            if a.spans.is_empty() {
                return Err(format!("{file}: non-neutral label lacks evidence in contract {id}"));
            }
            qrels.push(Qrel { query: q.clone(), doc: id.clone(), grade: 1 });
            for index in a.spans {
                let span = *ranges
                    .get(index)
                    .ok_or_else(|| format!("{file}: contract {id} has invalid evidence index {index}"))?;
                spans.push(SpanRel { query: q.clone(), doc: id.clone(), span, grade: 1 });
            }
        }
        docs.push(Doc { id, title: None, text: c.text });
    }
    let set = LoadedSet {
        name: name.into(),
        sha256,
        docs,
        queries,
        qrels: Qrels::new(qrels)?,
        spans: Some(SpanRels::new(spans)?),
        unplaced: vec![],
        absent: vec![],
    };
    set.check()?;
    Ok(set)
}
