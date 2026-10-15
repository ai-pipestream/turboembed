//! HiCBench in the layout HiChunk's README describes:
//!
//! - `doc/HiCBench/<doc_id>.txt`: the documents, plain text. Every
//!   `.txt` file there is a document, with its file stem as id.
//! - `qas/HiCBench.jsonl`: one question per line, `{"input", "answers",
//!   "facts", "evidences", "all_classes", "_id"}`, where `_id` is the
//!   document and `evidences` are sentences quoted from it. Only
//!   `input`, `evidences` and `_id` are read.
//!
//! A question is a query with id `q<line>`, judging its document with
//! grade 1. Each evidence sentence is placed at its first occurrence in
//! the document; one that does not occur is listed in `unplaced`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Deserialize;

use crate::Result;
use crate::metrics::{Qrel, Qrels};
use crate::read::{SetDir, json_lines};
use crate::set::{Doc, LoadedSet, Query};
use crate::spans::{Span, SpanRel, SpanRels};

pub const DOCS: &str = "doc/HiCBench";
pub const QAS: &str = "qas/HiCBench.jsonl";

#[derive(Deserialize)]
struct Qa {
    input: String,
    evidences: Vec<String>,
    #[serde(rename = "_id")]
    id: String,
}

/// Reads the set in `dir` as `name`.
pub fn read(name: &str, dir: &Path) -> Result<LoadedSet> {
    read_in(name, SetDir::new(dir))
}

/// Reads the set from `files` as `name`.
pub fn read_in(name: &str, mut files: SetDir) -> Result<LoadedSet> {
    let mut docs = Vec::new();
    for f in files.list(DOCS, "txt")? {
        let text = files.text(&format!("{DOCS}/{f}"))?;
        let id = f.strip_suffix(".txt").unwrap_or(&f).to_owned();
        docs.push(Doc { id, title: None, text });
    }
    let text_of: BTreeMap<&str, &str> = docs.iter().map(|d| (d.id.as_str(), d.text.as_str())).collect();

    let mut queries = Vec::new();
    let mut qrels = Vec::new();
    let mut spans = Vec::new();
    let mut unplaced = Vec::new();
    for (line, qa) in json_lines::<Qa>(QAS, &files.text(QAS)?)? {
        let Some(text) = text_of.get(qa.id.as_str()) else {
            return Err(format!("{QAS}:{line}: document {} is not in {DOCS}", qa.id));
        };
        let q = format!("q{line}");
        let mut placed = BTreeSet::new();
        for e in qa.evidences.iter().filter(|e| !e.trim().is_empty()) {
            match text.find(e.as_str()) {
                Some(at) => {
                    placed.insert(Span::new(at, at + e.len())?);
                }
                None => unplaced.push(format!("query {q} document {}: {e:?}", qa.id)),
            }
        }
        spans.extend(placed.into_iter().map(|span| SpanRel { query: q.clone(), doc: qa.id.clone(), span, grade: 1 }));
        qrels.push(Qrel { query: q.clone(), doc: qa.id, grade: 1 });
        queries.push(Query { id: q, text: qa.input });
    }

    let set = LoadedSet {
        name: name.to_owned(),
        sha256: files.sha256(),
        docs,
        queries,
        qrels: Qrels::new(qrels)?,
        spans: Some(SpanRels::new(spans)?),
        unplaced,
        absent: Vec::new(),
    };
    set.check()?;
    Ok(set)
}
