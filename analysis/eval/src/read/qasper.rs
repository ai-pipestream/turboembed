//! Qasper v0.3, test split: `qasper-test-and-evaluator-v0.3.tgz`, whose
//! member `qasper-test-v0.3.json` maps arXiv ids to papers
//! (`{"title", "abstract", "full_text": [{"section_name", "paragraphs"}],
//! "qas": [...]}`; fields not read here are allowed).
//!
//! A paper becomes one document. Its text is built from units joined by
//! a blank line (`\n\n`): the abstract, then for each section its name
//! (when there is one) and its paragraphs, empty units left out. The
//! title is the document's title and is not in the text.
//!
//! A question is a query, judging its paper with grade 1. Its evidence
//! (every annotator's, `FLOAT SELECTED` figure and table captions left
//! out) is quoted text with no offsets, placed here as follows: every
//! unit whose text equals the evidence is a relevant span (a paragraph
//! can appear twice in a paper); failing that, the first place the text
//! occurs in the document; failing that, the evidence is listed in
//! `unplaced`. Some evidence is a heading, part of a paragraph, or
//! differs from the paragraph in white space, which is why the later
//! steps exist.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Deserialize;

use crate::Result;
use crate::metrics::{Qrel, Qrels};
use crate::read::{SetDir, json, tar, utf8};
use crate::set::{Doc, LoadedSet, Query};
use crate::spans::{Span, SpanRel, SpanRels};

pub const ARCHIVE: &str = "qasper-test-and-evaluator-v0.3.tgz";
pub const MEMBER: &str = "qasper-test-v0.3.json";

const FLOAT: &str = "FLOAT SELECTED";

#[derive(Deserialize)]
struct Paper {
    title: String,
    #[serde(rename = "abstract")]
    abstract_: String,
    full_text: Vec<Section>,
    qas: Vec<Qa>,
}

#[derive(Deserialize)]
struct Section {
    section_name: Option<String>,
    paragraphs: Vec<String>,
}

#[derive(Deserialize)]
struct Qa {
    question: String,
    question_id: String,
    answers: Vec<AnswerBy>,
}

#[derive(Deserialize)]
struct AnswerBy {
    answer: Answer,
}

#[derive(Deserialize)]
struct Answer {
    evidence: Vec<String>,
}

/// Reads the archive in `dir` as `name`.
pub fn read(name: &str, dir: &Path) -> Result<LoadedSet> {
    let mut files = SetDir::new(dir);
    let archive = files.bytes(ARCHIVE)?;
    let member = tar::gzip_member(ARCHIVE, &archive, MEMBER)?;
    let label = format!("{ARCHIVE}:{MEMBER}");
    let text = utf8(&label, member)?;
    read_json(name, &label, &text, files.sha256())
}

/// Reads the papers in `text`, named `file` in errors, with the set's
/// digest `sha256`.
pub fn read_json(name: &str, file: &str, text: &str, sha256: String) -> Result<LoadedSet> {
    let papers: BTreeMap<String, Paper> = json(file, text)?;
    let mut docs = Vec::with_capacity(papers.len());
    let mut queries = Vec::new();
    let mut qrels = Vec::new();
    let mut spans = Vec::new();
    let mut unplaced = Vec::new();
    for (id, p) in papers {
        let (text, units) = layout(&p);
        for qa in &p.qas {
            let q = &qa.question_id;
            queries.push(Query { id: q.clone(), text: qa.question.clone() });
            qrels.push(Qrel { query: q.clone(), doc: id.clone(), grade: 1 });
            let mut placed = BTreeSet::new();
            let evidence: BTreeSet<&str> = qa
                .answers
                .iter()
                .flat_map(|a| a.answer.evidence.iter().map(String::as_str))
                .filter(|e| !e.starts_with(FLOAT) && !e.trim().is_empty())
                .collect();
            for e in evidence {
                let whole: Vec<Span> = units.iter().filter(|(_, u)| u == e).map(|(s, _)| *s).collect();
                if !whole.is_empty() {
                    placed.extend(whole);
                } else if let Some(at) = text.find(e) {
                    placed.insert(Span::new(at, at + e.len())?);
                } else {
                    unplaced.push(format!("query {q} paper {id}: {e:?}"));
                }
            }
            spans.extend(placed.into_iter().map(|span| SpanRel { query: q.clone(), doc: id.clone(), span, grade: 1 }));
        }
        docs.push(Doc { id, title: Some(p.title), text });
    }
    let set = LoadedSet {
        name: name.to_owned(),
        sha256,
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

/// The paper's text, and each unit's span and text.
fn layout(p: &Paper) -> (String, Vec<(Span, String)>) {
    let mut units: Vec<&str> = vec![&p.abstract_];
    for s in &p.full_text {
        units.extend(s.section_name.as_deref());
        units.extend(s.paragraphs.iter().map(String::as_str));
    }
    let mut text = String::new();
    let mut spans = Vec::new();
    for u in units.into_iter().filter(|u| !u.is_empty()) {
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        let start = text.len();
        text.push_str(u);
        spans.push((Span { start, end: text.len() }, u.to_owned()));
    }
    (text, spans)
}
