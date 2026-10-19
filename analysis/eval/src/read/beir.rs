//! The BEIR layout, which every BEIR set shares:
//!
//! - `corpus.jsonl`: one `{"_id", "title", "text"}` object per line,
//!   with an optional `metadata` object that is not read.
//! - `queries.jsonl`: one `{"_id", "text"}` object per line, with an
//!   optional `metadata` object. It holds the queries of every split.
//! - `qrels/test.tsv`: a header line `query-id<TAB>corpus-id<TAB>score`,
//!   then one judgement per line with an integer grade.
//!
//! Only the queries the test judgements name are kept, in the order of
//! `queries.jsonl`. A judgement naming a query the other files do not
//! have is an error. One naming a document `corpus.jsonl` does not have
//! is kept and listed in `LoadedSet.absent`: it counts as relevant and is
//! never retrieved, as BEIR scores it with `trec_eval` (ArguAna's test
//! judgements name 5 such documents). Document level judgements only;
//! BEIR has no spans.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Deserialize;

use crate::Result;
use crate::metrics::{Qrel, Qrels};
use crate::read::{SetDir, json_lines};
use crate::set::{Doc, LoadedSet, Query};

pub const CORPUS: &str = "corpus.jsonl";
pub const QUERIES: &str = "queries.jsonl";
pub const QRELS: &str = "qrels/test.tsv";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusRow {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    text: String,
    #[serde(default, rename = "metadata")]
    _metadata: serde_json::Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QueryRow {
    #[serde(rename = "_id")]
    id: String,
    text: String,
    #[serde(default, rename = "metadata")]
    _metadata: serde_json::Value,
}

/// Reads the set in `dir` as `name`.
pub fn read(name: &str, dir: &Path) -> Result<LoadedSet> {
    read_in(name, SetDir::new(dir))
}

/// Reads the set from `files` as `name`.
pub fn read_in(name: &str, mut files: SetDir) -> Result<LoadedSet> {
    let docs: Vec<Doc> = json_lines::<CorpusRow>(CORPUS, &files.text(CORPUS)?)?
        .into_iter()
        .map(|(_, r)| Doc { id: r.id, title: (!r.title.is_empty()).then_some(r.title), text: r.text })
        .collect();
    let doc_ids: BTreeSet<&str> = docs.iter().map(|d| d.id.as_str()).collect();

    let mut all_queries: BTreeMap<String, (usize, String)> = BTreeMap::new();
    for (line, r) in json_lines::<QueryRow>(QUERIES, &files.text(QUERIES)?)? {
        if all_queries.insert(r.id.clone(), (line, r.text)).is_some() {
            return Err(format!("{QUERIES}:{line}: query {} appears twice", r.id));
        }
    }

    let (qrels, absent) = read_qrels(&files.text(QRELS)?, &all_queries, &doc_ids)?;
    let judged: BTreeSet<&str> = qrels.iter().map(|q| q.query.as_str()).collect();
    let mut queries: Vec<(usize, Query)> = all_queries
        .iter()
        .filter(|(id, _)| judged.contains(id.as_str()))
        .map(|(id, (line, text))| (*line, Query { id: id.clone(), text: text.clone() }))
        .collect();
    queries.sort_by_key(|(line, _)| *line);

    let set = LoadedSet {
        name: name.to_owned(),
        sha256: files.sha256(),
        docs,
        queries: queries.into_iter().map(|(_, q)| q).collect(),
        qrels: Qrels::new(qrels)?,
        spans: None,
        unplaced: Vec::new(),
        absent,
    };
    set.check()?;
    Ok(set)
}

/// The judgements, and a line for each one naming a document not in the
/// corpus.
fn read_qrels(
    text: &str,
    queries: &BTreeMap<String, (usize, String)>,
    docs: &BTreeSet<&str>,
) -> Result<(Vec<Qrel>, Vec<String>)> {
    let mut lines = text.lines().enumerate();
    match lines.next() {
        Some((_, h)) if h.trim_end() == "query-id\tcorpus-id\tscore" => {}
        _ => return Err(format!("{QRELS}:1: expected the header query-id<TAB>corpus-id<TAB>score")),
    }
    let mut rows = Vec::new();
    let mut absent = Vec::new();
    for (i, line) in lines {
        let n = i + 1;
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.trim_end_matches('\r').split('\t').collect();
        let [query, doc, score] = f[..] else {
            return Err(format!("{QRELS}:{n}: expected 3 tab separated fields, found {}", f.len()));
        };
        let grade: i32 = score.parse().map_err(|_| format!("{QRELS}:{n}: score {score:?} is not an integer"))?;
        if !queries.contains_key(query) {
            return Err(format!("{QRELS}:{n}: query {query} is not in {QUERIES}"));
        }
        if !docs.contains(doc) {
            absent.push(format!("{QRELS}:{n}: query {query} document {doc}"));
        }
        rows.push(Qrel { query: query.to_owned(), doc: doc.to_owned(), grade });
    }
    Ok((rows, absent))
}
