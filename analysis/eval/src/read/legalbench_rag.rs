//! LegalBench-RAG's layout, and the mini subset its benchmark script
//! runs.
//!
//! - `benchmarks/<name>.json` for `privacy_qa`, `contractnli`, `maud`
//!   and `cuad`: `{"tests": [{"query", "snippets": [{"file_path",
//!   "span": [start, end], "answer"?}], "tags"?}]}`.
//! - `corpus/<file_path>`: the documents, plain text.
//!
//! Spans are `[start, end)` in code points of the document as Python's
//! `open(path).read()` returns it, so with universal newlines; the
//! document text here is translated the same way and the offsets are
//! converted to bytes of it. When a snippet carries its `answer`, the
//! text at the span must equal it.
//!
//! The mini subset is what `legalbenchrag/benchmark.py` runs with
//! `MAX_TESTS_PER_BENCHMARK = 194` and `SORT_BY_DOCUMENT = True`: a
//! benchmark with more tests is sorted, stably, by one Python
//! `random.random()` draw seeded with the test's first snippet's file
//! path, and the first 194 are kept. The corpus is the documents the
//! kept tests name. Query ids are `<benchmark>/<index in its file>`; a
//! query judges each document it has a snippet in with grade 1, and
//! each snippet with grade 1.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Deserialize;

use crate::Result;
use crate::metrics::{Qrel, Qrels};
use crate::read::offsets::{CodePoints, universal_newlines};
use crate::read::pyrandom::PyRandom;
use crate::read::{SetDir, json};
use crate::set::{Doc, LoadedSet, Query};
use crate::spans::{Span, SpanRel, SpanRels};

/// The benchmarks, in the order `benchmark.py` reads them.
pub const BENCHMARKS: [&str; 4] = ["privacy_qa", "contractnli", "maud", "cuad"];

/// The tests kept per benchmark in the mini subset.
pub const MINI_TESTS_PER_BENCHMARK: usize = 194;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Benchmark {
    tests: Vec<Test>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Test {
    query: String,
    snippets: Vec<Snippet>,
    #[serde(default, rename = "tags")]
    _tags: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Snippet {
    file_path: String,
    span: (usize, usize),
    answer: Option<String>,
}

/// Reads the mini subset of the set in `dir` as `name`.
pub fn read(name: &str, dir: &Path) -> Result<LoadedSet> {
    let mut files = SetDir::new(dir);
    let mut kept: Vec<(String, Test)> = Vec::new();
    for b in BENCHMARKS {
        let file = format!("benchmarks/{b}.json");
        let bench: Benchmark = json(&file, &files.text(&file)?)?;
        let mut tests: Vec<(usize, Test)> = bench.tests.into_iter().enumerate().collect();
        for (i, t) in &tests {
            if t.snippets.is_empty() {
                return Err(format!("{file}: test {i} has no snippets"));
            }
        }
        if tests.len() > MINI_TESTS_PER_BENCHMARK {
            // sorted() with a key is stable; equal keys come only from
            // equal first file paths.
            let mut keyed: Vec<(f64, (usize, Test))> =
                tests.into_iter().map(|t| (PyRandom::from_str_seed(&t.1.snippets[0].file_path).random(), t)).collect();
            keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
            tests = keyed.into_iter().map(|(_, t)| t).take(MINI_TESTS_PER_BENCHMARK).collect();
        }
        kept.extend(tests.into_iter().map(|(i, t)| (format!("{b}/{i}"), t)));
    }

    let paths: BTreeSet<&str> =
        kept.iter().flat_map(|(_, t)| t.snippets.iter().map(|s| s.file_path.as_str())).collect();
    let mut docs = Vec::with_capacity(paths.len());
    let mut points = BTreeMap::new();
    for p in paths {
        let text = universal_newlines(files.text(&format!("corpus/{p}"))?);
        points.insert(p.to_owned(), CodePoints::new(&text));
        docs.push(Doc { id: p.to_owned(), title: None, text });
    }
    let text_of: BTreeMap<&str, &str> = docs.iter().map(|d| (d.id.as_str(), d.text.as_str())).collect();

    let mut queries = Vec::with_capacity(kept.len());
    let mut qrels = BTreeSet::new();
    let mut spans = Vec::new();
    for (id, t) in &kept {
        for s in &t.snippets {
            let cp = &points[&s.file_path];
            let (start, end) = s.span;
            let at = || format!("query {id}: snippet [{start}, {end}) of {}", s.file_path);
            let (Some(b0), Some(b1)) = (cp.byte(start), cp.byte(end)) else {
                return Err(format!("{}: the document has {} code points", at(), cp.len()));
            };
            let span = Span::new(b0, b1).map_err(|e| format!("{}: {e}", at()))?;
            let quoted = &text_of[s.file_path.as_str()][b0..b1];
            if let Some(answer) = &s.answer
                && answer != quoted
            {
                return Err(format!("{}: the text there is {quoted:?}, the answer is {answer:?}", at()));
            }
            qrels.insert((id.clone(), s.file_path.clone()));
            spans.push(SpanRel { query: id.clone(), doc: s.file_path.clone(), span, grade: 1 });
        }
        queries.push(Query { id: id.clone(), text: t.query.clone() });
    }

    let set = LoadedSet {
        name: name.to_owned(),
        sha256: files.sha256(),
        docs,
        queries,
        qrels: Qrels::new(qrels.into_iter().map(|(query, doc)| Qrel { query, doc, grade: 1 }))?,
        spans: Some(SpanRels::new(spans)?),
        unplaced: Vec::new(),
    };
    set.check()?;
    Ok(set)
}
