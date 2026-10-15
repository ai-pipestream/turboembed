//! ConTEB's layout: a `documents` table of chunks and a `queries` table,
//! each published as Parquet:
//!
//! - documents: `chunk_id`, `chunk`, other columns allowed. A chunk id is
//!   `<doc id>_<position>`, as ConTEB's `DataFormatter.parse_id` splits
//!   it: exactly one `_`, and an integer position.
//! - queries: `chunk_id`, `query`, other columns allowed; the chunk is
//!   the one that answers the query.
//!
//! A document is its chunks in position order joined by
//! [`CHUNK_SEPARATOR`](crate::read::CHUNK_SEPARATOR). A query (id: its
//! row number from 0) judges its document with grade 1 and its chunk's
//! byte range with grade 1.

use std::collections::BTreeMap;
use std::path::Path;

use crate::Result;
use crate::metrics::{Qrel, Qrels};
use crate::read::parquet::Parquet;
use crate::read::{SetDir, join_chunks};
use crate::set::{Doc, LoadedSet, Query};
use crate::spans::{Span, SpanRel, SpanRels};

fn parse_id(file: &str, row: usize, id: &str) -> Result<(String, u64)> {
    let bad = || format!("{file}: row {row}: chunk id {id:?} is not <doc id>_<position>");
    let (doc, pos) = id.split_once('_').ok_or_else(bad)?;
    if pos.contains('_') {
        return Err(bad());
    }
    Ok((doc.to_owned(), pos.parse().map_err(|_| bad())?))
}

/// Reads the tables `documents` and `queries` (paths under `dir`) as
/// `name`.
pub fn read(name: &str, dir: &Path, documents: &str, queries: &str) -> Result<LoadedSet> {
    read_in(name, SetDir::new(dir), documents, queries)
}

/// Reads the tables `documents` and `queries` from `files` as `name`.
pub fn read_in(name: &str, mut files: SetDir, documents: &str, queries: &str) -> Result<LoadedSet> {
    let bytes = files.bytes(documents)?;
    let t = Parquet::open(documents, &bytes)?;
    let (ids, chunks) = (t.strings("chunk_id")?, t.strings("chunk")?);
    let mut by_doc: BTreeMap<String, BTreeMap<u64, &str>> = BTreeMap::new();
    for row in 0..t.rows() {
        let chunk_id = ids.at(row)?;
        let (doc, pos) = parse_id(documents, row, chunk_id)?;
        if by_doc.entry(doc).or_default().insert(pos, chunks.at(row)?).is_some() {
            return Err(format!("{documents}: row {row}: chunk {chunk_id} appears twice"));
        }
    }
    let mut docs = Vec::with_capacity(by_doc.len());
    let mut ranges: BTreeMap<(String, u64), (usize, usize)> = BTreeMap::new();
    for (id, chunks) in by_doc {
        let (text, r) = join_chunks(chunks.values().copied());
        for (pos, range) in chunks.keys().zip(r) {
            ranges.insert((id.clone(), *pos), range);
        }
        docs.push(Doc { id, title: None, text });
    }

    let qbytes = files.bytes(queries)?;
    let t = Parquet::open(queries, &qbytes)?;
    let (ids, texts) = (t.strings("chunk_id")?, t.strings("query")?);
    let mut qs = Vec::new();
    let mut qrels = Vec::new();
    let mut spans = Vec::new();
    for row in 0..t.rows() {
        let chunk_id = ids.at(row)?;
        let (doc, pos) = parse_id(queries, row, chunk_id)?;
        let Some(&(start, end)) = ranges.get(&(doc.clone(), pos)) else {
            return Err(format!("{queries}: row {row}: chunk {chunk_id} is not in {documents}"));
        };
        let span = Span::new(start, end).map_err(|e| format!("{queries}: row {row}: chunk {chunk_id}: {e}"))?;
        let id = row.to_string();
        spans.push(SpanRel { query: id.clone(), doc: doc.clone(), span, grade: 1 });
        qrels.push(Qrel { query: id.clone(), doc, grade: 1 });
        qs.push(Query { id, text: texts.at(row)?.to_owned() });
    }

    let set = LoadedSet {
        name: name.to_owned(),
        sha256: files.sha256(),
        docs,
        queries: qs,
        qrels: Qrels::new(qrels)?,
        spans: Some(SpanRels::new(spans)?),
        unplaced: Vec::new(),
        absent: Vec::new(),
    };
    set.check()?;
    Ok(set)
}
