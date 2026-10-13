//! ConTEB's layout: a `documents` table of chunks and a `queries` table,
//! published as Parquet. The workspace has no Parquet reader, so this
//! reads their rows as JSON Lines, one object per row with the column
//! names as keys (what `datasets`' `to_json` writes for a split):
//!
//! - documents: `{"chunk_id", "chunk"}`, other columns allowed. A chunk
//!   id is `<doc id>_<position>`, as ConTEB's `DataFormatter.parse_id`
//!   splits it: exactly one `_`, and an integer position.
//! - queries: `{"chunk_id", "query"}`, other columns allowed; the chunk
//!   is the one that answers the query.
//!
//! A document is its chunks in position order joined by
//! [`CHUNK_SEPARATOR`](crate::read::CHUNK_SEPARATOR). A query (id: its
//! row number from 0) judges its document with grade 1 and its chunk's
//! byte range with grade 1.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::Result;
use crate::metrics::{Qrel, Qrels};
use crate::read::{SetDir, join_chunks, json_lines};
use crate::set::{Doc, LoadedSet, Query};
use crate::spans::{Span, SpanRel, SpanRels};

#[derive(Deserialize)]
struct ChunkRow {
    chunk_id: String,
    chunk: String,
}

#[derive(Deserialize)]
struct QueryRow {
    chunk_id: String,
    query: String,
}

fn parse_id(file: &str, line: usize, id: &str) -> Result<(String, u64)> {
    let bad = || format!("{file}:{line}: chunk id {id:?} is not <doc id>_<position>");
    let (doc, pos) = id.split_once('_').ok_or_else(bad)?;
    if pos.contains('_') {
        return Err(bad());
    }
    Ok((doc.to_owned(), pos.parse().map_err(|_| bad())?))
}

/// Reads the tables `documents` and `queries` (paths under `dir`) as
/// `name`.
pub fn read(name: &str, dir: &Path, documents: &str, queries: &str) -> Result<LoadedSet> {
    let mut files = SetDir::new(dir);
    let mut by_doc: BTreeMap<String, BTreeMap<u64, String>> = BTreeMap::new();
    for (line, r) in json_lines::<ChunkRow>(documents, &files.text(documents)?)? {
        let (doc, pos) = parse_id(documents, line, &r.chunk_id)?;
        if by_doc.entry(doc).or_default().insert(pos, r.chunk).is_some() {
            return Err(format!("{documents}:{line}: chunk {} appears twice", r.chunk_id));
        }
    }
    let mut docs = Vec::with_capacity(by_doc.len());
    let mut ranges: BTreeMap<(String, u64), (usize, usize)> = BTreeMap::new();
    for (id, chunks) in by_doc {
        let (text, r) = join_chunks(chunks.values().map(String::as_str));
        for (pos, range) in chunks.keys().zip(r) {
            ranges.insert((id.clone(), *pos), range);
        }
        docs.push(Doc { id, title: None, text });
    }

    let mut qs = Vec::new();
    let mut qrels = Vec::new();
    let mut spans = Vec::new();
    for (n, (line, r)) in json_lines::<QueryRow>(queries, &files.text(queries)?)?.into_iter().enumerate() {
        let (doc, pos) = parse_id(queries, line, &r.chunk_id)?;
        let Some(&(start, end)) = ranges.get(&(doc.clone(), pos)) else {
            return Err(format!("{queries}:{line}: chunk {} is not in {documents}", r.chunk_id));
        };
        let span = Span::new(start, end).map_err(|e| format!("{queries}:{line}: chunk {}: {e}", r.chunk_id))?;
        let id = n.to_string();
        spans.push(SpanRel { query: id.clone(), doc: doc.clone(), span, grade: 1 });
        qrels.push(Qrel { query: id.clone(), doc, grade: 1 });
        qs.push(Query { id, text: r.query });
    }

    let set = LoadedSet {
        name: name.to_owned(),
        sha256: files.sha256(),
        docs,
        queries: qs,
        qrels: Qrels::new(qrels)?,
        spans: Some(SpanRels::new(spans)?),
        unplaced: Vec::new(),
    };
    set.check()?;
    Ok(set)
}
