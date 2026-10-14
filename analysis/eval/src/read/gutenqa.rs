//! GutenQA: 100 Project Gutenberg books cut into chunks by LumberChunker,
//! and 30 questions per book. Published as two Parquet tables; the
//! workspace has no Parquet reader, so this reads their rows as JSON
//! Lines, one object per row with the column names as keys:
//!
//! - `gutenqa_chunks.jsonl`: `{"Book Name", "Book ID", "Chunk ID",
//!   "Chapter", "Chunk"}`.
//! - `questions.jsonl`: `{"Book Name", "Book ID", "Chunk ID", "Question",
//!   "Answer", "Chunk Must Contain"}`.
//!
//! A book is a document (id: its Book ID, title: its Book Name), its
//! chunks in Chunk ID order joined by
//! [`CHUNK_SEPARATOR`](crate::read::CHUNK_SEPARATOR). A question (id: its
//! row number from 0) judges its book with grade 1. Its relevant span is
//! "Chunk Must Contain", the passage the answer is in, at its first
//! occurrence in the question's chunk; one the chunk does not contain is
//! listed in `unplaced`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::Result;
use crate::metrics::{Qrel, Qrels};
use crate::read::{SetDir, join_chunks, json_lines};
use crate::set::{Doc, LoadedSet, Query};
use crate::spans::{Span, SpanRel, SpanRels};

pub const CHUNKS: &str = "gutenqa_chunks.jsonl";
pub const QUESTIONS: &str = "questions.jsonl";

#[derive(Deserialize)]
struct ChunkRow {
    #[serde(rename = "Book Name")]
    book_name: String,
    #[serde(rename = "Book ID")]
    book: u32,
    #[serde(rename = "Chunk ID")]
    chunk: u32,
    #[serde(rename = "Chunk")]
    text: String,
}

#[derive(Deserialize)]
struct QuestionRow {
    #[serde(rename = "Book ID")]
    book: u32,
    #[serde(rename = "Chunk ID")]
    chunk: u32,
    #[serde(rename = "Question")]
    question: String,
    #[serde(rename = "Chunk Must Contain")]
    must_contain: String,
}

struct Book {
    name: String,
    chunks: BTreeMap<u32, String>,
}

/// Reads the set in `dir` as `name`.
pub fn read(name: &str, dir: &Path) -> Result<LoadedSet> {
    let mut files = SetDir::new(dir);
    let mut books: BTreeMap<u32, Book> = BTreeMap::new();
    for (line, r) in json_lines::<ChunkRow>(CHUNKS, &files.text(CHUNKS)?)? {
        let b = books.entry(r.book).or_insert_with(|| Book { name: r.book_name.clone(), chunks: BTreeMap::new() });
        if b.name != r.book_name {
            return Err(format!(
                "{CHUNKS}:{line}: book {} is named {:?} here and {:?} before",
                r.book, r.book_name, b.name
            ));
        }
        if b.chunks.insert(r.chunk, r.text).is_some() {
            return Err(format!("{CHUNKS}:{line}: book {} has chunk {} twice", r.book, r.chunk));
        }
    }
    let mut docs = Vec::with_capacity(books.len());
    let mut ranges: BTreeMap<(u32, u32), (usize, usize)> = BTreeMap::new();
    for (id, b) in books {
        let (text, r) = join_chunks(b.chunks.values().map(String::as_str));
        for (c, range) in b.chunks.keys().zip(r) {
            ranges.insert((id, *c), range);
        }
        docs.push(Doc { id: id.to_string(), title: Some(b.name), text });
    }
    let text_of: BTreeMap<String, &str> = docs.iter().map(|d| (d.id.clone(), d.text.as_str())).collect();

    let mut queries = Vec::new();
    let mut qrels = Vec::new();
    let mut spans = Vec::new();
    let mut unplaced = Vec::new();
    for (n, (line, r)) in json_lines::<QuestionRow>(QUESTIONS, &files.text(QUESTIONS)?)?.into_iter().enumerate() {
        let Some(&(start, end)) = ranges.get(&(r.book, r.chunk)) else {
            return Err(format!("{QUESTIONS}:{line}: book {} chunk {} is not in {CHUNKS}", r.book, r.chunk));
        };
        let id = n.to_string();
        let doc = r.book.to_string();
        let chunk = &text_of[&doc][start..end];
        match chunk.find(r.must_contain.as_str()).filter(|_| !r.must_contain.is_empty()) {
            Some(at) => {
                let span = Span::new(start + at, start + at + r.must_contain.len())?;
                spans.push(SpanRel { query: id.clone(), doc: doc.clone(), span, grade: 1 });
            }
            None => unplaced.push(format!("query {id} book {doc} chunk {}: {:?}", r.chunk, r.must_contain)),
        }
        qrels.push(Qrel { query: id.clone(), doc, grade: 1 });
        queries.push(Query { id, text: r.question });
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
