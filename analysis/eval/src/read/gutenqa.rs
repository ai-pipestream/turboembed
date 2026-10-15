//! GutenQA: 100 Project Gutenberg books cut into chunks by LumberChunker,
//! and 30 questions per book, as the two Parquet tables it is published
//! as:
//!
//! - `gutenqa_chunks.parquet`: `Book Name`, `Book ID`, `Chunk ID`,
//!   `Chapter`, `Chunk`.
//! - `questions.parquet`: `Book Name`, `Book ID`, `Chunk ID`,
//!   `Question`, `Answer`, `Chunk Must Contain`.
//!
//! Only the columns used are read. A book is a document (id: its Book
//! ID, title: its Book Name), its chunks in Chunk ID order joined by
//! [`CHUNK_SEPARATOR`](crate::read::CHUNK_SEPARATOR). A question (id: its
//! row number from 0) judges its book with grade 1. Its relevant span is
//! "Chunk Must Contain", the passage the answer is in, at its first
//! occurrence in the question's chunk; one the chunk does not contain is
//! listed in `unplaced`.

use std::collections::BTreeMap;
use std::path::Path;

use crate::Result;
use crate::metrics::{Qrel, Qrels};
use crate::read::parquet::Parquet;
use crate::read::{SetDir, join_chunks};
use crate::set::{Doc, LoadedSet, Query};
use crate::spans::{Span, SpanRel, SpanRels};

pub const CHUNKS: &str = "gutenqa_chunks.parquet";
pub const QUESTIONS: &str = "questions.parquet";

struct Book<'a> {
    name: &'a str,
    chunks: BTreeMap<u32, &'a str>,
}

/// Reads the set in `dir` as `name`.
pub fn read(name: &str, dir: &Path) -> Result<LoadedSet> {
    let mut files = SetDir::new(dir);
    let bytes = files.bytes(CHUNKS)?;
    let t = Parquet::open(CHUNKS, &bytes)?;
    let (names, ids, chunk_ids, chunk_texts) =
        (t.strings("Book Name")?, t.ints("Book ID")?, t.ints("Chunk ID")?, t.strings("Chunk")?);
    let mut books: BTreeMap<u32, Book> = BTreeMap::new();
    for row in 0..t.rows() {
        let (book, chunk, book_name) = (ids.at_u32(row)?, chunk_ids.at_u32(row)?, names.at(row)?);
        let b = books.entry(book).or_insert_with(|| Book { name: book_name, chunks: BTreeMap::new() });
        if b.name != book_name {
            return Err(format!(
                "{CHUNKS}: row {row}: book {book} is named {book_name:?} here and {:?} before",
                b.name
            ));
        }
        if b.chunks.insert(chunk, chunk_texts.at(row)?).is_some() {
            return Err(format!("{CHUNKS}: row {row}: book {book} has chunk {chunk} twice"));
        }
    }
    let mut docs = Vec::with_capacity(books.len());
    let mut ranges: BTreeMap<(u32, u32), (usize, usize)> = BTreeMap::new();
    for (id, b) in &books {
        let (text, r) = join_chunks(b.chunks.values().copied());
        for (c, range) in b.chunks.keys().zip(r) {
            ranges.insert((*id, *c), range);
        }
        docs.push(Doc { id: id.to_string(), title: Some(b.name.to_owned()), text });
    }
    let text_of: BTreeMap<String, &str> = docs.iter().map(|d| (d.id.clone(), d.text.as_str())).collect();

    let qbytes = files.bytes(QUESTIONS)?;
    let t = Parquet::open(QUESTIONS, &qbytes)?;
    let (ids, chunk_ids, questions, must) =
        (t.ints("Book ID")?, t.ints("Chunk ID")?, t.strings("Question")?, t.strings("Chunk Must Contain")?);
    let mut queries = Vec::new();
    let mut qrels = Vec::new();
    let mut spans = Vec::new();
    let mut unplaced = Vec::new();
    for row in 0..t.rows() {
        let (book, chunk, must_contain) = (ids.at_u32(row)?, chunk_ids.at_u32(row)?, must.at(row)?);
        let Some(&(start, end)) = ranges.get(&(book, chunk)) else {
            return Err(format!("{QUESTIONS}: row {row}: book {book} chunk {chunk} is not in {CHUNKS}"));
        };
        let id = row.to_string();
        let doc = book.to_string();
        let text = &text_of[&doc][start..end];
        match text.find(must_contain).filter(|_| !must_contain.is_empty()) {
            Some(at) => {
                let span = Span::new(start + at, start + at + must_contain.len())?;
                spans.push(SpanRel { query: id.clone(), doc: doc.clone(), span, grade: 1 });
            }
            None => unplaced.push(format!("query {id} book {doc} chunk {chunk}: {must_contain:?}")),
        }
        qrels.push(Qrel { query: id.clone(), doc, grade: 1 });
        queries.push(Query { id, text: questions.at(row)?.to_owned() });
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
