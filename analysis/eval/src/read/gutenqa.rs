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
//! occurrence in the question's chunk. A passage not found as written is
//! looked for again with both texts [`loose`]: quotation marks dropped
//! and each run of whitespace one space, since the passages often swap
//! the books' curly double quotes for straight single ones, add a closing
//! quote, or have a plain space where the book has a line break or a
//! no-break space. The span is then the original bytes of the chunk from
//! the first to the last character matched. A passage not found either
//! way is listed in `unplaced`.

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
        match locate(text, must_contain) {
            Some((from, to)) => {
                let span = Span::new(start + from, start + to)?;
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

/// The byte range of `passage` in `chunk`: its first occurrence as
/// written, else its first occurrence with both texts [`loose`]. `None`
/// when neither finds it, or the passage is empty.
pub fn locate(chunk: &str, passage: &str) -> Option<(usize, usize)> {
    if passage.is_empty() {
        return None;
    }
    if let Some(at) = chunk.find(passage) {
        return Some((at, at + passage.len()));
    }
    let (needle, _) = loose(passage);
    let needle = needle.trim_matches(' ');
    if needle.is_empty() {
        return None;
    }
    let (hay, from) = loose(chunk);
    let at = hay.find(needle)?;
    Some((from[at].0, from[at + needle.len() - 1].1))
}

/// Quotation marks a GutenQA passage may have where its book has
/// another, or none.
fn is_quote(c: char) -> bool {
    matches!(c, '\'' | '"' | '`' | '\u{2018}' | '\u{2019}' | '\u{201B}' | '\u{201C}' | '\u{201D}' | '\u{201F}')
}

/// `text` with every quotation mark dropped and each run of whitespace
/// (no-break spaces and line breaks included) one space, and for each
/// byte of the result the byte range of `text` it stands for. Apostrophes
/// are dropped too, from both texts alike.
pub fn loose(text: &str) -> (String, Vec<(usize, usize)>) {
    let mut out = String::with_capacity(text.len());
    let mut from: Vec<(usize, usize)> = Vec::with_capacity(text.len());
    for (i, c) in text.char_indices() {
        let end = i + c.len_utf8();
        if is_quote(c) {
            continue;
        }
        if c.is_whitespace() {
            if out.ends_with(' ') {
                from.last_mut().expect("a space was pushed").1 = end;
            } else {
                out.push(' ');
                from.push((i, end));
            }
            continue;
        }
        out.push(c);
        from.extend(std::iter::repeat_n((i, end), c.len_utf8()));
    }
    (out, from)
}
