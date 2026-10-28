//! Training documents for the static-embed trainer: Tatoeba
//! sentence exports (one sentence per document) and FineWeb
//! parquet documents split into sentences (adjacent pairs).

use std::path::Path;

use turbo_eval::read::parquet::Parquet;

use crate::error::Error;

use super::split_sentences;

fn fail<T>(m: String) -> Result<T, Error> {
    Err(Error::Model(m))
}

/// Documents of `path`: a Tatoeba export (tab-separated
/// id/language/text rows) keeps `eng` rows as one-sentence
/// documents; any other text file is one document per non-empty
/// line, itself split into sentences.
pub fn read_sentences(path: &Path) -> Result<Vec<Vec<String>>, Error> {
    let text = std::fs::read_to_string(path).map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    let tatoeba = text.lines().take(50).filter(|l| l.split('\t').count() >= 3).count() >= 25;
    let mut docs = Vec::new();
    if tatoeba {
        for line in text.lines() {
            let mut f = line.split('\t');
            let _id = f.next();
            let Some(("eng", s)) = f.next().zip(f.next()) else { continue };
            let sents = split_sentences(s);
            if !sents.is_empty() {
                docs.push(sents);
            }
        }
    } else {
        for line in text.lines() {
            let sents = split_sentences(line);
            if !sents.is_empty() {
                docs.push(sents);
            }
        }
    }
    Ok(docs)
}

/// The first `max_docs` documents with at least two sentences from a
/// FineWeb parquet file's `text` column.
pub fn read_parquet(path: &Path, max_docs: usize) -> Result<Vec<Vec<String>>, Error> {
    let bytes = std::fs::read(path).map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    let name = path.to_string_lossy().into_owned();
    let table = Parquet::open(&name, &bytes).map_err(|e| Error::Model(e.to_string()))?;
    let texts = table.strings("text").map_err(|e| Error::Model(e.to_string()))?;
    let mut docs = Vec::new();
    for r in 0..texts.len() {
        if docs.len() >= max_docs {
            break;
        }
        let Some(t) = texts.get(r) else { continue };
        let sents = split_sentences(t);
        if sents.len() >= 2 {
            docs.push(sents);
        }
    }
    if docs.is_empty() {
        return fail(format!("{}: no documents with two sentences", path.display()));
    }
    Ok(docs)
}
