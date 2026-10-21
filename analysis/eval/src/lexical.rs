//! Lexical retrieval: BM25 over the chunks' terms, with turbo-chunker's
//! posting index (`Bm25PostingIndex`) and Lucene's parameters by default.
//! No model and no vectors; the embedder only tokenizes for the analysis
//! and counts chunk tokens for the cost.
//!
//! A chunk's terms are its UAX #29 word tokens that hold a letter or a
//! digit, lowercased with `str::to_lowercase` (`TermRule::Folded`; close
//! to, but not the same as, the full case fold of the folded view
//! turbo-chunker's term statistics are read from: `ß` stays `ß`, and a
//! word-final capital sigma becomes `ς`). `TermRule::English` adds what Lucene's
//! `EnglishAnalyzer` does after its tokenizer: a trailing `'s` dropped, its
//! 33 stop words left out, and Porter's stemmer. The tokenizer is not
//! Lucene's `StandardTokenizer`, so the terms are close to Lucene's but
//! not the same.
//!
//! A document's score is its best chunk's, and the rankings keep the top
//! [`DEPTH`](crate::retrieve::DEPTH) by the tie rules the dense ranking uses: score, then document
//! id descending, then (for chunks) span ascending. A document whose id is
//! the query's own is left out, as in the dense ranking. A chunk that
//! shares no term with the query is not ranked.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::num::NonZeroU32;

use turbo_analysis::Ix;
use turbo_analysis_layers::Strs;
use turbo_chunker::terms::{Bm25, Bm25PostingIndex, Bm25PostingIndexBuilder, TermStats, Vocabulary};
use turbo_stem::{PorterStemmer, Stemmer};

use crate::Result;
use crate::metrics::{Hit, Ranking, Run};
use crate::record::Setting;
use crate::retrieve::top;
use crate::spans::{ChunkHit, ChunkRanking, ChunkRun, Span};

/// Which terms a text gives.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TermRule {
    Folded,
    English,
}

/// Lucene's English stop words (`EnglishAnalyzer.ENGLISH_STOP_WORDS_SET`).
const ENGLISH_STOP: [&str; 33] = [
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of",
    "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with",
];

/// The longest word token, in UTF-16 units, as Lucene's tokenizer cuts.
const MAX_TOKEN: NonZeroU32 = NonZeroU32::new(255).unwrap();

impl TermRule {
    pub fn name(self) -> &'static str {
        match self {
            TermRule::Folded => "folded",
            TermRule::English => "english",
        }
    }

    /// Appends the terms of `text` to `out`, in text order.
    pub fn terms(self, text: &str, out: &mut Vec<String>) {
        let (mut spans, mut types) = (Vec::new(), Vec::new());
        turbo_tokenize::word_tokens(text, MAX_TOKEN, &mut spans, &mut types);
        for s in spans {
            let token = &text[s.start as usize..s.end as usize];
            if !token.chars().any(char::is_alphanumeric) {
                continue;
            }
            let mut term = token.to_lowercase();
            if self == TermRule::English {
                for possessive in ["'s", "\u{2019}s"] {
                    if let Some(stripped) = term.strip_suffix(possessive) {
                        term.truncate(stripped.len());
                        break;
                    }
                }
                if term.is_empty() || ENGLISH_STOP.contains(&term.as_str()) {
                    continue;
                }
                term = PorterStemmer.stem(&term);
            }
            out.push(term);
        }
    }
}

/// BM25 retrieval as a pipeline runs it.
#[derive(Clone, Debug, PartialEq)]
pub struct Lexical {
    pub model: Bm25,
    pub terms: TermRule,
    /// A directory to write each set's terms to, for a reference BM25 to
    /// index the same terms (`<set>.chunks.tsv`: document id, a tab, the
    /// chunk's terms joined by spaces; `<set>.queries.tsv` the same for the
    /// queries). Not part of the record.
    pub dump: Option<std::path::PathBuf>,
}

/// The unit a document's chunks are counted over: one row a chunk.
struct Chunk;

impl Lexical {
    /// Every setting the retrieval runs with, for the record.
    pub fn config(&self) -> Vec<Setting> {
        let s = |name: &str, value: String| Setting { name: name.into(), value };
        vec![s("k1", self.model.k1.to_string()), s("b", self.model.b.to_string()), s("terms", self.terms.name().into())]
    }

    /// The posting index of the chunks, `texts[c]` the text of chunk `c`,
    /// whose document is `doc[c]`; chunks of one document are adjacent, so
    /// row `c` of the index is chunk `c`.
    pub fn index(&self, texts: &[String], doc: &[u32]) -> Result<Bm25PostingIndex> {
        if texts.len() != doc.len() {
            return Err(format!("{} chunk texts and {} document numbers", texts.len(), doc.len()));
        }
        let mut builder = Bm25PostingIndexBuilder::new();
        let mut at = 0;
        let mut terms = Vec::new();
        while at < texts.len() {
            let end = at + doc[at..].iter().take_while(|&&d| d == doc[at]).count();
            let mut units: Vec<BTreeMap<String, u32>> = Vec::with_capacity(end - at);
            for t in &texts[at..end] {
                terms.clear();
                self.terms.terms(t, &mut terms);
                let mut counts = BTreeMap::new();
                for term in terms.drain(..) {
                    *counts.entry(term).or_insert(0u32) += 1;
                }
                units.push(counts);
            }
            let (vocabulary, stats) = counted(&units);
            let first = builder.add(&stats, &vocabulary)?;
            if first as usize != at {
                return Err(format!("chunk {at} became row {first}"));
            }
            at = end;
        }
        Ok(builder.build())
    }

    /// Writes the terms of the chunks and the queries to `dump`, when set.
    pub fn dump(
        &self,
        set: &str,
        ids: &[String],
        texts: &[String],
        doc: &[u32],
        queries: &[(&str, &str)],
    ) -> Result<()> {
        let Some(dir) = &self.dump else { return Ok(()) };
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let mut terms = Vec::new();
        let mut line = |id: &str, text: &str, out: &mut String| {
            terms.clear();
            self.terms.terms(text, &mut terms);
            out.push_str(id);
            out.push('\t');
            out.push_str(&terms.join(" "));
            out.push('\n');
        };
        let (mut chunks, mut qs) = (String::new(), String::new());
        for (t, &d) in texts.iter().zip(doc) {
            line(&ids[d as usize], t, &mut chunks);
        }
        for &(id, text) in queries {
            line(id, text, &mut qs);
        }
        for (name, body) in [("chunks", chunks), ("queries", qs)] {
            let p = dir.join(format!("{set}.{name}.tsv"));
            std::fs::write(&p, body).map_err(|e| format!("{}: {e}", p.display()))?;
        }
        Ok(())
    }

    /// Ranks every query against the index, on as many threads as the
    /// machine has.
    pub fn search(&self, index: &Search<'_>, queries: &[(&str, &str)]) -> Result<(Run, ChunkRun)> {
        let rows = index.index.rows();
        if index.doc.len() != rows || index.span.len() != rows {
            return Err(format!(
                "an index of {rows} rows with {} document numbers and {} spans",
                index.doc.len(),
                index.span.len()
            ));
        }
        if let Some(&d) = index.doc.iter().find(|&&d| d as usize >= index.docs.len()) {
            return Err(format!("document number {d} of {} documents", index.docs.len()));
        }
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(queries.len()).max(1);
        let per = queries.len().div_ceil(threads).max(1);
        let parts: Vec<Vec<(Ranking, ChunkRanking)>> = std::thread::scope(|scope| {
            let handles: Vec<_> = queries
                .chunks(per)
                .map(|qs| scope.spawn(move || qs.iter().map(|&(id, text)| self.rank(index, id, text)).collect()))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))).collect()
        });
        let (docs, chunks): (Vec<_>, Vec<_>) = parts.into_iter().flatten().unzip();
        Ok((Run::new(docs)?, ChunkRun::new(chunks)?))
    }

    fn rank(&self, s: &Search<'_>, query: &str, text: &str) -> (Ranking, ChunkRanking) {
        let mut terms = Vec::new();
        self.terms.terms(text, &mut terms);
        let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
        let rows: Vec<(u32, f64)> = s
            .index
            .top_k(self.model, &refs, s.index.rows())
            .into_iter()
            .filter(|&(r, _)| s.docs[s.doc[r as usize] as usize] != query)
            .collect();
        let mut best: BTreeMap<u32, f64> = BTreeMap::new();
        for &(r, score) in &rows {
            let d = best.entry(s.doc[r as usize]).or_insert(f64::NEG_INFINITY);
            *d = d.max(score);
        }
        let id = |d: u32| &s.docs[d as usize];
        let mut docs: Vec<(u32, f64)> = best.into_iter().collect();
        let doc_order = |a: &(u32, f64), b: &(u32, f64)| b.1.total_cmp(&a.1).then_with(|| id(b.0).cmp(id(a.0)));
        sort_top(&mut docs, doc_order);
        let hits = docs.iter().map(|&(d, score)| Hit { doc: id(d).clone(), score }).collect();
        let mut chunks = rows;
        let chunk_order = |a: &(u32, f64), b: &(u32, f64)| {
            let (da, db) = (s.doc[a.0 as usize], s.doc[b.0 as usize]);
            b.1.total_cmp(&a.1)
                .then_with(|| id(db).cmp(id(da)))
                .then_with(|| s.span[a.0 as usize].cmp(&s.span[b.0 as usize]))
        };
        sort_top(&mut chunks, chunk_order);
        let chunk_hits = chunks
            .iter()
            .map(|&(r, score)| ChunkHit { doc: id(s.doc[r as usize]).clone(), span: s.span[r as usize], score })
            .collect();
        (Ranking { query: query.into(), hits }, ChunkRanking { query: query.into(), hits: chunk_hits })
    }
}

/// What a search reads: the index and, per row, its document and span.
pub struct Search<'a> {
    pub index: &'a Bm25PostingIndex,
    pub doc: &'a [u32],
    pub span: &'a [Span],
    /// The set's document ids.
    pub docs: &'a [String],
}

/// Keeps the first [`DEPTH`](crate::retrieve::DEPTH) of `items` by `order`, sorted.
fn sort_top<T: Copy>(items: &mut Vec<T>, order: impl Fn(&T, &T) -> Ordering) {
    let mut ix: Vec<usize> = (0..items.len()).collect();
    top(&mut ix, |&a, &b| order(&items[a], &items[b]));
    *items = ix.into_iter().map(|i| items[i]).collect();
}

/// One document's vocabulary and term counts, a unit per chunk.
fn counted(units: &[BTreeMap<String, u32>]) -> (Vocabulary, TermStats<Chunk>) {
    let mut all: Vec<&str> = units.iter().flat_map(|u| u.keys().map(String::as_str)).collect();
    all.sort_unstable();
    all.dedup();
    let mut text = Strs::default();
    for t in &all {
        text.push(t);
    }
    let vocabulary = Vocabulary { text };
    let mut stats = TermStats {
        row: vec![0],
        term: Vec::new(),
        tf: Vec::new(),
        length: Vec::new(),
        df: vec![0; all.len()],
        _of: PhantomData,
    };
    for u in units {
        for (term, &n) in u {
            let t = all.binary_search(&term.as_str()).expect("a term of the vocabulary");
            stats.term.push(Ix::new(t as u32));
            stats.tf.push(n);
            stats.df[t] += 1;
        }
        stats.length.push(u.values().sum());
        stats.row.push(stats.term.len() as u32);
    }
    (vocabulary, stats)
}

#[cfg(test)]
#[path = "lexical_tests.rs"]
mod tests;
