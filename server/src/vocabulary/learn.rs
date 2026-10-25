//! LearnVocabulary's counting: term and document frequencies of the
//! UAX #29 words of every document, and the n-grams whose words are
//! adjacent with only whitespace between them, kept by their pointwise
//! mutual information
//!
//! `pmi(w1 .. wn) = log2(P(w1 .. wn) / (P(w1) * .. * P(wn)))`,
//!
//! where `P(g)` is the n-gram's count over the n-gram positions of its
//! size and `P(w)` a word's count over all words. The idf column is
//! BM25's, with each document one unit.
//!
//! The vocabulary is tab separated text: two `#` lines naming the format
//! and the settings, a column header, the single terms, then the n-grams,
//! each section by term frequency, most frequent first, ties in byte
//! order. The same documents in the same order with the same settings
//! give the same bytes.

use std::collections::HashMap;
use std::fmt::Write;
use std::num::NonZeroU32;

use turbo_analysis::Span;
use turbo_chunker::terms::Bm25;
use turbo_stopword::{StopwordLanguage, StopwordList};
use turbo_tokenize::{WordType, word_tokens};

use crate::proto::vocabulary::LearnVocabularyHeader;

/// The longest word, in UTF-16 units, before the tokenizer cuts it.
const MAX_WORD: NonZeroU32 = NonZeroU32::new(255).unwrap();

pub(crate) struct Options {
    min_tf: u64,
    max_terms: usize,
    lowercase: bool,
    stopwords: Option<&'static StopwordList>,
    /// The n-gram sizes counted, smallest and largest.
    ngrams: Option<(usize, usize)>,
    min_pmi: f64,
    pub(crate) language: String,
    pub(crate) seed: String,
}

impl Options {
    /// The header's settings, or why they are refused.
    pub(crate) fn new(h: &LearnVocabularyHeader, max_ngram_size: u32) -> Result<Options, String> {
        let stopwords = if h.filter_stopwords {
            let primary = h.language.split(['-', '_']).next().unwrap_or_default().to_ascii_lowercase();
            let lang = StopwordLanguage::from_code(&primary)
                .ok_or_else(|| format!("no bundled stopword list for language {:?}", h.language))?;
            Some(StopwordList::bundled(lang))
        } else {
            None
        };
        let ngrams = if h.include_ngrams {
            let (lo, hi) = (h.min_ngram_size, h.max_ngram_size);
            if lo < 2 || hi < lo || hi > max_ngram_size {
                return Err(format!("n-gram sizes {lo} to {hi}: need 2 <= min <= max <= {max_ngram_size}"));
            }
            if !h.min_pmi.is_finite() {
                return Err(format!("min_pmi {} is not finite", h.min_pmi));
            }
            Some((lo as usize, hi as usize))
        } else {
            None
        };
        Ok(Options {
            min_tf: u64::from(h.min_term_frequency.max(1)),
            max_terms: h.max_terms as usize,
            lowercase: h.lowercase,
            stopwords,
            ngrams,
            min_pmi: h.min_pmi,
            language: h.language.clone(),
            seed: h.dictionary_artifact_id.clone(),
        })
    }

    fn stop(&self, w: &str) -> bool {
        self.stopwords.is_some_and(|s| s.contains(w))
    }
}

#[derive(Clone, Copy, Default)]
struct Count {
    tf: u64,
    df: u64,
}

pub(crate) struct Learner {
    o: Options,
    terms: HashMap<String, Count>,
    ngrams: HashMap<String, Count>,
    /// n-gram positions seen, by size.
    positions: Vec<u64>,
    documents: u64,
    tokens: u64,
    spans: Vec<Span>,
    types: Vec<WordType>,
}

/// A learned vocabulary's bytes and what they hold.
pub(crate) struct Learned {
    pub(crate) bytes: Vec<u8>,
    pub(crate) documents: u64,
    pub(crate) tokens: u64,
    pub(crate) terms: u64,
    pub(crate) ngrams: u64,
}

struct Row<'a> {
    term: &'a str,
    count: Count,
    pmi: Option<f64>,
    seed: bool,
}

fn merge(into: &mut HashMap<String, Count>, local: HashMap<String, u64>) {
    for (k, n) in local {
        let c = into.entry(k).or_default();
        c.tf += n;
        c.df += 1;
    }
}

impl Learner {
    pub(crate) fn new(o: Options) -> Learner {
        let positions = vec![0; o.ngrams.map_or(0, |(_, hi)| hi + 1)];
        Learner {
            o,
            terms: HashMap::new(),
            ngrams: HashMap::new(),
            positions,
            documents: 0,
            tokens: 0,
            spans: Vec::new(),
            types: Vec::new(),
        }
    }

    /// Counts one document.
    pub(crate) fn add(&mut self, text: &str) {
        self.spans.clear();
        self.types.clear();
        word_tokens(text, MAX_WORD, &mut self.spans, &mut self.types);
        let words: Vec<String> = self
            .spans
            .iter()
            .map(|s| {
                let w = &text[s.start as usize..s.end as usize];
                if self.o.lowercase { w.to_lowercase() } else { w.to_owned() }
            })
            .collect();
        // Words share a run while only whitespace lies between them.
        let mut run = Vec::with_capacity(words.len());
        let mut r = 0u32;
        for (i, s) in self.spans.iter().enumerate() {
            if i > 0 && !text[self.spans[i - 1].end as usize..s.start as usize].chars().all(char::is_whitespace) {
                r += 1;
            }
            run.push(r);
        }
        let mut local: HashMap<String, u64> = HashMap::new();
        for w in &words {
            *local.entry(w.clone()).or_default() += 1;
        }
        merge(&mut self.terms, local);
        if let Some((lo, hi)) = self.o.ngrams {
            let mut local: HashMap<String, u64> = HashMap::new();
            for n in lo..=hi.min(words.len()) {
                for i in 0..=words.len() - n {
                    if run[i] == run[i + n - 1] {
                        self.positions[n] += 1;
                        *local.entry(words[i..i + n].join(" ")).or_default() += 1;
                    }
                }
            }
            merge(&mut self.ngrams, local);
        }
        self.documents += 1;
        self.tokens += words.len() as u64;
    }

    fn pmi(&self, gram: &str, count: Count) -> Option<f64> {
        let words: Vec<&str> = gram.split(' ').collect();
        let positions = *self.positions.get(words.len())?;
        if positions == 0 || count.tf == 0 {
            return None;
        }
        let mut p = (count.tf as f64 / positions as f64).log2();
        for w in &words {
            let tf = self.terms.get(*w)?.tf;
            p -= (tf as f64 / self.tokens as f64).log2();
        }
        Some(p)
    }

    /// The vocabulary, with `seeds` kept whatever their counts.
    pub(crate) fn finish(self, seeds: &[String]) -> Learned {
        let o = &self.o;
        let mut terms: Vec<Row> = self
            .terms
            .iter()
            .filter(|(t, c)| c.tf >= o.min_tf && !o.stop(t))
            .map(|(t, &count)| Row { term: t, count, pmi: None, seed: false })
            .collect();
        let mut grams: Vec<Row> = Vec::new();
        for (g, &count) in &self.ngrams {
            let (first, last) = (g.split(' ').next().unwrap_or(g), g.rsplit(' ').next().unwrap_or(g));
            if count.tf < o.min_tf || o.stop(first) || o.stop(last) {
                continue;
            }
            if let Some(pmi) = self.pmi(g, count).filter(|&p| p >= o.min_pmi) {
                grams.push(Row { term: g, count, pmi: Some(pmi), seed: false });
            }
        }
        let order = |a: &Row, b: &Row| b.count.tf.cmp(&a.count.tf).then_with(|| a.term.cmp(b.term));
        for rows in [&mut terms, &mut grams] {
            rows.sort_unstable_by(order);
            if o.max_terms > 0 {
                rows.truncate(o.max_terms);
            }
        }
        let folded: Vec<String> =
            seeds.iter().map(|s| if o.lowercase { s.to_lowercase() } else { s.clone() }).collect();
        for s in &folded {
            let (rows, counts) = if s.contains(' ') { (&mut grams, &self.ngrams) } else { (&mut terms, &self.terms) };
            if let Some(row) = rows.iter_mut().find(|r| r.term == s.as_str()) {
                row.seed = true;
                continue;
            }
            let count = counts.get(s.as_str()).copied().unwrap_or_default();
            let pmi = if s.contains(' ') { self.pmi(s, count) } else { None };
            rows.push(Row { term: s, count, pmi, seed: true });
        }
        terms.sort_unstable_by(order);
        grams.sort_unstable_by(order);
        let mut out = String::from("# turbo-vocabulary 1\n");
        let ngrams = o.ngrams.map_or("none".to_owned(), |(lo, hi)| format!("{lo}-{hi}"));
        let _ = writeln!(
            out,
            "# language={}\tdocuments={}\ttokens={}\tlowercase={}\tfilter_stopwords={}\tmin_term_frequency={}\t\
             max_terms={}\tngrams={ngrams}\tmin_pmi={}\tseed={}",
            o.language,
            self.documents,
            self.tokens,
            o.lowercase,
            o.stopwords.is_some(),
            o.min_tf,
            o.max_terms,
            o.min_pmi,
            o.seed,
        );
        out.push_str("term\tkind\ttf\tdf\tidf\tpmi\tseed\n");
        for (kind, rows) in [("term", &terms), ("ngram", &grams)] {
            for r in rows.iter() {
                let idf = Bm25::idf(self.documents, r.count.df);
                let pmi = r.pmi.map_or(String::new(), |p| format!("{p:.6}"));
                let _ = writeln!(
                    out,
                    "{}\t{kind}\t{}\t{}\t{idf:.6}\t{pmi}\t{}",
                    r.term,
                    r.count.tf,
                    r.count.df,
                    u8::from(r.seed)
                );
            }
        }
        Learned {
            bytes: out.into_bytes(),
            documents: self.documents,
            tokens: self.tokens,
            terms: terms.len() as u64,
            ngrams: grams.len() as u64,
        }
    }
}

#[cfg(test)]
#[path = "learn_tests.rs"]
mod tests;
