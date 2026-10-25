//! An inverted index over the units of many documents, for BM25 top-k
//! without scoring every unit. `Bm25PostingIndexBuilder` adds each
//! document's `TermStats` (every unit becomes a row, numbered from 0 in the
//! order added) and `build` gives a `Bm25PostingIndex`: for each term, the
//! rows holding it with their counts, in row order.
//!
//! `top_k` walks the query's posting lists in row order (document at a
//! time) with MaxScore pruning: once k rows are held, a term whose bound,
//! added to the bounds of the terms weaker than it, cannot beat the k-th
//! score no longer brings in rows of its own, and its list is only
//! searched for rows another term brought. A term's bound is its idf times
//! the weight of its largest count in its shortest row. Every row it
//! returns is scored as `Query::score` scores it, term by term in query
//! order, so the scores are the same bits a full scan gives; equal scores
//! go to the earlier row.

#[path = "posting_ext.rs"]
mod ext;

use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;

use turbo_analysis::{Ix, IxRange, Layer, TopK};
use turbo_analysis_layers::{Strs, Tokens};

use super::bm25::{Bm25, Collection};
use super::collection::{CollectionStatistics, CollectionStatisticsBuilder};
use super::layer::{IndexedTerms, TermStats, Vocabulary};

pub(super) type RowPositions = Vec<(u32, Vec<u32>)>;

/// Rows being added.
#[derive(Default)]
pub struct Bm25PostingIndexBuilder {
    terms: HashMap<String, Vec<(u32, u32)>>,
    positions: HashMap<String, RowPositions>,
    length: Vec<u32>,
    statistics: CollectionStatisticsBuilder,
}

impl Bm25PostingIndexBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds every unit of `stats` as the next rows and returns the first
    /// one's number; unit `i` is row `first + i`. `vocabulary` is the
    /// document's. `Err`, with nothing added, when the two do not pass
    /// `TermStats::validate_with` (the statistics builder checks them
    /// before anything here changes), or the rows would pass `u32` or the
    /// statistics `u64`.
    pub fn add<L>(&mut self, stats: &TermStats<L>, vocabulary: &Vocabulary) -> Result<u32, String> {
        let first = self.length.len();
        let end = first.checked_add(stats.length.len()).filter(|&e| e <= u32::MAX as usize);
        let first = u32::try_from(first).ok().filter(|_| end.is_some()).ok_or("more rows than a u32 numbers")?;
        self.statistics.add(stats, vocabulary)?;
        for (i, unit) in stats.units().enumerate() {
            let row = first + i as u32;
            let (terms, tf) = stats.unit(unit);
            for (t, &n) in terms.iter().zip(tf) {
                self.terms.entry(vocabulary.get(*t).to_owned()).or_default().push((row, n));
            }
        }
        self.length.extend_from_slice(&stats.length);
        Ok(first)
    }

    /// Adds every unit of `stats` along with per-unit token positions read
    /// from `indexed` over `unit_tokens`.
    pub fn add_with_positions<L>(
        &mut self,
        stats: &TermStats<L>,
        vocabulary: &Vocabulary,
        indexed: &IndexedTerms,
        unit_tokens: &[IxRange<Tokens>],
    ) -> Result<u32, String> {
        if unit_tokens.len() != stats.length.len() {
            return Err("unit_tokens length does not match stats units".into());
        }
        let max_tok = indexed.len();
        if unit_tokens.iter().any(|r| r.end.get() > max_tok) {
            return Err("unit token range exceeds indexed terms length".into());
        }
        let first = self.add(stats, vocabulary)?;
        for (i, range) in unit_tokens.iter().enumerate() {
            let row = first + i as u32;
            let mut by_term: BTreeMap<Ix<Vocabulary>, Vec<u32>> = BTreeMap::new();
            let mut pos = 0u32;
            for tok_i in range.start.get()..range.end.get() {
                if let Some(t_ix) = indexed.term[tok_i] {
                    by_term.entry(t_ix).or_default().push(pos);
                    pos += 1;
                }
            }
            for (t_ix, pos_list) in by_term {
                self.positions.entry(vocabulary.get(t_ix).to_owned()).or_default().push((row, pos_list));
            }
        }
        Ok(first)
    }

    /// Adds a single row directly from its ordered sequence of normalized
    /// term strings, recording both term frequencies and 0-based positions
    /// for phrase queries.
    pub fn add_row_tokens(&mut self, row_terms: &[&str]) -> Result<u32, String> {
        let mut counts: BTreeMap<&str, (u32, Vec<u32>)> = BTreeMap::new();
        let mut kept = 0u32;
        for &raw in row_terms {
            if raw.is_empty() {
                continue;
            }
            let entry = counts.entry(raw).or_insert_with(|| (0, Vec::new()));
            entry.0 = entry.0.checked_add(1).ok_or("term count overflow")?;
            entry.1.push(kept);
            kept = kept.checked_add(1).ok_or("row position overflow")?;
        }
        let mut vocab_strs = Strs::default();
        let mut term_col = Vec::with_capacity(counts.len());
        let mut tf_col = Vec::with_capacity(counts.len());
        let mut df_col = Vec::with_capacity(counts.len());
        for (idx, (&t, &(tf, _))) in counts.iter().enumerate() {
            vocab_strs.push(t);
            term_col.push(Ix::new(idx as u32));
            tf_col.push(tf);
            df_col.push(1);
        }
        let vocab = Vocabulary { text: vocab_strs };
        let stats = TermStats::<()> {
            row: vec![0, term_col.len() as u32],
            term: term_col,
            tf: tf_col,
            length: vec![kept],
            df: df_col,
            _of: PhantomData,
        };
        let row = self.add(&stats, &vocab)?;
        for (t, (_, pos_list)) in counts {
            self.positions.entry(t.to_owned()).or_default().push((row, pos_list));
        }
        Ok(row)
    }

    pub fn build(mut self) -> Bm25PostingIndex {
        let mut terms: Vec<(String, Posting)> = self
            .terms
            .into_iter()
            .map(|(term, rows)| {
                let max_tf = rows.iter().map(|&(_, n)| n).max().unwrap_or(0);
                let min_length = rows.iter().map(|&(r, _)| self.length[r as usize]).min().unwrap_or(0);
                let positions = self.positions.remove(&term).unwrap_or_default();
                (term, Posting { rows, positions, max_tf, min_length })
            })
            .collect();
        terms.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        Bm25PostingIndex { terms, length: self.length, statistics: self.statistics.build() }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Posting {
    /// `(row, count)`, rows ascending.
    pub(super) rows: Vec<(u32, u32)>,
    /// `(row, sorted_positions)`, rows ascending.
    pub(super) positions: RowPositions,
    pub(super) max_tf: u32,
    pub(super) min_length: u32,
}

/// The posting lists of the rows added, terms in byte order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bm25PostingIndex {
    pub(super) terms: Vec<(String, Posting)>,
    pub(super) length: Vec<u32>,
    pub(super) statistics: CollectionStatistics,
}

impl Bm25PostingIndex {
    /// The rows: every unit added, those with no terms included.
    pub fn rows(&self) -> usize {
        self.length.len()
    }

    /// The length (in indexed terms) of `row`.
    pub fn row_length(&self, row: u32) -> Option<u32> {
        self.length.get(row as usize).copied()
    }

    /// The statistics of the rows added, as a collection of its own.
    pub fn statistics(&self) -> &CollectionStatistics {
        &self.statistics
    }

    /// The rows holding `term`, with its count in each, rows ascending.
    pub fn postings(&self, term: &str) -> &[(u32, u32)] {
        self.posting(term).map_or(&[], |p| &p.rows[..])
    }

    /// The token positions of `term` in `row`, if positional postings were
    /// recorded for that row.
    pub fn positions(&self, term: &str, row: u32) -> &[u32] {
        let Some(p) = self.posting(term) else { return &[] };
        p.positions.binary_search_by_key(&row, |&(r, _)| r).ok().map_or(&[], |i| &p.positions[i].1[..])
    }

    pub(super) fn posting(&self, term: &str) -> Option<&Posting> {
        self.terms.binary_search_by(|(t, _)| t.as_str().cmp(term)).ok().map(|i| &self.terms[i].1)
    }

    /// The `k` rows that score highest for `query` under `model`, with the
    /// idf and mean length of the rows added, best first.
    pub fn top_k(&self, model: Bm25, query: &[&str], k: usize) -> Vec<(u32, f64)> {
        self.top_k_with(model, &self.statistics, query, k)
    }

    /// `top_k` with the idf and mean length of `collection`, a larger
    /// corpus the rows are part of. A term given twice counts twice. Rows
    /// holding no query term are never returned.
    pub fn top_k_with(&self, model: Bm25, collection: &impl Collection, query: &[&str], k: usize) -> Vec<(u32, f64)> {
        let (units, mean) = (collection.units(), collection.mean_length());
        // Query order, each term's list, idf and bound.
        let mut cursors: Vec<Cursor<'_>> = query
            .iter()
            .filter_map(|&term| {
                let p = self.posting(term)?;
                let idf = Bm25::idf(units, collection.df(term));
                let bound = idf * model.weight(p.max_tf, p.min_length, mean);
                Some(Cursor { rows: &p.rows, at: 0, idf, bound })
            })
            .collect();
        if k == 0 || cursors.is_empty() {
            return Vec::new();
        }
        // The bounds hold while a term's weight grows with its count and
        // shrinks with the row's length (k1 >= 0, 0 <= b <= 1) and no idf
        // is negative (a df past the units); otherwise every row is scored.
        let monotone = model.k1 >= 0.0 && (0.0..=1.0).contains(&model.b);
        let prune = monotone && cursors.iter().all(|c| c.idf >= 0.0 && c.bound.is_finite());
        let mut by_bound: Vec<usize> = (0..cursors.len()).collect();
        by_bound.sort_by(|&a, &b| cursors[a].bound.total_cmp(&cursors[b].bound));
        let mut cumulative = Vec::with_capacity(by_bound.len());
        let mut sum = 0.0;
        for &c in &by_bound {
            sum += cursors[c].bound;
            cumulative.push(sum);
        }
        let mut top = TopK::new(k);
        // The first entry of `by_bound` that still brings in rows.
        let mut essential = 0;
        loop {
            if prune && let Some(threshold) = top.threshold() {
                // Slack for rounding: a bound computed in another order
                // than a score must never drop a row that could enter.
                while essential < by_bound.len() && cumulative[essential] * (1.0 + 1e-9) <= threshold {
                    essential += 1;
                }
            }
            let Some(row) = by_bound[essential..].iter().filter_map(|&c| cursors[c].current()).min() else {
                break;
            };
            let mut score = -0.0;
            for c in &mut cursors {
                if let Some(tf) = c.seek(row) {
                    score += c.idf * model.weight(tf, self.length[row as usize], mean);
                }
            }
            top.offer(row, score);
            for &c in &by_bound[essential..] {
                if cursors[c].current() == Some(row) {
                    cursors[c].at += 1;
                }
            }
        }
        top.into_sorted()
    }
}

struct Cursor<'a> {
    rows: &'a [(u32, u32)],
    at: usize,
    idf: f64,
    bound: f64,
}

impl Cursor<'_> {
    fn current(&self) -> Option<u32> {
        self.rows.get(self.at).map(|&(r, _)| r)
    }

    /// Moves to the first entry at or past `row` and returns the count
    /// there if it is `row`'s.
    fn seek(&mut self, row: u32) -> Option<u32> {
        self.at += self.rows[self.at..].partition_point(|&(r, _)| r < row);
        self.rows.get(self.at).filter(|&&(r, _)| r == row).map(|&(_, n)| n)
    }
}
