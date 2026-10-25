//! An inverted index over the units of many documents, for BM25 top-k
//! without scoring every unit, plus positional phrase matching (`phrase`)
//! and deterministic binary snapshot serialization (`to_bytes`, `from_bytes`).
//! `Bm25PostingIndexBuilder` adds each document's `TermStats` (every unit
//! becomes a row, numbered from 0 in the order added) and `build` gives a
//! `Bm25PostingIndex`: for each term, the rows holding it with their counts
//! (and optional token positions), in row order.

use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;

use turbo_analysis::{Ix, IxRange, Layer, TopK};
use turbo_analysis_layers::{Strs, Tokens};

use super::bm25::{Bm25, Collection};
use super::collection::{CollectionStatistics, CollectionStatisticsBuilder};
use super::layer::{IndexedTerms, TermStats, Vocabulary};

const MAGIC: &[u8; 8] = b"TBM25PI\x01";
type RowPositions = Vec<(u32, Vec<u32>)>;

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
struct Posting {
    /// `(row, count)`, rows ascending.
    rows: Vec<(u32, u32)>,
    /// `(row, sorted_positions)`, rows ascending.
    positions: RowPositions,
    max_tf: u32,
    min_length: u32,
}

/// The posting lists of the rows added, terms in byte order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bm25PostingIndex {
    terms: Vec<(String, Posting)>,
    length: Vec<u32>,
    statistics: CollectionStatistics,
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

    fn posting(&self, term: &str) -> Option<&Posting> {
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
        let mut essential = 0;
        loop {
            if prune && let Some(threshold) = top.threshold() {
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

    /// Evaluates a positional phrase query (`terms` within `slop` token
    /// movements) and returns the top `k` matching `(row, score)` pairs
    /// scored by BM25 phrase frequency.
    pub fn phrase(&self, model: Bm25, terms: &[&str], slop: u32, k: usize) -> Vec<(u32, f64)> {
        self.phrase_with(model, &self.statistics, terms, slop, k)
    }

    /// Evaluates a positional phrase query against `collection` statistics.
    pub fn phrase_with(
        &self,
        model: Bm25,
        collection: &impl Collection,
        terms: &[&str],
        slop: u32,
        k: usize,
    ) -> Vec<(u32, f64)> {
        if k == 0 || terms.is_empty() {
            return Vec::new();
        }
        let Some(postings) = terms.iter().map(|&t| self.posting(t)).collect::<Option<Vec<&Posting>>>() else {
            return Vec::new();
        };
        let (units, mean) = (collection.units(), collection.mean_length());
        let idf_sum: f64 = terms.iter().map(|&t| Bm25::idf(units, collection.df(t))).sum();
        let mut top = TopK::new(k);
        for &(row, first_tf) in &postings[0].rows {
            if !postings[1..].iter().all(|p| p.rows.binary_search_by_key(&row, |&(r, _)| r).is_ok()) {
                continue;
            }
            let freq = if terms.len() == 1 {
                first_tf
            } else {
                let pos_lists: Option<Vec<&[u32]>> = postings
                    .iter()
                    .map(|p| {
                        p.positions.binary_search_by_key(&row, |&(r, _)| r).ok().map(|i| p.positions[i].1.as_slice())
                    })
                    .collect();
                let Some(pos_lists) = pos_lists else { continue };
                phrase_matches(&pos_lists, slop)
            };
            if freq > 0 {
                top.offer(row, idf_sum * model.weight(freq, self.length[row as usize], mean));
            }
        }
        top.into_sorted()
    }

    /// Serializes this posting index (including collection statistics, row
    /// lengths, term frequencies, and positional postings) into a
    /// deterministic binary snapshot.
    pub fn to_bytes(&self) -> Vec<u8> {
        let stats_bytes = self.statistics.to_bytes();
        let mut out = Vec::with_capacity(MAGIC.len() + 24 + stats_bytes.len() + self.length.len() * 4);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(stats_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&stats_bytes);
        out.extend_from_slice(&(self.length.len() as u64).to_le_bytes());
        for &len in &self.length {
            out.extend_from_slice(&len.to_le_bytes());
        }
        out.extend_from_slice(&(self.terms.len() as u64).to_le_bytes());
        for (term, p) in &self.terms {
            out.extend_from_slice(&(term.len() as u32).to_le_bytes());
            out.extend_from_slice(term.as_bytes());
            out.extend_from_slice(&(p.rows.len() as u32).to_le_bytes());
            for &(row, tf) in &p.rows {
                out.extend_from_slice(&row.to_le_bytes());
                out.extend_from_slice(&tf.to_le_bytes());
            }
            out.extend_from_slice(&(p.positions.len() as u32).to_le_bytes());
            for (row, pos_list) in &p.positions {
                out.extend_from_slice(&row.to_le_bytes());
                out.extend_from_slice(&(pos_list.len() as u32).to_le_bytes());
                for &pos in pos_list {
                    out.extend_from_slice(&pos.to_le_bytes());
                }
            }
        }
        out
    }

    /// Restores a [`Bm25PostingIndex`] from [`Bm25PostingIndex::to_bytes`],
    /// validating magic, collection statistics, row bounds, term ordering,
    /// and positional lists.
    pub fn from_bytes(bytes: &[u8]) -> Result<Bm25PostingIndex, String> {
        let mut r = ByteReader(bytes);
        if r.take(MAGIC.len())? != MAGIC {
            return Err("not a Bm25PostingIndex snapshot of version 1".into());
        }
        let stats_len = usize::try_from(r.u64()?).map_err(|_| "stats length overflow")?;
        let statistics = CollectionStatistics::from_bytes(r.take(stats_len)?)?;
        let num_rows = usize::try_from(r.u64()?).map_err(|_| "row count overflow")?;
        if num_rows > r.0.len() / 4 || num_rows > u32::MAX as usize {
            return Err(format!("{num_rows} rows in {} bytes", r.0.len()));
        }
        let mut length = Vec::with_capacity(num_rows);
        for _ in 0..num_rows {
            length.push(r.u32()?);
        }
        let num_terms = usize::try_from(r.u64()?).map_err(|_| "term count overflow")?;
        if num_terms > r.0.len() / 12 {
            return Err(format!("{num_terms} terms in {} bytes", r.0.len()));
        }
        let mut terms: Vec<(String, Posting)> = Vec::with_capacity(num_terms);
        for t_ix in 0..num_terms {
            let t_len = r.u32()? as usize;
            let term = std::str::from_utf8(r.take(t_len)?).map_err(|_| format!("term {t_ix} is not UTF-8"))?;
            if term.is_empty() || terms.last().is_some_and(|(prev, _)| prev.as_str() >= term) {
                return Err(format!("term {t_ix} is empty or out of byte order"));
            }
            let rows_len = r.u32()? as usize;
            if rows_len == 0 || rows_len > r.0.len() / 8 {
                return Err(format!("term {t_ix} has invalid posting length {rows_len}"));
            }
            let mut rows = Vec::with_capacity(rows_len);
            let mut max_tf = 0u32;
            let mut min_length = u32::MAX;
            for _ in 0..rows_len {
                let (row, tf) = (r.u32()?, r.u32()?);
                if (row as usize) >= num_rows || tf == 0 {
                    return Err(format!("term {t_ix} has invalid posting ({row}, {tf})"));
                }
                if rows.last().is_some_and(|&(prev_r, _)| prev_r >= row) {
                    return Err(format!("term {t_ix} postings out of row order"));
                }
                max_tf = max_tf.max(tf);
                min_length = min_length.min(length[row as usize]);
                rows.push((row, tf));
            }
            let pos_rows_len = r.u32()? as usize;
            if pos_rows_len > rows_len || pos_rows_len > r.0.len() / 8 {
                return Err(format!("term {t_ix} has invalid position rows length {pos_rows_len}"));
            }
            let mut positions = Vec::with_capacity(pos_rows_len);
            for _ in 0..pos_rows_len {
                let (row, count) = (r.u32()?, r.u32()? as usize);
                if (row as usize) >= num_rows || count == 0 || count > r.0.len() / 4 {
                    return Err(format!("term {t_ix} has invalid position list for row {row}"));
                }
                if positions.last().is_some_and(|&(prev_r, _): &(u32, Vec<u32>)| prev_r >= row) {
                    return Err(format!("term {t_ix} position rows out of order"));
                }
                let mut pos_list = Vec::with_capacity(count);
                for _ in 0..count {
                    let p = r.u32()?;
                    if pos_list.last().is_some_and(|&prev_p| prev_p >= p) {
                        return Err(format!("term {t_ix} row {row} positions not strictly ascending"));
                    }
                    pos_list.push(p);
                }
                positions.push((row, pos_list));
            }
            terms.push((term.to_owned(), Posting { rows, positions, max_tf, min_length }));
        }
        if !r.0.is_empty() {
            return Err(format!("{} trailing bytes past end of Bm25PostingIndex", r.0.len()));
        }
        Ok(Bm25PostingIndex { terms, length, statistics })
    }
}

fn phrase_matches(pos_lists: &[&[u32]], slop: u32) -> u32 {
    if pos_lists.is_empty() || pos_lists.iter().any(|l| l.is_empty()) {
        return 0;
    }
    let mut count = 0u32;
    for &p0 in pos_lists[0] {
        if slop == 0 {
            let ok = pos_lists[1..]
                .iter()
                .enumerate()
                .all(|(i, list)| p0.checked_add((i + 1) as u32).is_some_and(|want| list.binary_search(&want).is_ok()));
            if ok {
                count += 1;
            }
        } else if match_chain(&pos_lists[1..], p0, slop) {
            count += 1;
        }
    }
    count
}

fn match_chain(rest: &[&[u32]], prev: u32, rem_slop: u32) -> bool {
    let Some((&first, tail)) = rest.split_first() else {
        return true;
    };
    for &p in first {
        if p == prev {
            continue;
        }
        let step = (p as i64) - (prev as i64);
        let cost = (step - 1).unsigned_abs() as u32;
        if cost <= rem_slop && match_chain(tail, p, rem_slop - cost) {
            return true;
        }
    }
    false
}

struct ByteReader<'a>(&'a [u8]);

impl<'a> ByteReader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.0.len() < n {
            return Err("truncated Bm25PostingIndex snapshot".into());
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
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

    fn seek(&mut self, row: u32) -> Option<u32> {
        self.at += self.rows[self.at..].partition_point(|&(r, _)| r < row);
        self.rows.get(self.at).filter(|&&(r, _)| r == row).map(|&(_, n)| n)
    }
}
