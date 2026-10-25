//! Positional phrase matching (`phrase`, `phrase_with`) and binary snapshot
//! persistence (`to_bytes`, `from_bytes`) for [`Bm25PostingIndex`].

use std::collections::HashSet;

use turbo_analysis::TopK;

use super::super::bm25::{Bm25, Collection};
use super::super::collection::CollectionStatistics;
use super::Bm25PostingIndex;
use super::Posting;

const MAGIC: &[u8; 8] = b"TBM25PI\x01";

/// Maximum positional slop permitted in a phrase query.
pub const MAX_PHRASE_SLOP: u32 = 64;

impl Bm25PostingIndex {
    /// Evaluates a positional phrase query (`terms` within `slop` token
    /// movements, capped at [`MAX_PHRASE_SLOP`]) and returns the top `k`
    /// matching `(row, score)` pairs scored by BM25 phrase frequency.
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
        let slop = slop.min(MAX_PHRASE_SLOP);
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
    let mut failed = HashSet::new();
    for &p0 in pos_lists[0] {
        if slop == 0 {
            let ok = pos_lists[1..]
                .iter()
                .enumerate()
                .all(|(i, list)| p0.checked_add((i + 1) as u32).is_some_and(|want| list.binary_search(&want).is_ok()));
            if ok {
                count += 1;
            }
        } else if match_chain(&pos_lists[1..], 1, p0, slop, &mut failed) {
            count += 1;
        }
    }
    count
}

fn match_chain(
    rest: &[&[u32]],
    depth: usize,
    prev: u32,
    rem_slop: u32,
    failed: &mut HashSet<(usize, usize, u32)>,
) -> bool {
    let Some((&first, tail)) = rest.split_first() else {
        return true;
    };
    let min_pos = prev.saturating_sub(rem_slop);
    let max_pos = prev.saturating_add(1).saturating_add(rem_slop);
    let start_ix = first.partition_point(|&p| p < min_pos);
    for (offset, &p) in first[start_ix..].iter().enumerate() {
        if p > max_pos {
            break;
        }
        if p == prev {
            continue;
        }
        let step = (p as i64) - (prev as i64);
        let cost = (step - 1).unsigned_abs() as u32;
        if cost <= rem_slop {
            let next_slop = rem_slop - cost;
            let pos_ix = start_ix + offset;
            if failed.contains(&(depth, pos_ix, next_slop)) {
                continue;
            }
            if match_chain(tail, depth + 1, p, next_slop, failed) {
                return true;
            }
            failed.insert((depth, pos_ix, next_slop));
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
