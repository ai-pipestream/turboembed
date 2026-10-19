//! Exact retrieval: every query against every chunk vector, by cosine,
//! with no index. A document's score is its best chunk's.
//!
//! Vectors are unit length, so the cosine is the dot product. Each dot
//! product is summed in f32 over eight lanes in a fixed order, then the
//! lanes in order, so the same vectors give the same scores on every
//! machine. Rankings keep the top [`DEPTH`] documents and chunks, by the
//! tie rules of `metrics` and `spans`.
//!
//! A document whose id is the query's own id is left out of both
//! rankings before the cut, as BEIR's exact search leaves it out (its
//! `ignore_identical_ids`, on by default): in ArguAna, say, each query is
//! also a document and would otherwise rank itself first.

use std::cmp::Ordering;

use crate::metrics::{Hit, Ranking, Run};
use crate::spans::{ChunkHit, ChunkRanking, ChunkRun, Span};
use crate::{RECALL_KS, Result};

/// The hits kept per query: the deepest cutoff measured.
pub const DEPTH: usize = RECALL_KS[RECALL_KS.len() - 1];

/// The chunk vectors of a set, `[chunks, dim]`, with each chunk's
/// document (an index into `docs`) and span.
pub struct Index<'a> {
    pub dim: usize,
    pub vectors: Vec<f32>,
    pub doc: Vec<u32>,
    pub span: Vec<Span>,
    /// The set's document ids.
    pub docs: &'a [String],
}

/// The dot product of two vectors of one length.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let (a8, a_rest) = a.as_chunks::<8>();
    let (b8, b_rest) = b.as_chunks::<8>();
    let mut lanes = [0f32; 8];
    for (x, y) in a8.iter().zip(b8) {
        for k in 0..8 {
            lanes[k] += x[k] * y[k];
        }
    }
    let mut sum = lanes.iter().sum::<f32>();
    for (x, y) in a_rest.iter().zip(b_rest) {
        sum += x * y;
    }
    sum
}

impl Index<'_> {
    fn len(&self) -> usize {
        self.doc.len()
    }

    /// Score descending, then document id descending, as `metrics` ranks.
    fn doc_order(&self, scores: &[f64]) -> impl Fn(&usize, &usize) -> Ordering {
        move |&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| self.docs[b].cmp(&self.docs[a]))
    }

    /// The document and chunk rankings of one query vector.
    fn rank(&self, query: &str, q: &[f32]) -> (Ranking, ChunkRanking) {
        let scores: Vec<f64> = self.vectors.chunks_exact(self.dim).map(|v| f64::from(dot(q, v))).collect();
        let mut best = vec![f64::NEG_INFINITY; self.docs.len()];
        for (c, &s) in scores.iter().enumerate() {
            let d = self.doc[c] as usize;
            best[d] = best[d].max(s);
        }
        let mut docs: Vec<usize> =
            (0..self.docs.len()).filter(|&d| best[d].is_finite() && self.docs[d] != query).collect();
        top(&mut docs, self.doc_order(&best));
        let hits = docs.iter().map(|&d| Hit { doc: self.docs[d].clone(), score: best[d] }).collect();

        // Score descending, document id descending, span ascending, as
        // `spans` ranks chunks.
        let chunk_order = |&a: &usize, &b: &usize| {
            scores[b]
                .total_cmp(&scores[a])
                .then_with(|| self.docs[self.doc[b] as usize].cmp(&self.docs[self.doc[a] as usize]))
                .then_with(|| self.span[a].cmp(&self.span[b]))
        };
        let mut chunks: Vec<usize> = (0..self.len()).filter(|&c| self.docs[self.doc[c] as usize] != query).collect();
        top(&mut chunks, chunk_order);
        let chunk_hits = chunks
            .iter()
            .map(|&c| ChunkHit { doc: self.docs[self.doc[c] as usize].clone(), span: self.span[c], score: scores[c] })
            .collect();
        (Ranking { query: query.into(), hits }, ChunkRanking { query: query.into(), hits: chunk_hits })
    }

    /// Ranks every query (`queries[i]` with vector row i of `vectors`),
    /// on as many threads as the machine has.
    pub fn search(&self, queries: &[&str], vectors: &[f32]) -> Result<(Run, ChunkRun)> {
        if vectors.len() != queries.len() * self.dim {
            return Err(format!("{} values for {} queries of {} dimensions", vectors.len(), queries.len(), self.dim));
        }
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(queries.len()).max(1);
        let per = queries.len().div_ceil(threads).max(1);
        let parts: Vec<Vec<(Ranking, ChunkRanking)>> = std::thread::scope(|scope| {
            let handles: Vec<_> = queries
                .chunks(per)
                .enumerate()
                .map(|(p, qs)| {
                    scope.spawn(move || {
                        qs.iter()
                            .enumerate()
                            .map(|(k, q)| {
                                let i = p * per + k;
                                self.rank(q, &vectors[i * self.dim..(i + 1) * self.dim])
                            })
                            .collect()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))).collect()
        });
        let (docs, chunks): (Vec<_>, Vec<_>) = parts.into_iter().flatten().unzip();
        Ok((Run::new(docs)?, ChunkRun::new(chunks)?))
    }
}

/// Keeps the first [`DEPTH`] of `items` by `order`, sorted.
fn top(items: &mut Vec<usize>, order: impl Fn(&usize, &usize) -> Ordering) {
    if items.len() > DEPTH {
        items.select_nth_unstable_by(DEPTH - 1, &order);
        items.truncate(DEPTH);
    }
    items.sort_unstable_by(order);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_sums_every_lane_and_the_rest() {
        let a: Vec<f32> = (1..=11).map(|x| x as f32).collect();
        let b = vec![1.0; 11];
        assert_eq!(dot(&a, &b), 66.0);
        assert_eq!(dot(&[], &[]), 0.0);
    }

    #[test]
    fn documents_score_their_best_chunk() {
        // Two dimensions. Document a has chunks (1, 0) and (0, 1); b has
        // (0.6, 0.8). The query (0, 1): a's best is 1, b's 0.8, so a
        // ranks first on its second chunk. c has no chunk and no rank.
        let docs = ["a".to_string(), "b".into(), "c".into()];
        let s = |x, y| Span::new(x, y).unwrap();
        let index = Index {
            dim: 2,
            vectors: vec![1.0, 0.0, 0.0, 1.0, 0.6, 0.8],
            doc: vec![0, 0, 1],
            span: vec![s(0, 5), s(5, 9), s(0, 3)],
            docs: &docs,
        };
        let (run, chunks) = index.search(&["q"], &[0.0, 1.0]).unwrap();
        let ranked: Vec<(&str, f64)> = run.hits("q").iter().map(|h| (h.doc.as_str(), h.score)).collect();
        assert_eq!(ranked, [("a", 1.0), ("b", f64::from(0.8f32))]);
        let c: Vec<(&str, Span)> = chunks.hits("q").iter().map(|h| (h.doc.as_str(), h.span)).collect();
        assert_eq!(c, [("a", s(5, 9)), ("b", s(0, 3)), ("a", s(0, 5))]);
    }

    #[test]
    fn the_query_s_own_document_is_left_out() {
        // Query b is also document b, the best match; it is dropped from
        // both rankings, and a is ranked as if b were not there.
        let docs = ["a".to_string(), "b".into()];
        let s = |x, y| Span::new(x, y).unwrap();
        let index = Index {
            dim: 2,
            vectors: vec![0.6, 0.8, 0.0, 1.0],
            doc: vec![0, 1],
            span: vec![s(0, 3), s(0, 4)],
            docs: &docs,
        };
        let (run, chunks) = index.search(&["b", "q"], &[0.0, 1.0, 0.0, 1.0]).unwrap();
        let ids = |q| run.hits(q).iter().map(|h| h.doc.as_str()).collect::<Vec<_>>();
        assert_eq!(ids("b"), ["a"]);
        assert_eq!(ids("q"), ["b", "a"]);
        assert_eq!(chunks.hits("b").iter().map(|h| h.doc.as_str()).collect::<Vec<_>>(), ["a"]);
        assert_eq!(chunks.hits("q").len(), 2);
    }

    #[test]
    fn only_the_top_depth_are_kept_with_ties_by_id() {
        // 150 documents of one chunk each, all scoring the same: the top
        // 100 by the tie rule are the 100 highest ids.
        let docs: Vec<String> = (0..150).map(|i| format!("d{i:03}")).collect();
        let index = Index {
            dim: 1,
            vectors: vec![1.0; 150],
            doc: (0..150).collect(),
            span: vec![Span::new(0, 1).unwrap(); 150],
            docs: &docs,
        };
        let (run, chunks) = index.search(&["q"], &[1.0]).unwrap();
        let hits = run.hits("q");
        assert_eq!(hits.len(), DEPTH);
        assert_eq!((hits[0].doc.as_str(), hits[99].doc.as_str()), ("d149", "d050"));
        assert_eq!(chunks.hits("q").len(), DEPTH);
        assert_eq!(chunks.hits("q")[0].doc, "d149");
    }
}
