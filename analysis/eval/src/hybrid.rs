//! Hybrid retrieval: BM25 and exact dense cosine fused per chunk before
//! each document is scored by its best chunk (#799).
//!
//! Each chunk receives a dense cosine score from its unit vector (`dot`)
//! and a BM25 score over its terms (`Bm25PostingIndex`, `0.0` when the
//! chunk shares no term with the query). A document whose id is the
//! query's own is left out before fusion, as in the dense and BM25
//! rankings. The remaining chunks are fused by either:
//!
//! - [`Fusion::Rrf`]: reciprocal rank fusion over the query's 1-based
//!   dense and BM25 chunk rankings,
//!   `(1 - bm25_weight) / (k + rank_dense) + bm25_weight / (k + rank_bm25)`,
//!   where chunks with no query term receive `0` from the BM25 side; or
//! - [`Fusion::Weighted`]: a convex combination of min-max normalized
//!   per-query chunk scores in `[0, 1]`,
//!   `(1 - bm25_weight) * norm_dense + bm25_weight * norm_bm25`.
//!
//! Each document then takes the score of its best chunk, and the top
//! [`DEPTH`](crate::retrieve::DEPTH) documents and chunks are kept by the
//! tie rules of `retrieve` and `lexical`.

use std::cmp::Ordering;

use turbo_chunker::terms::{Bm25, Bm25PostingIndex};

use crate::Result;
use crate::lexical::{Lexical, TermRule};
use crate::metrics::{Hit, Ranking, Run};
use crate::record::Setting;
use crate::retrieve::{dot, top};
use crate::spans::{ChunkHit, ChunkRanking, ChunkRun, Span};

/// Default RRF constant `k` (`60`, Cormack et al.).
pub const DEFAULT_RRF_K: u32 = 60;

/// Default BM25 weight (`0.35`) when fusing BM25 and dense chunk scores.
pub const DEFAULT_BM25_WEIGHT: f64 = 0.35;

/// How a chunk's dense cosine and BM25 scores are fused into one score.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Fusion {
    /// Reciprocal rank fusion over 1-based dense and BM25 chunk ranks:
    /// `(1 - bm25_weight) / (k + rank_dense) + bm25_weight / (k + rank_bm25)`.
    Rrf { k: u32, bm25_weight: f64 },
    /// Convex combination of min-max normalized per-query chunk scores:
    /// `(1 - bm25_weight) * norm_dense + bm25_weight * norm_bm25`.
    Weighted { bm25_weight: f64 },
}

impl Fusion {
    /// Checks that the fusion parameters are valid.
    pub fn check(self) -> Result<Self> {
        match self {
            Fusion::Rrf { k, bm25_weight } => {
                if k == 0 {
                    return Err("RRF k must be at least 1".into());
                }
                if !(bm25_weight.is_finite() && (0.0..=1.0).contains(&bm25_weight)) {
                    return Err(format!("bm25_weight {bm25_weight}: must be between 0 and 1"));
                }
            }
            Fusion::Weighted { bm25_weight } => {
                if !(bm25_weight.is_finite() && (0.0..=1.0).contains(&bm25_weight)) {
                    return Err(format!("bm25_weight {bm25_weight}: must be between 0 and 1"));
                }
            }
        }
        Ok(self)
    }
}

/// Hybrid BM25 + dense retrieval configuration for a [`Pipeline`](crate::pipeline::Pipeline).
#[derive(Clone, Debug, PartialEq)]
pub struct Hybrid {
    pub model: Bm25,
    pub terms: TermRule,
    pub fusion: Fusion,
}

/// What a hybrid search reads: the BM25 posting index, the dense chunk
/// vectors `[chunks, dim]`, and each chunk's document number and span.
pub struct HybridSearch<'a> {
    pub bm25: &'a Bm25PostingIndex,
    pub dim: usize,
    pub vectors: &'a [f32],
    pub doc: &'a [u32],
    pub span: &'a [Span],
    pub docs: &'a [String],
}

impl Hybrid {
    /// Every setting the hybrid retrieval runs with, for the record.
    pub fn config(&self) -> Vec<Setting> {
        let s = |name: &str, value: String| Setting { name: name.into(), value };
        let mut out = match self.fusion {
            Fusion::Rrf { k, bm25_weight } => {
                vec![s("fusion", "rrf".into()), s("rrf_k", k.to_string()), s("bm25_weight", bm25_weight.to_string())]
            }
            Fusion::Weighted { bm25_weight } => {
                vec![s("fusion", "weighted".into()), s("bm25_weight", bm25_weight.to_string())]
            }
        };
        out.push(s("k1", self.model.k1.to_string()));
        out.push(s("b", self.model.b.to_string()));
        out.push(s("terms", self.terms.name().into()));
        out
    }

    /// Builds the BM25 posting index over the chunk texts (`texts[c]` in
    /// document `doc[c]`), using [`Lexical::index`].
    pub fn index(&self, texts: &[String], doc: &[u32]) -> Result<Bm25PostingIndex> {
        Lexical { model: self.model, terms: self.terms, dump: None }.index(texts, doc)
    }

    /// Ranks every query (`queries[i] = (id, text)` with dense query vector
    /// row `i` of `query_vectors`) across available threads.
    pub fn search(
        &self,
        s: &HybridSearch<'_>,
        queries: &[(&str, &str)],
        query_vectors: &[f32],
    ) -> Result<(Run, ChunkRun)> {
        self.fusion.check()?;
        let rows = s.bm25.rows();
        if s.dim == 0 {
            return Err("hybrid search dim must be at least 1".into());
        }
        if s.doc.len() != rows || s.span.len() != rows {
            return Err(format!(
                "an index of {rows} rows with {} document numbers and {} spans",
                s.doc.len(),
                s.span.len()
            ));
        }
        if s.vectors.len() != rows * s.dim {
            return Err(format!("{} vector values for {rows} chunks of {} dimensions", s.vectors.len(), s.dim));
        }
        if query_vectors.len() != queries.len() * s.dim {
            return Err(format!(
                "{} vector values for {} queries of {} dimensions",
                query_vectors.len(),
                queries.len(),
                s.dim
            ));
        }
        if let Some(&d) = s.doc.iter().find(|&&d| d as usize >= s.docs.len()) {
            return Err(format!("document number {d} of {} documents", s.docs.len()));
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
                            .map(|(k, &(id, text))| {
                                let i = p * per + k;
                                let q_vec = &query_vectors[i * s.dim..(i + 1) * s.dim];
                                self.rank(s, id, text, q_vec)
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

    fn rank(&self, s: &HybridSearch<'_>, query: &str, text: &str, q_vec: &[f32]) -> (Ranking, ChunkRanking) {
        let rows = s.bm25.rows();
        let dense_raw: Vec<f64> = s.vectors.chunks_exact(s.dim).map(|v| f64::from(dot(q_vec, v))).collect();

        let mut terms = Vec::new();
        self.terms.terms(text, &mut terms);
        let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
        let bm25_hits = s.bm25.top_k(self.model, &refs, rows);
        let mut bm25_raw = vec![0.0f64; rows];
        for (r, score) in bm25_hits {
            bm25_raw[r as usize] = score;
        }

        let eligible: Vec<usize> = (0..rows).filter(|&c| s.docs[s.doc[c] as usize] != query).collect();
        let id = |d: u32| &s.docs[d as usize];
        let tie_chunk =
            |a: usize, b: usize| -> Ordering { id(s.doc[b]).cmp(id(s.doc[a])).then_with(|| s.span[a].cmp(&s.span[b])) };

        let mut fused = vec![f64::NEG_INFINITY; rows];
        match self.fusion {
            Fusion::Weighted { bm25_weight } => {
                let (mut d_min, mut d_max) = (f64::INFINITY, f64::NEG_INFINITY);
                let (mut b_min, mut b_max) = (f64::INFINITY, f64::NEG_INFINITY);
                for &c in &eligible {
                    d_min = d_min.min(dense_raw[c]);
                    d_max = d_max.max(dense_raw[c]);
                    b_min = b_min.min(bm25_raw[c]);
                    b_max = b_max.max(bm25_raw[c]);
                }
                let d_span = d_max - d_min;
                let b_span = b_max - b_min;
                let w_d = 1.0 - bm25_weight;
                for &c in &eligible {
                    let nd = if d_span > 0.0 { (dense_raw[c] - d_min) / d_span } else { 0.0 };
                    let nb = if b_span > 0.0 { (bm25_raw[c] - b_min) / b_span } else { 0.0 };
                    fused[c] = w_d * nd + bm25_weight * nb;
                }
            }
            Fusion::Rrf { k, bm25_weight } => {
                let k_f = f64::from(k);
                let w_d = 1.0 - bm25_weight;
                let mut dense_ord = eligible.clone();
                dense_ord
                    .sort_unstable_by(|&a, &b| dense_raw[b].total_cmp(&dense_raw[a]).then_with(|| tie_chunk(a, b)));
                for (idx, &c) in dense_ord.iter().enumerate() {
                    fused[c] = w_d / (k_f + (idx + 1) as f64);
                }
                let mut bm25_ord: Vec<usize> = eligible.iter().copied().filter(|&c| bm25_raw[c] > 0.0).collect();
                bm25_ord.sort_unstable_by(|&a, &b| bm25_raw[b].total_cmp(&bm25_raw[a]).then_with(|| tie_chunk(a, b)));
                for (idx, &c) in bm25_ord.iter().enumerate() {
                    fused[c] += bm25_weight / (k_f + (idx + 1) as f64);
                }
            }
        }

        let mut best = vec![f64::NEG_INFINITY; s.docs.len()];
        for &c in &eligible {
            let d = s.doc[c] as usize;
            best[d] = best[d].max(fused[c]);
        }
        let mut docs: Vec<usize> = (0..s.docs.len()).filter(|&d| best[d].is_finite() && s.docs[d] != query).collect();
        top(&mut docs, |&a, &b| best[b].total_cmp(&best[a]).then_with(|| s.docs[b].cmp(&s.docs[a])));
        let hits = docs.iter().map(|&d| Hit { doc: s.docs[d].clone(), score: best[d] }).collect();

        let mut chunks = eligible;
        top(&mut chunks, |&a, &b| fused[b].total_cmp(&fused[a]).then_with(|| tie_chunk(a, b)));
        let chunk_hits =
            chunks.iter().map(|&c| ChunkHit { doc: id(s.doc[c]).clone(), span: s.span[c], score: fused[c] }).collect();
        (Ranking { query: query.into(), hits }, ChunkRanking { query: query.into(), hits: chunk_hits })
    }
}

#[cfg(test)]
#[path = "hybrid_tests.rs"]
mod tests;
