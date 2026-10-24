//! Corpus-scale static-to-contextual cascade retrieval (#865): the
//! pool, merge, tier and space rules of `turbo_chunker::Cascade`,
//! over every chunk of a set at once instead of one document's
//! levels.
//!
//! Stage 1 admits the union of two shortlists of `candidates`
//! chunks: the static space's exact top-k (`VectorIndex`) and
//! BM25's top-k (`Bm25PostingIndex`), each score mapped to
//! `[0, 1]` as `Cascade` maps them (cosine through `(s + 1) / 2`,
//! BM25 divided by the query's maximum). Stage 2 scores only the
//! pool in the contextual space. A chunk's ranking score is its
//! contextual score when stage 2 scored it, else the mean of the
//! stage-1 signals present; contextually scored chunks always
//! outrank unscored ones (zero or empty rows), exactly as in
//! `Cascade`. Documents take their best chunk, by the tie rules
//! of `retrieve` and `spans`, and a query's own document is left
//! out, as in the dense and hybrid rankings.
//!
//! The two spaces are checked the way `Cascade::new` checks
//! them: the static and contextual models must differ, and each
//! query names its space to its index, which refuses a vector
//! from another space instead of scoring it.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use turbo_chunker::terms::{Bm25, Bm25PostingIndex};
use turbo_vector_index::{IndexBuilder, Metric, Space, VectorIndex};

use crate::Result;
use crate::lexical::{Lexical, TermRule};
use crate::metrics::{Hit, Ranking, Run};
use crate::record::Setting;
use crate::retrieve::top;
use crate::spans::{ChunkHit, ChunkRanking, ChunkRun, Span};

/// Corpus cascade configuration for a [`Pipeline`](crate::pipeline::Pipeline).
#[derive(Clone, Debug, PartialEq)]
pub struct Cascade {
    /// Per-signal stage-1 shortlist size.
    pub candidates: usize,
    pub model: Bm25,
    pub terms: TermRule,
}

impl Cascade {
    pub fn check(&self) -> Result<()> {
        if self.candidates == 0 {
            return Err("cascade candidates must be at least 1".into());
        }
        if !self.model.k1.is_finite() || self.model.k1 < 0.0 || !(0.0..=1.0).contains(&self.model.b) {
            return Err("bm25 parameters out of range".into());
        }
        Ok(())
    }

    /// Every setting the cascade retrieval runs with, for the record.
    pub fn config(&self, static_model: &str) -> Vec<Setting> {
        let s = |name: &str, value: String| Setting { name: name.into(), value };
        vec![
            s("candidates", self.candidates.to_string()),
            s("static_model", static_model.into()),
            s("k1", self.model.k1.to_string()),
            s("b", self.model.b.to_string()),
            s("terms", self.terms.name().into()),
        ]
    }

    /// Builds the BM25 posting index over the chunk texts.
    pub fn index(&self, texts: &[String], doc: &[u32]) -> Result<Bm25PostingIndex> {
        Lexical { model: self.model, terms: self.terms, dump: None }.index(texts, doc)
    }
}

/// The two exact indexes of one cascade search, over the chunks
/// whose rows the indexes could hold: zero or non-finite rows are
/// left out and score as `Cascade`'s empty rows do (unscored).
pub struct CascadeIndex {
    statics: VectorIndex<u32>,
    contextuals: VectorIndex<u32>,
    static_at: Vec<Option<u32>>,
    contextual_at: Vec<Option<u32>>,
    rows: usize,
}

fn build(space: Space, vectors: &[f32], rows: usize) -> Result<(VectorIndex<u32>, Vec<Option<u32>>)> {
    let dim = space.dim as usize;
    let want = rows.checked_mul(dim).ok_or("chunk count times dim overflows usize")?;
    if dim == 0 || vectors.len() != want {
        return Err(format!("{} vector values for {rows} chunks of {dim} dimensions", vectors.len()));
    }
    let mut b = IndexBuilder::new(space, Metric::Cosine).map_err(|e| e.to_string())?;
    let mut at = vec![None; rows];
    for (i, v) in vectors.chunks_exact(dim).enumerate() {
        if v.iter().any(|x| !x.is_finite()) || v.iter().all(|&x| x == 0.0) {
            continue;
        }
        at[i] = Some(b.push(v, i as u32).map_err(|e| e.to_string())?);
    }
    Ok((b.build(), at))
}

/// What a cascade search reads, besides its indexes: the BM25
/// index and chunk layout, and the queries' vectors in each
/// stage's own space.
pub struct CascadeSearch<'a> {
    pub bm25: &'a Bm25PostingIndex,
    pub doc: &'a [u32],
    pub span: &'a [Span],
    pub docs: &'a [String],
    pub static_space: &'a Space,
    pub static_queries: &'a [f32],
    pub contextual_space: &'a Space,
    pub contextual_queries: &'a [f32],
}

impl Cascade {
    /// Builds the static and contextual indexes over all chunks,
    /// refusing one model in both stages, as `Cascade::new` does.
    pub fn build(
        &self,
        static_space: Space,
        static_vectors: &[f32],
        contextual_space: Space,
        contextual_vectors: &[f32],
        rows: usize,
    ) -> Result<CascadeIndex> {
        self.check()?;
        if static_space.model == contextual_space.model {
            return Err("the static and contextual stages must be different spaces".into());
        }
        let (statics, static_at) = build(static_space, static_vectors, rows)?;
        let (contextuals, contextual_at) = build(contextual_space, contextual_vectors, rows)?;
        Ok(CascadeIndex { statics, contextuals, static_at, contextual_at, rows })
    }

    /// Ranks every query. `static_queries` and `contextual_queries`
    /// are the queries' vectors in their own spaces, row `i` for
    /// `queries[i]`; a vector of the wrong space is refused by its
    /// index, never scored.
    pub fn search(
        &self,
        index: &CascadeIndex,
        s: &CascadeSearch<'_>,
        queries: &[(&str, &str)],
    ) -> Result<(Run, ChunkRun)> {
        let (static_space, static_queries) = (s.static_space, s.static_queries);
        let (contextual_space, contextual_queries) = (s.contextual_space, s.contextual_queries);
        self.check()?;
        let rows = index.rows;
        if s.bm25.rows() != rows || s.doc.len() != rows || s.span.len() != rows {
            return Err(format!(
                "cascade rows {rows}, bm25 {}, documents {}, spans {}",
                s.bm25.rows(),
                s.doc.len(),
                s.span.len()
            ));
        }
        if let Some(&d) = s.doc.iter().find(|&&d| d as usize >= s.docs.len()) {
            return Err(format!("document number {d} of {} documents", s.docs.len()));
        }
        let (sd, cd) = (static_space.dim as usize, contextual_space.dim as usize);
        if static_queries.len() != queries.len() * sd || contextual_queries.len() != queries.len() * cd {
            return Err("query vectors do not match the queries and spaces".into());
        }
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(queries.len()).max(1);
        let per = queries.len().div_ceil(threads).max(1);
        let parts: Vec<Result<Vec<(Ranking, ChunkRanking)>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = queries
                .chunks(per)
                .enumerate()
                .map(|(p, qs)| {
                    scope.spawn(move || {
                        qs.iter()
                            .enumerate()
                            .map(|(k, &(id, text))| {
                                let i = p * per + k;
                                self.rank(
                                    index,
                                    s,
                                    id,
                                    text,
                                    static_space,
                                    &static_queries[i * sd..(i + 1) * sd],
                                    contextual_space,
                                    &contextual_queries[i * cd..(i + 1) * cd],
                                )
                            })
                            .collect()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))).collect()
        });
        let (mut docs, mut chunks) = (Vec::with_capacity(queries.len()), Vec::with_capacity(queries.len()));
        for part in parts {
            for (d, c) in part? {
                docs.push(d);
                chunks.push(c);
            }
        }
        Ok((Run::new(docs)?, ChunkRun::new(chunks)?))
    }

    #[allow(clippy::too_many_arguments)]
    fn rank(
        &self,
        index: &CascadeIndex,
        s: &CascadeSearch<'_>,
        query: &str,
        text: &str,
        static_space: &Space,
        sq: &[f32],
        contextual_space: &Space,
        cq: &[f32],
    ) -> Result<(Ranking, ChunkRanking)> {
        let own = |c: u32| s.docs[s.doc[c as usize] as usize] != query;
        let unit = |score: f32| (f64::from(score.clamp(-1.0, 1.0)) + 1.0) / 2.0;

        let mut static_score: BTreeMap<u32, f64> = BTreeMap::new();
        let mut pool: Vec<u32> = Vec::new();
        let mut in_pool = vec![false; index.rows];
        let shortlist = index.statics.top_k(static_space, sq, self.candidates).map_err(|e| e.to_string())?;
        for hit in shortlist {
            let row = *index.statics.meta(hit.row);
            static_score.insert(row, unit(hit.score));
            if own(row) && !in_pool[row as usize] {
                in_pool[row as usize] = true;
                pool.push(row);
            }
        }

        let mut terms = Vec::new();
        self.terms.terms(text, &mut terms);
        let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
        let bm25_all = s.bm25.top_k(self.model, &refs, index.rows);
        let max = bm25_all.first().map_or(0.0, |h| h.1);
        let mut lexical_score: BTreeMap<u32, f64> = BTreeMap::new();
        if max > 0.0 {
            for (row, score) in &bm25_all {
                lexical_score.insert(*row, (score / max).clamp(0.0, 1.0));
            }
            for (row, _) in bm25_all.iter().take(self.candidates) {
                if own(*row) && !in_pool[*row as usize] {
                    in_pool[*row as usize] = true;
                    pool.push(*row);
                }
            }
        }

        // Stage 2 scores the pool only; stage-1 scores for pool rows
        // admitted by the other signal are read from their indexes.
        let mut scored: Vec<(u32, u8, f64)> = Vec::with_capacity(pool.len());
        for &row in &pool {
            let st = match static_score.get(&row) {
                Some(&v) => Some(v),
                None => match index.static_at[row as usize] {
                    Some(at) => Some(unit(index.statics.score(static_space, sq, at).map_err(|e| e.to_string())?)),
                    None => None,
                },
            };
            let lx = lexical_score.get(&row).copied();
            let stage1 = match (st, lx) {
                (Some(a), Some(b)) => (a + b) / 2.0,
                (Some(a), None) => a,
                (None, Some(b)) => b,
                (None, None) => 0.0,
            };
            let ctx = match index.contextual_at[row as usize] {
                Some(at) => Some(unit(index.contextuals.score(contextual_space, cq, at).map_err(|e| e.to_string())?)),
                None => None,
            };
            let tier = u8::from(ctx.is_none());
            scored.push((row, tier, ctx.unwrap_or(stage1)));
        }

        let id = |d: u32| &s.docs[d as usize];
        let tie_chunk = |a: u32, b: u32| -> Ordering {
            id(s.doc[b as usize]).cmp(id(s.doc[a as usize])).then_with(|| s.span[a as usize].cmp(&s.span[b as usize]))
        };
        let order = |a: &(u32, u8, f64), b: &(u32, u8, f64)| {
            a.1.cmp(&b.1).then(b.2.total_cmp(&a.2)).then_with(|| tie_chunk(a.0, b.0))
        };
        let mut best: Vec<(u8, f64)> = vec![(2, f64::NEG_INFINITY); s.docs.len()];
        for &(row, tier, score) in &scored {
            let d = &mut best[s.doc[row as usize] as usize];
            if tier < d.0 || (tier == d.0 && score > d.1) {
                *d = (tier, score);
            }
        }
        let mut docs: Vec<usize> = (0..s.docs.len()).filter(|&d| best[d].1.is_finite() && s.docs[d] != query).collect();
        top(&mut docs, |&a, &b| {
            best[a].0.cmp(&best[b].0).then(best[b].1.total_cmp(&best[a].1)).then_with(|| s.docs[b].cmp(&s.docs[a]))
        });
        let hits = docs.iter().map(|&d| Hit { doc: s.docs[d].clone(), score: best[d].1 }).collect();

        let mut chunks: Vec<usize> = (0..scored.len()).collect();
        top(&mut chunks, |&a, &b| order(&scored[a], &scored[b]));
        let chunk_hits = chunks
            .iter()
            .map(|&i| {
                let (row, _, score) = scored[i];
                ChunkHit { doc: id(s.doc[row as usize]).clone(), span: s.span[row as usize], score }
            })
            .collect();
        Ok((Ranking { query: query.into(), hits }, ChunkRanking { query: query.into(), hits: chunk_hits }))
    }
}

impl CascadeIndex {
    /// How many contextual rows a search scored is the pool's size;
    /// exposed for tests that assert stage 2 stays inside it.
    pub fn rows(&self) -> usize {
        self.rows
    }
}

#[cfg(test)]
#[path = "cascade_tests.rs"]
mod tests;
