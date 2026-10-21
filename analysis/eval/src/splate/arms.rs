//! Concrete index builders, cross-encoder scorers, and query runners for the
//! four retrieval arms of the SPLATE cascade evaluation study (#711).

use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use turbo::rerank::{RerankOptions, Reranker};
use turbo_chunker::terms::{Bm25PostingIndex, Bm25PostingIndexBuilder};
use turbo_vector_index::{IndexBuilder, Metric, Normalization, Space, TokenIndex, TokenIndexBuilder, VectorIndex};

use super::corpus::{EncodedCorpus, EncodedDoc, EncodedQuery};
use super::posting::{SparsePostingIndex, SparsePostingIndexBuilder, SplateAdapter};
use super::{ArmReport, CascadeArm, SplateConfig};
use crate::Result;
use crate::metrics::{Hit, Qrels, Ranking, Run, evaluate};
use crate::retrieve::dot;

/// Trait abstracting Stage-2 cross-encoder scoring for Arm 2 (`StaticThenCrossEncoder`).
pub trait CrossEncoderScorer: Send + Sync {
    /// Scores `(query, candidate_doc)` for each candidate index in `candidates`.
    fn score_candidates(&self, corpus: &EncodedCorpus, query: &EncodedQuery, candidates: &[u32]) -> Result<Vec<f32>>;

    /// Byte footprint of any stored cross-encoder weights/tables.
    fn byte_size(&self) -> usize;
}

/// Joint query-passage cross-attention and ordered-bigram interaction scorer
/// over contextual token vectors and token IDs.
#[derive(Clone, Debug)]
pub struct InteractionCrossEncoder {
    dim: usize,
    mlp_w1: Vec<f32>,
    mlp_w2: [f32; 8],
}

impl InteractionCrossEncoder {
    pub fn new(dim: usize) -> Self {
        let mut mlp_w1 = vec![0.0f32; 8 * 4];
        for h in 0..8 {
            mlp_w1[h * 4] = 1.10 + 0.05 * h as f32;
            mlp_w1[h * 4 + 1] = 0.65 + 0.03 * h as f32;
            mlp_w1[h * 4 + 2] = 0.85 - 0.02 * h as f32;
            mlp_w1[h * 4 + 3] = 0.45 + 0.02 * h as f32;
        }
        let mlp_w2 = [0.22, 0.18, 0.16, 0.14, 0.12, 0.08, 0.06, 0.04];
        Self { dim, mlp_w1, mlp_w2 }
    }
}

impl CrossEncoderScorer for InteractionCrossEncoder {
    fn score_candidates(&self, corpus: &EncodedCorpus, query: &EncodedQuery, candidates: &[u32]) -> Result<Vec<f32>> {
        let dim = self.dim;
        let q_toks: Vec<&[f32]> = query.token_vecs.chunks_exact(dim).collect();
        let nq = q_toks.len().max(1) as f32;
        let mut out = Vec::with_capacity(candidates.len());

        for &cand in candidates {
            let doc: &EncodedDoc = &corpus.docs[cand as usize];
            let d_toks: Vec<&[f32]> = doc.token_vecs.chunks_exact(dim).collect();
            let nd = d_toks.len().max(1) as f32;
            let (mut max_sum, mut soft_sum, mut bigram_bonus) = (0.0f32, 0.0f32, 0.0f32);

            for (qi, qt) in q_toks.iter().enumerate() {
                let mut best = f32::NEG_INFINITY;
                let mut best_dj = 0usize;
                let mut attn_num = 0.0f32;
                let mut attn_den = 0.0f32;
                for (dj, dt) in d_toks.iter().enumerate() {
                    let s = dot(qt, dt);
                    if s > best {
                        best = s;
                        best_dj = dj;
                    }
                    let w = (s * 4.0).exp();
                    attn_num += w * s;
                    attn_den += w;
                }
                if best.is_finite() {
                    max_sum += best;
                }
                if attn_den > 0.0 {
                    soft_sum += attn_num / attn_den;
                }
                if qi + 1 < q_toks.len() && best_dj + 1 < d_toks.len() {
                    let next_sim = dot(q_toks[qi + 1], d_toks[best_dj + 1]);
                    if next_sim > 0.45 {
                        bigram_bonus += next_sim;
                    }
                }
            }
            let exact_hits = query.token_ids.iter().filter(|id| doc.token_ids.contains(id)).count() as f32 / nq;
            let feats = [max_sum / nq, soft_sum / nq, bigram_bonus / nq, exact_hits / (1.0 + 0.005 * nd)];
            let mut logit = 0.0f32;
            for (h, &w2) in self.mlp_w2.iter().enumerate() {
                let row = &self.mlp_w1[h * 4..(h + 1) * 4];
                let z = row[0] * feats[0] + row[1] * feats[1] + row[2] * feats[2] + row[3] * feats[3];
                let act = if z > 0.0 { z } else { 0.1 * z };
                logit += w2 * act;
            }
            out.push(logit);
        }
        Ok(out)
    }

    fn byte_size(&self) -> usize {
        std::mem::size_of::<Self>() + self.mlp_w1.len() * std::mem::size_of::<f32>()
    }
}

/// Bundle-backed [`turbo::rerank::Reranker`] cross-encoder scorer (#707).
pub struct BundleCrossEncoder {
    session: Mutex<Reranker>,
    options: RerankOptions,
}

impl BundleCrossEncoder {
    pub fn open(bundle_dir: &Path, threads: u32, options: RerankOptions) -> Result<Self> {
        let session = Reranker::open(bundle_dir, threads).map_err(|e| e.message)?;
        Ok(Self { session: Mutex::new(session), options })
    }
}

impl CrossEncoderScorer for BundleCrossEncoder {
    fn score_candidates(&self, corpus: &EncodedCorpus, query: &EncodedQuery, candidates: &[u32]) -> Result<Vec<f32>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let pairs: Vec<(&str, &str)> =
            candidates.iter().map(|&c| (query.text.as_str(), corpus.docs[c as usize].text.as_str())).collect();
        let mut guard = self.session.lock().map_err(|_| "reranker mutex poisoned".to_owned())?;
        let res = guard.score(&pairs, self.options).map_err(|e| e.message)?;
        Ok(res.scores)
    }

    fn byte_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

pub(crate) struct BuiltArm1 {
    pub index: Bm25PostingIndex,
    pub index_bytes: usize,
    pub build_ms: f64,
}

pub(crate) struct BuiltArm2<'a> {
    pub space: Space,
    pub static_index: VectorIndex<u32>,
    pub scorer: &'a dyn CrossEncoderScorer,
    pub index_bytes: usize,
    pub build_ms: f64,
}

pub(crate) struct BuiltArm3 {
    pub space: Space,
    pub sparse_index: SparsePostingIndex,
    pub token_index: TokenIndex<u32>,
    pub index_bytes: usize,
    pub sparse_bytes: usize,
    pub token_bytes: usize,
    pub build_ms: f64,
}

pub(crate) struct BuiltArm4 {
    pub space: Space,
    pub adapter: SplateAdapter,
    pub splate_index: SparsePostingIndex,
    pub token_index: TokenIndex<u32>,
    pub index_bytes: usize,
    pub first_stage_bytes: usize,
    pub token_bytes: usize,
    pub build_ms: f64,
}

fn bm25_index_bytes(corpus: &EncodedCorpus, index: &Bm25PostingIndex) -> usize {
    let mut postings = 0usize;
    let mut dict_bytes = 0usize;
    let mut seen = std::collections::BTreeSet::new();
    for d in &corpus.docs {
        for t in d.bm25_vocab.text.iter() {
            if seen.insert(t.to_owned()) {
                dict_bytes += t.len() + 24;
                postings += index.postings(t).len();
            }
        }
    }
    dict_bytes + postings * std::mem::size_of::<(u32, u32)>() + index.rows() * std::mem::size_of::<u32>()
}

fn token_index_bytes(index: &TokenIndex<u32>) -> usize {
    let dim = index.space().dim as usize;
    let bpr_bytes = dim.div_ceil(16) * 64;
    let token_vec_bytes = index.total_tokens() * bpr_bytes;
    let source_bytes = index.total_tokens() * std::mem::size_of::<turbo_vector_index::Source>();
    let offset_bytes = (index.len() + 1) * std::mem::size_of::<u32>() + index.len() * std::mem::size_of::<u32>();
    token_vec_bytes + source_bytes + offset_bytes
}

pub(crate) fn build_arm1(corpus: &EncodedCorpus) -> Result<BuiltArm1> {
    let t0 = Instant::now();
    let mut builder = Bm25PostingIndexBuilder::new();
    for d in &corpus.docs {
        builder.add(&d.bm25_stats, &d.bm25_vocab)?;
    }
    let index = builder.build();
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let index_bytes = bm25_index_bytes(corpus, &index);
    Ok(BuiltArm1 { index, index_bytes, build_ms })
}

pub(crate) fn build_arm2<'a>(corpus: &EncodedCorpus, scorer: &'a dyn CrossEncoderScorer) -> Result<BuiltArm2<'a>> {
    let t0 = Instant::now();
    let space = Space::new(corpus.model.clone(), corpus.dim as u32, Normalization::Unit);
    let mut builder = IndexBuilder::new(space.clone(), Metric::Cosine).map_err(|e| e.to_string())?;
    for (i, d) in corpus.docs.iter().enumerate() {
        builder.push(&d.static_vec, i as u32).map_err(|e| e.to_string())?;
    }
    let static_index = builder.build();
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let bpr_bytes = corpus.dim.div_ceil(16) * 64;
    let vec_bytes = corpus.docs.len() * (bpr_bytes + std::mem::size_of::<u32>());
    let token_payload_bytes: usize = corpus.docs.iter().map(|d| d.token_vecs.len() * std::mem::size_of::<f32>()).sum();
    let index_bytes = vec_bytes + token_payload_bytes + scorer.byte_size();
    Ok(BuiltArm2 { space, static_index, scorer, index_bytes, build_ms })
}

pub(crate) fn build_arm3(corpus: &EncodedCorpus) -> Result<BuiltArm3> {
    let t0 = Instant::now();
    let space = Space::new(corpus.model.clone(), corpus.dim as u32, Normalization::Unit);
    let mut s_builder = SparsePostingIndexBuilder::new();
    let mut t_builder = TokenIndexBuilder::new(space.clone()).map_err(|e| e.to_string())?;
    for (i, d) in corpus.docs.iter().enumerate() {
        s_builder.add(&d.splade_terms)?;
        t_builder.push(&d.token_vecs, i as u32).map_err(|e| e.to_string())?;
    }
    let sparse_index = s_builder.build();
    let token_index = t_builder.build();
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let sparse_bytes = sparse_index.byte_size();
    let token_bytes = token_index_bytes(&token_index);
    let index_bytes = sparse_bytes + token_bytes;
    Ok(BuiltArm3 { space, sparse_index, token_index, index_bytes, sparse_bytes, token_bytes, build_ms })
}

pub(crate) fn build_arm4(corpus: &EncodedCorpus, config: &SplateConfig) -> Result<BuiltArm4> {
    let space = Space::new(corpus.model.clone(), corpus.dim as u32, Normalization::Unit);
    let adapter = SplateAdapter::from_basis(
        &corpus.vocab_basis,
        corpus.vocab_size,
        corpus.dim,
        config.splate_scale,
        config.splate_bias,
    )?;
    let mut doc_terms = Vec::with_capacity(corpus.docs.len());
    for d in &corpus.docs {
        doc_terms.push(adapter.encode_tokens(&d.token_vecs, &config.splate_doc_cut)?);
    }
    let t0 = Instant::now();
    let mut s_builder = SparsePostingIndexBuilder::new();
    let mut t_builder = TokenIndexBuilder::new(space.clone()).map_err(|e| e.to_string())?;
    for (i, (d, terms)) in corpus.docs.iter().zip(&doc_terms).enumerate() {
        s_builder.add(terms)?;
        t_builder.push(&d.token_vecs, i as u32).map_err(|e| e.to_string())?;
    }
    let splate_index = s_builder.build();
    let token_index = t_builder.build();
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let first_stage_bytes = adapter.byte_size() + splate_index.byte_size();
    let token_bytes = token_index_bytes(&token_index);
    let index_bytes = first_stage_bytes + token_bytes;
    Ok(BuiltArm4 { space, adapter, splate_index, token_index, index_bytes, first_stage_bytes, token_bytes, build_ms })
}

impl BuiltArm1 {
    pub fn query(&self, corpus: &EncodedCorpus, q: &EncodedQuery, config: &SplateConfig) -> (Ranking, usize, u64, u64) {
        let refs: Vec<&str> = q.bm25_terms.iter().map(String::as_str).collect();
        let t0 = Instant::now();
        let raw = self.index.top_k(config.bm25, &refs, config.top_k);
        let retrieve_ns = t0.elapsed().as_nanos() as u64;
        let hits: Vec<Hit> = raw
            .into_iter()
            .take(config.top_k)
            .map(|(row, score)| Hit { doc: corpus.docs[row as usize].id.clone(), score })
            .collect();
        let pool = hits.len();
        (Ranking { query: q.id.clone(), hits }, pool, retrieve_ns, 0)
    }
}

impl BuiltArm2<'_> {
    pub fn query(
        &self,
        corpus: &EncodedCorpus,
        q: &EncodedQuery,
        config: &SplateConfig,
    ) -> Result<(Ranking, usize, u64, u64)> {
        let t0 = Instant::now();
        let raw = self.static_index.top_k(&self.space, &q.static_vec, config.shortlist_k).map_err(|e| e.to_string())?;
        let shortlist: Vec<u32> = raw.into_iter().map(|h| h.row).take(config.shortlist_k).collect();
        let retrieve_ns = t0.elapsed().as_nanos() as u64;
        let pool = shortlist.len();
        let t1 = Instant::now();
        let ce_scores = self.scorer.score_candidates(corpus, q, &shortlist)?;
        let mut order: Vec<usize> = (0..shortlist.len()).collect();
        order.sort_by(|&a, &b| ce_scores[b].total_cmp(&ce_scores[a]).then_with(|| a.cmp(&b)));
        let rerank_ns = t1.elapsed().as_nanos() as u64;
        let hits = order
            .into_iter()
            .take(config.top_k)
            .map(|i| Hit { doc: corpus.docs[shortlist[i] as usize].id.clone(), score: f64::from(ce_scores[i]) })
            .collect();
        Ok((Ranking { query: q.id.clone(), hits }, pool, retrieve_ns, rerank_ns))
    }
}

impl BuiltArm3 {
    pub fn query(
        &self,
        corpus: &EncodedCorpus,
        q: &EncodedQuery,
        config: &SplateConfig,
    ) -> Result<(Ranking, usize, u64, u64)> {
        let t0 = Instant::now();
        let raw = self.sparse_index.top_k(&q.splade_terms, config.shortlist_k);
        let shortlist: Vec<u32> = raw.into_iter().map(|(row, _)| row).take(config.shortlist_k).collect();
        let retrieve_ns = t0.elapsed().as_nanos() as u64;
        let pool = shortlist.len();
        if shortlist.is_empty() {
            return Ok((Ranking { query: q.id.clone(), hits: Vec::new() }, 0, retrieve_ns, 0));
        }
        let t1 = Instant::now();
        let reranked =
            self.token_index.rerank(&self.space, &q.token_vecs, &shortlist, config.top_k).map_err(|e| e.to_string())?;
        let rerank_ns = t1.elapsed().as_nanos() as u64;
        let hits = reranked
            .into_iter()
            .map(|h| Hit {
                doc: corpus.docs[*self.token_index.meta(h.candidate) as usize].id.clone(),
                score: f64::from(h.score),
            })
            .collect();
        Ok((Ranking { query: q.id.clone(), hits }, pool, retrieve_ns, rerank_ns))
    }
}

impl BuiltArm4 {
    pub fn query(
        &self,
        corpus: &EncodedCorpus,
        q: &EncodedQuery,
        q_splate_terms: &[turbo::sparse::SparseTerm],
        config: &SplateConfig,
    ) -> Result<(Ranking, usize, u64, u64)> {
        let t0 = Instant::now();
        let raw = self.splate_index.top_k(q_splate_terms, config.shortlist_k);
        let shortlist: Vec<u32> = raw.into_iter().map(|(row, _)| row).take(config.shortlist_k).collect();
        let retrieve_ns = t0.elapsed().as_nanos() as u64;
        let pool = shortlist.len();
        if shortlist.is_empty() {
            return Ok((Ranking { query: q.id.clone(), hits: Vec::new() }, 0, retrieve_ns, 0));
        }
        let t1 = Instant::now();
        let reranked =
            self.token_index.rerank(&self.space, &q.token_vecs, &shortlist, config.top_k).map_err(|e| e.to_string())?;
        let rerank_ns = t1.elapsed().as_nanos() as u64;
        let hits = reranked
            .into_iter()
            .map(|h| Hit {
                doc: corpus.docs[*self.token_index.meta(h.candidate) as usize].id.clone(),
                score: f64::from(h.score),
            })
            .collect();
        Ok((Ranking { query: q.id.clone(), hits }, pool, retrieve_ns, rerank_ns))
    }
}

fn percentile_us(samples_ns: &mut [u64], pct: usize) -> f64 {
    if samples_ns.is_empty() {
        return 0.0;
    }
    samples_ns.sort_unstable();
    let idx = ((samples_ns.len() - 1) * pct) / 100;
    samples_ns[idx] as f64 / 1000.0
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_arm(
    arm: CascadeArm,
    shortlist_k: usize,
    rankings: Vec<Ranking>,
    pools: &[usize],
    encode_ns: &mut [u64],
    retrieve_ns: &mut [u64],
    rerank_ns: &mut [u64],
    samples_ns: &mut [u64],
    index_bytes: usize,
    first_stage_bytes: usize,
    token_index_bytes: usize,
    build_ms: f64,
    qrels: &Qrels,
) -> Result<ArmReport> {
    let run = Run::new(rankings)?;
    let scores = evaluate(&run, qrels)?;
    let mean_pool_size = if pools.is_empty() { 0.0 } else { pools.iter().sum::<usize>() as f64 / pools.len() as f64 };
    let encode_p50_us = percentile_us(encode_ns, 50);
    let retrieve_p50_us = percentile_us(retrieve_ns, 50);
    let rerank_p50_us = percentile_us(rerank_ns, 50);
    let p50_us = percentile_us(samples_ns, 50);
    let p99_us = percentile_us(samples_ns, 99);
    Ok(ArmReport {
        arm,
        shortlist_k,
        scores,
        index_bytes,
        first_stage_bytes,
        token_index_bytes,
        build_ms,
        encode_p50_us,
        retrieve_p50_us,
        rerank_p50_us,
        p50_us,
        p99_us,
        mean_pool_size,
    })
}
