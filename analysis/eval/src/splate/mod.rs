//! Four-arm SPLATE retrieval cascade study (`#711`).
//!
//! Compares four retrieval architectures on identical corpus, query, and relevance
//! judgments:
//! - **Arm 1 (`Bm25Only`)**: BM25 keyword retrieval over [`turbo_chunker::terms::Bm25PostingIndex`].
//! - **Arm 2 (`StaticThenCrossEncoder`)**: Static dense embeddings (`turbo-vector-index`) followed
//!   by a top-$C$ cross-encoder reranker (`turbo::rerank::Reranker` or [`InteractionCrossEncoder`]).
//! - **Arm 3 (`LearnedSparseThenMaxSim`)**: Learned sparse expansion ([`SparsePostingIndex`])
//!   followed by late-interaction `MaxSim` over the top-$C$ candidates (`TokenIndex`).
//! - **Arm 4 (`SplateThenMaxSim`)**: Sparse late-interaction adapter ([`SplateAdapter`]) mapping
//!   ColBERT-style token vectors into vocabulary space ([`SparsePostingIndex`]) for candidate
//!   generation, followed by exact `MaxSim` reranking over the same token vectors.

pub mod arms;
pub mod corpus;
pub mod posting;

use std::fmt::Write as _;
use std::time::Instant;

use turbo::sparse::SparseOptions;
use turbo_chunker::terms::Bm25;

use crate::Result;
use crate::metrics::{DocScores, Qrels, Ranking};
use crate::set::LoadedSet;

pub use arms::{BundleCrossEncoder, CrossEncoderScorer, InteractionCrossEncoder};
pub use corpus::{EncodedCorpus, EncodedDoc, EncodedQuery, encode_loaded_set, synthetic_benchmark_set};
pub use posting::{SparsePostingIndex, SparsePostingIndexBuilder, SplateAdapter};

use arms::{build_arm1, build_arm2, build_arm3, build_arm4, evaluate_arm};

/// Identifiers for the four retrieval cascade arms compared in `#711`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CascadeArm {
    /// Arm 1: BM25 lexical baseline (`Bm25PostingIndex`).
    Bm25Only,
    /// Arm 2: Static dense first stage (`VectorIndex`) + top-$C$ cross-encoder reranker.
    StaticThenCrossEncoder,
    /// Arm 3: Learned sparse first stage (`SparsePostingIndex`) + top-$C$ `MaxSim` reranker.
    LearnedSparseThenMaxSim,
    /// Arm 4: SPLATE (`SplateAdapter` + `SparsePostingIndex`) + top-$C$ `MaxSim` reranker.
    SplateThenMaxSim,
}

impl CascadeArm {
    /// Ordered list of all four study arms.
    pub const ALL: [Self; 4] =
        [Self::Bm25Only, Self::StaticThenCrossEncoder, Self::LearnedSparseThenMaxSim, Self::SplateThenMaxSim];

    /// Short machine-readable identifier for tables and reports.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Bm25Only => "arm1-bm25",
            Self::StaticThenCrossEncoder => "arm2-static-ce",
            Self::LearnedSparseThenMaxSim => "arm3-splade-maxsim",
            Self::SplateThenMaxSim => "arm4-splate-maxsim",
        }
    }

    /// Human-readable label describing the cascade stages.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Bm25Only => "Arm 1: BM25 only",
            Self::StaticThenCrossEncoder => "Arm 2: Static dense + Cross-encoder",
            Self::LearnedSparseThenMaxSim => "Arm 3: Learned sparse + ColBERT MaxSim",
            Self::SplateThenMaxSim => "Arm 4: SPLATE adapter + ColBERT MaxSim",
        }
    }
}

/// Configuration for a four-arm SPLATE evaluation run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplateConfig {
    /// Embedding dimension for static and contextual token vectors (multiple of 8).
    pub dim: usize,
    /// Maximum vocabulary size for learned sparse and SPLATE projections.
    pub max_vocab: usize,
    /// Number of candidates ($C$) retrieved by stage 1 and rescored by stage 2.
    pub shortlist_k: usize,
    /// Final number of ranked documents ($k$) returned per query.
    pub top_k: usize,
    /// BM25 hyperparameters (`k1`, `b`).
    pub bm25: Bm25,
    /// Sparsity cut (`top_k`, `floor`) for SPLADE document representations.
    pub splade_doc_cut: SparseOptions,
    /// Sparsity cut (`top_k`, `floor`) for SPLADE query representations.
    pub splade_query_cut: SparseOptions,
    /// Sparsity cut (`top_k`, `floor`) for SPLATE document representations.
    pub splate_doc_cut: SparseOptions,
    /// Sparsity cut (`top_k`, `floor`) for SPLATE query representations.
    pub splate_query_cut: SparseOptions,
    /// Gain factor applied to the SPLATE adapter projection weights.
    pub splate_scale: f32,
    /// Uniform bias shift applied before ReLU in the SPLATE adapter.
    pub splate_bias: f32,
    /// Number of interleaved measurement rounds across all four arms (`>= 1`).
    pub rounds: usize,
}

impl Default for SplateConfig {
    fn default() -> Self {
        Self {
            dim: 32,
            max_vocab: 2_048,
            shortlist_k: 50,
            top_k: 100,
            bm25: Bm25::LUCENE,
            splade_doc_cut: SparseOptions { top_k: 64, floor: 0.05 },
            splade_query_cut: SparseOptions { top_k: 24, floor: 0.05 },
            splate_doc_cut: SparseOptions { top_k: 64, floor: 0.05 },
            splate_query_cut: SparseOptions { top_k: 24, floor: 0.05 },
            splate_scale: 2.2,
            splate_bias: -0.65,
            rounds: 5,
        }
    }
}

/// Evaluation and systems cost report for a single cascade arm.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmReport {
    /// Cascade arm identifier.
    pub arm: CascadeArm,
    /// Candidate shortlist depth ($C$) evaluated by the second stage.
    pub shortlist_k: usize,
    /// Document-level retrieval quality scores (`ndcg_at_10`, `recall.{at_1, at_5, at_10, at_100}`).
    pub scores: DocScores,
    /// Total in-memory index footprint in bytes across all structures required by the arm.
    pub index_bytes: usize,
    /// First-stage index bytes (BM25 postings, static dense vectors, or sparse postings + adapter).
    pub first_stage_bytes: usize,
    /// Late-interaction token index bytes (`0` for Arms 1 and 2).
    pub token_index_bytes: usize,
    /// Wall-clock index structure construction time in milliseconds from encoded document inputs.
    pub build_ms: f64,
    /// Median (`p50`) per-query representation/adapter encoding latency in microseconds.
    pub encode_p50_us: f64,
    /// Median (`p50`) Stage-1 shortlist retrieval latency in microseconds.
    pub retrieve_p50_us: f64,
    /// Median (`p50`) Stage-2 reranking latency in microseconds (`0.0` for Arm 1).
    pub rerank_p50_us: f64,
    /// Median (`p50`) per-query search (`retrieve + rerank`) latency in microseconds.
    pub p50_us: f64,
    /// Tail (`p99`) per-query search (`retrieve + rerank`) latency in microseconds.
    pub p99_us: f64,
    /// Mean number of candidates passed to stage 2 per query.
    pub mean_pool_size: f64,
}

/// Complete four-arm comparison report on a single dataset and shortlist depth.
#[derive(Debug, Clone, PartialEq)]
pub struct StudyReport {
    /// Dataset name.
    pub dataset: String,
    /// Number of indexed documents.
    pub doc_count: usize,
    /// Number of evaluated queries.
    pub query_count: usize,
    /// Study configuration used for this run.
    pub config: SplateConfig,
    /// Reports for Arms 1 through 4 in order.
    pub arms: [ArmReport; 4],
}

impl StudyReport {
    /// Formats the four-arm comparison as a GitHub-flavored Markdown table.
    #[must_use]
    pub fn to_markdown_table(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "| Arm | C | Recall@1 | Recall@5 | Recall@10 | Recall@100 | nDCG@10 | Encode p50 (us) | Retrieve p50 (us) | Rerank p50 (us) | Search p50 (us) | Search p99 (us) | Index (KiB) | Build (ms) |"
        );
        let _ = writeln!(
            out,
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        );
        for rep in &self.arms {
            let _ = writeln!(
                out,
                "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.2} |",
                rep.arm.label(),
                rep.shortlist_k,
                rep.scores.recall.at_1,
                rep.scores.recall.at_5,
                rep.scores.recall.at_10,
                rep.scores.recall.at_100,
                rep.scores.ndcg_at_10,
                rep.encode_p50_us,
                rep.retrieve_p50_us,
                rep.rerank_p50_us,
                rep.p50_us,
                rep.p99_us,
                rep.index_bytes as f64 / 1024.0,
                rep.build_ms,
            );
        }
        out
    }
}

/// Runs the four-arm SPLATE study on `set` with the default synthetic interaction cross-encoder.
pub fn run_study(set: &LoadedSet, config: &SplateConfig) -> Result<StudyReport> {
    let encoded =
        encode_loaded_set(set, config.dim, config.max_vocab, &config.splade_doc_cut, &config.splade_query_cut)?;
    let ce = InteractionCrossEncoder::new(encoded.dim);
    run_study_with_encoded(&encoded, &set.qrels, &ce, config)
}

/// Runs the four-arm SPLATE study across multiple shortlist depths $C$ while encoding the
/// corpus once.
pub fn run_shortlist_sweep(
    set: &LoadedSet,
    shortlist_depths: &[usize],
    base_config: &SplateConfig,
) -> Result<Vec<StudyReport>> {
    if shortlist_depths.is_empty() {
        return Err("shortlist_depths must not be empty".into());
    }
    let encoded = encode_loaded_set(
        set,
        base_config.dim,
        base_config.max_vocab,
        &base_config.splade_doc_cut,
        &base_config.splade_query_cut,
    )?;
    let ce = InteractionCrossEncoder::new(encoded.dim);
    let mut reports = Vec::with_capacity(shortlist_depths.len());
    for &c in shortlist_depths {
        let cfg = SplateConfig { shortlist_k: c, top_k: base_config.top_k.max(c), ..*base_config };
        reports.push(run_study_with_encoded(&encoded, &set.qrels, &ce, &cfg)?);
    }
    Ok(reports)
}

/// Runs the four-arm SPLATE study using a caller-supplied [`EncodedCorpus`] and
/// [`CrossEncoderScorer`].
pub fn run_study_with_encoded(
    encoded: &EncodedCorpus,
    qrels: &Qrels,
    cross_encoder: &dyn CrossEncoderScorer,
    config: &SplateConfig,
) -> Result<StudyReport> {
    if config.shortlist_k == 0 || config.top_k == 0 {
        return Err("shortlist_k and top_k must be >= 1".into());
    }
    let rounds = config.rounds.max(1);

    let arm1 = build_arm1(encoded)?;
    let arm2 = build_arm2(encoded, cross_encoder)?;
    let arm3 = build_arm3(encoded)?;
    let arm4 = build_arm4(encoded, config)?;

    let q_len = encoded.queries.len();
    let mut q_splate_terms = Vec::with_capacity(q_len);
    let mut enc_ns: [Vec<u64>; 4] =
        [Vec::with_capacity(q_len), Vec::with_capacity(q_len), Vec::with_capacity(q_len), Vec::with_capacity(q_len)];
    for q in &encoded.queries {
        let t_proj = Instant::now();
        let terms = arm4.adapter.encode_tokens(&q.token_vecs, &config.splate_query_cut)?;
        let proj_ns = t_proj.elapsed().as_nanos() as u64;
        q_splate_terms.push(terms);
        enc_ns[0].push(q.encode_ns[0]);
        enc_ns[1].push(q.encode_ns[1]);
        enc_ns[2].push(q.encode_ns[2]);
        enc_ns[3].push(q.encode_ns[3] + proj_ns);
    }

    let mut ret_ns: [Vec<u64>; 4] = std::array::from_fn(|_| Vec::with_capacity(q_len * rounds));
    let mut rer_ns: [Vec<u64>; 4] = std::array::from_fn(|_| Vec::with_capacity(q_len * rounds));
    let mut tot_ns: [Vec<u64>; 4] = std::array::from_fn(|_| Vec::with_capacity(q_len * rounds));
    let mut ranks: [Vec<Ranking>; 4] = std::array::from_fn(|_| Vec::with_capacity(q_len));
    let mut pools: [Vec<usize>; 4] = std::array::from_fn(|_| Vec::with_capacity(q_len));

    for round in 0..rounds {
        for (qi, q) in encoded.queries.iter().enumerate() {
            let (r1, p1, rt1, rk1) = arm1.query(encoded, q, config);
            let (r2, p2, rt2, rk2) = arm2.query(encoded, q, config)?;
            let (r3, p3, rt3, rk3) = arm3.query(encoded, q, config)?;
            let (r4, p4, rt4, rk4) = arm4.query(encoded, q, &q_splate_terms[qi], config)?;

            for (idx, (rt, rk)) in [(rt1, rk1), (rt2, rk2), (rt3, rk3), (rt4, rk4)].into_iter().enumerate() {
                ret_ns[idx].push(rt);
                rer_ns[idx].push(rk);
                tot_ns[idx].push(rt + rk);
            }
            if round == 0 {
                for (idx, (r, p)) in [(r1, p1), (r2, p2), (r3, p3), (r4, p4)].into_iter().enumerate() {
                    ranks[idx].push(r);
                    pools[idx].push(p);
                }
            }
        }
    }

    let [ranks1, ranks2, ranks3, ranks4] = ranks;
    let rep1 = evaluate_arm(
        CascadeArm::Bm25Only,
        config.shortlist_k,
        ranks1,
        &pools[0],
        &mut enc_ns[0],
        &mut ret_ns[0],
        &mut rer_ns[0],
        &mut tot_ns[0],
        arm1.index_bytes,
        arm1.index_bytes,
        0,
        arm1.build_ms,
        qrels,
    )?;
    let rep2 = evaluate_arm(
        CascadeArm::StaticThenCrossEncoder,
        config.shortlist_k,
        ranks2,
        &pools[1],
        &mut enc_ns[1],
        &mut ret_ns[1],
        &mut rer_ns[1],
        &mut tot_ns[1],
        arm2.index_bytes,
        arm2.index_bytes,
        0,
        arm2.build_ms,
        qrels,
    )?;
    let rep3 = evaluate_arm(
        CascadeArm::LearnedSparseThenMaxSim,
        config.shortlist_k,
        ranks3,
        &pools[2],
        &mut enc_ns[2],
        &mut ret_ns[2],
        &mut rer_ns[2],
        &mut tot_ns[2],
        arm3.index_bytes,
        arm3.sparse_bytes,
        arm3.token_bytes,
        arm3.build_ms,
        qrels,
    )?;
    let rep4 = evaluate_arm(
        CascadeArm::SplateThenMaxSim,
        config.shortlist_k,
        ranks4,
        &pools[3],
        &mut enc_ns[3],
        &mut ret_ns[3],
        &mut rer_ns[3],
        &mut tot_ns[3],
        arm4.index_bytes,
        arm4.first_stage_bytes,
        arm4.token_bytes,
        arm4.build_ms,
        qrels,
    )?;

    Ok(StudyReport {
        dataset: encoded.set_name.clone(),
        doc_count: encoded.docs.len(),
        query_count: encoded.queries.len(),
        config: *config,
        arms: [rep1, rep2, rep3, rep4],
    })
}
