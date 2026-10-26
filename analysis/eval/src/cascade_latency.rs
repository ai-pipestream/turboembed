//! Per-query cascade latency measurement for the trained-encoder
//! cascade record (#879, speed bar of #712).
//!
//! `EvalRecord` carries whole-set stage times only (cost.rs), so the
//! p50/p95 per-query latencies the speed bar is stated in are measured
//! here instead: chunks and indexes are built once (untimed, batched),
//! then every arm answers every query one at a time, arms interleaved
//! per query per round, with query encoding inside the timed span.
//! Arms: static only, contextual only, and the corpus cascade
//! (cascade.rs) at each candidate count. Single-query searches keep
//! the measurement single-threaded in effect: no batching anywhere on
//! the timed path. Quality from the first round's rankings is reported
//! beside the latencies as a sanity check; the committed quality
//! records come from `turbo-eval run`.

use std::collections::BTreeSet;
use std::time::Instant;

use turbo_vector_index::{IndexBuilder, Metric, Normalization, Space, VectorIndex};

use crate::Result;
use crate::analysis::Analysis;
use crate::cascade::{Cascade, CascadeSearch};
use crate::chunkers::Chunker;
use crate::embedder::{Embedder, Role};
use crate::metrics::{DocScores, Hit, Ranking, Run, evaluate};
use crate::set::LoadedSet;

/// One measured arm: a single tier, or the cascade at a candidate count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyArm {
    StaticOnly,
    ContextualOnly,
    Cascade(usize),
}

impl LatencyArm {
    pub fn label(self) -> String {
        match self {
            LatencyArm::StaticOnly => "static only".into(),
            LatencyArm::ContextualOnly => "contextual only".into(),
            LatencyArm::Cascade(c) => format!("cascade C={c}"),
        }
    }
}

/// Latencies (microseconds) and first-round quality of one arm on one set.
#[derive(Debug, Clone)]
pub struct ArmLatency {
    pub arm: LatencyArm,
    pub encode_p50_us: f64,
    pub encode_p95_us: f64,
    pub search_p50_us: f64,
    pub search_p95_us: f64,
    /// Encode plus search, the speed bar's quantity.
    pub total_p50_us: f64,
    pub total_p95_us: f64,
    pub scores: DocScores,
}

/// The latency report of one set: every arm, plus the corpus shape.
#[derive(Debug, Clone)]
pub struct LatencyReport {
    pub set: String,
    pub docs: usize,
    pub chunks: usize,
    pub queries: usize,
    pub rounds: usize,
    pub arms: Vec<ArmLatency>,
}

impl LatencyReport {
    pub fn to_markdown_table(&self) -> String {
        let mut out = String::new();
        out.push_str(
            "| Arm | Encode p50 (us) | Encode p95 (us) | Search p50 (us) | Search p95 (us) | Total p50 (us) | Total p95 (us) | Recall@10 | nDCG@10 |\n",
        );
        out.push_str("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for a in &self.arms {
            out.push_str(&format!(
                "| {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.4} | {:.4} |\n",
                a.arm.label(),
                a.encode_p50_us,
                a.encode_p95_us,
                a.search_p50_us,
                a.search_p95_us,
                a.total_p50_us,
                a.total_p95_us,
                a.scores.recall.at_10,
                a.scores.ndcg_at_10,
            ));
        }
        out
    }
}

fn percentile(sorted_us: &[u64], q: f64) -> f64 {
    if sorted_us.is_empty() {
        return 0.0;
    }
    let at = ((sorted_us.len() as f64 * q).ceil() as usize).clamp(1, sorted_us.len()) - 1;
    sorted_us[at] as f64 / 1000.0
}

fn embed_all(embedder: &mut dyn Embedder, texts: &[String], role: Role) -> Result<Vec<f32>> {
    let dim = embedder.dim();
    let mut out = Vec::with_capacity(texts.len() * dim);
    let mut part = Vec::new();
    for group in texts.chunks(1024) {
        let refs: Vec<&str> = group.iter().map(String::as_str).collect();
        embedder.embed(&refs, role, &mut part)?;
        out.extend_from_slice(&part);
    }
    Ok(out)
}

fn dense_index(space: Space, vectors: &[f32], rows: usize) -> Result<VectorIndex<u32>> {
    let dim = space.dim as usize;
    let mut b = IndexBuilder::new(space, Metric::Cosine).map_err(|e| e.to_string())?;
    for (i, v) in vectors.chunks_exact(dim).enumerate().take(rows) {
        if v.iter().any(|x| !x.is_finite()) || v.iter().all(|&x| x == 0.0) {
            continue;
        }
        b.push(v, i as u32).map_err(|e| e.to_string())?;
    }
    Ok(b.build())
}

/// Document ranking of one dense chunk search: best chunk per
/// document in score order, the query's own document left out.
fn dense_ranking(
    index: &VectorIndex<u32>,
    space: &Space,
    qvec: &[f32],
    doc_of: &[u32],
    ids: &[String],
    qid: &str,
) -> Result<Ranking> {
    let fetch = (ids.len()).min(400);
    let hits = index.top_k(space, qvec, fetch).map_err(|e| e.to_string())?;
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for h in hits {
        let doc = doc_of[h.row as usize] as usize;
        if ids[doc] == qid || !seen.insert(doc) {
            continue;
        }
        out.push(Hit { doc: ids[doc].clone(), score: f64::from(h.score) });
        if out.len() >= 100 {
            break;
        }
    }
    Ok(Ranking { query: qid.into(), hits: out })
}

/// Measures every arm in `depths` (plus the two single tiers) on `set`:
/// chunked by `chunker` over `analysis`, static chunks and queries in
/// `statics`' space, contextual ones in `contextual`'s. `rounds`
/// interleaved repetitions; round 0's rankings are scored.
#[allow(clippy::too_many_arguments)]
pub fn measure(
    set: &LoadedSet,
    analysis: &Analysis,
    chunker: &dyn Chunker,
    statics: &mut dyn Embedder,
    contextual: &mut dyn Embedder,
    depths: &[usize],
    rounds: usize,
) -> Result<LatencyReport> {
    set.check()?;
    if depths.is_empty() || rounds == 0 {
        return Err("depths and rounds must be non-empty".into());
    }
    // Untimed setup, mirroring pipeline::evaluate's chunking.
    let docs = set
        .docs
        .iter()
        .map(|d| analysis.analyze(&d.text, contextual).map_err(|e| format!("document {}: {e}", d.id)))
        .collect::<Result<Vec<_>>>()?;
    let (mut doc_of, mut spans) = (Vec::new(), Vec::new());
    for (i, (d, doc)) in set.docs.iter().zip(&docs).enumerate() {
        let (chunks, pooled) = chunker.chunk_vectors(doc, contextual).map_err(|e| format!("document {}: {e}", d.id))?;
        if pooled.is_some() {
            return Err("vectors a chunker pools are retrieved by dense retrieval only".into());
        }
        for s in chunks {
            doc_of.push(i as u32);
            spans.push(s);
        }
    }
    drop(docs);
    if spans.is_empty() {
        return Err(format!("set {}: the chunker made no chunks", set.name));
    }
    let ids: Vec<String> = set.docs.iter().map(|d| d.id.clone()).collect();
    let texts: Vec<String> =
        doc_of.iter().zip(&spans).map(|(&d, s)| set.docs[d as usize].text[s.start..s.end].to_owned()).collect();
    let rows = texts.len();
    let static_vectors = embed_all(statics, &texts, Role::Document)?;
    let contextual_vectors = embed_all(contextual, &texts, Role::Document)?;
    let static_space = Space::new(statics.identity().model_ref(), statics.dim() as u32, Normalization::None);
    let contextual_space = Space::new(contextual.identity().model_ref(), contextual.dim() as u32, Normalization::None);
    let static_index = dense_index(static_space.clone(), &static_vectors, rows)?;
    let contextual_index = dense_index(contextual_space.clone(), &contextual_vectors, rows)?;
    let template = Cascade {
        candidates: depths[0],
        model: turbo_chunker::terms::Bm25::LUCENE,
        terms: crate::lexical::TermRule::Folded,
    };
    let cascade_index =
        template.build(static_space.clone(), &static_vectors, contextual_space.clone(), &contextual_vectors, rows)?;
    let bm25_index = template.index(&texts, &doc_of)?;
    drop(texts);
    let cascades: Vec<Cascade> =
        depths.iter().map(|&c| Cascade { candidates: c, model: template.model, terms: template.terms }).collect();

    let arms: Vec<LatencyArm> = [LatencyArm::StaticOnly, LatencyArm::ContextualOnly]
        .into_iter()
        .chain(depths.iter().map(|&c| LatencyArm::Cascade(c)))
        .collect();
    let n_arms = arms.len();
    let mut enc_ns: Vec<Vec<u64>> = vec![Vec::new(); n_arms];
    let mut sea_ns: Vec<Vec<u64>> = vec![Vec::new(); n_arms];
    let mut tot_ns: Vec<Vec<u64>> = vec![Vec::new(); n_arms];
    let mut rankings: Vec<Vec<Ranking>> = vec![Vec::new(); n_arms];
    let (sdim, cdim) = (statics.dim(), contextual.dim());
    let (mut sv, mut cv) = (Vec::new(), Vec::new());

    for round in 0..rounds {
        for q in &set.queries {
            for (ai, arm) in arms.iter().enumerate() {
                let t0 = Instant::now();
                let enc: u64;
                let sea: u64;
                let ranking = match arm {
                    LatencyArm::StaticOnly => {
                        statics.embed(&[q.text.as_str()], Role::Query, &mut sv)?;
                        enc = t0.elapsed().as_nanos() as u64;
                        let t1 = Instant::now();
                        let r = dense_ranking(&static_index, &static_space, &sv[..sdim], &doc_of, &ids, &q.id)?;
                        sea = t1.elapsed().as_nanos() as u64;
                        r
                    }
                    LatencyArm::ContextualOnly => {
                        contextual.embed(&[q.text.as_str()], Role::Query, &mut cv)?;
                        enc = t0.elapsed().as_nanos() as u64;
                        let t1 = Instant::now();
                        let r = dense_ranking(&contextual_index, &contextual_space, &cv[..cdim], &doc_of, &ids, &q.id)?;
                        sea = t1.elapsed().as_nanos() as u64;
                        r
                    }
                    LatencyArm::Cascade(_) => {
                        statics.embed(&[q.text.as_str()], Role::Query, &mut sv)?;
                        contextual.embed(&[q.text.as_str()], Role::Query, &mut cv)?;
                        enc = t0.elapsed().as_nanos() as u64;
                        let cascade = &cascades[ai - 2];
                        let search = CascadeSearch {
                            bm25: &bm25_index,
                            doc: &doc_of,
                            span: &spans,
                            docs: &ids,
                            static_space: &static_space,
                            static_queries: &sv[..sdim],
                            contextual_space: &contextual_space,
                            contextual_queries: &cv[..cdim],
                        };
                        let t1 = Instant::now();
                        let (run, _chunks) =
                            cascade.search(&cascade_index, &search, &[(q.id.as_str(), q.text.as_str())])?;
                        sea = t1.elapsed().as_nanos() as u64;
                        Ranking { query: q.id.clone(), hits: run.hits(&q.id).to_vec() }
                    }
                };
                enc_ns[ai].push(enc);
                sea_ns[ai].push(sea);
                tot_ns[ai].push(enc + sea);
                if round == 0 {
                    rankings[ai].push(ranking);
                }
            }
        }
    }

    let mut out = Vec::with_capacity(n_arms);
    for (ai, arm) in arms.iter().enumerate() {
        for v in [&mut enc_ns[ai], &mut sea_ns[ai], &mut tot_ns[ai]] {
            v.sort_unstable();
        }
        let scores = evaluate(&Run::new(std::mem::take(&mut rankings[ai]))?, &set.qrels)?;
        out.push(ArmLatency {
            arm: *arm,
            encode_p50_us: percentile(&enc_ns[ai], 0.50),
            encode_p95_us: percentile(&enc_ns[ai], 0.95),
            search_p50_us: percentile(&sea_ns[ai], 0.50),
            search_p95_us: percentile(&sea_ns[ai], 0.95),
            total_p50_us: percentile(&tot_ns[ai], 0.50),
            total_p95_us: percentile(&tot_ns[ai], 0.95),
            scores,
        });
    }
    Ok(LatencyReport {
        set: set.name.clone(),
        docs: set.docs.len(),
        chunks: rows,
        queries: set.queries.len(),
        rounds,
        arms: out,
    })
}
