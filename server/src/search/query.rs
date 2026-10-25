//! Hybrid query AST execution, Reciprocal Rank Fusion (`1 / (k + rank)`),
//! `turbo-query` filter extraction, parent expansion, cross-encoder reranking,
//! `MatchedSpan` attribution, and `DocumentHeat` generation (`#1221`).

use std::collections::{BTreeMap, HashMap};

use turbo::{TURBO_TASK_RERANK, turbo_rerank_options, turbo_text, turbo_text_pair};
use turbo_query::FilterKind;

use crate::api::Result;
use crate::proto::search::query_node::Clause;
use crate::proto::search::{
    BoostClause, ChunkAttribution, DocumentHeat, ExtractedFilterProto, FusionStrategy, HeatSegment, JoinClause,
    JoinOperator, MatchedSpan, QueryNode, SearchHit, SearchRequest, SearchResponse, SemanticClause, SpanProto,
    TermClause, TermMatchMode,
};
use crate::search::clauses::{
    eval_filter_clause, eval_late_clause, eval_phrase_clause, eval_sparse_clause, eval_term_clause,
};
use crate::search::registry::{CompiledDense, HybridIndex, ScoreMap, bad_arg, parse_space, shared_query_profile};
use crate::state::State;

pub(crate) fn execute_search(
    resolved_index: String,
    ix: &HybridIndex,
    state: &State,
    req: SearchRequest,
) -> Result<SearchResponse> {
    let top_k = if req.top_k == 0 { 10usize } else { req.top_k as usize };
    if ix.chunks.is_empty() {
        return Ok(SearchResponse {
            resolved_index,
            hits: Vec::new(),
            document_heat: Vec::new(),
            analyzed_query_terms: Vec::new(),
            extracted_filters: Vec::new(),
        });
    }

    let mut analyzed_query_terms = Vec::new();
    let mut extracted_filters = Vec::new();
    let mut top_nodes: Vec<ScoreMap> = Vec::new();

    if !req.natural_query.is_empty() {
        let qa = shared_query_profile()
            .analyze(&req.natural_query)
            .map_err(|e| bad_arg(format!("query analysis failed: {e}")))?;
        analyzed_query_terms = qa.bm25_terms.clone();
        for f in &qa.filters {
            let (kind, summary) = match &f.kind {
                FilterKind::Temporal(t) => ("temporal", t.value_text.clone()),
                FilterKind::Money(m) => ("money", format!("{} {}", m.decimal_text, m.currency.as_str())),
                FilterKind::Quantity(q) => ("quantity", format!("{}{}", q.decimal_text, q.unit)),
                FilterKind::Place(p) => ("place", p.canonical_name.clone()),
                FilterKind::Glossary(g) => ("glossary", g.term.clone()),
                FilterKind::Entity(e) => ("entity", format!("{}:{}", e.entity_type, f.surface)),
            };
            extracted_filters.push(ExtractedFilterProto {
                kind: kind.to_owned(),
                summary,
                span: Some(SpanProto { start: f.span.start, end: f.span.end }),
            });
        }
        if !analyzed_query_terms.is_empty() {
            let term_clause = TermClause {
                terms: analyzed_query_terms.clone(),
                mode: TermMatchMode::Any as i32,
                top_k: (top_k * 4) as u32,
            };
            top_nodes.push(eval_term_clause(ix, &term_clause)?);
        }
        for f in &qa.filters {
            let mut m = ScoreMap::new();
            let needle = f.surface.to_ascii_lowercase();
            for (row, ch) in ix.chunks.iter().enumerate() {
                if let Some(pos) = ch.text.to_ascii_lowercase().find(&needle) {
                    let s = ch.span.start + pos as u32;
                    let e = s + f.surface.len() as u32;
                    m.insert(
                        row as u32,
                        (
                            0.25,
                            vec![MatchedSpan {
                                source: "filter".into(),
                                label: f.surface.clone(),
                                span: Some(SpanProto { start: s, end: e }),
                                score: 0.25,
                            }],
                        ),
                    );
                }
            }
            if !m.is_empty() {
                top_nodes.push(m);
            }
        }
    }

    if !req.query_vector.is_empty() {
        let sem = SemanticClause {
            query_vector: req.query_vector.clone(),
            query_space: req.query_space.clone(),
            query_text: String::new(),
            top_k: (top_k * 4) as u32,
        };
        top_nodes.push(eval_semantic_clause(ix, state, &sem)?);
    }

    if let Some(ast) = &req.query_ast {
        top_nodes.push(eval_node(ix, state, ast, top_k * 4)?);
    }

    let combined = if top_nodes.is_empty() {
        ScoreMap::new()
    } else if top_nodes.len() == 1 {
        top_nodes.pop().unwrap()
    } else {
        fuse_score_maps(&top_nodes, JoinOperator::Or, FusionStrategy::ReciprocalRank, 60.0)
    };

    let mut ranked: Vec<(u32, f64, Vec<MatchedSpan>)> =
        combined.into_iter().map(|(r, (s, spans))| (r, s, spans)).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(top_k);

    if !req.rerank_bundle.is_empty() && !ranked.is_empty() {
        let q_text = if !req.natural_query.is_empty() { req.natural_query.as_str() } else { "query" };
        rerank_candidates(ix, state, &req.rerank_bundle, q_text, &mut ranked)?;
    }

    let expand_parent = req.parent_expansion.as_ref().is_some_and(|p| p.enabled);
    let max_parent_tokens = req.parent_expansion.as_ref().map_or(0, |p| p.max_parent_tokens as usize);

    let mut hits = Vec::with_capacity(ranked.len());
    let mut doc_heat_map: BTreeMap<String, (f32, Vec<HeatSegment>)> = BTreeMap::new();

    for (row, score, matched_spans) in ranked {
        let ch = &ix.chunks[row as usize];
        let mut signal_scores = HashMap::new();
        for ms in &matched_spans {
            *signal_scores.entry(ms.source.clone()).or_insert(0.0f32) += ms.score;
        }
        let (has_parent, parent_span_proto, parent_text) = match ch.parent_span {
            Some(ps) => {
                let p_words = ch.parent_text.split_whitespace().count();
                let allow = !expand_parent || max_parent_tokens == 0 || p_words <= max_parent_tokens;
                let p_txt = if expand_parent && allow { ch.parent_text.clone() } else { String::new() };
                (true, Some(SpanProto { start: ps.start, end: ps.end }), p_txt)
            }
            None => (false, None, String::new()),
        };

        let f_score = score as f32;
        let entry = doc_heat_map.entry(ch.doc_id.clone()).or_insert_with(|| (f_score, Vec::new()));
        if f_score > entry.0 {
            entry.0 = f_score;
        }
        entry.1.push(HeatSegment {
            span: Some(SpanProto { start: ch.span.start, end: ch.span.end }),
            score: f_score,
            chunk_id: ch.chunk_id.clone(),
        });

        hits.push(SearchHit {
            chunk_id: ch.chunk_id.clone(),
            doc_id: ch.doc_id.clone(),
            level_kind: ch.level_kind.clone(),
            score: f_score,
            chunk_span: Some(SpanProto { start: ch.span.start, end: ch.span.end }),
            parent_span: parent_span_proto,
            has_parent,
            text: ch.text.clone(),
            parent_text,
            attribution: Some(ChunkAttribution { matched_spans, signal_scores }),
            attributes: ch.attributes.clone().into_iter().collect(),
        });
    }

    let mut document_heat: Vec<DocumentHeat> = doc_heat_map
        .into_iter()
        .map(|(doc_id, (max_score, mut segments))| {
            segments.sort_by_key(|s| s.span.as_ref().map_or(0, |sp| sp.start));
            DocumentHeat { doc_id, max_score, segments }
        })
        .collect();
    document_heat.sort_by(|a, b| b.max_score.total_cmp(&a.max_score).then_with(|| a.doc_id.cmp(&b.doc_id)));

    Ok(SearchResponse { resolved_index, hits, document_heat, analyzed_query_terms, extracted_filters })
}

fn eval_node(ix: &HybridIndex, state: &State, node: &QueryNode, default_k: usize) -> Result<ScoreMap> {
    let Some(clause) = &node.clause else {
        return Err(bad_arg("QueryNode.clause must be set"));
    };
    match clause {
        Clause::Semantic(c) => eval_semantic_clause(ix, state, c),
        Clause::Term(c) => eval_term_clause(ix, c),
        Clause::Phrase(c) => eval_phrase_clause(ix, c, default_k),
        Clause::Sparse(c) => eval_sparse_clause(ix, c, default_k),
        Clause::LateInteraction(c) => eval_late_clause(ix, c, default_k),
        Clause::Filter(c) => Ok(eval_filter_clause(ix, c)),
        Clause::Boost(b) => {
            let BoostClause { child, boost } = b.as_ref();
            let child_node = child.as_deref().ok_or_else(|| bad_arg("BoostClause.child is required"))?;
            if !boost.is_finite() || *boost < 0.0 {
                return Err(bad_arg(format!("BoostClause.boost must be finite and >= 0, got {boost}")));
            }
            let mut m = eval_node(ix, state, child_node, default_k)?;
            for (score, spans) in m.values_mut() {
                *score *= f64::from(*boost);
                for sp in spans {
                    sp.score *= *boost;
                }
            }
            Ok(m)
        }
        Clause::Join(JoinClause { op, fusion, children, rrf_k }) => {
            if children.is_empty() {
                return Ok(ScoreMap::new());
            }
            let join_op = JoinOperator::try_from(*op).map_err(|_| bad_arg(format!("invalid JoinOperator {op}")))?;
            let strat =
                FusionStrategy::try_from(*fusion).map_err(|_| bad_arg(format!("invalid FusionStrategy {fusion}")))?;
            let k_val = if *rrf_k <= 0.0 || !rrf_k.is_finite() { 60.0 } else { f64::from(*rrf_k) };
            let mut maps = Vec::with_capacity(children.len());
            for ch in children {
                maps.push(eval_node(ix, state, ch, default_k)?);
            }
            Ok(fuse_score_maps(&maps, join_op, strat, k_val))
        }
    }
}

fn eval_semantic_clause(ix: &HybridIndex, state: &State, c: &SemanticClause) -> Result<ScoreMap> {
    let q_space = match &c.query_space {
        Some(sp) => parse_space(Some(sp))?,
        None => ix.space.clone(),
    };
    let q_vec = if !c.query_vector.is_empty() {
        c.query_vector.clone()
    } else if !c.query_text.is_empty() && !ix.spec.embed_bundle.is_empty() {
        embed_query_text(state, &ix.spec.embed_bundle, &c.query_text)?
    } else {
        return Err(bad_arg("SemanticClause requires query_vector or query_text with embed_bundle"));
    };

    let k = if c.top_k == 0 { ix.chunks.len().max(1) } else { c.top_k as usize };
    let mut out = ScoreMap::new();
    match &ix.dense {
        CompiledDense::Empty => {}
        CompiledDense::Flat(vec_ix) => {
            let hits = vec_ix.top_k(&q_space, &q_vec, k).map_err(|e| bad_arg(e.to_string()))?;
            for h in hits {
                let chunk_row = *vec_ix.meta(h.row);
                let ch = &ix.chunks[chunk_row as usize];
                out.insert(
                    chunk_row,
                    (
                        f64::from(h.score),
                        vec![MatchedSpan {
                            source: "dense".into(),
                            label: ix.space.model.id.clone(),
                            span: Some(SpanProto { start: ch.span.start, end: ch.span.end }),
                            score: h.score,
                        }],
                    ),
                );
            }
        }
        CompiledDense::Quantized(q_ix) => {
            let hits = q_ix.top_k(&q_space, &q_vec, k).map_err(|e| bad_arg(e.to_string()))?;
            for h in hits {
                let chunk_row = *q_ix.meta(h.row);
                let ch = &ix.chunks[chunk_row as usize];
                out.insert(
                    chunk_row,
                    (
                        f64::from(h.score),
                        vec![MatchedSpan {
                            source: "turboquant".into(),
                            label: format!("{}b", q_ix.config().bits),
                            span: Some(SpanProto { start: ch.span.start, end: ch.span.end }),
                            score: h.score,
                        }],
                    ),
                );
            }
        }
    }
    Ok(out)
}

fn fuse_score_maps(maps: &[ScoreMap], op: JoinOperator, fusion: FusionStrategy, rrf_k: f64) -> ScoreMap {
    let mut rank_maps: Vec<HashMap<u32, usize>> = Vec::new();
    if fusion == FusionStrategy::ReciprocalRank {
        for m in maps {
            let mut order: Vec<(u32, f64)> = m.iter().map(|(&r, &(s, _))| (r, s)).collect();
            order.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let rm: HashMap<u32, usize> = order.into_iter().enumerate().map(|(i, (r, _))| (r, i + 1)).collect();
            rank_maps.push(rm);
        }
    }

    let mut candidates: Vec<u32> = maps.iter().flat_map(|m| m.keys().copied()).collect();
    candidates.sort_unstable();
    candidates.dedup();

    let mut out = ScoreMap::new();
    for row in candidates {
        if op == JoinOperator::And && !maps.iter().all(|m| m.contains_key(&row)) {
            continue;
        }
        let mut fused = 0.0f64;
        let mut all_spans = Vec::new();
        for (mi, m) in maps.iter().enumerate() {
            if let Some((s, spans)) = m.get(&row) {
                match fusion {
                    FusionStrategy::Score => fused += *s,
                    FusionStrategy::ReciprocalRank => {
                        let r = rank_maps[mi][&row] as f64;
                        fused += 1.0 / (rrf_k + r);
                    }
                }
                all_spans.extend(spans.iter().cloned());
            }
        }
        out.insert(row, (fused, all_spans));
    }
    out
}

fn embed_query_text(state: &State, bundle_name: &str, text: &str) -> Result<Vec<f32>> {
    let (_, loaded) = state.ready(bundle_name, "")?;
    let held = loaded.take()?;
    let slot = held.slot();
    let opts = turbo::turbo_embed_options {
        struct_size: size_of::<turbo::turbo_embed_options>() as u32,
        truncate: 0,
        max_tokens: 0,
        prompt_role: turbo::TURBO_PROMPT_QUERY,
        normalize: 0,
        pooling: 0,
        output_dim: 0,
    };
    let views = [turbo_text { ptr: text.as_ptr().cast(), len: text.len() as u64 }];
    slot.session.write_text(&views, &opts)?;
    let res = slot.session.run()?;
    let info = res.info()?;
    let bytes = res.read(info.bytes)?;
    Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

fn rerank_candidates(
    ix: &HybridIndex,
    state: &State,
    bundle_name: &str,
    query_text: &str,
    ranked: &mut [(u32, f64, Vec<MatchedSpan>)],
) -> Result<()> {
    let (_, loaded) = state.ready(bundle_name, "")?;
    if loaded.task != TURBO_TASK_RERANK {
        return Err(bad_arg(format!("bundle `{bundle_name}` is not a TASK_RERANK model")));
    }
    let held = loaded.take()?;
    let slot = held.slot();
    let opts = turbo_rerank_options {
        struct_size: size_of::<turbo_rerank_options>() as u32,
        query_tokens: 0,
        max_tokens: 0,
        stride: 0,
    };
    let q_view = turbo_text { ptr: query_text.as_ptr().cast(), len: query_text.len() as u64 };
    let pairs: Vec<turbo_text_pair> = ranked
        .iter()
        .map(|&(r, _, _)| {
            let p = ix.chunks[r as usize].text.as_str();
            turbo_text_pair { query: q_view, passage: turbo_text { ptr: p.as_ptr().cast(), len: p.len() as u64 } }
        })
        .collect();
    slot.session.write_pairs(&pairs, &opts)?;
    let res = slot.session.run()?;
    let info = res.info()?;
    let bytes = res.read(info.bytes)?;
    for ((row, score, spans), b) in ranked.iter_mut().zip(bytes.as_chunks::<4>().0) {
        let rr = f32::from_le_bytes(*b);
        *score = f64::from(rr);
        let ch = &ix.chunks[*row as usize];
        spans.push(MatchedSpan {
            source: "rerank".into(),
            label: bundle_name.to_owned(),
            span: Some(SpanProto { start: ch.span.start, end: ch.span.end }),
            score: rr,
        });
    }
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Ok(())
}
