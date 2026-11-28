//! Hybrid query AST execution, Reciprocal Rank Fusion (`1 / (k + rank)`),
//! `turbo-query` filter extraction, parent expansion, `MatchedSpan`
//! attribution, and `DocumentHeat` generation (`#1221`, `#2609`).

use std::collections::{BTreeMap, HashMap};

use turbo_query::FilterKind;

use crate::api::Result;
use crate::proto::search::query_node::Clause;
use crate::proto::search::{
    BoostClause, ChunkAttribution, DocumentHeat, ExtractedFilterProto, FusionStrategy, HeatSegment, JoinClause,
    JoinOperator, MatchedSpan, QueryNode, SearchHit, SearchRequest, SearchResponse, SemanticClause, SpanProto,
    TermClause, TermMatchMode,
};
use crate::search::clauses::{
    MAX_TOP_K, bounded_top_k, eval_filter_clause, eval_late_clause, eval_phrase_clause, eval_semantic_clause,
    eval_sparse_clause, eval_term_clause,
};
use crate::search::registry::{HybridIndex, ScoreMap, bad_arg, parse_space, shared_query_profile};

pub(crate) type RankedRow = (u32, f64, Vec<MatchedSpan>);

pub(crate) enum SearchStep {
    Done(SearchResponse),
    NeedsRerank {
        resolved_index: String,
        query_text: String,
        ranked: Vec<RankedRow>,
        passages: Vec<String>,
        analyzed_query_terms: Vec<String>,
        extracted_filters: Vec<ExtractedFilterProto>,
    },
}

pub(crate) fn validate_search_header(req: &SearchRequest) -> Result<(usize, Option<String>)> {
    let top_k = bounded_top_k(req.top_k, 10)?;
    let rerank_query = resolve_rerank_query(req)?;
    Ok((top_k, rerank_query))
}

pub(crate) fn empty_search_response(resolved_index: String) -> SearchResponse {
    SearchResponse {
        resolved_index,
        hits: Vec::new(),
        document_heat: Vec::new(),
        analyzed_query_terms: Vec::new(),
        extracted_filters: Vec::new(),
    }
}

pub(crate) fn collect_semantic_query_texts<'a>(
    node: &'a QueryNode,
    has_embed_bundle: bool,
    chunk_len: usize,
    out: &mut Vec<&'a str>,
) -> Result<()> {
    let Some(clause) = &node.clause else {
        return Err(bad_arg("QueryNode.clause must be set"));
    };
    match clause {
        Clause::Semantic(c) => {
            bounded_top_k(c.top_k, chunk_len.max(1))?;
            if c.query_vector.is_empty() && !c.query_text.is_empty() && has_embed_bundle {
                if let Some(s) = &c.query_space {
                    parse_space(Some(s))?;
                }
                out.push(&c.query_text);
            }
        }
        Clause::Boost(b) => {
            let child = b.child.as_deref().ok_or_else(|| bad_arg("BoostClause.child is required"))?;
            if !b.boost.is_finite() || b.boost < 0.0 {
                return Err(bad_arg(format!("BoostClause.boost must be finite and >= 0, got {}", b.boost)));
            }
            collect_semantic_query_texts(child, has_embed_bundle, chunk_len, out)?;
        }
        Clause::Join(JoinClause { op, fusion, children, .. }) if !children.is_empty() => {
            JoinOperator::try_from(*op).map_err(|_| bad_arg(format!("invalid JoinOperator {op}")))?;
            FusionStrategy::try_from(*fusion).map_err(|_| bad_arg(format!("invalid FusionStrategy {fusion}")))?;
            for ch in children {
                collect_semantic_query_texts(ch, has_embed_bundle, chunk_len, out)?;
            }
        }
        _ => {}
    }
    Ok(())
}

pub(crate) fn retrieve_candidates(
    resolved_index: String,
    ix: &HybridIndex,
    req: &SearchRequest,
    top_k: usize,
    rerank_query: Option<String>,
    embedded_queries: &HashMap<String, Vec<f32>>,
) -> Result<SearchStep> {
    if ix.chunks.is_empty() {
        return Ok(SearchStep::Done(empty_search_response(resolved_index)));
    }

    let cand_k = top_k.saturating_mul(4).min(MAX_TOP_K);
    let mut analyzed_query_terms = Vec::new();
    let mut extracted_filters = Vec::new();
    let mut top_nodes: Vec<ScoreMap> = Vec::new();
    let mut narrowing_filters: Vec<ScoreMap> = Vec::new();

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
            let mut m = ScoreMap::new();
            let needle = f.surface.to_ascii_lowercase();
            for (row, ch) in ix.chunks.iter().enumerate() {
                if let Some(pos) = ch.text.to_ascii_lowercase().find(&needle) {
                    let s = ch.span.start + pos as u32;
                    let e = s + f.surface.len() as u32;
                    m.insert(
                        row as u32,
                        (
                            1.0,
                            vec![MatchedSpan {
                                source: "filter".into(),
                                label: f.surface.clone(),
                                span: Some(SpanProto { start: s, end: e }),
                                score: 1.0,
                            }],
                        ),
                    );
                }
            }
            narrowing_filters.push(m);
        }
        if !analyzed_query_terms.is_empty() {
            let term_clause = TermClause {
                terms: analyzed_query_terms.clone(),
                mode: TermMatchMode::Any as i32,
                top_k: cand_k as u32,
            };
            top_nodes.push(eval_term_clause(ix, &term_clause)?);
        }
    }

    if !req.query_vector.is_empty() {
        let sem = SemanticClause {
            query_vector: req.query_vector.clone(),
            query_space: req.query_space.clone(),
            query_text: String::new(),
            top_k: cand_k as u32,
        };
        top_nodes.push(eval_semantic_clause(ix, &sem, embedded_queries)?);
    }

    if let Some(ast) = &req.query_ast {
        if let Some(Clause::Filter(fc)) = &ast.clause
            && !top_nodes.is_empty()
        {
            narrowing_filters.push(eval_filter_clause(ix, fc));
        } else {
            top_nodes.push(eval_node(ix, ast, cand_k, embedded_queries)?);
        }
    }

    let mut combined = if !top_nodes.is_empty() {
        if top_nodes.len() == 1 {
            top_nodes.pop().unwrap()
        } else {
            fuse_score_maps(&top_nodes, JoinOperator::Or, FusionStrategy::ReciprocalRank, 60.0)
        }
    } else if !narrowing_filters.is_empty() {
        fuse_score_maps(&narrowing_filters, JoinOperator::And, FusionStrategy::Score, 60.0)
    } else {
        ScoreMap::new()
    };

    if !top_nodes_were_empty(req, &analyzed_query_terms) && !narrowing_filters.is_empty() {
        for nf in &narrowing_filters {
            combined.retain(|row, (_, spans)| {
                if let Some((_, f_spans)) = nf.get(row) {
                    spans.extend(f_spans.iter().cloned());
                    true
                } else {
                    false
                }
            });
        }
    }

    let mut ranked: Vec<RankedRow> = combined.into_iter().map(|(r, (s, spans))| (r, s, spans)).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(top_k);

    if let Some(query_text) = rerank_query
        && !ranked.is_empty()
    {
        let passages = ranked.iter().map(|&(r, _, _)| ix.chunks[r as usize].text.clone()).collect();
        return Ok(SearchStep::NeedsRerank {
            resolved_index,
            query_text,
            ranked,
            passages,
            analyzed_query_terms,
            extracted_filters,
        });
    }

    Ok(SearchStep::Done(materialize_search_response(
        resolved_index,
        ix,
        req,
        ranked,
        analyzed_query_terms,
        extracted_filters,
    )))
}

pub(crate) fn materialize_search_response(
    resolved_index: String,
    ix: &HybridIndex,
    req: &SearchRequest,
    ranked: Vec<RankedRow>,
    analyzed_query_terms: Vec<String>,
    extracted_filters: Vec<ExtractedFilterProto>,
) -> SearchResponse {
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
            page_start: ch.page_span.map_or(0, |(a, _)| a),
            page_end: ch.page_span.map_or(0, |(_, b)| b),
            has_page_span: ch.page_span.is_some(),
            page_boxes: ch.page_boxes.clone(),
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

    SearchResponse { resolved_index, hits, document_heat, analyzed_query_terms, extracted_filters }
}

fn top_nodes_were_empty(req: &SearchRequest, analyzed_query_terms: &[String]) -> bool {
    analyzed_query_terms.is_empty() && req.query_vector.is_empty() && req.query_ast.is_none()
}

fn resolve_rerank_query(req: &SearchRequest) -> Result<Option<String>> {
    if req.rerank_bundle.is_empty() {
        return Ok(None);
    }
    if !req.natural_query.trim().is_empty() {
        return Ok(Some(req.natural_query.clone()));
    }
    if let Some(ast) = &req.query_ast
        && let Some(q) = extract_ast_query_text(ast)
    {
        return Ok(Some(q));
    }
    Err(bad_arg("rerank_bundle requires a non-empty natural_query or SemanticClause.query_text"))
}

fn extract_ast_query_text(node: &QueryNode) -> Option<String> {
    match node.clause.as_ref()? {
        Clause::Semantic(s) if !s.query_text.trim().is_empty() => Some(s.query_text.clone()),
        Clause::Boost(b) => b.child.as_deref().and_then(extract_ast_query_text),
        Clause::Join(j) => j.children.iter().find_map(extract_ast_query_text),
        _ => None,
    }
}

fn eval_node(
    ix: &HybridIndex,
    node: &QueryNode,
    default_k: usize,
    embedded_queries: &HashMap<String, Vec<f32>>,
) -> Result<ScoreMap> {
    let Some(clause) = &node.clause else {
        return Err(bad_arg("QueryNode.clause must be set"));
    };
    match clause {
        Clause::Semantic(c) => eval_semantic_clause(ix, c, embedded_queries),
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
            let mut m = eval_node(ix, child_node, default_k, embedded_queries)?;
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
                maps.push(eval_node(ix, ch, default_k, embedded_queries)?);
            }
            Ok(fuse_score_maps(&maps, join_op, strat, k_val))
        }
    }
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
