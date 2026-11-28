//! Async session-pool acquisition and `max_batch`-chunked execution for
//! `TurboSearchService` (`Search`, `IndexDocuments`, and `Reindex`, `#2609`).

use std::collections::HashMap;
use std::sync::Arc;

use turbo::{TURBO_PROMPT_QUERY, TURBO_TASK_RERANK, turbo_rerank_options};
use turbo_vector_index::{Normalization, Space};

use crate::api::{Failure, Result, TURBO_E_INTERNAL};
use crate::deadline::Deadline;
use crate::priority::Priority;
use crate::proto::search::{
    IndexDocumentRequest, IndexDocumentsResponse, IndexSpec, MatchedSpan, ReindexRequest, ReindexResponse,
    SearchRequest, SearchResponse, SpanProto,
};
use crate::search::ingest::{
    StagedDocument, commit_staged_documents, embed_prepared_documents, finalize_prepared_document, prepare_document,
};
use crate::search::query::{
    RankedRow, SearchStep, collect_semantic_query_texts, empty_search_response, materialize_search_response,
    retrieve_candidates, validate_search_header,
};
use crate::search::registry::{
    HybridIndex, SearchRegistry, already_exists, bad_arg, bad_state, parse_space, verify_embed_bundle_matches_space,
};
use crate::state::State;
use crate::tasks::embed::exec::{embed_f32_batches, prompt_embed_options};
use crate::tasks::score_rerank_batches;

fn task_failed(what: &str, err: tokio::task::JoinError) -> Failure {
    Failure::new(TURBO_E_INTERNAL, 0, format!("{what} task failed: {err}"))
}

fn apply_rerank_scores(ix: &HybridIndex, bundle_name: &str, ranked: &mut [RankedRow], logits: &[f32]) {
    for ((row, score, spans), &rr) in ranked.iter_mut().zip(logits) {
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
}

async fn stage_documents_async(
    state: &Arc<State>,
    spec: IndexSpec,
    space: Space,
    sealed: bool,
    items: Vec<IndexDocumentRequest>,
    priority: Priority,
    deadline: &Deadline,
) -> Result<Vec<StagedDocument>> {
    let spec_prep = spec.clone();
    let mut prepared = tokio::task::spawn_blocking(move || {
        let mut out = Vec::with_capacity(items.len());
        for doc_req in items {
            out.push(prepare_document(&spec_prep, sealed, doc_req)?);
        }
        Ok::<_, Failure>(out)
    })
    .await
    .map_err(|e| task_failed("index_documents prepare", e))??;

    let any_chunks = prepared.iter().any(|d| !d.prepared_chunks.is_empty());
    if !spec.embed_bundle.is_empty() && any_chunks {
        let loaded = verify_embed_bundle_matches_space(state, &spec.embed_bundle, &space)?;
        let need_embed = prepared.iter().any(|d| d.prepared_chunks.iter().any(|c| c.dense_vector.is_empty()));
        if need_embed {
            let held = loaded.acquire_async(priority, deadline).await?;
            let state_e = Arc::clone(state);
            let space_e = space.clone();
            let deadline_e = deadline.clone();
            prepared = tokio::task::spawn_blocking(move || {
                embed_prepared_documents(&state_e, &loaded, held.slot(), &space_e, &mut prepared, &deadline_e)?;
                drop(held);
                Ok::<_, Failure>(prepared)
            })
            .await
            .map_err(|e| task_failed("index_documents embed", e))??;
        }
    }

    tokio::task::spawn_blocking(move || {
        let mut staged = Vec::with_capacity(prepared.len());
        for doc in prepared {
            staged.push(finalize_prepared_document(&space, doc)?);
        }
        Ok::<_, Failure>(staged)
    })
    .await
    .map_err(|e| task_failed("index_documents finalize", e))?
}

pub(crate) async fn run_index_documents(
    registry: Arc<SearchRegistry>,
    state: Arc<State>,
    items: Vec<IndexDocumentRequest>,
    priority: Priority,
    deadline: Deadline,
) -> Result<IndexDocumentsResponse> {
    let target_arg = items[0].index_name.clone();
    let (resolved_name, ix_lock) = registry.resolve_index(&target_arg)?;
    let (spec, space, sealed) = {
        let ix = ix_lock.read().unwrap_or_else(|e| e.into_inner());
        (ix.spec.clone(), ix.space.clone(), ix.sealed)
    };

    let staged = stage_documents_async(&state, spec, space, sealed, items, priority, &deadline).await?;

    tokio::task::spawn_blocking(move || {
        let mut ix = ix_lock.write().unwrap_or_else(|e| e.into_inner());
        if ix.sealed {
            return Err(bad_state(format!("search index `{}` is sealed and accepts no writes", ix.spec.index_name)));
        }
        let mut batch_chunk_terms = Vec::new();
        let (indexed_docs, indexed_chunks) = commit_staged_documents(&mut ix, staged, &mut batch_chunk_terms)?;
        let col_name =
            if !ix.spec.collection_name.is_empty() { ix.spec.collection_name.clone() } else { resolved_name.clone() };
        drop(ix);
        let drift = registry.record_collection_update(&col_name, &resolved_name, &batch_chunk_terms);
        Ok(IndexDocumentsResponse { index_name: resolved_name, indexed_documents: indexed_docs, indexed_chunks, drift })
    })
    .await
    .map_err(|e| task_failed("index_documents commit", e))?
}

pub(crate) async fn run_reindex(
    registry: Arc<SearchRegistry>,
    state: Arc<State>,
    req: ReindexRequest,
    priority: Priority,
    deadline: Deadline,
) -> Result<ReindexResponse> {
    let target_spec = req.target_spec.ok_or_else(|| bad_arg("ReindexRequest.target_spec is required"))?;
    let target_space = parse_space(target_spec.space.as_ref())?;
    let target_name = target_spec.index_name.clone();
    let (_, src_lock) = registry.resolve_index(&req.source_index)?;
    let (src_space, src_budget, src_had_vectors, retained_docs) = {
        let src = src_lock.read().unwrap_or_else(|e| e.into_inner());
        let had_vecs = src.chunks.iter().any(|c| c.dense_vector.is_some() || !c.token_vectors.is_empty());
        (src.space.clone(), src.spec.chunk_budget_tokens, had_vecs, src.docs.clone())
    };
    let space_changed = target_space != src_space;
    let budget_changed = target_spec.chunk_budget_tokens != src_budget;
    if space_changed && src_had_vectors && target_spec.embed_bundle.is_empty() {
        return Err(bad_arg(
            "cannot reindex vectors into a different SpaceSpec without a matching target embed_bundle",
        ));
    }

    if registry.indexes.read().unwrap_or_else(|e| e.into_inner()).contains_key(&target_name) {
        return Err(already_exists(format!("search index `{target_name}` already exists")));
    }

    let state_empty = Arc::clone(&state);
    let spec_empty = target_spec.clone();
    let dst = tokio::task::spawn_blocking(move || HybridIndex::empty(&state_empty, spec_empty))
        .await
        .map_err(|e| task_failed("reindex init", e))??;

    let doc_reqs: Vec<IndexDocumentRequest> = retained_docs
        .into_iter()
        .map(|d| {
            let chunks = if space_changed {
                let mut stripped = d.raw_chunks;
                for c in &mut stripped {
                    c.dense_vector.clear();
                    c.token_vectors.clear();
                }
                stripped
            } else if budget_changed && d.raw_chunks.is_empty() {
                Vec::new()
            } else {
                d.chunks
            };
            IndexDocumentRequest {
                index_name: target_name.clone(),
                doc_id: d.doc_id,
                text: d.text,
                graph: d.graph,
                chunks,
                attributes: d.attributes.into_iter().collect(),
                layout_boxes: Vec::new(),
            }
        })
        .collect();

    let dst_spec = dst.spec.clone();
    let dst_space = dst.space.clone();
    let staged = stage_documents_async(&state, dst_spec, dst_space, false, doc_reqs, priority, &deadline).await?;

    let swap_alias = req.swap_alias;
    tokio::task::spawn_blocking(move || {
        let mut dst = dst;
        let mut batch_terms = Vec::new();
        commit_staged_documents(&mut dst, staged, &mut batch_terms)?;
        let summary = registry.publish(dst)?;
        if !swap_alias.is_empty() {
            let mut aliases = registry.aliases.write().unwrap_or_else(|e| e.into_inner());
            aliases.insert(swap_alias.clone(), target_name);
        }
        Ok(ReindexResponse { summary: Some(summary), swapped_alias: swap_alias })
    })
    .await
    .map_err(|e| task_failed("reindex commit", e))?
}

pub(crate) async fn run_search(
    registry: Arc<SearchRegistry>,
    state: Arc<State>,
    req: SearchRequest,
    priority: Priority,
    deadline: Deadline,
) -> Result<SearchResponse> {
    let (resolved_name, ix_lock) = registry.resolve_index(&req.index_name)?;
    let (top_k, rerank_query) = validate_search_header(&req)?;
    let (embed_bundle, space, chunk_len) = {
        let ix = ix_lock.read().unwrap_or_else(|e| e.into_inner());
        if ix.chunks.is_empty() {
            return Ok(empty_search_response(resolved_name));
        }
        (ix.spec.embed_bundle.clone(), ix.space.clone(), ix.chunks.len())
    };

    let mut unique_texts: Vec<String> = Vec::new();
    if let Some(ast) = &req.query_ast {
        let mut raw_texts = Vec::new();
        collect_semantic_query_texts(ast, !embed_bundle.is_empty(), chunk_len, &mut raw_texts)?;
        for t in raw_texts {
            if !unique_texts.iter().any(|seen| seen == t) {
                unique_texts.push(t.to_owned());
            }
        }
    }

    let mut embedded_queries: HashMap<String, Vec<f32>> = HashMap::new();
    if !unique_texts.is_empty() {
        let loaded = verify_embed_bundle_matches_space(&state, &embed_bundle, &space)?;
        let held = loaded.acquire_async(priority, &deadline).await?;
        let state_q = Arc::clone(&state);
        let deadline_q = deadline.clone();
        let unit = matches!(space.normalization, Normalization::Unit);
        let dim = space.dim;
        embedded_queries = tokio::task::spawn_blocking(move || {
            let opts = prompt_embed_options(TURBO_PROMPT_QUERY, unit, loaded.model_info.dim, dim);
            let _docs = loaded.metrics.track_docs(unique_texts.len(), Some(loaded.device));
            let vecs = embed_f32_batches(&state_q, &loaded, held.slot(), &unique_texts, &opts, &deadline_q)?;
            drop(held);
            Ok::<_, Failure>(unique_texts.into_iter().zip(vecs).collect())
        })
        .await
        .map_err(|e| task_failed("search query embed", e))??;
    }

    let req = Arc::new(req);
    let req_ret = Arc::clone(&req);
    let ix_ret = Arc::clone(&ix_lock);
    let step = tokio::task::spawn_blocking(move || {
        let ix = ix_ret.read().unwrap_or_else(|e| e.into_inner());
        retrieve_candidates(resolved_name, &ix, &req_ret, top_k, rerank_query, &embedded_queries)
    })
    .await
    .map_err(|e| task_failed("search retrieve", e))??;

    let (resolved_index, query_text, mut ranked, passages, analyzed_query_terms, extracted_filters) = match step {
        SearchStep::Done(resp) => return Ok(resp),
        SearchStep::NeedsRerank {
            resolved_index,
            query_text,
            ranked,
            passages,
            analyzed_query_terms,
            extracted_filters,
        } => (resolved_index, query_text, ranked, passages, analyzed_query_terms, extracted_filters),
    };

    let (_, rerank_loaded) = state.ready(&req.rerank_bundle, "")?;
    if rerank_loaded.task != TURBO_TASK_RERANK {
        return Err(bad_arg(format!("bundle `{}` is not a TASK_RERANK model", req.rerank_bundle)));
    }
    let held = rerank_loaded.acquire_async(priority, &deadline).await?;
    tokio::task::spawn_blocking(move || {
        let opts = turbo_rerank_options {
            struct_size: size_of::<turbo_rerank_options>() as u32,
            query_tokens: 0,
            max_tokens: 0,
            stride: 0,
        };
        let _docs = rerank_loaded.metrics.track_docs(passages.len(), Some(rerank_loaded.device));
        let logits =
            score_rerank_batches(&state, &rerank_loaded, held.slot(), &query_text, &passages, &opts, &deadline)?;
        drop(held);
        let ix = ix_lock.read().unwrap_or_else(|e| e.into_inner());
        apply_rerank_scores(&ix, &req.rerank_bundle, &mut ranked, &logits);
        Ok(materialize_search_response(resolved_index, &ix, &req, ranked, analyzed_query_terms, extracted_filters))
    })
    .await
    .map_err(|e| task_failed("search rerank", e))?
}
