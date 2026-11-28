//! `AnalysisService::RerankAndHighlight` execution and wire round-trip
//! validation (`#2342`).

mod calibrator;
mod highlight;
mod maxsim;
mod validate;

pub(crate) use calibrator::BundleCalibration;

use std::sync::Arc;

use turbo::{TURBO_TASK_RERANK, turbo_rerank_options};
use turbo_analysis::{ModelRef, Span};
use turbo_analysis_embed::SourcePassage;

use crate::api::{Failure, Result, TURBO_E_INTERNAL, TURBO_E_INVALID_ARGUMENT, TURBO_E_UNSUPPORTED_OPTION, field};
use crate::deadline::Deadline;
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::result::{
    RerankAndHighlightRequest, RerankAndHighlightResponse, RerankCandidateInputProto, RerankHighlightMode,
    RerankedCandidateProto, SemanticHighlightSpanProto,
};
use crate::results::{MAX_RERANK_ITEMS, model_to_proto, over};
use crate::state::State;
use crate::tasks::score_rerank_batches;

fn bad(msg: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, msg)
}

pub(super) fn prepare_source_passage<'a>(
    cand: &'a RerankCandidateInputProto,
    offsets: OffsetMode,
    lang: &str,
) -> Result<SourcePassage<'a>> {
    if cand.sentence_span_start.len() != cand.sentence_span_end.len() {
        return Err(bad("candidate sentence_span_start and sentence_span_end lengths must match"));
    }
    let explicit = if cand.sentence_span_start.is_empty() {
        Vec::new()
    } else {
        Offsetter::new(&cand.text, offsets).spans_from_wire(
            &cand.text,
            "candidate.sentence_span",
            &cand.sentence_span_start,
            &cand.sentence_span_end,
        )?
    };
    SourcePassage::new(&cand.text, lang, &explicit).map_err(|e| bad(e.to_string()))
}

pub(super) fn to_wire_highlights(
    cand: &RerankCandidateInputProto,
    offsets: OffsetMode,
    spans: Vec<turbo_analysis_embed::HighlightCandidateSpan>,
) -> Vec<SemanticHighlightSpanProto> {
    if spans.is_empty() {
        return Vec::new();
    }
    let src_spans: Vec<Span> = spans.iter().map(|s| s.source_span).collect();
    let (starts, ends) = Offsetter::new(&cand.text, offsets).spans_to_wire(&cand.text, &src_spans);
    spans
        .into_iter()
        .zip(starts.into_iter().zip(ends))
        .map(|(h, (span_start, span_end))| SemanticHighlightSpanProto {
            span_start,
            span_end,
            parent_span_start: cand.parent_offset.saturating_add(u64::from(span_start)),
            parent_span_end: cand.parent_offset.saturating_add(u64::from(span_end)),
            score: h.score,
            surface: h.surface,
        })
        .collect()
}

/// Executes calibrated cross-encoder reranking plus top-K semantic highlight
/// attribution (`#2342`).
pub(crate) async fn rerank_and_highlight(
    state: Arc<State>,
    req: RerankAndHighlightRequest,
    priority: crate::priority::Priority,
    deadline: Deadline,
) -> Result<RerankAndHighlightResponse> {
    let req = Arc::new(req);
    let offsets = OffsetMode::from_proto(req.offsets)?;
    let mode = RerankHighlightMode::try_from(req.highlight_mode)
        .map_err(|_| bad(format!("unknown RerankHighlightMode {}", req.highlight_mode)))?;
    let rerank_name = req.rerank_model.trim();
    if rerank_name.is_empty() {
        return Err(bad("rerank_model must be non-empty"));
    }
    if req.candidates.len() as u64 > MAX_RERANK_ITEMS {
        return Err(over("max_rerank_items", "candidates", req.candidates.len() as u64));
    }

    let (_, rerank_loaded) = state.ready(rerank_name, "")?;
    if rerank_loaded.task != TURBO_TASK_RERANK {
        return Err(Failure::new(
            TURBO_E_UNSUPPORTED_OPTION,
            0,
            format!("model {rerank_name} is not a TASK_RERANK bundle"),
        ));
    }
    let mi = &rerank_loaded.model_info;
    let rerank_mref =
        ModelRef { id: field(&mi.model_id), revision: field(&mi.revision), sha256: field(&mi.manifest_sha256) };
    let calibrator = calibrator::resolve_calibrator(&rerank_loaded, &rerank_mref, req.calibration.as_ref())?;

    if req.candidates.is_empty() {
        return Ok(RerankAndHighlightResponse {
            query: req.query.clone(),
            rerank_model: Some(model_to_proto(&rerank_mref)),
            highlight_model: None,
            offsets: req.offsets,
            results: Vec::new(),
        });
    }

    let held = rerank_loaded.acquire_async(priority, &deadline).await?;
    let state_c = Arc::clone(&state);
    let loaded_c = Arc::clone(&rerank_loaded);
    let req_c = Arc::clone(&req);
    let deadline_c = deadline.clone();
    let mut results = tokio::task::spawn_blocking(move || {
        let rerank_opts = turbo_rerank_options {
            struct_size: std::mem::size_of::<turbo_rerank_options>() as u32,
            query_tokens: req_c.query_tokens,
            max_tokens: req_c.max_tokens,
            stride: req_c.stride,
        };
        let _docs = loaded_c.metrics.track_docs(req_c.candidates.len(), Some(loaded_c.device));
        let passages: Vec<&str> = req_c.candidates.iter().map(|c| c.text.as_str()).collect();
        let logits =
            score_rerank_batches(&state_c, &loaded_c, held.slot(), &req_c.query, &passages, &rerank_opts, &deadline_c)?;
        drop(held);

        let mut results = Vec::with_capacity(req_c.candidates.len());
        for (idx, (cand, &raw_logit)) in req_c.candidates.iter().zip(&logits).enumerate() {
            let cal = calibrator.score(raw_logit)?;
            results.push(RerankedCandidateProto {
                input_index: idx as u32,
                doc_id: cand.doc_id.clone(),
                raw_logit: cal.raw_logit,
                calibrated_probability: cal.calibrated_probability,
                conformal_p_value: cal.conformal_p_value,
                highlights: Vec::new(),
                qa_abstained: false,
                qa_null_probability: None,
            });
        }
        results.sort_by(|a, b| {
            b.calibrated_probability
                .total_cmp(&a.calibrated_probability)
                .then_with(|| b.raw_logit.total_cmp(&a.raw_logit))
                .then_with(|| a.input_index.cmp(&b.input_index))
        });
        Ok(results)
    })
    .await
    .map_err(|e| Failure::new(TURBO_E_INTERNAL, 0, format!("rerank_and_highlight task failed: {e}")))??;

    let top_k = if req.highlight_top_k == 0 { 10 } else { req.highlight_top_k as usize }.min(results.len());
    let max_snippets = if req.max_snippets_per_candidate == 0 { 2 } else { req.max_snippets_per_candidate as usize };

    let highlight_model = if mode == RerankHighlightMode::None || top_k == 0 || max_snippets == 0 {
        None
    } else {
        highlight::attribute_top_k(
            &state,
            &rerank_loaded,
            &rerank_mref,
            &req,
            offsets,
            mode,
            &mut results[..top_k],
            max_snippets,
            priority,
            &deadline,
        )
        .await?
    };

    Ok(RerankAndHighlightResponse {
        query: req.query.clone(),
        rerank_model: Some(model_to_proto(&rerank_mref)),
        highlight_model: highlight_model.as_ref().map(model_to_proto),
        offsets: req.offsets,
        results,
    })
}
