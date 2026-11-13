//! `AnalysisService::RerankAndHighlight` execution and wire round-trip
//! validation (`#2342`).

mod highlight;
mod validate;

use std::path::Path;
use std::sync::atomic::Ordering;

use turbo::{TURBO_DTYPE_F32, TURBO_TASK_RERANK, turbo_rerank_options, turbo_text, turbo_text_pair};
use turbo_analysis::{ModelRef, Span};
use turbo_analysis_embed::{RerankCalibrator, SourcePassage};
use turbo_calibration::Transform;

use crate::api::{
    Failure, Result, TURBO_E_CAPACITY, TURBO_E_INTERNAL, TURBO_E_INVALID_ARGUMENT, TURBO_E_UNSUPPORTED_OPTION, field,
};
use crate::deadline::Deadline;
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::result::rerank_calibration_spec_proto::Kind as CalKind;
use crate::proto::result::{
    RerankAndHighlightRequest, RerankAndHighlightResponse, RerankCalibrationSpecProto, RerankCandidateInputProto,
    RerankHighlightMode, RerankedCandidateProto, SemanticHighlightSpanProto,
};
use crate::results::{MAX_RERANK_ITEMS, f32s, model_to_proto, over};
use crate::state::{Loaded, State};
use crate::stream::run::read_output;

fn bad(msg: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, msg)
}

fn build_calibrator(
    loaded: &Loaded,
    mref: &ModelRef,
    spec: Option<&RerankCalibrationSpecProto>,
) -> Result<RerankCalibrator> {
    if let Some(s) = spec
        && let Some(kind) = &s.kind
    {
        let domain = if s.domain.trim().is_empty() { "search" } else { s.domain.trim() };
        return match kind {
            CalKind::Temperature(t) => RerankCalibrator::from_transform(Transform::Temperature(*t)).map_err(bad),
            CalKind::Platt(p) => {
                RerankCalibrator::from_transform(Transform::Platt { slope: p.slope, intercept: p.intercept })
                    .map_err(bad)
            }
            CalKind::Isotonic(iso) => RerankCalibrator::from_transform(Transform::Isotonic {
                upper: iso.upper.clone(),
                value: iso.value.clone(),
            })
            .map_err(bad),
            CalKind::TcalArtifact(bytes) => {
                RerankCalibrator::from_tcal_bytes(bytes, mref, domain, s.inference_hash).map_err(bad)
            }
        };
    }
    let sidecar = Path::new(&loaded.bundle).join("calibration.tcal");
    if sidecar.is_file() {
        let bytes = std::fs::read(&sidecar)
            .map_err(|e| Failure::new(TURBO_E_INTERNAL, 0, format!("read {}: {e}", sidecar.display())))?;
        let domain = spec.map(|s| s.domain.trim()).filter(|d| !d.is_empty()).unwrap_or("search");
        let hash = spec.map_or(0, |s| s.inference_hash);
        return RerankCalibrator::from_tcal_bytes(&bytes, mref, domain, hash).map_err(bad);
    }
    Ok(RerankCalibrator::uncalibrated())
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
pub(crate) fn rerank_and_highlight(
    state: &State,
    req: &RerankAndHighlightRequest,
    deadline: &Deadline,
) -> Result<RerankAndHighlightResponse> {
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
    let calibrator = build_calibrator(&rerank_loaded, &rerank_mref, req.calibration.as_ref())?;

    if req.candidates.is_empty() {
        return Ok(RerankAndHighlightResponse {
            query: req.query.clone(),
            rerank_model: Some(model_to_proto(&rerank_mref)),
            highlight_model: None,
            offsets: req.offsets,
            results: Vec::new(),
        });
    }

    let rerank_opts = turbo_rerank_options {
        struct_size: std::mem::size_of::<turbo_rerank_options>() as u32,
        query_tokens: req.query_tokens,
        max_tokens: req.max_tokens,
        stride: req.stride,
    };

    let _docs = rerank_loaded.metrics.track_docs(req.candidates.len(), Some(rerank_loaded.device));
    deadline.check_alive()?;
    let held = rerank_loaded.take()?;
    let slot = held.slot();
    let max_batch = (slot.info.max_batch as usize).max(1);
    if max_batch > MAX_RERANK_ITEMS as usize {
        return Err(Failure::new(TURBO_E_CAPACITY, 0, "invalid session max_batch"));
    }

    let q_view = turbo_text { ptr: req.query.as_ptr().cast(), len: req.query.len() as u64 };
    let mut logits = Vec::with_capacity(req.candidates.len());
    for chunk in req.candidates.chunks(max_batch) {
        deadline.check_alive()?;
        let pairs: Vec<turbo_text_pair> = chunk
            .iter()
            .map(|c| turbo_text_pair {
                query: q_view,
                passage: turbo_text { ptr: c.text.as_ptr().cast(), len: c.text.len() as u64 },
            })
            .collect();
        slot.session.write_pairs(&pairs, &rerank_opts)?;
        deadline.check_alive()?;
        rerank_loaded.metrics.record_batch_size(pairs.len());
        state.stream.runs.fetch_add(1, Ordering::Relaxed);
        let output = slot.session.run()?;
        let (dim, dtype, bytes) = read_output(&output, pairs.len())?;
        drop(output);
        if dtype != TURBO_DTYPE_F32 || (dim != 1 && dim != 2) {
            return Err(Failure::new(
                TURBO_E_INTERNAL,
                0,
                format!("unexpected rerank output shape: dim={dim}, dtype={dtype}"),
            ));
        }
        let vals = f32s(&bytes);
        if dim == 1 {
            logits.extend_from_slice(&vals);
        } else {
            for &[neg, pos] in vals.as_chunks::<2>().0 {
                logits.push(pos - neg);
            }
        }
    }
    drop(held);

    let mut results = Vec::with_capacity(req.candidates.len());
    for (idx, (cand, &raw_logit)) in req.candidates.iter().zip(&logits).enumerate() {
        let cal = calibrator.score(raw_logit).map_err(bad)?;
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

    let top_k = if req.highlight_top_k == 0 { 10 } else { req.highlight_top_k as usize }.min(results.len());
    let max_snippets = if req.max_snippets_per_candidate == 0 { 2 } else { req.max_snippets_per_candidate as usize };

    let highlight_model = if mode == RerankHighlightMode::None || top_k == 0 || max_snippets == 0 {
        None
    } else {
        highlight::attribute_top_k(
            state,
            &rerank_loaded,
            &rerank_mref,
            req,
            offsets,
            mode,
            &mut results[..top_k],
            max_snippets,
            deadline,
        )?
    };

    Ok(RerankAndHighlightResponse {
        query: req.query.clone(),
        rerank_model: Some(model_to_proto(&rerank_mref)),
        highlight_model: highlight_model.as_ref().map(model_to_proto),
        offsets: req.offsets,
        results,
    })
}
