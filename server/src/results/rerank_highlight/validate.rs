//! Wire round-trip validation for [`RerankAndHighlightResponse`] (`#2342`).

use std::collections::BTreeSet;

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::OffsetMode;
use crate::proto::result::{ModelRef, RerankAndHighlightResponse, RerankedCandidateProto, SemanticHighlightSpanProto};
use crate::results::{MAX_RERANK_ITEMS, Wire, over};

fn bad(msg: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, msg)
}

fn check_model_ref(what: &str, m: Option<&ModelRef>) -> Result<(), Failure> {
    let Some(m) = m else {
        return Err(bad(format!("{what} is required")));
    };
    if m.id.trim().is_empty() {
        return Err(bad(format!("{what}.id must be non-empty")));
    }
    Ok(())
}

fn wire_surface_units(surface: &str, mode: OffsetMode) -> usize {
    match mode {
        OffsetMode::Utf8 => surface.len(),
        OffsetMode::Utf16 => surface.encode_utf16().count(),
        OffsetMode::CodePoint => surface.chars().count(),
    }
}

fn check_highlight(mode: OffsetMode, h: &SemanticHighlightSpanProto) -> Result<(), Failure> {
    if h.span_start >= h.span_end {
        return Err(bad("SemanticHighlightSpanProto.span_start must be < span_end"));
    }
    if h.surface.is_empty() {
        return Err(bad("SemanticHighlightSpanProto.surface must be non-empty"));
    }
    if !h.score.is_finite() {
        return Err(bad("SemanticHighlightSpanProto.score must be finite"));
    }
    let width = (h.span_end - h.span_start) as usize;
    if wire_surface_units(&h.surface, mode) != width {
        return Err(bad(
            "SemanticHighlightSpanProto.surface length does not match span_end - span_start in OffsetMode",
        ));
    }
    let parent_width = h
        .parent_span_end
        .checked_sub(h.parent_span_start)
        .ok_or_else(|| bad("SemanticHighlightSpanProto.parent_span_end must be >= parent_span_start"))?;
    if parent_width != width as u64 {
        return Err(bad("SemanticHighlightSpanProto parent span width must match local span width"));
    }
    if h.parent_span_start < u64::from(h.span_start) {
        return Err(bad("SemanticHighlightSpanProto.parent_span_start cannot be smaller than span_start"));
    }
    Ok(())
}

fn check_candidate(mode: OffsetMode, c: &RerankedCandidateProto) -> Result<(), Failure> {
    if !c.raw_logit.is_finite() {
        return Err(bad("RerankedCandidateProto.raw_logit must be finite"));
    }
    if !c.calibrated_probability.is_finite() || !(0.0..=1.0).contains(&c.calibrated_probability) {
        return Err(bad("RerankedCandidateProto.calibrated_probability must lie in [0.0, 1.0]"));
    }
    if let Some(pv) = c.conformal_p_value
        && (!pv.is_finite() || pv <= 0.0 || pv > 1.0)
    {
        return Err(bad("RerankedCandidateProto.conformal_p_value must lie in (0.0, 1.0]"));
    }
    if let Some(np) = c.qa_null_probability
        && (!np.is_finite() || !(0.0..=1.0).contains(&np))
    {
        return Err(bad("RerankedCandidateProto.qa_null_probability must lie in [0.0, 1.0]"));
    }
    let mut prev_score: Option<f64> = None;
    for (i, h) in c.highlights.iter().enumerate() {
        check_highlight(mode, h)?;
        if let Some(prev) = prev_score
            && h.score > prev
        {
            return Err(bad("RerankedCandidateProto.highlights must be ordered by non-increasing score"));
        }
        prev_score = Some(h.score);
        for other in &c.highlights[..i] {
            if h.span_start < other.span_end && other.span_start < h.span_end {
                return Err(bad("RerankedCandidateProto.highlights must not overlap"));
            }
        }
    }
    Ok(())
}

impl Wire for RerankAndHighlightResponse {
    type Msg = RerankAndHighlightResponse;

    fn to_msg(&self) -> Self::Msg {
        self.clone()
    }

    fn from_msg(msg: &Self::Msg) -> Result<Self, Failure> {
        if msg.results.len() as u64 > MAX_RERANK_ITEMS {
            return Err(over("max_rerank_items", "results", msg.results.len() as u64));
        }
        let mode = OffsetMode::from_proto(msg.offsets)?;
        check_model_ref("rerank_model", msg.rerank_model.as_ref())?;
        if let Some(hm) = &msg.highlight_model
            && hm.id.trim().is_empty()
        {
            return Err(bad("highlight_model.id must be non-empty when present"));
        }
        let mut seen = BTreeSet::new();
        for (k, r) in msg.results.iter().enumerate() {
            if !seen.insert(r.input_index) {
                return Err(bad(format!("duplicate RerankedCandidateProto.input_index {}", r.input_index)));
            }
            check_candidate(mode, r)?;
            if k > 0 {
                let prev = &msg.results[k - 1];
                let ord = prev
                    .calibrated_probability
                    .total_cmp(&r.calibrated_probability)
                    .then_with(|| prev.raw_logit.total_cmp(&r.raw_logit));
                if ord.is_lt() {
                    return Err(bad("RerankAndHighlightResponse.results must be sorted descending by score"));
                }
            }
        }
        Ok(msg.clone())
    }
}
