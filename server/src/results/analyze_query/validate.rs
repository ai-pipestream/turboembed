//! Wire round-trip validation for [`AnalyzeQueryResponse`] (`#2339`).

use std::collections::HashSet;

use turbo_query::QueryComparisonOp;

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::result::{
    AnalyzeQueryResponse, QueryDenseResultProto, QueryFieldPlanProto, QueryOrdinalTermProto, QuerySpladeResultProto,
    QueryStructuredFilterProto, QueryVectorSliceProto,
};
use crate::results::{Wire, f32s, mismatched};

fn bad(msg: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, msg)
}

fn check_term(
    offsetter: &Offsetter,
    query: &str,
    total: Option<u32>,
    t: &QueryOrdinalTermProto,
) -> Result<(), Failure> {
    if t.term.is_empty() {
        return Err(bad("QueryOrdinalTermProto.term must be non-empty"));
    }
    if t.pos_len == 0 {
        return Err(bad("QueryOrdinalTermProto.pos_len must be >= 1"));
    }
    if let Some(tot) = total
        && t.ordinal >= tot
    {
        return Err(bad(format!("QueryOrdinalTermProto.ordinal {} >= total_token_ordinals {tot}", t.ordinal)));
    }
    let sp = offsetter.from_wire(query, "query_term.span", t.span_start, t.span_end)?;
    if query[sp.range()] != t.surface {
        return Err(bad("QueryOrdinalTermProto.surface does not match query[span]"));
    }
    Ok(())
}

fn check_field(offsetter: &Offsetter, query: &str, f: &QueryFieldPlanProto) -> Result<(), Failure> {
    if f.field.trim().is_empty() {
        return Err(bad("QueryFieldPlanProto.field must be non-empty"));
    }
    let mut seen_ord = HashSet::new();
    let mut prev_empty = None;
    for &ord in &f.empty_normalized_ordinals {
        if ord >= f.total_token_ordinals {
            return Err(bad(format!(
                "empty_normalized_ordinal {ord} >= total_token_ordinals {}",
                f.total_token_ordinals
            )));
        }
        if let Some(prev) = prev_empty
            && ord <= prev
        {
            return Err(bad("empty_normalized_ordinals must be strictly ascending"));
        }
        prev_empty = Some(ord);
        seen_ord.insert(ord);
    }
    let mut prev_bm25 = None;
    for t in &f.bm25_terms {
        check_term(offsetter, query, Some(f.total_token_ordinals), t)?;
        if let Some(prev) = prev_bm25
            && t.ordinal <= prev
        {
            return Err(bad("bm25_terms ordinals must be strictly ascending"));
        }
        if !seen_ord.insert(t.ordinal) {
            return Err(bad(format!("ordinal {} appears in both empty_normalized_ordinals and bm25_terms", t.ordinal)));
        }
        prev_bm25 = Some(t.ordinal);
    }
    for t in &f.residual_terms {
        check_term(offsetter, query, Some(f.total_token_ordinals), t)?;
    }
    for s in &f.synonyms {
        if s.start_ordinal >= s.end_ordinal || s.end_ordinal > f.total_token_ordinals {
            return Err(bad("QuerySynonymMatchProto ordinal range is invalid"));
        }
        if s.pos_len != s.end_ordinal - s.start_ordinal {
            return Err(bad("QuerySynonymMatchProto.pos_len must equal end_ordinal - start_ordinal"));
        }
        if s.terms.is_empty() || s.terms.iter().any(String::is_empty) {
            return Err(bad("QuerySynonymMatchProto.terms must be non-empty"));
        }
        let sp = offsetter.from_wire(query, "synonym.span", s.span_start, s.span_end)?;
        if query[sp.range()] != s.surface {
            return Err(bad("QuerySynonymMatchProto.surface does not match query[span]"));
        }
    }
    Ok(())
}

fn check_filter(offsetter: &Offsetter, query: &str, f: &QueryStructuredFilterProto) -> Result<(), Failure> {
    let sp = offsetter.from_wire(query, "filter.span", f.span_start, f.span_end)?;
    let vsp = offsetter.from_wire(query, "filter.value_span", f.value_span_start, f.value_span_end)?;
    if vsp.start < sp.start || vsp.end > sp.end {
        return Err(bad("filter.value_span must lie within filter.span"));
    }
    if query[sp.range()] != f.surface {
        return Err(bad("QueryStructuredFilterProto.surface does not match query[span]"));
    }
    if QueryComparisonOp::parse(&f.comparison).is_none() {
        return Err(bad(format!("unknown filter comparison {:?}", f.comparison)));
    }
    if f.payload.is_none() {
        return Err(bad("QueryStructuredFilterProto.payload must be set"));
    }
    Ok(())
}

fn check_slice(what: &str, s: &QueryVectorSliceProto) -> Result<(), Failure> {
    if s.dim == 0 {
        return Err(bad(format!("{what}.dim must be > 0")));
    }
    if s.data.len() != s.dim as usize * 4 {
        return Err(bad(format!("{what}.data byte length {} != dim * 4", s.data.len())));
    }
    let vals = f32s(&s.data);
    if vals.iter().any(|v| !v.is_finite()) || !s.inv_norm.is_finite() || s.inv_norm < 0.0 {
        return Err(bad(format!("{what} contains non-finite float values")));
    }
    if s.sha256.len() != 32 {
        return Err(bad(format!("{what}.sha256 must be 32 bytes")));
    }
    if !matches!(s.turboquant_bits, 0 | 2..=4) {
        return Err(bad(format!("{what}.turboquant_bits {} must be 0 or 2..=4", s.turboquant_bits)));
    }
    Ok(())
}

fn check_model_ref(what: &str, m: Option<&crate::proto::result::ModelRef>) -> Result<(), Failure> {
    let Some(m) = m else {
        return Err(bad(format!("{what}.model is required")));
    };
    if m.id.trim().is_empty() {
        return Err(bad(format!("{what}.model.id must be non-empty")));
    }
    Ok(())
}

fn check_dense(d: &QueryDenseResultProto) -> Result<(), Failure> {
    if d.target_id.is_empty() {
        return Err(bad("QueryDenseResultProto.target_id must be non-empty"));
    }
    check_model_ref("QueryDenseResultProto", d.model.as_ref())?;
    let full = d.full.as_ref().ok_or_else(|| bad("QueryDenseResultProto.full is required"))?;
    check_slice("dense.full", full)?;
    let mut prev_dim = 0u32;
    for m in &d.matryoshka {
        check_slice("dense.matryoshka", m)?;
        if m.dim >= full.dim || m.dim <= prev_dim {
            return Err(bad("dense.matryoshka dims must be strictly ascending and < full.dim"));
        }
        prev_dim = m.dim;
    }
    Ok(())
}

fn check_splade(offsetter: &Offsetter, query: &str, s: &QuerySpladeResultProto) -> Result<(), Failure> {
    if s.target_id.is_empty() {
        return Err(bad("QuerySpladeResultProto.target_id must be non-empty"));
    }
    check_model_ref("QuerySpladeResultProto", s.model.as_ref())?;
    let n = s.term_id.len();
    if s.weight.len() != n {
        return Err(mismatched("splade.weight", s.weight.len(), n));
    }
    let mut prev = None;
    for (&tid, &w) in s.term_id.iter().zip(&s.weight) {
        if let Some(p) = prev
            && tid <= p
        {
            return Err(bad("splade.term_id must be strictly ascending"));
        }
        if !w.is_finite() || w <= 0.0 {
            return Err(bad("splade.weight must be finite and > 0"));
        }
        prev = Some(tid);
    }
    if !s.has_span.is_empty() || !s.span_start.is_empty() || !s.span_end.is_empty() {
        if s.has_span.len() != n || s.span_start.len() != n || s.span_end.len() != n {
            return Err(bad("splade span columns must be empty or match term_id length"));
        }
        for i in 0..n {
            if s.has_span[i] {
                offsetter.from_wire(query, "splade.span", s.span_start[i], s.span_end[i])?;
            }
        }
    }
    match s.impact_bits {
        0 => {
            if !s.impact.is_empty() {
                return Err(bad("splade.impact must be empty when impact_bits is 0"));
            }
        }
        8 | 16 => {
            let max_imp = if s.impact_bits == 8 { u32::from(u8::MAX) } else { u32::from(u16::MAX) };
            if s.impact.len() != n || s.impact.iter().any(|&v| v == 0 || v > max_imp) {
                return Err(bad("splade.impact values out of range for impact_bits"));
            }
        }
        other => return Err(bad(format!("unsupported splade.impact_bits {other}"))),
    }
    if !s.term_keys.is_empty() && (s.term_keys.len() != n || s.term_keys.iter().any(String::is_empty)) {
        return Err(bad("splade.term_keys must be empty or match term_id length with non-empty keys"));
    }
    Ok(())
}

impl Wire for AnalyzeQueryResponse {
    type Msg = AnalyzeQueryResponse;

    fn to_msg(&self) -> Self::Msg {
        self.clone()
    }

    fn from_msg(msg: &Self::Msg) -> Result<Self, Failure> {
        let mode = OffsetMode::from_proto(msg.offsets)?;
        let offsetter = Offsetter::new(&msg.query, mode);
        if !msg.language_score.is_finite() || !(0.0..=1.0).contains(&msg.language_score) {
            return Err(bad("AnalyzeQueryResponse.language_score must be in [0.0, 1.0]"));
        }
        for f in &msg.fields {
            check_field(&offsetter, &msg.query, f)?;
        }
        for flt in &msg.filters {
            check_filter(&offsetter, &msg.query, flt)?;
        }
        let first_total = msg.fields.first().map(|f| f.total_token_ordinals);
        for t in &msg.residual_terms {
            check_term(&offsetter, &msg.query, first_total, t)?;
        }
        for d in &msg.dense_results {
            check_dense(d)?;
        }
        for sp in &msg.splade_results {
            check_splade(&offsetter, &msg.query, sp)?;
        }
        Ok(msg.clone())
    }
}
