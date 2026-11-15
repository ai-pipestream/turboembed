//! SparseTermsResult: the turbo-analysis-layers `SparseTerms` layer's
//! columns (issue #708), document level (`SparseTerms<View>`), with its
//! source spans, occurrence spans, and optional calibrated SPLADE `u8` / `u16`
//! integer impacts and upper-bound error certificate (`#2343`).

use turbo::tokenizer::Tokenizer;
use turbo_analysis::{Ix, Layer, ModelRef, Span, View};
use turbo_analysis_embed::{ImpactBits, ImpactQuantizer, QuantizedSparseTerms};
use turbo_analysis_layers::SparseTerms;

use super::{MAX_SPARSE_ENTRIES, MAX_SPARSE_OCCURRENCES, Wire, mismatched, model_from_proto, model_to_proto, over};
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::result::SparseTermsResult;

/// Document-level [`SparseTerms`] paired with optional view text, wire
/// [`OffsetMode`], and optional calibrated integer impacts (`#2343`).
pub struct SparseTermsWire {
    pub text: String,
    pub offsets: OffsetMode,
    pub terms: SparseTerms<View>,
    pub impacts: Option<QuantizedSparseTerms>,
}

impl SparseTermsWire {
    pub fn new(terms: SparseTerms<View>, text: impl Into<String>, offsets: OffsetMode) -> Self {
        Self { text: text.into(), offsets, terms, impacts: None }
    }

    /// Attaches pre-computed quantized SPLADE impacts and error certificate (`#2343`).
    pub fn with_impacts(mut self, impacts: QuantizedSparseTerms) -> Self {
        self.impacts = Some(impacts);
        self
    }

    /// Prunes and quantizes document `terms` (unit 0) using `quantizer`,
    /// preserving source and occurrence spans for surviving terms (`#2343`).
    pub fn quantize_doc(
        terms: &SparseTerms<View>,
        text: impl Into<String>,
        offsets: OffsetMode,
        quantizer: &ImpactQuantizer,
        tokenizer: Option<&Tokenizer>,
    ) -> Result<Self, Failure> {
        let quantized = quantizer
            .quantize_doc_layer(terms, Ix::new(0), tokenizer)
            .map_err(|e| Failure::new(e.code, e.field, e.message))?;
        let pruned = prune_layer_to_quantized(terms, &quantized);
        Ok(Self { text: text.into(), offsets, terms: pruned, impacts: Some(quantized) })
    }

    /// Prunes and quantizes query `terms` (unit 0) using `quantizer`,
    /// preserving source and occurrence spans for surviving terms (`#2343`).
    pub fn quantize_query(
        terms: &SparseTerms<View>,
        text: impl Into<String>,
        offsets: OffsetMode,
        quantizer: &ImpactQuantizer,
        tokenizer: Option<&Tokenizer>,
    ) -> Result<Self, Failure> {
        let quantized = quantizer
            .quantize_query_layer(terms, Ix::new(0), tokenizer)
            .map_err(|e| Failure::new(e.code, e.field, e.message))?;
        let pruned = prune_layer_to_quantized(terms, &quantized);
        Ok(Self { text: text.into(), offsets, terms: pruned, impacts: Some(quantized) })
    }
}

fn prune_layer_to_quantized(terms: &SparseTerms<View>, quantized: &QuantizedSparseTerms) -> SparseTerms<View> {
    let mut out = SparseTerms::new(terms.model.clone());
    let (ids, _) = terms.entries(Ix::new(0));
    let spans = terms.spans(Ix::new(0));
    let has_occ = terms.has_occurrences();
    let mut i = 0usize;
    for &(vocab_id, _) in quantized.impacts() {
        while i < ids.len() && ids[i] < vocab_id {
            i += 1;
        }
        if i < ids.len() && ids[i] == vocab_id {
            out.push(vocab_id, terms.weight[i], spans.get(i).copied().flatten());
            if has_occ {
                out.push_occurrences(terms.occurrences(i));
            }
        }
    }
    out.unit();
    out
}

fn spans_to_wire(spans: &[Span], text: &str, offsets: OffsetMode) -> (Vec<u32>, Vec<u32>) {
    if text.is_empty() && offsets == OffsetMode::Utf8 {
        (spans.iter().map(|s| s.start).collect(), spans.iter().map(|s| s.end).collect())
    } else {
        Offsetter::new(text, offsets).spans_to_wire(text, spans)
    }
}

fn sparse_to_msg(
    t: &SparseTerms<View>,
    text: &str,
    offsets: OffsetMode,
    impacts: Option<&QuantizedSparseTerms>,
) -> SparseTermsResult {
    let has_occ = t.has_occurrences();
    let any_span = t.span.iter().any(|s| s.is_some());
    let present_spans: Vec<Span> = t.span.iter().flatten().copied().collect();
    let (span_start, span_end) =
        if any_span { spans_to_wire(&present_spans, text, offsets) } else { (Vec::new(), Vec::new()) };
    let (occ_start, occ_end) = if has_occ { spans_to_wire(&t.occ, text, offsets) } else { (Vec::new(), Vec::new()) };
    let (impact, impact_bits, impact_scale, impact_w_max, impact_l1_rounding_bound, impact_dot_error_bound, term_key) =
        match impacts {
            Some(imp) => {
                let cert = imp.certificate();
                (
                    imp.impacts().iter().map(|&(_, q)| q as u32).collect(),
                    imp.bits().bits(),
                    imp.scale(),
                    imp.w_max(),
                    cert.l1_rounding_bound,
                    cert.dot_error_bound,
                    imp.term_keys().to_vec(),
                )
            }
            None => (Vec::new(), 0, 0.0, 0.0, 0.0, 0.0, Vec::new()),
        };
    SparseTermsResult {
        model: Some(model_to_proto(&t.model)),
        row: t.row.clone(),
        term: t.term.clone(),
        weight: t.weight.clone(),
        has_span: if any_span { t.span.iter().map(|s| s.is_some()).collect() } else { Vec::new() },
        span_start,
        span_end,
        occ_start,
        occ_end,
        occ_at: if has_occ { t.occ_at.clone() } else { Vec::new() },
        text: text.to_string(),
        offsets: offsets.to_proto(),
        impact,
        impact_bits,
        impact_scale,
        impact_w_max,
        impact_l1_rounding_bound,
        impact_dot_error_bound,
        term_key,
    }
}

fn spans_from_wire(
    what: &str,
    starts: &[u32],
    ends: &[u32],
    text: &str,
    offsets: OffsetMode,
) -> Result<Vec<Span>, Failure> {
    if text.is_empty() && offsets == OffsetMode::Utf8 {
        Ok(starts.iter().zip(ends).map(|(&s, &e)| Span { start: s, end: e }).collect())
    } else {
        Offsetter::new(text, offsets).spans_from_wire(text, what, starts, ends)
    }
}

fn decode_impacts(msg: &SparseTermsResult, n: usize) -> Result<Option<QuantizedSparseTerms>, Failure> {
    let any_impact = msg.impact_bits != 0
        || !msg.impact.is_empty()
        || msg.impact_scale != 0.0
        || msg.impact_w_max != 0.0
        || msg.impact_l1_rounding_bound != 0.0
        || msg.impact_dot_error_bound != 0.0
        || !msg.term_key.is_empty();
    if !any_impact {
        return Ok(None);
    }
    let bits =
        ImpactBits::from_bits(msg.impact_bits).map_err(|e| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, e.message))?;
    if msg.impact.len() != n {
        return Err(mismatched("impact", msg.impact.len(), n));
    }
    if !msg.term_key.is_empty() && msg.term_key.len() != n {
        return Err(mismatched("term_key", msg.term_key.len(), n));
    }
    let max_q = bits.max_impact() as u32;
    let mut pairs = Vec::with_capacity(n);
    for (k, (&vocab_id, &q)) in msg.term.iter().zip(&msg.impact).enumerate() {
        if q == 0 || q > max_q {
            return Err(Failure::new(
                TURBO_E_INVALID_ARGUMENT,
                0,
                format!("impact[{k}] (term {vocab_id}) is {q}, outside 1..={max_q}"),
            ));
        }
        pairs.push((vocab_id, q as u16));
    }
    let quantized = QuantizedSparseTerms::from_raw_parts(
        pairs,
        msg.weight.clone(),
        msg.term_key.clone(),
        bits,
        msg.impact_scale,
        msg.impact_w_max,
        msg.impact_l1_rounding_bound,
        msg.impact_dot_error_bound,
    )
    .map_err(|e| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, e.message))?;
    Ok(Some(quantized))
}

impl Wire for SparseTermsWire {
    type Msg = SparseTermsResult;

    fn to_msg(&self) -> SparseTermsResult {
        sparse_to_msg(&self.terms, &self.text, self.offsets, self.impacts.as_ref())
    }

    fn from_msg(msg: &SparseTermsResult) -> Result<Self, Failure> {
        let offsets = OffsetMode::from_proto(msg.offsets)?;
        let n = msg.term.len();
        if n as u64 > MAX_SPARSE_ENTRIES {
            return Err(over("max_sparse_entries", "sparse term entries", n as u64));
        }
        if msg.occ_start.len() as u64 > MAX_SPARSE_OCCURRENCES {
            return Err(over("max_sparse_occurrences", "sparse occurrence spans", msg.occ_start.len() as u64));
        }
        if msg.weight.len() != n {
            return Err(mismatched("weight", msg.weight.len(), n));
        }
        let spans_given = !msg.has_span.is_empty() || !msg.span_start.is_empty() || !msg.span_end.is_empty();
        if spans_given && !(msg.has_span.len() == n && msg.span_start.len() == msg.span_end.len()) {
            return Err(mismatched("has_span", msg.has_span.len(), n));
        }
        let spans = if msg.has_span.is_empty() {
            Vec::new()
        } else {
            let present = spans_from_wire("sparse term span", &msg.span_start, &msg.span_end, &msg.text, offsets)?;
            let mut spans = Vec::with_capacity(n);
            let mut k = 0;
            for &is_present in &msg.has_span {
                if !is_present {
                    spans.push(None);
                } else {
                    let Some(&span) = present.get(k) else {
                        return Err(mismatched("span_start", msg.span_start.len(), msg.span_end.len()));
                    };
                    spans.push(Some(span));
                    k += 1;
                }
            }
            if k != present.len() {
                return Err(mismatched("span_start", msg.span_start.len(), k));
            }
            spans
        };
        if msg.occ_start.len() != msg.occ_end.len() {
            return Err(mismatched("occ_start", msg.occ_start.len(), msg.occ_end.len()));
        }
        let (occ, occ_at) = if msg.occ_start.is_empty() {
            if msg.occ_at.is_empty() {
                (Vec::new(), Vec::new())
            } else {
                if msg.occ_at.len() != n || msg.occ_at.iter().any(|&e| e != 0) {
                    return Err(Failure::new(
                        TURBO_E_INVALID_ARGUMENT,
                        0,
                        "occ_at: zero-occurrence offsets are 0, one per entry",
                    ));
                }
                (Vec::new(), msg.occ_at.clone())
            }
        } else {
            if msg.occ_at.len() != n {
                return Err(mismatched("occ_at", msg.occ_at.len(), n));
            }
            if !msg.occ_at.windows(2).all(|w| w[0] <= w[1]) || msg.occ_at.last() != Some(&(msg.occ_start.len() as u32))
            {
                return Err(Failure::new(
                    TURBO_E_INVALID_ARGUMENT,
                    0,
                    "occ_at: occurrence ends in entry order, the last one past the last occurrence",
                ));
            }
            let occ = spans_from_wire("sparse occurrence span", &msg.occ_start, &msg.occ_end, &msg.text, offsets)?;
            (occ, msg.occ_at.clone())
        };
        if msg.row.is_empty() || msg.row[0] != 0 || msg.row.last().is_some_and(|&r| r as usize != n) {
            return Err(Failure::new(
                TURBO_E_INVALID_ARGUMENT,
                0,
                format!("row: one start per unit and one past the last of the {n} entries, the first 0"),
            ));
        }
        if msg.row.iter().any(|&r| r as usize > n) || !msg.row.windows(2).all(|w| w[0] <= w[1]) {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "row: starts in unit order, each an entry index"));
        }
        let model: ModelRef = msg.model.as_ref().map(model_from_proto).unwrap_or(ModelRef {
            id: String::new(),
            revision: String::new(),
            sha256: String::new(),
        });
        let has_occ = !occ_at.is_empty();
        let mut out = SparseTerms::new(model);
        let mut e = 0;
        for w in msg.row.windows(2) {
            for k in w[0] as usize..w[1] as usize {
                out.push(msg.term[k], msg.weight[k], spans.get(e).copied().flatten());
                if has_occ {
                    let lo = if e == 0 { 0 } else { occ_at[e - 1] as usize };
                    out.push_occurrences(&occ[lo..occ_at[e] as usize]);
                }
                e += 1;
            }
            out.unit();
        }
        out.check().map_err(|e| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("sparse terms: {e}")))?;
        let impacts = decode_impacts(msg, n)?;
        Ok(SparseTermsWire { text: msg.text.clone(), offsets, terms: out, impacts })
    }
}

impl Wire for SparseTerms<View> {
    type Msg = SparseTermsResult;

    fn to_msg(&self) -> SparseTermsResult {
        sparse_to_msg(self, "", OffsetMode::Utf8, None)
    }

    fn from_msg(msg: &SparseTermsResult) -> Result<Self, Failure> {
        SparseTermsWire::from_msg(msg).map(|w| w.terms)
    }
}

#[cfg(test)]
#[path = "sparse_tests.rs"]
mod tests;
