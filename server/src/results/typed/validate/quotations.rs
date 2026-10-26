//! Quotation confidence/kind and optional projected coordinates.
use super::*;
use crate::proto::result::QuotationAnnotation;
use turbo_analysis::Layer;
use turbo_analysis_layers::{Quotations, QuoteKind};

pub(super) fn layer(layer: &TypedLayer) -> Result<(), Failure> {
    if layer.id != Quotations::ID || (layer.status != LayerWireStatus::Ok as i32 && layer.annotations.is_empty()) {
        return Ok(());
    }
    if layer.scope != AnchorKind::Text as i32
        || layer.annotations.iter().any(|a| !matches!(&a.value, Some(Value::Quotation(_))))
    {
        return Err(bad("quotations need text scope and matching quotation payloads"));
    }
    Ok(())
}

pub(super) fn record(
    ctx: &LayerCtx<'_>,
    i: usize,
    ann: &TypedAnnotation,
    q: &QuotationAnnotation,
) -> Result<(), Failure> {
    if ctx.lid != Quotations::ID || ann.spans.is_empty() {
        return Err(bad("quotation payload needs its matching layer and projected content"));
    }
    if q.kind.and_then(|k| u8::try_from(k).ok()).and_then(QuoteKind::from_byte).is_none() {
        return Err(bad("quotation kind is missing or unknown"));
    }
    let confidence = q.confidence.ok_or_else(|| bad("quotation confidence is missing"))?;
    check_f32_prob("quotation confidence", ctx.lid, i, confidence)?;
    for (name, span) in [("speaker", q.speaker.as_ref()), ("cue_verb", q.cue_verb.as_ref())] {
        if let Some(s) = span {
            ctx.offsetter.from_wire(ctx.text, &format!("{}[{i}].{name}", ctx.lid), s.start, s.end)?;
        }
    }
    // Projected content can retain several fragments or collapse to one empty
    // fragment. A deleted leading fragment can also move its visible start
    // past a later record. Generic span validation applies; no row re-sort.
    // speaker_chain remains the native raw contextual row, not a node index.
    Ok(())
}
