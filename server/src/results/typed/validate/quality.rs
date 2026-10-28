//! Preserve the shared metric domains, including signed readability values.

use turbo_analysis::Layer;
use turbo_analysis_layers::{
    GrammarCategory, GrammarDiagnostics, QualityFlags, QualityMetrics, SpanQuality, TextQuality,
};

use super::*;
use crate::proto::result::{GrammarDiagnosticAnnotation, QualityMetricsAnnotation};

const MAX_RULE_ID_BYTES: usize = 64;

pub(super) fn layer(layer: &TypedLayer) -> Result<(), Failure> {
    if layer.status != LayerWireStatus::Ok as i32 {
        return Ok(());
    }
    if layer.id == GrammarDiagnostics::ID {
        if layer.scope != AnchorKind::Text as i32 {
            return Err(bad("grammar diagnostics layer needs text scope"));
        }
        for ann in &layer.annotations {
            if !matches!(&ann.value, Some(Value::GrammarDiagnostic(_))) {
                return Err(bad("grammar diagnostics layer has a different typed payload"));
            }
        }
        return Ok(());
    }
    let text = layer.id == TextQuality::ID;
    if !text && layer.id != SpanQuality::ID {
        return Ok(());
    }
    if layer.scope != AnchorKind::Text as i32 || (text && layer.annotations.len() != 1) {
        return Err(bad("quality layers need text scope and text quality needs one summary"));
    }
    for ann in &layer.annotations {
        if !matches!((&ann.value, text), (Some(Value::TextQuality(_)), true) | (Some(Value::SpanQuality(_)), false)) {
            return Err(bad("quality layer has a different typed payload"));
        }
    }
    Ok(())
}

pub(super) fn grammar_diagnostic(
    ctx: &LayerCtx<'_>,
    ann: &TypedAnnotation,
    d: &GrammarDiagnosticAnnotation,
) -> Result<(), Failure> {
    if ctx.lid != GrammarDiagnostics::ID || ann.spans.len() != 1 {
        return Err(bad("grammar diagnostic payload needs turbo:grammar_diagnostics layer and one projected span"));
    }
    let trimmed = d.rule_id.trim();
    if trimmed.is_empty() || trimmed.len() != d.rule_id.len() || d.rule_id.len() > MAX_RULE_ID_BYTES {
        return Err(bad("grammar diagnostic rule_id must be nonempty, trimmed, and at most 64 bytes"));
    }
    if GrammarCategory::from_name(&d.category).is_none() {
        return Err(bad("grammar diagnostic category is unknown"));
    }
    if d.message.trim().is_empty() {
        return Err(bad("grammar diagnostic message must be nonempty"));
    }
    if d.replacements.iter().any(|r| r.is_empty()) {
        return Err(bad("grammar diagnostic replacements must be nonempty strings"));
    }
    if !d.confidence.is_finite() || !(0.0..=1.0).contains(&d.confidence) {
        return Err(bad("grammar diagnostic confidence must be finite and in [0, 1]"));
    }
    Ok(())
}

pub(super) fn metrics(
    ctx: &LayerCtx<'_>,
    ann: &TypedAnnotation,
    m: &QualityMetricsAnnotation,
    text: bool,
) -> Result<(), Failure> {
    let expected = if text { TextQuality::ID } else { SpanQuality::ID };
    if ctx.lid != expected || ann.spans.len() != 1 {
        return Err(bad("quality payload needs its matching layer and one projected span"));
    }
    let required = |value: Option<f32>| value.ok_or_else(|| bad("quality metric is missing"));
    let flags = u16::try_from(m.flags).map_err(|_| bad("quality flags exceed the shared bit set"))?;
    QualityMetrics {
        composite_score: required(m.composite_score)?,
        mtld_lexical_diversity: required(m.mtld_lexical_diversity)?,
        flesch_kincaid_grade: required(m.flesch_kincaid_grade)?,
        automated_readability_index: required(m.automated_readability_index)?,
        coleman_liau_index: required(m.coleman_liau_index)?,
        information_density: required(m.information_density)?,
        alpha_ratio: required(m.alpha_ratio)?,
        digit_ratio: required(m.digit_ratio)?,
        symbol_ratio: required(m.symbol_ratio)?,
        whitespace_ratio: required(m.whitespace_ratio)?,
        replacement_ratio: required(m.replacement_ratio)?,
        flags: QualityFlags(flags),
    }
    .check()
    .map_err(bad)
}
