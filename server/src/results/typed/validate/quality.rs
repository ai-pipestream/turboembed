//! Preserve the shared metric domains, including signed readability values.

use turbo_analysis::Layer;
use turbo_analysis_layers::{QualityFlags, QualityMetrics, SpanQuality, TextQuality};

use super::*;
use crate::proto::result::QualityMetricsAnnotation;

pub(super) fn layer(layer: &TypedLayer) -> Result<(), Failure> {
    let text = layer.id == TextQuality::ID;
    if (!text && layer.id != SpanQuality::ID) || layer.status != LayerWireStatus::Ok as i32 {
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
