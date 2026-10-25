//! Category evidence keeps exact producer and supporting-layer references.

use super::*;
use crate::proto::result::EvidenceAnnotation;
use std::sync::OnceLock;
use turbo_analysis::{Evidence, Layer, View};
use turbo_analysis_layers::{Categories, Chunks, Sentences, Structure};

fn known(lid: &str) -> bool {
    static IDS: OnceLock<[String; 4]> = OnceLock::new();
    IDS.get_or_init(|| {
        [
            Evidence::<Categories<View>>::NAME.to_string(),
            Evidence::<Categories<Sentences>>::NAME.to_string(),
            Evidence::<Categories<Structure>>::NAME.to_string(),
            Evidence::<Categories<Chunks>>::NAME.to_string(),
        ]
    })
    .iter()
    .any(|id| id == lid)
}

pub(super) fn layer(layer: &TypedLayer) -> Result<(), Failure> {
    if known(&layer.id)
        && layer.status == LayerWireStatus::Ok as i32
        && (layer.scope != AnchorKind::Aligned as i32
            || layer.annotations.iter().any(|a| !matches!(a.value, Some(Value::Evidence(_)))))
    {
        return Err(bad("category evidence needs aligned scope and evidence payloads"));
    }
    Ok(())
}

pub(super) fn record(
    ctx: &LayerCtx<'_>,
    i: usize,
    ann: &TypedAnnotation,
    record: &EvidenceAnnotation,
) -> Result<(), Failure> {
    if !known(ctx.lid) || ann.model_ref.is_none() {
        return Err(bad("category evidence needs its matching layer and producer reference"));
    }
    check_aligned_index("target_index", ctx.lid, i, record.target_index, ctx.layer_len)?;
    if record.raw_score.is_some_and(|v| !v.is_finite()) {
        return Err(bad("evidence raw score must be finite"));
    }
    if let Some(p) = record.calibrated_probability {
        check_prob("evidence calibrated probability", ctx.lid, i, p)?;
    }
    for support in &record.support {
        let len = ctx
            .layer_lengths
            .get(&(support.layer.as_str(), support.instance.as_str()))
            .ok_or_else(|| bad("evidence support refers to a missing layer instance"))?;
        if support.start > support.end || support.end as usize > *len {
            return Err(bad("evidence support range exceeds its selected layer instance"));
        }
    }
    Ok(())
}
