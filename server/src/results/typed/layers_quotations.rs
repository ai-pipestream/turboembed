//! Typed quotations use the same projected view for content, speaker and cue.
use super::{ProjectionCx, build_layer};
use crate::proto::result::{QuotationAnnotation, TypedAnnotation, TypedLayer, typed_annotation::Value};
use std::any::TypeId;
use turbo_analysis::{Ix, KeyId, Layer};
use turbo_analysis_layers::Quotations;

pub(super) fn project_quotations(cx: &mut ProjectionCx<'_>, key: KeyId) -> Option<TypedLayer> {
    if key.ty != TypeId::of::<Quotations>() {
        return None;
    }
    Some(build_layer::<Quotations>(cx, key, |cx, layer, mref| {
        (0..layer.len())
            .map(|i| TypedAnnotation {
                spans: cx.project_spans(&layer.spans, Ix::new(i as u32)),
                // Keep native confidence in its f32 payload; no calibrated
                // probability is supplied by the shared quotation layer.
                probability: None,
                model_ref: mref.map(ToOwned::to_owned),
                value: Some(Value::Quotation(QuotationAnnotation {
                    kind: Some(i32::from(layer.kind[i].to_byte())),
                    confidence: Some(layer.confidence[i]),
                    speaker: layer.speaker[i].map(|span| cx.project_wire_span(layer.spans.view(), span)),
                    cue_verb: layer.cue_verb[i].map(|span| cx.project_wire_span(layer.spans.view(), span)),
                    speaker_chain: layer.speaker_chain[i],
                })),
            })
            .collect()
    }))
}
