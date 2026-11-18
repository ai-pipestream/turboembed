//! Typed projection for `turbo:token-graph` (`TokenGraph`, issue #2340).
use super::{ProjectionCx, build_layer};
use crate::proto::result::{TokenArcAnnotation, TypedAnnotation, TypedLayer, typed_annotation::Value};
use std::any::TypeId;
use turbo_analysis::{Ix, KeyId, Layer};
use turbo_analysis_layers::TokenGraph;

pub(super) fn project_token_graph(cx: &mut ProjectionCx<'_>, key: KeyId) -> Option<TypedLayer> {
    if key.ty != TypeId::of::<TokenGraph>() {
        return None;
    }
    Some(build_layer::<TokenGraph>(cx, key, |cx, layer, mref| {
        let num_positions = layer.num_positions;
        (0..layer.len())
            .map(|i| {
                let ix = Ix::new(i as u32);
                let arc = layer.get(ix);
                TypedAnnotation {
                    spans: cx.project_spans(&layer.spans, ix),
                    probability: None,
                    model_ref: mref.map(ToOwned::to_owned),
                    value: Some(Value::TokenArc(TokenArcAnnotation {
                        position: arc.position,
                        position_length: arc.position_length,
                        term: arc.term.to_owned(),
                        kind: u32::from(arc.kind.to_byte()),
                        num_positions,
                    })),
                }
            })
            .collect()
    }))
}
