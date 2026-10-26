//! Category evidence is a typed aligned layer; entity votes stay embedded.

use std::any::TypeId;

use turbo_analysis::{Evidence, KeyId, View, ViewId};
use turbo_analysis_layers::{Categories, CategoryScope, Chunks, Sentences, Structure};

use super::{ProjectionCx, build_layer, coalesce_wire_spans};
use crate::proto::result::typed_annotation::Value;
use crate::proto::result::{EvidenceAnnotation, EvidenceSupport, TypedAnnotation, TypedLayer};

fn project<L: CategoryScope>(cx: &mut ProjectionCx<'_>, key: KeyId) -> TypedLayer {
    build_layer::<Evidence<Categories<L>>>(cx, key, |cx, evidence, _| {
        (0..evidence.records())
            .map(|i| {
                let producer = evidence.producer_of(i);
                let model = cx.register_producer(&producer.annotator, producer.model.as_ref());
                TypedAnnotation {
                    spans: coalesce_wire_spans(
                        evidence.spans_of(i).iter().map(|&s| cx.project_wire_span(ViewId::SOURCE, s)).collect(),
                    ),
                    probability: None,
                    model_ref: Some(model),
                    value: Some(Value::Evidence(EvidenceAnnotation {
                        target_index: i as u32,
                        raw_score: evidence.raw_of(i),
                        calibrated_probability: evidence.calibrated_of(i),
                        abstention: evidence.abstention_of(i).map(ToOwned::to_owned),
                        support: evidence
                            .support_of(i)
                            .iter()
                            .map(|s| {
                                let target = &evidence.layers[s.layer as usize];
                                EvidenceSupport {
                                    layer: target.layer.clone(),
                                    instance: target.instance.clone(),
                                    start: s.start,
                                    end: s.end,
                                }
                            })
                            .collect(),
                    })),
                }
            })
            .collect()
    })
}

pub(crate) fn project_evidence_layer(cx: &mut ProjectionCx<'_>, key: KeyId) -> Option<TypedLayer> {
    let ty = key.ty;
    if ty == TypeId::of::<Evidence<Categories<View>>>() {
        return Some(project::<View>(cx, key));
    }
    if ty == TypeId::of::<Evidence<Categories<Sentences>>>() {
        return Some(project::<Sentences>(cx, key));
    }
    if ty == TypeId::of::<Evidence<Categories<Structure>>>() {
        return Some(project::<Structure>(cx, key));
    }
    if ty == TypeId::of::<Evidence<Categories<Chunks>>>() {
        return Some(project::<Chunks>(cx, key));
    }
    None
}
