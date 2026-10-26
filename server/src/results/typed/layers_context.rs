//! Shared entity salience and source-preserving chunk/table header context.

use std::any::TypeId;

use turbo_analysis::{Ix, Key, KeyId};
use turbo_analysis_layers::{ChunkTableHeaders, DocumentEntities, EntitySalience};

use super::{ProjectionCx, build_layer, coalesce_wire_spans};
use crate::proto::result::typed_annotation::Value;
use crate::proto::result::{ChunkTableHeaderAnnotation, EntitySalienceAnnotation, TypedAnnotation, TypedLayer};

pub(crate) fn project_context_layer(cx: &mut ProjectionCx<'_>, key: KeyId) -> Option<TypedLayer> {
    if key.ty == TypeId::of::<EntitySalience>() {
        let entity_key = cx
            .doc
            .made_on(key)
            .iter()
            .find(|k| k.ty == TypeId::of::<DocumentEntities>())
            .copied()
            .unwrap_or(Key::<DocumentEntities>::DEFAULT.id());
        return Some(build_layer::<EntitySalience>(cx, key, |cx, layer, model| {
            let entities = cx.doc.get_key(&Key::<DocumentEntities>::named(entity_key.name));
            layer
                .scores
                .iter()
                .map(|score| TypedAnnotation {
                    spans: entities
                        .and_then(|e| {
                            e.entities.get(score.entity_ix as usize).map(|entity| {
                                coalesce_wire_spans(
                                    entity.mentions.iter().map(|m| cx.project_wire_span(e.view, m.span)).collect(),
                                )
                            })
                        })
                        .unwrap_or_default(),
                    probability: Some(f64::from(score.salience)),
                    model_ref: model.map(ToOwned::to_owned),
                    value: Some(Value::EntitySalience(EntitySalienceAnnotation {
                        entity_index: score.entity_ix,
                        salience: score.salience,
                        rank: score.rank,
                        position_score: score.position_score,
                        syntax_score: score.syntax_score,
                        frequency_score: score.frequency_score,
                        centrality_score: score.centrality_score,
                    })),
                })
                .collect()
        }));
    }
    if key.ty == TypeId::of::<ChunkTableHeaders>() {
        return Some(build_layer::<ChunkTableHeaders>(cx, key, |cx, layer, model| {
            layer
                .contexts
                .iter()
                .enumerate()
                .map(|(i, context)| TypedAnnotation {
                    spans: cx.project_spans(&layer.spans, Ix::new(i as u32)),
                    probability: None,
                    model_ref: model.map(ToOwned::to_owned),
                    value: Some(Value::ChunkTableHeader(ChunkTableHeaderAnnotation {
                        chunk_index: context.chunk_ix,
                        table_id: context.table_id,
                        is_continuation: context.is_continuation,
                        header_prefix: context.header_prefix.to_string(),
                        header_cells: context
                            .header_cells
                            .iter()
                            .map(|&s| cx.project_wire_span(layer.spans.view(), s))
                            .collect(),
                    })),
                })
                .collect()
        }));
    }
    None
}
