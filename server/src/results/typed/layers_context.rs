//! Shared entity salience and source-preserving chunk/table header context.

use std::any::TypeId;

use turbo_analysis::{Document, Ix, Key, KeyId, Layer};
use turbo_analysis_layers::{
    ChunkTableHeaders, Chunks, DocumentEntities, EntitySalience, LayoutBoxes, PageBoundingBox,
};
use turbo_chunker::ChunkCitations;

use super::layers_structure::aligned_spans;
use super::{ProjectionCx, build_layer, coalesce_wire_spans};
use crate::proto::result::typed_annotation::Value;
use crate::proto::result::{
    ChunkCitationAnnotation, ChunkTableHeaderAnnotation, EntitySalienceAnnotation, LayoutBoxAnnotation,
    PageBoundingBoxProto, TypedAnnotation, TypedLayer,
};

fn to_proto_boxes(boxes: &[PageBoundingBox]) -> Vec<PageBoundingBoxProto> {
    boxes.iter().map(|b| PageBoundingBoxProto { page: b.page, x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1 }).collect()
}

pub(super) fn chunk_citation_fields(
    doc: &Document,
    chunks: &Chunks,
    i: usize,
) -> (Option<u32>, Option<u32>, Vec<PageBoundingBoxProto>) {
    let ix = Ix::new(i as u32);
    if let Some(citations) = doc.get::<ChunkCitations>()
        && citations.len() == chunks.len()
    {
        let span = citations.page_span(ix);
        return (span.map(|(a, _)| a), span.map(|(_, b)| b), to_proto_boxes(citations.page_boxes(ix)));
    }
    let Some(layout) = doc.get::<LayoutBoxes>() else {
        return (None, None, Vec::new());
    };
    let chunk_span = chunks.spans.get(ix);
    let mapped = if chunks.spans.view() == layout.spans.view() {
        Some(chunk_span)
    } else {
        doc.map_span(chunks.spans.view(), layout.spans.view(), chunk_span).ok()
    };
    let Some(span) = mapped else {
        return (None, None, Vec::new());
    };
    let union = layout.union_boxes_by_page(span);
    let page_span = union.first().zip(union.last()).map(|(f, l)| (f.page, l.page));
    (page_span.map(|(a, _)| a), page_span.map(|(_, b)| b), to_proto_boxes(&union))
}

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
    if key.ty == TypeId::of::<LayoutBoxes>() {
        return Some(build_layer::<LayoutBoxes>(cx, key, |cx, layer, model| {
            let model = model.map(ToOwned::to_owned);
            (0..layer.len())
                .map(|i| TypedAnnotation {
                    spans: cx.project_spans(&layer.spans, Ix::new(i as u32)),
                    probability: None,
                    model_ref: model.clone(),
                    value: Some(Value::LayoutBox(LayoutBoxAnnotation {
                        page: layer.page[i],
                        x0: layer.x0[i],
                        y0: layer.y0[i],
                        x1: layer.x1[i],
                        y1: layer.y1[i],
                    })),
                })
                .collect()
        }));
    }
    if key.ty == TypeId::of::<ChunkCitations>() {
        return Some(build_layer::<ChunkCitations>(cx, key, |cx, layer, model| {
            let model = model.map(ToOwned::to_owned);
            (0..layer.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let page_span = layer.page_span(ix);
                    TypedAnnotation {
                        spans: aligned_spans::<Chunks>(cx, key, i, |c| &c.spans),
                        probability: None,
                        model_ref: model.clone(),
                        value: Some(Value::ChunkCitation(ChunkCitationAnnotation {
                            chunk_index: i as u32,
                            page_start: page_span.map(|(a, _)| a),
                            page_end: page_span.map(|(_, b)| b),
                            page_boxes: to_proto_boxes(layer.page_boxes(ix)),
                        })),
                    }
                })
                .collect()
        }));
    }
    None
}
