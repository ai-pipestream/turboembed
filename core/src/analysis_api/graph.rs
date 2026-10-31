//! Typed extraction from an imported or freshly analyzed `Document`.
//!
//! Extraction never slices a caller-supplied text. Coordinates come from
//! the document's own source and views, and every view range is projected
//! to source with `Document::map_span` before it is exposed over the ABI.

use std::any::TypeId;

use turbo_analysis::{Document, Ix, Key, KeyId, Layer, Span, Spans, ViewId};
use turbo_analysis_layers::{
    Confidence, Dependencies, Entities, Lemmas, Pii, PiiConfidence, PosTags, Sentences, Tokens,
};

use super::{
    AnalysisResultInner, CoordRange, LayerRef, PoolRef, SpanDraft, SpanPoolBuilder, TURBO_ANALYSIS_LAYER_DEPENDENCIES,
    TURBO_ANALYSIS_LAYER_ENTITIES, TURBO_ANALYSIS_LAYER_LEMMAS, TURBO_ANALYSIS_LAYER_PII, TURBO_ANALYSIS_LAYER_POS,
    TURBO_ANALYSIS_LAYER_SENTENCES, TURBO_ANALYSIS_LAYER_TOKENS, TURBO_PII_CONFIDENCE_CHECKSUM,
    TURBO_PII_CONFIDENCE_FORMAT, TURBO_PII_CONFIDENCE_LABELLED, coordinate_range,
};
use crate::status::{BUNDLE_INVALID, CAPACITY, Error, Result};

fn invalid(message: impl Into<String>) -> Error {
    Error::new(BUNDLE_INVALID, message.into())
}

fn index_u32(index: usize, what: &str) -> Result<u32> {
    u32::try_from(index).map_err(|_| Error::new(CAPACITY, format!("{what} exceeds u32")))
}

fn get_by_id<L: Layer>(doc: &Document, id: KeyId) -> Option<&L> {
    doc.get_key(&Key::<L>::named(id.name))
}

fn confidence<L: Layer>(doc: &Document, id: KeyId, expected: usize) -> Result<Option<&[f64]>> {
    let Some(confidence) = get_by_id::<Confidence<L>>(doc, id) else {
        return Ok(None);
    };
    if confidence.value.len() != expected {
        return Err(invalid(format!("confidence for {id} has {} values for {expected} items", confidence.value.len())));
    }
    Ok(Some(&confidence.value))
}

struct Mapper<'a> {
    doc: &'a Document,
}

impl Mapper<'_> {
    fn coordinates(&self, view: ViewId, span: Span) -> Result<(CoordRange, CoordRange)> {
        let view_range = coordinate_range(self.doc.text(view), self.doc.offsets(view), (span.start, span.end))?;
        let source_span = self
            .doc
            .map_span(view, ViewId::SOURCE, span)
            .map_err(|e| invalid(format!("graph span cannot project to source: {e}")))?;
        let source_range = coordinate_range(
            self.doc.source(),
            self.doc.offsets(ViewId::SOURCE),
            (source_span.start, source_span.end),
        )?;
        Ok((view_range, source_range))
    }
}

struct ItemFields {
    tokens: Option<(u32, u32)>,
    label: Option<PoolRef>,
    value: Option<PoolRef>,
    classification: u32,
    prob: Option<f64>,
}

fn push_fragments<L: Layer>(
    builder: &mut SpanPoolBuilder,
    mapper: &Mapper<'_>,
    layer: LayerRef,
    spans: &Spans<L>,
    item: usize,
    fields: ItemFields,
) -> Result<()> {
    let item_index = index_u32(item, "graph item count")?;
    for (fragment, span) in spans.fragments(Ix::new(item_index)).enumerate() {
        let (view, source) = mapper.coordinates(spans.view(), span)?;
        builder.push(
            layer,
            SpanDraft {
                item_index,
                fragment_index: index_u32(fragment, "graph fragment count")?,
                classification: fields.classification,
                view: Some(view),
                source: Some(source),
                tokens: fields.tokens,
                label: fields.label,
                value: fields.value,
                prob: fields.prob,
            },
        )?;
    }
    Ok(())
}

fn anchor_tokens(doc: &Document, id: KeyId) -> Option<&Tokens> {
    for anchor in doc.made_on(id) {
        if anchor.ty == TypeId::of::<Tokens>()
            && let Some(tokens) = get_by_id::<Tokens>(doc, *anchor)
        {
            return Some(tokens);
        }
    }
    doc.get::<Tokens>()
}

fn layer_ref(builder: &mut SpanPoolBuilder, kind: u32, id: KeyId, view: ViewId) -> Result<LayerRef> {
    builder.layer_ref(kind, &id.layer.to_string(), id.name, view.0 as u32)
}

fn extract_sentences(builder: &mut SpanPoolBuilder, mapper: &Mapper<'_>, doc: &Document, id: KeyId) -> Result<()> {
    let Some(layer) = get_by_id::<Sentences>(doc, id) else { return Ok(()) };
    let probs = confidence::<Sentences>(doc, id, layer.spans.len())?;
    let layer_ref = layer_ref(builder, TURBO_ANALYSIS_LAYER_SENTENCES, id, layer.spans.view())?;
    for item in 0..layer.spans.len() {
        push_fragments(
            builder,
            mapper,
            layer_ref,
            &layer.spans,
            item,
            ItemFields {
                tokens: None,
                label: None,
                value: None,
                classification: 0,
                prob: probs.map(|values| values[item]),
            },
        )?;
    }
    Ok(())
}

fn extract_tokens(builder: &mut SpanPoolBuilder, mapper: &Mapper<'_>, doc: &Document, id: KeyId) -> Result<()> {
    let Some(layer) = get_by_id::<Tokens>(doc, id) else { return Ok(()) };
    let probs = confidence::<Tokens>(doc, id, layer.spans.len())?;
    let layer_ref = layer_ref(builder, TURBO_ANALYSIS_LAYER_TOKENS, id, layer.spans.view())?;
    for item in 0..layer.spans.len() {
        let item_u32 = index_u32(item, "graph token count")?;
        push_fragments(
            builder,
            mapper,
            layer_ref,
            &layer.spans,
            item,
            ItemFields {
                tokens: Some((item_u32, item_u32 + 1)),
                label: None,
                value: None,
                classification: 0,
                prob: probs.map(|values| values[item]),
            },
        )?;
    }
    Ok(())
}

fn extract_pos(builder: &mut SpanPoolBuilder, mapper: &Mapper<'_>, doc: &Document, id: KeyId) -> Result<()> {
    let Some(layer) = get_by_id::<PosTags>(doc, id) else { return Ok(()) };
    let Some(anchor) = anchor_tokens(doc, id) else {
        return Err(invalid(format!("aligned layer {id} has no token anchor")));
    };
    if layer.tags.len() != anchor.spans.len() {
        return Err(invalid(format!(
            "POS layer {id} has {} tags for {} anchor tokens",
            layer.tags.len(),
            anchor.spans.len()
        )));
    }
    let layer_ref = layer_ref(builder, TURBO_ANALYSIS_LAYER_POS, id, anchor.spans.view())?;
    for item in 0..layer.tags.len() {
        let item_u32 = index_u32(item, "graph POS count")?;
        let label = builder.string_ref(layer.tagset.label(layer.tags[item]))?;
        push_fragments(
            builder,
            mapper,
            layer_ref,
            &anchor.spans,
            item,
            ItemFields {
                tokens: Some((item_u32, item_u32 + 1)),
                label: Some(label),
                value: None,
                classification: 0,
                prob: layer.prob.as_ref().map(|values| values[item]),
            },
        )?;
    }
    Ok(())
}

fn extract_lemmas(builder: &mut SpanPoolBuilder, mapper: &Mapper<'_>, doc: &Document, id: KeyId) -> Result<()> {
    let Some(layer) = get_by_id::<Lemmas>(doc, id) else { return Ok(()) };
    let Some(anchor) = anchor_tokens(doc, id) else {
        return Err(invalid(format!("aligned layer {id} has no token anchor")));
    };
    if layer.lemma.len() != anchor.spans.len() {
        return Err(invalid(format!(
            "lemma layer {id} has {} values for {} anchor tokens",
            layer.lemma.len(),
            anchor.spans.len()
        )));
    }
    let probs = confidence::<Lemmas>(doc, id, layer.lemma.len())?;
    let layer_ref = layer_ref(builder, TURBO_ANALYSIS_LAYER_LEMMAS, id, anchor.spans.view())?;
    for item in 0..layer.lemma.len() {
        let item_u32 = index_u32(item, "graph lemma count")?;
        let raw = layer.lemma.get(item);
        let label = if raw == "_" { None } else { Some(builder.string_ref(raw)?) };
        push_fragments(
            builder,
            mapper,
            layer_ref,
            &anchor.spans,
            item,
            ItemFields {
                tokens: Some((item_u32, item_u32 + 1)),
                label,
                value: None,
                classification: 0,
                prob: probs.map(|values| values[item]),
            },
        )?;
    }
    Ok(())
}

/// A dependency row per token: its relation as the label and, as the
/// classification, its head token's index plus one, 0 under the root.
fn extract_dependencies(builder: &mut SpanPoolBuilder, mapper: &Mapper<'_>, doc: &Document, id: KeyId) -> Result<()> {
    let Some(layer) = get_by_id::<Dependencies<Tokens>>(doc, id) else { return Ok(()) };
    let Some(anchor) = anchor_tokens(doc, id) else {
        return Err(invalid(format!("aligned layer {id} has no token anchor")));
    };
    if layer.head.len() != anchor.spans.len() || layer.relation.len() != anchor.spans.len() {
        return Err(invalid(format!(
            "dependency layer {id} has {} heads for {} anchor tokens",
            layer.head.len(),
            anchor.spans.len()
        )));
    }
    let layer_ref = layer_ref(builder, TURBO_ANALYSIS_LAYER_DEPENDENCIES, id, anchor.spans.view())?;
    for item in 0..layer.head.len() {
        let item_u32 = index_u32(item, "graph dependency count")?;
        let label = builder.string_ref(layer.relations.label(layer.relation[item]))?;
        let classification = match layer.head[item] {
            Some(head) => index_u32(head.get(), "graph dependency head")?
                .checked_add(1)
                .ok_or_else(|| Error::new(CAPACITY, "graph dependency head exceeds u32"))?,
            None => 0,
        };
        push_fragments(
            builder,
            mapper,
            layer_ref,
            &anchor.spans,
            item,
            ItemFields {
                tokens: Some((item_u32, item_u32 + 1)),
                label: Some(label),
                value: None,
                classification,
                prob: None,
            },
        )?;
    }
    Ok(())
}

fn extract_entities(builder: &mut SpanPoolBuilder, mapper: &Mapper<'_>, doc: &Document, id: KeyId) -> Result<()> {
    let Some(layer) = get_by_id::<Entities>(doc, id) else { return Ok(()) };
    if layer.tokens.len() != layer.spans.len() || layer.kind.len() != layer.spans.len() {
        return Err(invalid(format!("entity layer {id} has inconsistent columns")));
    }
    if !layer.score.is_empty() && layer.score.len() != layer.spans.len() {
        return Err(invalid(format!("entity layer {id} has inconsistent scores")));
    }
    let anchor_len = anchor_tokens(doc, id).map(|tokens| tokens.spans.len());
    let confidence = confidence::<Entities>(doc, id, layer.spans.len())?;
    let layer_ref = layer_ref(builder, TURBO_ANALYSIS_LAYER_ENTITIES, id, layer.spans.view())?;
    for item in 0..layer.spans.len() {
        let range = layer.tokens[item];
        let (start, end) = (range.start.get(), range.end.get());
        let tokens = match anchor_len {
            Some(count) => {
                if start > end || end > count {
                    return Err(invalid(format!("entity layer {id} token range is outside its anchor")));
                }
                Some((index_u32(start, "graph entity token range")?, index_u32(end, "graph entity token range")?))
            }
            None => None,
        };
        let label = builder.string_ref(layer.kinds.label(layer.kind[item]))?;
        let prob = if layer.score.is_empty() { confidence.map(|values| values[item]) } else { Some(layer.score[item]) };
        push_fragments(
            builder,
            mapper,
            layer_ref,
            &layer.spans,
            item,
            ItemFields { tokens, label: Some(label), value: None, classification: 0, prob },
        )?;
    }
    Ok(())
}

fn extract_pii(builder: &mut SpanPoolBuilder, mapper: &Mapper<'_>, doc: &Document, id: KeyId) -> Result<()> {
    let Some(layer) = get_by_id::<Pii>(doc, id) else { return Ok(()) };
    let layer_ref = layer_ref(builder, TURBO_ANALYSIS_LAYER_PII, id, layer.spans.view())?;
    for item in 0..layer.spans.len() {
        let mention = layer.get(Ix::new(index_u32(item, "graph PII count")?));
        let label = builder.string_ref(mention.kind.name())?;
        let value = builder.string_ref(mention.normalized)?;
        let classification = match mention.confidence {
            PiiConfidence::Format => TURBO_PII_CONFIDENCE_FORMAT,
            PiiConfidence::Checksum => TURBO_PII_CONFIDENCE_CHECKSUM,
            PiiConfidence::LabelledChecksum => TURBO_PII_CONFIDENCE_LABELLED,
        };
        push_fragments(
            builder,
            mapper,
            layer_ref,
            &layer.spans,
            item,
            ItemFields { tokens: None, label: Some(label), value: Some(value), classification, prob: None },
        )?;
    }
    Ok(())
}

pub(crate) fn extract_graph(doc: &Document, graph: Vec<u8>) -> Result<AnalysisResultInner> {
    let mapper = Mapper { doc };
    let mut builder = SpanPoolBuilder::default();
    for id in doc.layers() {
        if id.ty == TypeId::of::<Sentences>() {
            extract_sentences(&mut builder, &mapper, doc, id)?;
        } else if id.ty == TypeId::of::<Tokens>() {
            extract_tokens(&mut builder, &mapper, doc, id)?;
        } else if id.ty == TypeId::of::<PosTags>() {
            extract_pos(&mut builder, &mapper, doc, id)?;
        } else if id.ty == TypeId::of::<Lemmas>() {
            extract_lemmas(&mut builder, &mapper, doc, id)?;
        } else if id.ty == TypeId::of::<Dependencies<Tokens>>() {
            extract_dependencies(&mut builder, &mapper, doc, id)?;
        } else if id.ty == TypeId::of::<Entities>() {
            extract_entities(&mut builder, &mapper, doc, id)?;
        } else if id.ty == TypeId::of::<Pii>() {
            extract_pii(&mut builder, &mapper, doc, id)?;
        }
    }
    Ok(builder.finish(graph))
}

#[cfg(test)]
mod tests;
