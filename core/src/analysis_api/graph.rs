//! Typed extraction from an imported or freshly analyzed `Document`.
//!
//! Extraction never slices a caller-supplied text. Coordinates come from
//! the document's own source and views, and every view range is projected
//! to source with `Document::map_span` before it is exposed over the ABI.
//! A caller's graph that holds what no row may carry (a probability
//! outside 0..=1, which graph import does not refuse) is the caller's
//! invalid argument; in the analyzer's own document it is the analyzer's
//! fault.

use std::any::TypeId;

use turbo_analysis::{Document, Ix, Key, KeyId, Layer, OffsetTable, Span, Spans, ViewId};
use turbo_analysis_layers::{
    Confidence, Dependencies, Entities, Lemmas, MorphFeatures, Pii, PiiConfidence, PosTags, Sentences, Tokens,
};

use super::{
    AnalysisResultInner, CoordRange, LayerRef, PiiRows, PoolRef, SpanDraft, SpanPoolBuilder,
    TURBO_ANALYSIS_LAYER_DEPENDENCIES, TURBO_ANALYSIS_LAYER_ENTITIES, TURBO_ANALYSIS_LAYER_LEMMAS,
    TURBO_ANALYSIS_LAYER_MORPHOLOGY, TURBO_ANALYSIS_LAYER_PII, TURBO_ANALYSIS_LAYER_POS,
    TURBO_ANALYSIS_LAYER_SENTENCES, TURBO_ANALYSIS_LAYER_TOKENS, TURBO_PII_CONFIDENCE_CHECKSUM,
    TURBO_PII_CONFIDENCE_FORMAT, TURBO_PII_CONFIDENCE_LABELLED, TURBO_PII_CONFIDENCE_LABELLED_CHECKSUM,
    coordinate_range,
};
use crate::status::{BUNDLE_INVALID, CAPACITY, Error, INVALID_ARGUMENT, Result};

fn index_u32(index: usize, what: &str) -> Result<u32> {
    u32::try_from(index).map_err(|_| Error::new(CAPACITY, format!("{what} exceeds u32")))
}

fn get_by_id<L: Layer>(doc: &Document, id: KeyId) -> Option<&L> {
    doc.get_key(&Key::<L>::named(id.name))
}

fn confidence<'d, L: Layer>(mapper: &Mapper<'d>, id: KeyId, expected: usize) -> Result<Option<&'d [f64]>> {
    let Some(confidence) = get_by_id::<Confidence<L>>(mapper.doc, id) else {
        return Ok(None);
    };
    if confidence.value.len() != expected {
        let n = confidence.value.len();
        return Err(mapper.invalid(format!("confidence for {id} has {n} values for {expected} items")));
    }
    Ok(Some(&confidence.value))
}

/// Whose document the rows are read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Origin {
    /// The analyzer's own, just analyzed.
    Analyzed,
    /// Imported from the caller's graph bytes.
    Caller,
}

struct Mapper<'a> {
    doc: &'a Document,
    origin: Origin,
    /// Whether a PII row carries its normalized value
    /// (`TURBO_ANALYZER_PII_VALUES`, #1904; otherwise the value is withheld)
    /// and its numeric kind (`TURBO_ANALYZER_PII_KIND_IDS`, #1940).
    pii: PiiRows,
}

impl Mapper<'_> {
    /// The refusal of what the document holds: `code` for the analyzer's
    /// own document, TURBO_E_INVALID_ARGUMENT for a caller's graph.
    fn refuse(&self, code: i32, message: String) -> Error {
        Error::new(if self.origin == Origin::Caller { INVALID_ARGUMENT } else { code }, message)
    }

    fn invalid(&self, message: String) -> Error {
        self.refuse(BUNDLE_INVALID, message)
    }

    fn coordinates(&self, view: ViewId, span: Span) -> Result<(CoordRange, CoordRange)> {
        let range = |text: &str, offsets: &OffsetTable, span: Span| {
            coordinate_range(text, offsets, (span.start, span.end)).map_err(|e| self.refuse(e.code, e.message))
        };
        let view_range = range(self.doc.text(view), self.doc.offsets(view), span)?;
        let source_span = self
            .doc
            .map_span(view, ViewId::SOURCE, span)
            .map_err(|e| self.invalid(format!("graph span cannot project to source: {e}")))?;
        let source_range = range(self.doc.source(), self.doc.offsets(ViewId::SOURCE), source_span)?;
        Ok((view_range, source_range))
    }
}

struct ItemFields {
    tokens: Option<(u32, u32)>,
    label: Option<PoolRef>,
    value: Option<PoolRef>,
    classification: u32,
    prob: Option<f64>,
    /// A PII row's numeric kind (`TURBO_SPAN_PII_KIND`, #1940).
    pii_kind: Option<u32>,
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
    if let Some(prob) = fields.prob
        && !(0.0..=1.0).contains(&prob)
    {
        let message = format!("graph item {item} of layer kind {} has probability {prob}, outside 0..=1", layer.kind);
        return Err(mapper.refuse(CAPACITY, message));
    }
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
                pii_kind: fields.pii_kind,
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
    let probs = confidence::<Sentences>(mapper, id, layer.spans.len())?;
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
                pii_kind: None,
            },
        )?;
    }
    Ok(())
}

fn extract_tokens(builder: &mut SpanPoolBuilder, mapper: &Mapper<'_>, doc: &Document, id: KeyId) -> Result<()> {
    let Some(layer) = get_by_id::<Tokens>(doc, id) else { return Ok(()) };
    let probs = confidence::<Tokens>(mapper, id, layer.spans.len())?;
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
                pii_kind: None,
            },
        )?;
    }
    Ok(())
}

fn extract_pos(builder: &mut SpanPoolBuilder, mapper: &Mapper<'_>, doc: &Document, id: KeyId) -> Result<()> {
    let Some(layer) = get_by_id::<PosTags>(doc, id) else { return Ok(()) };
    let Some(anchor) = anchor_tokens(doc, id) else {
        return Err(mapper.invalid(format!("aligned layer {id} has no token anchor")));
    };
    if layer.tags.len() != anchor.spans.len() {
        return Err(mapper.invalid(format!(
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
                pii_kind: None,
            },
        )?;
    }
    Ok(())
}

fn extract_lemmas(builder: &mut SpanPoolBuilder, mapper: &Mapper<'_>, doc: &Document, id: KeyId) -> Result<()> {
    let Some(layer) = get_by_id::<Lemmas>(doc, id) else { return Ok(()) };
    let Some(anchor) = anchor_tokens(doc, id) else {
        return Err(mapper.invalid(format!("aligned layer {id} has no token anchor")));
    };
    if layer.lemma.len() != anchor.spans.len() {
        return Err(mapper.invalid(format!(
            "lemma layer {id} has {} values for {} anchor tokens",
            layer.lemma.len(),
            anchor.spans.len()
        )));
    }
    let probs = confidence::<Lemmas>(mapper, id, layer.lemma.len())?;
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
                pii_kind: None,
            },
        )?;
    }
    Ok(())
}

/// A morphology row per token, with the feature string as the label.
fn extract_morphology(builder: &mut SpanPoolBuilder, mapper: &Mapper<'_>, doc: &Document, id: KeyId) -> Result<()> {
    let Some(layer) = get_by_id::<MorphFeatures<Tokens>>(doc, id) else { return Ok(()) };
    let Some(anchor) = anchor_tokens(doc, id) else {
        return Err(mapper.invalid(format!("aligned layer {id} has no token anchor")));
    };
    if layer.feats.len() != anchor.spans.len() {
        return Err(mapper.invalid(format!(
            "morphology layer {id} has {} values for {} anchor tokens",
            layer.feats.len(),
            anchor.spans.len()
        )));
    }
    let probs = confidence::<MorphFeatures<Tokens>>(mapper, id, layer.feats.len())?;
    let layer_ref = layer_ref(builder, TURBO_ANALYSIS_LAYER_MORPHOLOGY, id, anchor.spans.view())?;
    for item in 0..layer.feats.len() {
        let item_u32 = index_u32(item, "graph morphology count")?;
        let label = builder.string_ref(layer.feats.get(item))?;
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
                prob: probs.map(|values| values[item]),
                pii_kind: None,
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
        return Err(mapper.invalid(format!("aligned layer {id} has no token anchor")));
    };
    if layer.head.len() != anchor.spans.len() || layer.relation.len() != anchor.spans.len() {
        return Err(mapper.invalid(format!(
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
                pii_kind: None,
            },
        )?;
    }
    Ok(())
}

fn extract_entities(builder: &mut SpanPoolBuilder, mapper: &Mapper<'_>, doc: &Document, id: KeyId) -> Result<()> {
    let Some(layer) = get_by_id::<Entities>(doc, id) else { return Ok(()) };
    if layer.tokens.len() != layer.spans.len() || layer.kind.len() != layer.spans.len() {
        return Err(mapper.invalid(format!("entity layer {id} has inconsistent columns")));
    }
    if !layer.score.is_empty() && layer.score.len() != layer.spans.len() {
        return Err(mapper.invalid(format!("entity layer {id} has inconsistent scores")));
    }
    let anchor_len = anchor_tokens(doc, id).map(|tokens| tokens.spans.len());
    let confidence = confidence::<Entities>(mapper, id, layer.spans.len())?;
    let layer_ref = layer_ref(builder, TURBO_ANALYSIS_LAYER_ENTITIES, id, layer.spans.view())?;
    for item in 0..layer.spans.len() {
        let range = layer.tokens[item];
        let (start, end) = (range.start.get(), range.end.get());
        let tokens = match anchor_len {
            Some(count) => {
                if start > end || end > count {
                    return Err(mapper.invalid(format!("entity layer {id} token range is outside its anchor")));
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
            ItemFields { tokens, label: Some(label), value: None, classification: 0, prob, pii_kind: None },
        )?;
    }
    Ok(())
}

fn extract_pii(builder: &mut SpanPoolBuilder, mapper: &Mapper<'_>, doc: &Document, id: KeyId) -> Result<()> {
    let Some(layer) = get_by_id::<Pii>(doc, id) else { return Ok(()) };
    let layer_ref = layer_ref(builder, TURBO_ANALYSIS_LAYER_PII, id, layer.spans.view())?;
    for item in 0..layer.spans.len() {
        let mention = layer.get(Ix::new(index_u32(item, "graph PII count")?));
        let name = layer.kind_name(mention.kind).ok_or_else(|| {
            Error::new(INVALID_ARGUMENT, format!("PII mention {item} has a custom kind outside its layer's table"))
        })?;
        let label = builder.string_ref(name)?;
        let value = if mapper.pii.values { Some(builder.string_ref(mention.normalized)?) } else { None };
        let classification = match mention.confidence() {
            PiiConfidence::Format => TURBO_PII_CONFIDENCE_FORMAT,
            PiiConfidence::Checksum => TURBO_PII_CONFIDENCE_CHECKSUM,
            PiiConfidence::LabelledChecksum => TURBO_PII_CONFIDENCE_LABELLED_CHECKSUM,
            PiiConfidence::Labelled => TURBO_PII_CONFIDENCE_LABELLED,
        };
        push_fragments(
            builder,
            mapper,
            layer_ref,
            &layer.spans,
            item,
            ItemFields {
                tokens: None,
                label: Some(label),
                value,
                classification,
                prob: None,
                pii_kind: mapper.pii.kind_ids.then(|| mention.kind.code()),
            },
        )?;
    }
    Ok(())
}

/// The rows of every layer of `doc` this ABI knows; a PII row's value and
/// numeric kind only as `pii` asks (#1904, #1940).
pub(crate) fn extract_graph(
    doc: &Document,
    graph: Vec<u8>,
    origin: Origin,
    pii: PiiRows,
) -> Result<AnalysisResultInner> {
    let mapper = Mapper { doc, origin, pii };
    let mut builder = SpanPoolBuilder::default();
    for id in doc.layers() {
        if id.ty == TypeId::of::<Sentences>() {
            extract_sentences(&mut builder, &mapper, doc, id)?;
        } else if id.ty == TypeId::of::<Tokens>() {
            extract_tokens(&mut builder, &mapper, doc, id)?;
        } else if id.ty == TypeId::of::<PosTags>() {
            extract_pos(&mut builder, &mapper, doc, id)?;
        } else if id.ty == TypeId::of::<MorphFeatures<Tokens>>() {
            extract_morphology(&mut builder, &mapper, doc, id)?;
        } else if id.ty == TypeId::of::<Lemmas>() {
            extract_lemmas(&mut builder, &mapper, doc, id)?;
        } else if id.ty == TypeId::of::<Dependencies<Tokens>>() {
            extract_dependencies(&mut builder, &mapper, doc, id)?;
        } else if id.ty == TypeId::of::<Entities>() {
            extract_entities(&mut builder, &mapper, doc, id)?;
        } else if id.ty == TypeId::of::<Pii>() {
            extract_pii(&mut builder, &mapper, doc, id)?;
        } else if id.ty == TypeId::of::<turbo_analysis_layers::Words>() {
            words::extract_words(&mut builder, &mapper, doc, id)?;
        }
    }
    Ok(builder.finish(graph))
}

#[cfg(test)]
mod tests;
pub(crate) mod words;
