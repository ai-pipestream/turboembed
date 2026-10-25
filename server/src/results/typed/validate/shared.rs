//! Nested payload checks use the same offset and probability rules as
//! outer annotations; their references retain the selected layer instance.

use super::*;
use crate::proto::result::{
    ChunkTableHeaderAnnotation, DocumentEntityAnnotation, EntitySalienceAnnotation, EventAnnotation,
    KeyValuePairAnnotation, NBestAnnotation, SparseTermsAnnotation,
};

pub(super) fn bound(ctx: &LayerCtx<'_>, layer: &str) -> Result<usize, Failure> {
    let mut lengths = ctx.target_lengths.iter().filter(|((id, _), _)| *id == layer).map(|(_, len)| *len);
    let len = lengths.next().ok_or_else(|| bad(format!("{} has no {layer} index target", ctx.lid)))?;
    if lengths.next().is_some() {
        return Err(bad(format!("{} has ambiguous {layer} index targets", ctx.lid)));
    }
    Ok(len)
}

pub(super) fn target(ctx: &LayerCtx<'_>, layer: &str, index: u32) -> Result<usize, Failure> {
    let len = bound(ctx, layer)?;
    if index as usize >= len {
        return Err(bad(format!("{} index {index} exceeds {layer} target {len}", ctx.lid)));
    }
    Ok(len)
}

pub(super) fn key_value(ctx: &LayerCtx<'_>, i: usize, pair: &KeyValuePairAnnotation) -> Result<(), Failure> {
    for (field, span) in [("key", &pair.key_span), ("separator", &pair.separator_span), ("value", &pair.value_span)] {
        let span = span.as_ref().ok_or_else(|| bad(format!("{}[{i}] key/value {field} span is missing", ctx.lid)))?;
        ctx.offsetter.from_wire(ctx.text, ctx.lid, span.start, span.end)?;
    }
    Ok(())
}

pub(super) fn event(ctx: &LayerCtx<'_>, i: usize, event: &EventAnnotation) -> Result<(), Failure> {
    check_f32_prob("confidence", ctx.lid, i, event.confidence)?;
    target(ctx, "opennlp:sentences", event.sentence_index)?;
    for argument in &event.arguments {
        let span = argument.span.as_ref().ok_or_else(|| bad("event argument has no span"))?;
        // Native arguments are nonempty, but text deletion can project them
        // to an empty span. Presence, boundaries and anchors still apply.
        ctx.offsetter.from_wire(ctx.text, ctx.lid, span.start, span.end)?;
        match (argument.anchored_layer.as_str(), argument.anchored_index) {
            ("none", None) => {}
            (layer @ ("turbo:temporal" | "turbo:money" | "turbo:quantities" | "opennlp:entities"), Some(index)) => {
                target(ctx, layer, index)?;
            }
            _ => return Err(bad("event argument anchor layer/index is missing or inconsistent")),
        }
    }
    Ok(())
}

pub(super) fn salience(ctx: &LayerCtx<'_>, i: usize, score: &EntitySalienceAnnotation) -> Result<(), Failure> {
    let len = target(ctx, "turbo:document-entities", score.entity_index)?;
    check_aligned_index("entity_index", ctx.lid, i, score.entity_index, len)?;
    if ctx.layer_len != len || score.rank as usize >= len {
        return Err(bad(format!("{}[{i}] salience rank or entity count differs from target", ctx.lid)));
    }
    for (name, p) in [
        ("salience", score.salience),
        ("position", score.position_score),
        ("syntax", score.syntax_score),
        ("frequency", score.frequency_score),
        ("centrality", score.centrality_score),
    ] {
        check_f32_prob(name, ctx.lid, i, p)?;
    }
    Ok(())
}

pub(super) fn salience_order(ctx: &LayerCtx<'_>, layer: &TypedLayer) -> Result<(), Failure> {
    if layer.id != "turbo:entity-salience" {
        return Ok(());
    }
    if bound(ctx, "turbo:document-entities")? != layer.annotations.len() {
        return Err(bad("salience needs one row for every selected consolidated entity"));
    }
    let mut scores = vec![None; layer.annotations.len()];
    for annotation in &layer.annotations {
        let Some(Value::EntitySalience(score)) = &annotation.value else {
            return Err(bad("salience layer has a different payload"));
        };
        let slot = scores.get_mut(score.rank as usize).ok_or_else(|| bad("salience rank exceeds layer length"))?;
        if slot.replace(score.salience).is_some() {
            return Err(bad("duplicate salience rank"));
        }
    }
    if scores.windows(2).any(|w| w[0] < w[1]) {
        return Err(bad("salience ranks are not best-first"));
    }
    Ok(())
}

pub(super) fn table_header(ctx: &LayerCtx<'_>, i: usize, header: &ChunkTableHeaderAnnotation) -> Result<(), Failure> {
    target(ctx, "turbo:chunks", header.chunk_index)?;
    if !ctx.table_ids.contains(&header.table_id) {
        return Err(bad(format!("{}[{i}] table_id has no selected table target", ctx.lid)));
    }
    for span in &header.header_cells {
        ctx.offsetter.from_wire(ctx.text, ctx.lid, span.start, span.end)?;
    }
    Ok(())
}

pub(super) fn nbest(ctx: &LayerCtx<'_>, i: usize, nbest: &NBestAnnotation) -> Result<(), Failure> {
    check_aligned_index("target_index", ctx.lid, i, nbest.target_index, ctx.layer_len)?;
    for (rank, candidate) in nbest.candidates.iter().enumerate() {
        if candidate.rank as usize != rank || candidate.sequence_score.is_nan() {
            return Err(bad(format!("{}[{i}] invalid NBest rank or NaN sequence score", ctx.lid)));
        }
        if let Some(p) = candidate.probability {
            check_prob("NBest candidate probability", ctx.lid, i, p)?;
        }
    }
    if nbest.candidates.windows(2).any(|w| w[0].sequence_score < w[1].sequence_score) {
        return Err(bad(format!("{}[{i}] NBest candidates are not best-first", ctx.lid)));
    }
    Ok(())
}

pub(super) fn sparse(ctx: &LayerCtx<'_>, i: usize, sparse: &SparseTermsAnnotation) -> Result<(), Failure> {
    check_aligned_index("target_index", ctx.lid, i, sparse.target_index, ctx.layer_len)?;
    for entry in &sparse.entries {
        if !entry.weight.is_finite() || entry.weight <= 0.0 {
            return Err(bad(format!("{}[{i}] sparse weight must be positive and finite", ctx.lid)));
        }
        if let Some(span) = &entry.span {
            ctx.offsetter.from_wire(ctx.text, ctx.lid, span.start, span.end)?;
        }
        for occurrence in &entry.occurrences {
            ctx.offsetter.from_wire(ctx.text, ctx.lid, occurrence.start, occurrence.end)?;
        }
    }
    Ok(())
}

pub(super) fn document_entity(ctx: &LayerCtx<'_>, i: usize, entity: &DocumentEntityAnnotation) -> Result<(), Failure> {
    check_f32_prob("document entity confidence", ctx.lid, i, entity.confidence)?;
    if entity.mentions.is_empty() {
        return Err(bad(format!("{}[{i}] document entity has no mention", ctx.lid)));
    }
    if !entity.mean_sentiment.is_finite() {
        return Err(bad(format!("{}[{i}] document entity mean_sentiment is non-finite", ctx.lid)));
    }
    for mention in &entity.mentions {
        let span = mention.span.as_ref().ok_or_else(|| bad(format!("{}[{i}] mention has no span", ctx.lid)))?;
        ctx.offsetter.from_wire(ctx.text, ctx.lid, span.start, span.end)?;
        if let Some(index) = mention.entity_index {
            let len = ctx.entity_len.ok_or_else(|| bad(format!("{}[{i}] mention has no entity target", ctx.lid)))?;
            if index as usize >= len {
                return Err(bad(format!("{}[{i}] mention entity_index {index} exceeds target {len}", ctx.lid)));
            }
        }
    }
    for relation in &entity.relations {
        if relation.subject as usize != i || relation.object as usize >= ctx.layer_len {
            return Err(bad(format!("{}[{i}] relation subject/object is outside its entity target", ctx.lid)));
        }
        if !relation.score.is_finite() {
            return Err(bad(format!("{}[{i}] relation score is non-finite", ctx.lid)));
        }
    }
    Ok(())
}
