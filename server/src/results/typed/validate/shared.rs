//! Nested payload checks use the same offset and probability rules as
//! outer annotations; their references retain the selected layer instance.

use super::*;
use crate::proto::result::{DocumentEntityAnnotation, NBestAnnotation, SparseTermsAnnotation};

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
