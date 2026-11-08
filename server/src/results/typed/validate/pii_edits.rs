//! `turbo:pii-edits` on the wire: each edit's source span, covered mention
//! rows, rule and strategy. Rows are ascending, non-empty and inside the
//! covered pii layer when that layer's length is known; the rule names the
//! default or a kind the PII validator would take; the strategy is one the
//! native layer knows. Across the layer, every value is an edit, the
//! source spans are in order and disjoint, the policy revision is one, and
//! the members are inside the covered pii layer, named by `indexes_layer`
//! or else the pii layer of the same instance. An error names the layer
//! and the row, never a kind name or a refused number.

use turbo_analysis::{Layer, ViewId};
use turbo_analysis_layers::{Pii, PiiEdits, PiiKind, PiiStrategyKind};

use super::*;
use crate::proto::result::pii_edit_rule::Rule;
use crate::proto::result::pii_kind_ref::Kind;
use crate::proto::result::{
    OffsetMode, PiiEditAnnotation, PiiKindRef, TypedDocumentResult, TypedLayer, mapping_payload,
};

/// The checks across a `turbo:pii-edits` layer; any other layer passes.
pub(super) fn layer(
    msg: &TypedDocumentResult,
    text: &str,
    offsetter: &Offsetter,
    layer: &TypedLayer,
    layer_lengths: &HashMap<(&str, &str), usize>,
) -> Result<(), Failure> {
    let lid = layer.id.as_str();
    if lid != PiiEdits::ID {
        return Ok(());
    }
    let named = layer.indexes_layer.iter().find(|b| b.layer == Pii::ID);
    let covered = match named {
        Some(b) => layer_lengths.get(&(b.layer.as_str(), b.instance.as_str())),
        None => layer_lengths.get(&(Pii::ID, layer.instance.as_str())).or_else(|| layer_lengths.get(&(Pii::ID, ""))),
    }
    .copied();
    let mut revision = None;
    let mut last_end = 0u32;
    for (i, ann) in layer.annotations.iter().enumerate() {
        let Some(Value::PiiEdit(e)) = &ann.value else {
            return Err(bad(format!("{lid}[{i}] is not a pii edit")));
        };
        if *revision.get_or_insert(e.policy_revision) != e.policy_revision {
            return Err(bad(format!("{lid}[{i}] pii edit policy_revision differs from the layer's")));
        }
        if let Some(s) = &e.source {
            if s.start < last_end {
                return Err(bad(format!("{lid}[{i}] pii edit source span overlaps or precedes the edit before it")));
            }
            last_end = s.end;
        }
        let Some(&last) = e.members.last() else { continue };
        let n = covered.ok_or_else(|| bad(format!("{lid}[{i}] pii edit members have no covered {} layer", Pii::ID)))?;
        if usize::try_from(last).map_or(true, |m| m >= n) {
            return Err(bad(format!("{lid}[{i}] pii edit member is outside the covered {} layer", Pii::ID)));
        }
    }
    // A text withheld for PII (#1904) stands in as one byte per unit, so
    // the chain's byte spans can be checked against it only when the
    // units are bytes.
    if msg.withheld_text_len.is_some() && msg.offsets != OffsetMode::Utf8 as i32 {
        return Ok(());
    }
    chain(msg, text, offsetter, layer)
}

/// Each edit's span in the projection and its source span, both in UTF-8
/// bytes of the text they are on.
struct Pair {
    span: Span,
    source: Span,
}

/// The checks against the chain the message shows. The edits' view is
/// the layer's `native_view` (#1907), a derived view of the mapping,
/// whatever other views the message has.
///
/// The edits are the alignment of their view one for one (the native
/// layer refuses any other), so where the message is projected onto that
/// view the unchanged stretches between edits must be as long in the
/// parent as in the view; where it is projected onto the source and the
/// source is the parent, every edit's span is its source span, inside the
/// text, and its members are exactly the mentions under it. Onto any
/// other view only the layer-wide checks and the graph check apply.
fn chain(msg: &TypedDocumentResult, text: &str, offsetter: &Offsetter, layer: &TypedLayer) -> Result<(), Failure> {
    let lid = layer.id.as_str();
    if layer.annotations.is_empty() {
        return Ok(());
    }
    let Some(native) = layer.native_view else {
        return Err(bad(format!("{lid} has no native_view")));
    };
    let Some(mapping_payload::Payload::Inline(inline)) = msg.mapping.as_ref().and_then(|m| m.payload.as_ref()) else {
        return Err(bad(format!("{lid} needs the inline mapping to name its view")));
    };
    let Some(view) = inline.views.get(native as usize) else {
        return Err(bad(format!("{lid} native_view {native} is not in the mapping")));
    };
    let Some(parent) = view.parent else {
        return Err(bad(format!("{lid} is on the source view; edits are over a derived view")));
    };
    let on_source = msg.view == u32::from(ViewId::SOURCE.0);
    if (on_source && parent != u32::from(ViewId::SOURCE.0)) || (!on_source && msg.view != native) {
        return Ok(());
    }
    let mut pairs = Vec::with_capacity(layer.annotations.len());
    for (i, ann) in layer.annotations.iter().enumerate() {
        let (Some(Value::PiiEdit(e)), [frag]) = (&ann.value, ann.spans.as_slice()) else {
            return Err(bad(format!("{lid}[{i}] pii edit must carry one span")));
        };
        let span = offsetter.from_wire(text, &format!("{lid}[{i}] span"), frag.start, frag.end)?;
        let s = e.source.as_ref().ok_or_else(|| bad(format!("{lid}[{i}] pii edit has no source span")))?;
        pairs.push(Pair { span, source: Span::new(s.start, s.end) });
    }
    if on_source {
        for (i, p) in pairs.iter().enumerate() {
            if p.span != p.source {
                return Err(bad(format!("{lid}[{i}] pii edit source span disagrees with its span on the source")));
            }
        }
        return members_on_source(msg, text, offsetter, layer, &pairs);
    }
    let (mut parent_end, mut child_end) = (0u32, 0u32);
    for (i, p) in pairs.iter().enumerate() {
        if p.source.start.checked_sub(parent_end) != p.span.start.checked_sub(child_end) {
            return Err(bad(format!("{lid}[{i}] pii edit source span disagrees with the view's alignment")));
        }
        (parent_end, child_end) = (p.source.end, p.span.end);
    }
    Ok(())
}

/// On a source projection, each edit lists exactly the mentions of the
/// covered pii layer that overlap its source span, and every mention is
/// under an edit.
fn members_on_source(
    msg: &TypedDocumentResult,
    text: &str,
    offsetter: &Offsetter,
    layer: &TypedLayer,
    pairs: &[Pair],
) -> Result<(), Failure> {
    let lid = layer.id.as_str();
    let named = layer.indexes_layer.iter().find(|b| b.layer == Pii::ID).map(|b| b.instance.as_str());
    let pii = msg
        .layers
        .iter()
        .filter(|l| l.id == Pii::ID)
        .find(|l| named.map_or(l.instance == layer.instance || l.instance.is_empty(), |n| l.instance == n));
    let Some(pii) = pii else {
        return Err(bad(format!("{lid} has no covered {} layer in the message", Pii::ID)));
    };
    let mut under: Vec<Vec<u32>> = vec![Vec::new(); pairs.len()];
    for (m, ann) in pii.annotations.iter().enumerate() {
        let (Some(first), Some(last)) = (ann.spans.first(), ann.spans.last()) else { continue };
        let span = offsetter.from_wire(text, &format!("{}[{m}] span", Pii::ID), first.start, last.end)?;
        // The sources are ordered and disjoint (checked across the layer
        // first), so the edits a mention overlaps are one run.
        let first = pairs.partition_point(|p| p.source.end <= span.start);
        let mut covered = false;
        for (k, p) in pairs.iter().enumerate().skip(first) {
            if p.source.start >= span.end {
                break;
            }
            under[k].push(u32::try_from(m).expect("fewer than 2^32 mentions"));
            covered = true;
        }
        if !covered {
            return Err(bad(format!("{lid} leaves {}[{m}] under no edit", Pii::ID)));
        }
    }
    for (i, ann) in layer.annotations.iter().enumerate() {
        if let Some(Value::PiiEdit(e)) = &ann.value
            && under[i] != e.members
        {
            return Err(bad(format!("{lid}[{i}] pii edit does not list exactly the mentions under its source span")));
        }
    }
    Ok(())
}

pub(super) fn edit(ctx: &LayerCtx<'_>, i: usize, e: &PiiEditAnnotation) -> Result<(), Failure> {
    let lid = ctx.lid;
    let at = || format!("{lid}[{i}]");
    if lid != PiiEdits::ID {
        return Err(bad(format!("{} is a pii edit outside a {} layer", at(), PiiEdits::ID)));
    }
    let source = e.source.as_ref().ok_or_else(|| bad(format!("{} pii edit has no source span", at())))?;
    // The source span is never projected: it is exact by definition.
    outcome::check_single(lid, &format!("[{i}] source"), source, true)?;
    if source.start >= source.end {
        return Err(bad(format!("{} pii edit source span is empty or inverted", at())));
    }
    if e.members.is_empty() {
        return Err(bad(format!("{} pii edit covers no mention", at())));
    }
    if e.members.windows(2).any(|w| w[0] >= w[1]) {
        return Err(bad(format!("{} pii edit members are out of order or repeated", at())));
    }
    let covered = unique_target_len(&ctx.target_lengths, lid, Pii::ID)?;
    if let (Some(n), Some(&last)) = (covered, e.members.last())
        && usize::try_from(last).map_or(true, |m| m >= n)
    {
        return Err(bad(format!("{} pii edit member is outside the covered {} layer", at(), Pii::ID)));
    }
    match e.rule.as_ref().and_then(|r| r.rule.as_ref()) {
        None => return Err(bad(format!("{} pii edit rule is missing", at()))),
        Some(Rule::DefaultRule(true)) => {}
        Some(Rule::DefaultRule(false)) => return Err(bad(format!("{} pii edit default_rule is false", at()))),
        Some(Rule::Kind(k)) => kind(ctx, k).map_err(|what| bad(format!("{} pii edit rule {what}", at())))?,
    }
    if u8::try_from(e.strategy).ok().and_then(PiiStrategyKind::from_id).is_none() {
        return Err(bad(format!("{} pii edit strategy is not a known strategy", at())));
    }
    Ok(())
}

fn kind(ctx: &LayerCtx<'_>, k: &PiiKindRef) -> Result<(), &'static str> {
    match k.kind {
        None => Err("kind is missing"),
        Some(Kind::Builtin(id)) => match u16::try_from(id).ok().and_then(PiiKind::from_id) {
            Some(_) => Ok(()),
            None => Err("kind id is not a built-in kind"),
        },
        Some(Kind::Custom(index)) => match usize::try_from(index) {
            Ok(n) if n < ctx.custom_pii_kinds => Ok(()),
            _ => Err("custom kind index is outside the layer's custom_pii_kinds"),
        },
    }
}
