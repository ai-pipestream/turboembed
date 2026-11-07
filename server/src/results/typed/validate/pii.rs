//! PII on the wire: the custom-kind table of a `turbo:pii` layer (or of
//! the `turbo:pii-edits` layer that covers one), and each
//! mention's kind reference and confidence tier. A kind is a built-in id
//! [`PiiKind::from_id`] knows or an index into the layer's
//! `custom_pii_kinds`; a tier is one [`PiiConfidence::from_id`] knows.
//! `UNSPECIFIED` (0) is neither. The normalized form is not judged here.
//! An error names the layer, the row and its wire spans, never the
//! mention's text, its normalized form, a custom kind name, or the refused
//! number, which are caller input.

use std::fmt::Write as _;

use turbo_analysis::Layer;
use turbo_analysis_layers::{CustomKinds, KindError, Pii, PiiConfidence, PiiEdits, PiiKind};

use super::*;
use crate::proto::result::pii_kind_ref::Kind;
use crate::proto::result::{PiiMention, TypedLayer};

/// The number of custom kinds `layer` names: its `custom_pii_kinds`, which
/// only a `turbo:pii` layer may carry, held to [`CustomKinds::from_names`].
pub(super) fn custom_kinds(layer: &TypedLayer) -> Result<usize, Failure> {
    let lid = layer.id.as_str();
    let names = &layer.custom_pii_kinds;
    if lid != Pii::ID && lid != PiiEdits::ID {
        if !names.is_empty() {
            return Err(bad(format!(
                "layer {lid} carries custom_pii_kinds; only {} and {} may",
                Pii::ID,
                PiiEdits::ID
            )));
        }
        return Ok(0);
    }
    CustomKinds::from_names(names).map_err(|e| bad(format!("layer {lid} custom_pii_kinds: {}", table_error(e))))?;
    Ok(names.len())
}

/// What is wrong with a custom-kind table, with no name in it.
fn table_error(e: KindError) -> String {
    match e {
        // The reserved kind is a built-in, never caller text.
        KindError::Reserved(_) => "a name is a built-in kind name".to_owned(),
        // Duplicate, too long, empty and full carry an index or a length.
        other => other.to_string(),
    }
}

// Reads the retired string fields only to refuse them.
#[allow(deprecated)]
pub(super) fn mention(ctx: &LayerCtx<'_>, i: usize, ann: &TypedAnnotation, m: &PiiMention) -> Result<(), Failure> {
    let at = || format!("{}[{i}] at {}", ctx.lid, wire_spans(&ann.spans));
    if ctx.lid != Pii::ID {
        return Err(bad(format!("{} is a pii mention outside a {} layer", at(), Pii::ID)));
    }
    if ann.spans.is_empty() {
        return Err(bad(format!("{} pii mention has no span", at())));
    }
    if !m.kind.is_empty() || !m.confidence_tier.is_empty() {
        return Err(bad(format!("{} pii mention sets a retired string kind or confidence_tier", at())));
    }
    match m.kind_ref.as_ref().and_then(|r| r.kind) {
        None => return Err(bad(format!("{} pii kind is missing", at()))),
        Some(Kind::Builtin(id)) => {
            if u16::try_from(id).ok().and_then(PiiKind::from_id).is_none() {
                return Err(bad(format!("{} pii kind id is not a built-in kind", at())));
            }
        }
        Some(Kind::Custom(index)) => {
            if usize::try_from(index).map_or(true, |n| n >= ctx.custom_pii_kinds) {
                return Err(bad(format!(
                    "{} pii custom kind index is outside the layer's {} custom_pii_kinds",
                    at(),
                    ctx.custom_pii_kinds
                )));
            }
        }
    }
    if u32::try_from(m.confidence).ok().and_then(PiiConfidence::from_id).is_none() {
        return Err(bad(format!("{} pii confidence is not a known tier", at())));
    }
    Ok(())
}

/// The wire spans as `start..end`, comma separated, or `no span`.
fn wire_spans(spans: &[SpanFragment]) -> String {
    if spans.is_empty() {
        return "no span".to_owned();
    }
    let mut out = String::new();
    for (n, s) in spans.iter().enumerate() {
        let sep = if n == 0 { "" } else { ", " };
        let _ = write!(out, "{sep}{}..{}", s.start, s.end);
    }
    out
}
