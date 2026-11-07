//! `turbo:pii-edits` on the wire: each edit's source span, covered mention
//! rows, rule and strategy. Rows are ascending, non-empty and inside the
//! covered pii layer when that layer's length is known; the rule names the
//! default or a kind the PII validator would take; the strategy is one the
//! native layer knows. An error names the layer and the row, never a kind
//! name or a refused number.

use turbo_analysis::Layer;
use turbo_analysis_layers::{Pii, PiiEdits, PiiKind, PiiStrategyKind};

use super::*;
use crate::proto::result::pii_edit_rule::Rule;
use crate::proto::result::pii_kind_ref::Kind;
use crate::proto::result::{PiiEditAnnotation, PiiKindRef};

pub(super) fn edit(ctx: &LayerCtx<'_>, i: usize, e: &PiiEditAnnotation) -> Result<(), Failure> {
    let lid = ctx.lid;
    let at = || format!("{lid}[{i}]");
    if lid != PiiEdits::ID {
        return Err(bad(format!("{} is a pii edit outside a {} layer", at(), PiiEdits::ID)));
    }
    let source = e.source.as_ref().ok_or_else(|| bad(format!("{} pii edit has no source span", at())))?;
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
