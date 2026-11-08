//! `turbo:pii-risk` on the wire (#1889): the layer's one annotation, with
//! no span, whose numbers hold together as the native layer's do: every
//! level, severity and suggestion known, the level the one its score
//! gives, kinds in order and inside the layer's `custom_pii_kinds`, and
//! the mention count their sum. An error names the layer and the field,
//! never a kind name or a refused number.

use turbo_analysis::Layer;
use turbo_analysis_layers::{
    CustomKindId, CustomKinds, PiiKind, PiiRisk, PiiRiskKind, PiiRiskLevel, PiiSeverity, PiiSuggestedRewrite,
};

use super::*;
use crate::proto::result::PiiRiskAnnotation;
use crate::proto::result::pii_kind_ref::Kind;

fn known<T>(lid: &str, what: &str, id: i32, from: impl Fn(u32) -> Option<T>) -> Result<T, Failure> {
    u32::try_from(id).ok().and_then(from).ok_or_else(|| bad(format!("{lid} pii risk {what} is not known")))
}

pub(super) fn risk(ctx: &LayerCtx<'_>, i: usize, ann: &TypedAnnotation, r: &PiiRiskAnnotation) -> Result<(), Failure> {
    let lid = ctx.lid;
    if lid != PiiRisk::ID {
        return Err(bad(format!("{lid}[{i}] is a pii risk outside a {} layer", PiiRisk::ID)));
    }
    if i > 0 || ctx.layer_len != 1 {
        return Err(bad(format!("{lid} has {} annotations; a pii risk layer has one", ctx.layer_len)));
    }
    if !ann.spans.is_empty() {
        return Err(bad(format!("{lid}[0] pii risk has spans; it covers the document")));
    }
    // Placeholder names stand in for the layer's table, whose size is all
    // a kind index is held to.
    let custom_kinds = CustomKinds::from_names((0..ctx.custom_pii_kinds).map(|n| format!("custom-{n}")))
        .map_err(|_| bad(format!("{lid} custom_pii_kinds is not a valid table")))?;
    let mut kinds = Vec::with_capacity(r.kinds.len());
    for (n, k) in r.kinds.iter().enumerate() {
        let kind = match k.kind.as_ref().and_then(|r| r.kind) {
            Some(Kind::Builtin(id)) => u16::try_from(id).ok().and_then(PiiKind::from_id),
            Some(Kind::Custom(index)) => u16::try_from(index)
                .ok()
                .filter(|&x| usize::from(x) < ctx.custom_pii_kinds)
                .map(|x| PiiKind::Custom(CustomKindId::new(x))),
            None => None,
        }
        .ok_or_else(|| bad(format!("{lid}[0] pii risk kinds[{n}] kind is missing or not known")))?;
        kinds.push(PiiRiskKind {
            kind,
            mentions: k.mentions,
            severity: known(lid, "severity", k.severity, PiiSeverity::from_id)?,
            weighted_count: k.weighted_count,
            contribution: k.contribution,
            suggested: known(lid, "suggested rewrite", k.suggested, PiiSuggestedRewrite::from_id)?,
        });
    }
    let native = PiiRisk {
        score: r.score,
        level: known(lid, "level", r.level, PiiRiskLevel::from_id)?,
        mentions: r.mentions,
        chars: r.chars,
        density_per_1000_chars: r.density_per_1000_chars,
        kinds,
        custom_kinds,
        severity_table_digest: r.severity_table_digest,
    };
    native.check().map_err(|e| bad(format!("{lid}[0] {e}")))
}
