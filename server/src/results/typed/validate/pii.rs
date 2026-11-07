//! PII mention payloads: a kind the layer can name, built in or a custom
//! kind name, and a confidence tier the layer writes. The normalized form
//! is not judged here. An error names the layer, the row and its wire
//! spans, never the mention's text, its normalized form, or the refused
//! kind or tier, which are caller input.

use std::fmt::Write as _;

use turbo_analysis_layers::{PiiConfidence, PiiKind};

use super::*;
use crate::proto::result::PiiMention;

pub(super) fn mention(ctx: &LayerCtx<'_>, i: usize, ann: &TypedAnnotation, m: &PiiMention) -> Result<(), Failure> {
    let at = || format!("{}[{i}] at {}", ctx.lid, wire_spans(&ann.spans));
    if m.kind.is_empty() {
        return Err(bad(format!("{} pii kind is empty", at())));
    }
    if !resolvable(&m.kind) {
        return Err(bad(format!(
            "{} pii kind of {} bytes is neither a built-in kind nor a custom kind name of at most {} bytes",
            at(),
            m.kind.len(),
            PiiKind::MAX_CUSTOM_NAME_BYTES
        )));
    }
    if !known_tier(&m.confidence_tier) {
        return Err(bad(format!(
            "{} pii confidence_tier of {} bytes is not one of {}",
            at(),
            m.confidence_tier.len(),
            TIERS.map(PiiConfidence::name).join(", ")
        )));
    }
    Ok(())
}

/// Whether a non-empty `kind` names a [`PiiKind`]: one of
/// [`PiiKind::ALL`], or a custom kind name the layer takes, at most
/// [`PiiKind::MAX_CUSTOM_NAME_BYTES`] bytes. A custom name is judged by its
/// form alone, never by the process's registry of custom kinds, so a
/// message validates the same wherever it is checked.
fn resolvable(kind: &str) -> bool {
    PiiKind::parse(kind).is_some() || kind.len() <= PiiKind::MAX_CUSTOM_NAME_BYTES
}

/// Every confidence tier the layer writes.
const TIERS: [PiiConfidence; 3] = [PiiConfidence::Format, PiiConfidence::Checksum, PiiConfidence::LabelledChecksum];

/// Whether `name` is the name of one of [`TIERS`], exactly.
fn known_tier(name: &str) -> bool {
    TIERS.iter().any(|&tier| {
        // Exhaustive, so a tier added to `PiiConfidence` fails to compile
        // here until it is added to `TIERS` as well.
        match tier {
            PiiConfidence::Format | PiiConfidence::Checksum | PiiConfidence::LabelledChecksum => tier.name() == name,
        }
    })
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
