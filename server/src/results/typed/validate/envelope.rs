//! Validation of the result envelope (issue #1907) on a
//! `TypedDocumentResult`: the input and result identities are whole and
//! self-consistent, every view and layer has one provenance entry whose
//! model reference resolves, the inline mapping rebuilds as a view tree
//! whose target view is the one `raw_text` is in, and every layer's
//! native view is in that tree.

use std::collections::HashSet;

use turbo_analysis::export::FORMAT_VERSION;

use super::super::chain::WireChain;
use super::super::identity::{ENVELOPE_SCHEMA_VERSION, verify_view_text};
use super::bad;
use crate::api::Failure;
use crate::proto::result::{InputIdentity, ResultIdentity, TypedDocumentResult};

fn check_digest(what: &str, digest: &[u8]) -> Result<(), Failure> {
    if digest.len() != 32 {
        return Err(bad(format!("{what} is {} bytes, not the 32 of a SHA-256", digest.len())));
    }
    Ok(())
}

fn check_input(input: Option<&InputIdentity>) -> Result<(), Failure> {
    let Some(input) = input else {
        return Err(bad("the typed document carries no input identity"));
    };
    check_digest("input.raw_bytes_digest", &input.raw_bytes_digest)?;
    check_digest("input.source_text_digest", &input.source_text_digest)?;
    if let Some(b) = &input.binding
        && !b.parse_artifact_digest.is_empty()
        && b.parse_artifact_id.is_empty()
    {
        return Err(bad("input.binding has a parse_artifact_digest and no parse_artifact_id"));
    }
    Ok(())
}

fn check_result(result: Option<&ResultIdentity>) -> Result<(), Failure> {
    let Some(result) = result else {
        return Err(bad("the typed document carries no result identity"));
    };
    check_digest("result.content_digest", &result.content_digest)?;
    let hex: String = result.content_digest.iter().map(|b| format!("{b:02x}")).collect();
    if result.result_id != hex {
        return Err(bad("result.result_id is not the hexadecimal of result.content_digest"));
    }
    if result.envelope_schema_version != ENVELOPE_SCHEMA_VERSION {
        return Err(bad(format!(
            "envelope schema version {} is not the served {ENVELOPE_SCHEMA_VERSION}",
            result.envelope_schema_version
        )));
    }
    if result.graph_format_version != u32::from(FORMAT_VERSION) {
        return Err(bad(format!(
            "graph format version {} is not the served {FORMAT_VERSION}",
            result.graph_format_version
        )));
    }
    Ok(())
}

/// Validates the envelope of `msg` against its own views, layers and
/// model table, and returns the rebuilt view chain for the span checks.
pub(super) fn validate_envelope(msg: &TypedDocumentResult, known_models: &HashSet<&str>) -> Result<WireChain, Failure> {
    check_input(msg.input.as_ref())?;
    check_result(msg.result.as_ref())?;

    let mut seen: HashSet<(&str, &str)> = HashSet::with_capacity(msg.provenance.len());
    for (i, p) in msg.provenance.iter().enumerate() {
        if p.layer.trim().is_empty() {
            return Err(bad(format!("provenance[{i}] has an empty layer")));
        }
        if !seen.insert((p.layer.as_str(), p.instance.as_str())) {
            return Err(bad(format!("duplicate provenance for ({:?}, {:?})", p.layer, p.instance)));
        }
        if !p.model_ref.is_empty() && !known_models.contains(p.model_ref.as_str()) {
            return Err(bad(format!("provenance[{i}] references unknown model_ref {:?}", p.model_ref)));
        }
        check_digest(&format!("provenance[{i}].provenance_digest"), &p.provenance_digest)?;
        if let Some(d) = &p.content_digest {
            check_digest(&format!("provenance[{i}].content_digest"), d)?;
        }
        if let Some(d) = &p.input_snapshot_digest {
            check_digest(&format!("provenance[{i}].input_snapshot_digest"), d)?;
        }
    }
    for v in &msg.views {
        if !seen.contains(&("turbo:view", v.instance.as_str())) {
            return Err(bad(format!("view {:?} has no provenance entry", v.instance)));
        }
    }
    for l in &msg.layers {
        if !seen.contains(&(l.id.as_str(), l.instance.as_str())) {
            return Err(bad(format!("layer {:?} instance {:?} has no provenance entry", l.id, l.instance)));
        }
    }

    let chain = WireChain::from_payload(msg.mapping.as_ref())?;
    let listed: HashSet<&str> = msg.views.iter().map(|v| v.instance.as_str()).collect();
    for v in chain.views() {
        if !listed.contains(v.instance.as_str()) {
            return Err(bad(format!("mapping view {:?} is not in views", v.instance)));
        }
    }
    let view_u16 = u16::try_from(msg.view).map_err(|_| bad(format!("view {} exceeds u16::MAX", msg.view)))?;
    let target = chain.view(turbo_analysis::ViewId(view_u16))?;
    if target.text_len as usize != msg.raw_text.len() {
        return Err(bad(format!(
            "raw_text is {} bytes and the mapping says view {} has {}",
            msg.raw_text.len(),
            msg.view,
            target.text_len
        )));
    }
    verify_view_text(&format!("mapping view {} text_digest", msg.view), &target.text_digest, &msg.raw_text)?;
    for l in &msg.layers {
        if let Some(v) = l.native_view {
            let id = u16::try_from(v).map_err(|_| bad(format!("layer {:?} native_view {v} exceeds u16::MAX", l.id)))?;
            chain
                .view(turbo_analysis::ViewId(id))
                .map_err(|e| bad(format!("layer {:?} native_view: {}", l.id, e.message)))?;
        }
    }
    Ok(chain)
}
