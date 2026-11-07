//! Scoped references (issue #1907): a `ScopedRef` names one annotation
//! by the result it is in, its layer instance and item index, and the
//! native view its spans are in by instance and text digest, so it means
//! the same thing in any message. `TypedDocumentResult.view`,
//! `ViewChain.view`, annotation indexes and `model_ref` values are
//! message-local; [`scoped_ref`] turns them into a durable reference and
//! [`resolve`] finds the annotation again, refusing a reference into
//! another result or a view whose text has changed.

use super::bad;
use crate::api::Failure;
use crate::proto::result::{ScopedRef, TypedAnnotation, TypedDocumentResult, TypedLayer, ViewChain, mapping_payload};

fn chain_views(msg: &TypedDocumentResult) -> &[ViewChain] {
    match msg.mapping.as_ref().and_then(|m| m.payload.as_ref()) {
        Some(mapping_payload::Payload::Inline(inline)) => &inline.views,
        None => &[],
    }
}

fn chain_view(msg: &TypedDocumentResult, view: u32) -> Result<&ViewChain, Failure> {
    chain_views(msg)
        .iter()
        .find(|v| v.view == view)
        .ok_or_else(|| bad(format!("view {view} is not in the typed document's mapping")))
}

/// The scoped reference of annotation `index` of `layer` in `msg`. A
/// layer without text spans carries no view: its reference names the
/// result and the item only.
pub fn scoped_ref(msg: &TypedDocumentResult, layer: &TypedLayer, index: u32) -> Result<ScopedRef, Failure> {
    let result_id = msg.result.as_ref().map(|r| r.result_id.clone()).unwrap_or_default();
    if result_id.is_empty() {
        return Err(bad("the typed document carries no result identity"));
    }
    if index as usize >= layer.annotations.len() {
        return Err(bad(format!(
            "{}[{index}] is outside the layer's {} annotations",
            layer.id,
            layer.annotations.len()
        )));
    }
    let (view_instance, view_text_digest) = match layer.native_view {
        Some(v) => {
            let view = chain_view(msg, v)?;
            (view.instance.clone(), view.text_digest.clone())
        }
        None => (String::new(), Vec::new()),
    };
    Ok(ScopedRef {
        result_id,
        layer: layer.id.clone(),
        instance: layer.instance.clone(),
        index,
        view_instance,
        view_text_digest,
    })
}

/// The layer and annotation `r` names in `msg`, or why it names none: a
/// different result, no such layer instance or item, or a view whose
/// instance or text digest is not the one the reference was made on.
pub fn resolve<'a>(
    msg: &'a TypedDocumentResult,
    r: &ScopedRef,
) -> Result<(&'a TypedLayer, &'a TypedAnnotation), Failure> {
    let result_id = msg.result.as_ref().map(|x| x.result_id.as_str()).unwrap_or_default();
    if r.result_id != result_id {
        return Err(bad("the reference is into another result"));
    }
    let layer = msg
        .layers
        .iter()
        .find(|l| l.id == r.layer && l.instance == r.instance)
        .ok_or_else(|| bad(format!("the typed document has no layer {:?} instance {:?}", r.layer, r.instance)))?;
    let annotation = layer.annotations.get(r.index as usize).ok_or_else(|| {
        bad(format!("{}[{}] is outside the layer's {} annotations", r.layer, r.index, layer.annotations.len()))
    })?;
    match layer.native_view {
        Some(v) => {
            let view = chain_view(msg, v)?;
            if view.instance != r.view_instance {
                return Err(bad(format!(
                    "the reference is on view {:?}, the layer is on {:?}",
                    r.view_instance, view.instance
                )));
            }
            if view.text_digest != r.view_text_digest {
                return Err(bad(format!(
                    "the text of view {:?} is not the text the reference was made on",
                    view.instance
                )));
            }
        }
        None => {
            if !r.view_instance.is_empty() || !r.view_text_digest.is_empty() {
                return Err(bad(format!("the reference names a view and layer {:?} has no text spans", r.layer)));
            }
        }
    }
    Ok((layer, annotation))
}
