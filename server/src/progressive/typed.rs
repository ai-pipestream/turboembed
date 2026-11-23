//! The typed projection the progressive Complete event carries when the
//! request asks for it (issue #2517 F15): the final graph as
//! TypedDocumentResult, so a proto client reads the annotations without a
//! DocumentGraph reader or a second InspectDocumentGraph call.

use turbo_analysis::{Document, ViewId};

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::OffsetMode;
use crate::proto::result::{InputBinding, TypedDocumentResult, TypedResultOptions};
use crate::results::disclosure::Disclosure;
use crate::results::typed::document_to_typed_bound;

/// The view and offsets the projection is in.
#[derive(Clone, Copy)]
pub(super) struct Target {
    view: ViewId,
    offsets: OffsetMode,
}

/// The target `options` asks for, checked before anything runs.
pub(super) fn target(options: &TypedResultOptions) -> Result<Target, Failure> {
    let view = u16::try_from(options.view).map_err(|_| {
        Failure::new(TURBO_E_INVALID_ARGUMENT, 18, format!("typed_result.view {} exceeds u16::MAX", options.view))
    })?;
    Ok(Target { view: ViewId(view), offsets: OffsetMode::from_proto(options.offsets)? })
}

/// The projection of `doc`, under the request's disclosure, with its
/// binding and its operation key on the receipt, as InspectDocumentGraph
/// answers for the same graph.
pub(super) fn project(
    doc_id: &str,
    doc: &Document,
    t: Target,
    binding: Option<&InputBinding>,
    operation_key: &str,
    disclosure: &Disclosure,
) -> Result<TypedDocumentResult, Failure> {
    document_to_typed_bound(doc_id, doc, t.view, t.offsets, binding, operation_key, disclosure)
}
