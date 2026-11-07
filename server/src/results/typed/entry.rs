//! The public entry points of the typed projection: projecting a document
//! under a disclosure (#1904), `InspectDocumentGraph` and
//! `RoundTripTypedDocument`.

use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis::{Document, ViewId};

use super::{bad, document_to_typed_bound, validate_typed_document};
use crate::api::Failure;
use crate::offsets::OffsetMode;
use crate::proto::result::{InspectDocumentGraphRequest, TypedDocumentResult};
use crate::results::disclosure::{Disclosure, PiiOutputPolicy};
use crate::results::document::codecs;
use crate::results::{MAX_GRAPH_BYTES, Wire, over};

/// Projects all shared layers on `doc` into a strongly-typed [`TypedDocumentResult`]
/// with spans mapped to `target_view` and converted into `offsets`, under
/// [`Disclosure::FINDINGS`], with the envelope of a document the caller did
/// not bind to a source. With debug assertions on, the result is also put
/// through [`validate_typed_document`], and a message it refuses is an
/// error.
pub fn document_to_typed(
    doc_id: &str,
    doc: &Document,
    target_view: ViewId,
    offsets: OffsetMode,
) -> Result<TypedDocumentResult, Failure> {
    document_to_typed_bound(doc_id, doc, target_view, offsets, None, &Disclosure::FINDINGS)
}

/// [`document_to_typed`] under `disclosure`: on a document holding PII,
/// `raw_text` is empty unless the disclosure lets the target view's text
/// leave, and status details are cut to identities without `full_text`.
pub fn document_to_typed_with(
    doc_id: &str,
    doc: &Document,
    target_view: ViewId,
    offsets: OffsetMode,
    disclosure: &Disclosure,
) -> Result<TypedDocumentResult, Failure> {
    document_to_typed_bound(doc_id, doc, target_view, offsets, None, disclosure)
}

/// Imports the binary `DocumentGraph` in `req` and projects it into a
/// [`TypedDocumentResult`] in the requested view and [`OffsetMode`], under
/// the disclosure its `pii_output` asks for and [`PiiOutputPolicy::ALL`]
/// allows.
pub fn inspect_document_graph(req: &InspectDocumentGraphRequest) -> Result<TypedDocumentResult, Failure> {
    inspect_document_graph_under(req, PiiOutputPolicy::ALL)
}

/// [`inspect_document_graph`] under the server's `policy`.
pub fn inspect_document_graph_under(
    req: &InspectDocumentGraphRequest,
    policy: PiiOutputPolicy,
) -> Result<TypedDocumentResult, Failure> {
    let disclosure = Disclosure::decide(req.pii_output.as_ref(), policy)?;
    if req.graph.len() as u64 > MAX_GRAPH_BYTES {
        return Err(over("max_graph_bytes", "document graph bytes", req.graph.len() as u64));
    }
    let offsets = OffsetMode::from_proto(req.offsets)?;
    let view_u16 = u16::try_from(req.view).map_err(|_| bad(format!("view {} exceeds u16::MAX", req.view)))?;
    let graph = DocumentGraph::from_bytes(&req.graph).map_err(|e| bad(format!("document graph: {e}")))?;
    let doc = import(&graph, codecs()).map_err(|e| bad(format!("document graph import: {e}")))?;
    document_to_typed_bound(&req.doc_id, &doc, ViewId(view_u16), offsets, req.binding.as_ref(), &disclosure)
}

/// Validates `msg` and returns a clone for `RoundTripTypedDocument`.
pub fn round_trip_typed_document(msg: &TypedDocumentResult) -> Result<TypedDocumentResult, Failure> {
    validate_typed_document(msg)?;
    Ok(msg.clone())
}

impl Wire for TypedDocumentResult {
    type Msg = TypedDocumentResult;

    fn to_msg(&self) -> Self::Msg {
        self.clone()
    }

    fn from_msg(msg: &Self::Msg) -> Result<Self, Failure> {
        round_trip_typed_document(msg)
    }
}
