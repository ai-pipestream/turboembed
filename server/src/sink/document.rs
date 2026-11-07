//! Items that carry a document graph: the graph as RoundTripDocumentGraph
//! answers it, with its layers and views listed, and, when the sender is
//! configured for it, the typed projection InspectDocumentGraph answers for
//! that graph (`DocumentSinkItem.typed`).

use turbo_analysis::{Document, Layer, ViewId};
use turbo_analysis_layers::Pii;

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_STATE};
use crate::proto::result::{
    DocumentGraphRequest, DocumentGraphResponse, InspectDocumentGraphRequest, OffsetMode, TypedDocumentResult,
};
use crate::proto::sink::DocumentSinkItem;
use crate::results::disclosure::Disclosure;
use crate::results::document::{export_document_response, round_trip_document};
use crate::results::typed::{document_to_typed_with, inspect_document_graph, round_trip_typed_document};

/// The typed projection a forwarder adds to every item carrying a graph:
/// `view` and `offsets` mean what they mean on InspectDocumentGraphRequest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SinkTyped {
    /// The view the projection's spans are mapped to; 0 is the source.
    pub view: u32,
    /// The unit the projection's span offsets are counted in.
    pub offsets: OffsetMode,
}

impl SinkTyped {
    /// Refuses a view that no document can have.
    pub(crate) fn check(&self) -> Result<(), String> {
        u16::try_from(self.view)
            .map(|_| ())
            .map_err(|_| format!("the sink's typed view {} exceeds u16::MAX", self.view))
    }
}

/// The item for `graph`, numbered as it is sent. The graph is imported and
/// re-exported, so the layers and views list the bytes the item carries,
/// and the typed projection, when asked for, is built from those bytes.
/// A graph holding PII follows `disclosure` (#1904): without `full_text`
/// the item carries no graph, and carries the findings projection in the
/// configured view, or on the source view in UTF-8 offsets when no
/// projection is configured.
pub(crate) fn graph_item(
    doc_id: &str,
    graph: Vec<u8>,
    typed: Option<SinkTyped>,
    disclosure: &Disclosure,
) -> Result<DocumentSinkItem, Failure> {
    let options = Some(disclosure.to_options());
    let request = DocumentGraphRequest { graph: graph.into(), pii_output: options };
    let mut listed = round_trip_document(&request)?;
    let source = if listed.graph.is_empty() { request.graph } else { listed.graph.clone() };
    let typed = match typed {
        Some(t) => Some(inspect_document_graph(&InspectDocumentGraphRequest {
            doc_id: doc_id.to_owned(),
            graph: source.to_vec(),
            view: t.view,
            offsets: t.offsets as i32,
            pii_output: options,
        })?),
        None => listed.findings.take().map(|f| TypedDocumentResult { doc_id: doc_id.to_owned(), ..f }),
    };
    Ok(DocumentSinkItem {
        sequence: 0,
        doc_id: doc_id.to_owned(),
        graph: listed.graph.to_vec(),
        layers: listed.layers,
        views: listed.views,
        typed,
        analyzed: None,
    })
}

/// The item for a graph this server exported from `doc`: `exported` is
/// that whole export, so its bytes, layers and views are carried as they
/// are when `disclosure` lets the graph leave, and the typed projection,
/// when asked for, is built from `doc` itself. Otherwise the item is the
/// one [`graph_item`] gives under `disclosure`.
pub(crate) fn exported_item(
    doc_id: &str,
    exported: DocumentGraphResponse,
    doc: &Document,
    typed: Option<SinkTyped>,
    disclosure: &Disclosure,
) -> Result<DocumentSinkItem, Failure> {
    let projected = match typed {
        Some(t) => {
            let view = u16::try_from(t.view)
                .map_err(|_| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("view {} exceeds u16::MAX", t.view)))?;
            let offsets = crate::offsets::OffsetMode::from_proto(t.offsets as i32)?;
            Some(document_to_typed_with(doc_id, doc, ViewId(view), offsets, disclosure)?)
        }
        None => None,
    };
    if !disclosure.graph(doc) {
        let listed = export_document_response(doc, disclosure)?;
        let typed =
            projected.or_else(|| listed.findings.map(|f| TypedDocumentResult { doc_id: doc_id.to_owned(), ..f }));
        return Ok(DocumentSinkItem {
            sequence: 0,
            doc_id: doc_id.to_owned(),
            graph: Vec::new(),
            layers: listed.layers,
            views: listed.views,
            typed,
            analyzed: None,
        });
    }
    Ok(DocumentSinkItem {
        sequence: 0,
        doc_id: doc_id.to_owned(),
        graph: exported.graph.to_vec(),
        layers: exported.layers,
        views: exported.views,
        typed: projected,
        analyzed: None,
    })
}

/// Whether `item` is one whose graph was withheld for PII: no graph, a
/// typed projection, and a `turbo:pii` layer in its listing.
pub(crate) fn pii_withheld(item: &DocumentSinkItem) -> bool {
    item.graph.is_empty() && item.typed.is_some() && item.layers.iter().any(|l| l.layer == Pii::ID)
}

/// Why a typed projection is not accepted on `item`, if it is not: one on
/// an item without a graph (unless the graph was withheld for PII), or one
/// RoundTripTypedDocument refuses.
pub(crate) fn check_typed(item: &DocumentSinkItem, typed: &TypedDocumentResult) -> Result<(), String> {
    if item.graph.is_empty() && !pii_withheld(item) {
        return Err("carries a typed projection without a graph".into());
    }
    round_trip_typed_document(typed).map(|_| ()).map_err(|e| format!("typed document: {}", e.message))
}

/// The refusal for a graph forwarded to a server that has no sink.
pub(crate) fn no_sink() -> Failure {
    Failure::new(TURBO_E_INVALID_STATE, 0, "the server forwards to no sink")
}
