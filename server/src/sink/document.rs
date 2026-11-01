//! Items that carry a document graph: the graph as RoundTripDocumentGraph
//! answers it, with its layers and views listed, and, when the sender is
//! configured for it, the typed projection InspectDocumentGraph answers for
//! that graph (`DocumentSinkItem.typed`).

use crate::api::{Failure, TURBO_E_INVALID_STATE};
use crate::proto::result::{DocumentGraphRequest, InspectDocumentGraphRequest, OffsetMode, TypedDocumentResult};
use crate::proto::sink::DocumentSinkItem;
use crate::results::document::round_trip_document;
use crate::results::typed::{inspect_document_graph, round_trip_typed_document};

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
pub(crate) fn graph_item(doc_id: &str, graph: Vec<u8>, typed: Option<SinkTyped>) -> Result<DocumentSinkItem, Failure> {
    let listed = round_trip_document(&DocumentGraphRequest { graph: graph.into() })?;
    let graph = listed.graph.to_vec();
    let typed = match typed {
        Some(t) => Some(inspect_document_graph(&InspectDocumentGraphRequest {
            doc_id: doc_id.to_owned(),
            graph: graph.clone(),
            view: t.view,
            offsets: t.offsets as i32,
        })?),
        None => None,
    };
    Ok(DocumentSinkItem {
        sequence: 0,
        doc_id: doc_id.to_owned(),
        graph,
        layers: listed.layers,
        views: listed.views,
        typed,
        analyzed: None,
    })
}

/// Why a typed projection is not accepted on `item`, if it is not: one on
/// an item without a graph, or one RoundTripTypedDocument refuses.
pub(crate) fn check_typed(item: &DocumentSinkItem, typed: &TypedDocumentResult) -> Result<(), String> {
    if item.graph.is_empty() {
        return Err("carries a typed projection without a graph".into());
    }
    round_trip_typed_document(typed).map(|_| ()).map_err(|e| format!("typed document: {}", e.message))
}

/// The refusal for a graph forwarded to a server that has no sink.
pub(crate) fn no_sink() -> Failure {
    Failure::new(TURBO_E_INVALID_STATE, 0, "the server forwards to no sink")
}
