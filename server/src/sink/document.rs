//! Items that carry a document graph: the graph as RoundTripDocumentGraph
//! answers it, with its layers and views listed, and, when the sender is
//! configured for it, the typed projection InspectDocumentGraph answers for
//! that graph (`DocumentSinkItem.typed`).

use turbo_analysis::{Document, Layer, ViewId};
use turbo_analysis_layers::Pii;

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_STATE};
use crate::proto::result::{
    DocumentGraphRequest, DocumentGraphResponse, InputBinding, InspectDocumentGraphRequest, OffsetMode,
    ProjectSearchDocumentRequest, SearchDocumentProjection, TypedDocumentResult,
};
use crate::proto::sink::DocumentSinkItem;
use crate::results::disclosure::Disclosure;
use crate::results::document::{export_document_response, round_trip_document};
use crate::results::search_projection::{
    document_to_search_projection_bound, project_search_document, round_trip_search_projection,
};
use crate::results::typed::identity::{bound, check_binding};
use crate::results::typed::{document_to_typed_bound, inspect_document_graph, round_trip_typed_document};

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
/// projection is configured. The caller's `binding` (#1939) is the item's
/// and its projection's, copied as given.
pub(crate) fn graph_item(
    doc_id: &str,
    graph: Vec<u8>,
    typed: Option<SinkTyped>,
    search_projection: Option<SinkTyped>,
    disclosure: &Disclosure,
    binding: Option<&InputBinding>,
) -> Result<DocumentSinkItem, Failure> {
    check_binding(binding, "binding")?;
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
            binding: binding.cloned(),
            operation_key: String::new(),
            pii_policy: None,
        })?),
        None => listed.findings.take().map(|f| bound(TypedDocumentResult { doc_id: doc_id.to_owned(), ..f }, binding)),
    };
    let search_projection = match search_projection {
        Some(sp) => match project_search_document(&ProjectSearchDocumentRequest {
            doc_id: doc_id.to_owned(),
            graph: source.to_vec(),
            view: sp.view,
            offsets: sp.offsets as i32,
            pii_output: options,
            binding: binding.cloned(),
            analyzer_config_digest: Vec::new(),
        }) {
            Ok(proj) => Some(proj),
            Err(f) if listed.graph.is_empty() && f.code == TURBO_E_INVALID_STATE => None,
            Err(f) => return Err(f),
        },
        None => None,
    };
    Ok(DocumentSinkItem {
        sequence: 0,
        doc_id: doc_id.to_owned(),
        graph: listed.graph.to_vec(),
        layers: listed.layers,
        views: listed.views,
        typed,
        analyzed: None,
        binding: binding.cloned(),
        search_projection,
    })
}

/// The item for a graph this server exported from `doc`: `exported` is
/// that whole export, so its bytes, layers and views are carried as they
/// are when `disclosure` lets the graph leave, and the typed projection,
/// when asked for, is built from `doc` itself. Otherwise the item is the
/// one [`graph_item`] gives under `disclosure`. `binding` is carried as
/// [`graph_item`] carries it.
pub(crate) fn exported_item(
    doc_id: &str,
    exported: DocumentGraphResponse,
    doc: &Document,
    typed: Option<SinkTyped>,
    search_projection: Option<SinkTyped>,
    disclosure: &Disclosure,
    binding: Option<&InputBinding>,
) -> Result<DocumentSinkItem, Failure> {
    check_binding(binding, "binding")?;
    let projected = match typed {
        Some(t) => {
            let view = u16::try_from(t.view)
                .map_err(|_| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("view {} exceeds u16::MAX", t.view)))?;
            let offsets = crate::offsets::OffsetMode::from_proto(t.offsets as i32)?;
            Some(document_to_typed_bound(doc_id, doc, ViewId(view), offsets, binding, "", disclosure)?)
        }
        None => None,
    };
    let projected_search = match search_projection {
        Some(sp) => {
            let view = u16::try_from(sp.view)
                .map_err(|_| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("view {} exceeds u16::MAX", sp.view)))?;
            let offsets = crate::offsets::OffsetMode::from_proto(sp.offsets as i32)?;
            if disclosure.view_text(doc, ViewId(view)) {
                Some(document_to_search_projection_bound(doc_id, doc, ViewId(view), offsets, binding, &[], disclosure)?)
            } else {
                None
            }
        }
        None => None,
    };
    if !disclosure.graph(doc) {
        let listed = export_document_response(doc, disclosure)?;
        let typed = projected.or_else(|| {
            listed.findings.map(|f| bound(TypedDocumentResult { doc_id: doc_id.to_owned(), ..f }, binding))
        });
        return Ok(DocumentSinkItem {
            sequence: 0,
            doc_id: doc_id.to_owned(),
            graph: Vec::new(),
            layers: listed.layers,
            views: listed.views,
            typed,
            analyzed: None,
            binding: binding.cloned(),
            search_projection: projected_search,
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
        binding: binding.cloned(),
        search_projection: projected_search,
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

/// Validates an attached `SearchDocumentProjection` (`#2345`). An item may
/// carry `search_projection` either alongside a `graph` or standalone (when
/// a thick-client or upstream worker streams ready-to-index envelopes).
pub(crate) fn check_search_projection(sp: &SearchDocumentProjection) -> Result<(), String> {
    round_trip_search_projection(sp).map(|_| ()).map_err(|e| format!("search projection: {}", e.message))
}

/// Why the binding on `item` is not accepted, if it is not (#1939): a
/// `parse_artifact_digest` without a `parse_artifact_id`, or a typed
/// projection, search projection, or AnalyzeStream answer that carries
/// another binding.
pub(crate) fn check_binding_agrees(item: &DocumentSinkItem) -> Result<(), String> {
    check_binding(item.binding.as_ref(), "binding").map_err(|f| f.message)?;
    let typed = item.typed.as_ref().and_then(|t| t.input.as_ref()).map(|i| (i.binding.as_ref(), "typed projection"));
    let search = item
        .search_projection
        .as_ref()
        .and_then(|s| s.provenance.as_ref())
        .and_then(|p| p.input.as_ref())
        .map(|i| (i.binding.as_ref(), "search projection"));
    let answer = item.analyzed.as_ref().map(|a| (a.binding.as_ref(), "answer"));
    match [typed, search, answer].into_iter().flatten().find(|(b, _)| *b != item.binding.as_ref()) {
        Some((_, what)) => Err(format!("the {what} carries another binding than the item")),
        None => Ok(()),
    }
}

/// The refusal for a graph forwarded to a server that has no sink.
pub(crate) fn no_sink() -> Failure {
    Failure::new(TURBO_E_INVALID_STATE, 0, "the server forwards to no sink")
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Key, Provenance, Span};
    use turbo_analysis_layers::{Pii, PiiEvidence, PiiKind, PiiMention};

    #[test]
    fn pii_document_with_search_projection_omits_graph_and_preserves_findings() {
        let text = "contact alice@example.com";
        let mut doc = Document::new(text);
        let mut pii = Pii::new(ViewId::SOURCE);
        let detector = pii.detector("test", "1").unwrap();
        let evidence = [PiiEvidence::Shape];
        pii.push(PiiMention {
            span: Span::new(8, 25),
            kind: PiiKind::Email,
            detector,
            evidence: &evidence,
            score: None,
            normalized: "alice@example.com",
        });
        doc.add(&Key::DEFAULT, pii, Provenance::by("turbo-pii:pii")).unwrap();
        let bytes = export_document_response(&doc, &Disclosure::FULL).unwrap().graph.to_vec();
        let sp = Some(SinkTyped { view: 0, offsets: OffsetMode::Utf8 });

        let item = graph_item("doc-pii", bytes, None, sp, &Disclosure::FINDINGS, None).unwrap();
        assert!(item.graph.is_empty(), "graph must be omitted when PII full_text is off");
        assert!(item.typed.is_some(), "findings projection must still be carried");
        assert!(item.search_projection.is_none(), "search_projection must be omitted without full_text");

        let exported = export_document_response(&doc, &Disclosure::FINDINGS).unwrap();
        let exp_item = exported_item("doc-pii", exported, &doc, None, sp, &Disclosure::FINDINGS, None).unwrap();
        assert!(exp_item.graph.is_empty());
        assert!(exp_item.typed.is_some());
        assert!(exp_item.search_projection.is_none());
    }
}
