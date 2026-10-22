//! DocumentGraphResult: a whole document as binary DocumentGraph
//! bytes, answered with the same document imported and re-exported
//! plus every view and layer's standard id, provenance and status
//! (Ok, Skipped, Failed), so a client sees what ran without
//! decoding the graph itself. Layers whose codecs are not in the
//! server's registry are refused by name, never dropped silently.

use turbo_analysis::export::{DocumentGraph, LayerRecord, StatusRecord, ViewRecord, export, import};
use turbo_analysis_layers::export::codecs;

use super::{MAX_GRAPH_BYTES, model_to_proto, over};
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::proto::result::{DocumentGraphRequest, DocumentGraphResponse, LayerInfo, LayerWireStatus, ViewInfo};

fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

fn status(status: &StatusRecord) -> (LayerWireStatus, String) {
    match status {
        StatusRecord::Complete => (LayerWireStatus::Ok, String::new()),
        StatusRecord::Skipped(why) => (LayerWireStatus::Skipped, why.clone()),
        StatusRecord::Failed(error) => (LayerWireStatus::Failed, format!("{error:?}")),
    }
}

fn layer_info(record: &LayerRecord) -> LayerInfo {
    let (status, status_detail) = status(&record.status);
    LayerInfo {
        layer: record.layer.clone(),
        instance: record.instance.clone(),
        status: status as i32,
        status_detail,
        annotator: record.provenance.annotator.clone(),
        version: record.provenance.version.clone(),
        model: record.provenance.model.as_ref().map(model_to_proto),
        config_hash: record.provenance.config,
        items: record.len,
    }
}

fn view_info(record: &ViewRecord) -> ViewInfo {
    let (status, status_detail) = status(&record.status);
    ViewInfo {
        instance: record.instance.clone(),
        status: status as i32,
        status_detail,
        annotator: record.provenance.annotator.clone(),
        parent: record.parent.clone().unwrap_or_default(),
    }
}

/// Imports the request's graph, re-exports it, and lists its
/// layers and views from the re-exported records, so the listing
/// always describes the bytes in the same answer.
pub fn round_trip_document(req: &DocumentGraphRequest) -> Result<DocumentGraphResponse, Failure> {
    if req.graph.len() as u64 > MAX_GRAPH_BYTES {
        return Err(over("max_graph_bytes", "document graph bytes", req.graph.len() as u64));
    }
    let codecs = codecs();
    let graph = DocumentGraph::from_bytes(&req.graph).map_err(|e| bad(format!("document graph: {e}")))?;
    let doc = import(&graph, &codecs).map_err(|e| bad(format!("document graph import: {e}")))?;
    let out = export(&doc, &codecs).map_err(|e| bad(format!("document graph export: {e}")))?;
    let bytes = out.to_bytes().map_err(|e| bad(format!("document graph encode: {e}")))?;
    Ok(DocumentGraphResponse {
        graph: bytes,
        layers: out.layers.iter().map(layer_info).collect(),
        views: out.views.iter().map(view_info).collect(),
        source_text: out.source.text.clone(),
    })
}
