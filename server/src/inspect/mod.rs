//! Document inspection and chunk projection RPC handlers
//! (`InspectDocumentWindow` and `ProjectChunks`, issue #1220).

mod bindings;
pub mod heatmap;
pub mod pca;
pub mod project;
pub mod stacks;
pub mod window;
pub mod xray;

pub use project::project_chunks;

use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis::{Span, ViewId};

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::result::{DocumentShapeSummaryProto, DocumentWindowRequest, DocumentWindowResponse};
use crate::results::document::{codecs, layer_info, view_info};
use crate::results::{MAX_GRAPH_BYTES, over};

use heatmap::build_heatmap_lanes;
use window::{combined_segments, compute_window, document_chips};
use xray::{compute_xray, resolve_views};

fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

/// Executes `InspectDocumentWindow` on a binary [`DocumentGraph`], returning
/// the shape summary, paginated viewport window, sweep-line combined
/// elementary segments, document-level chips, multi-lane heatmaps, and
/// optional [`NormalizationXrayProto`] in the requested [`OffsetMode`].
pub fn inspect_document_window(req: &DocumentWindowRequest) -> Result<DocumentWindowResponse, Failure> {
    if req.graph.len() as u64 > MAX_GRAPH_BYTES {
        return Err(over("max_graph_bytes", "document graph bytes", req.graph.len() as u64));
    }
    if req.graph.is_empty() {
        return Err(bad("graph: document graph bytes are required"));
    }
    let offset_mode = OffsetMode::from_proto(req.offsets)?;
    if req.heatmap_buckets > heatmap::MAX_HEATMAP_BUCKETS {
        return Err(over("max_heatmap_buckets", "heatmap_buckets", req.heatmap_buckets as u64));
    }

    let graph = DocumentGraph::from_bytes(&req.graph).map_err(|e| bad(format!("document graph: {e}")))?;
    let doc = import(&graph, codecs()).map_err(|e| bad(format!("document graph import: {e}")))?;

    let complete_views = resolve_views(&graph)?;
    let view_count = complete_views.len() as u32;
    if req.view >= view_count {
        return Err(bad(format!("view {} is out of range for document with {view_count} view(s)", req.view)));
    }
    let view_id = ViewId(req.view as u16);
    let text = doc.text(view_id);
    let offsetter = Offsetter::new(text, offset_mode);

    let full_span = Span::new(0, text.len() as u32);
    let (_, total_units) = offsetter.to_wire(text, full_span);

    let summary = DocumentShapeSummaryProto {
        view: req.view,
        total_bytes: text.len() as u32,
        total_units,
        view_count: graph.views.len() as u32,
        layer_count: graph.layers.len() as u32,
        layers: graph.layers.iter().map(layer_info).collect(),
        views: graph.views.iter().map(view_info).collect(),
    };

    let (window_proto, window_span) = compute_window(text, req.view, &offsetter, req.page_size, req.page)?;
    let segments = combined_segments(&doc, view_id, text, window_span, &offsetter, &req.filter_layers)?;
    let chips = document_chips(&doc);
    let heatmap_lanes = build_heatmap_lanes(&doc, view_id, text, &offsetter, req.heatmap_buckets)?;
    let xray = match req.xray_target_view {
        Some(target_view) => Some(compute_xray(&graph, req.view, target_view, offset_mode)?),
        None => None,
    };

    Ok(DocumentWindowResponse {
        summary: Some(summary),
        window: Some(window_proto),
        segments,
        document_chips: chips,
        heatmap_lanes,
        xray,
    })
}
