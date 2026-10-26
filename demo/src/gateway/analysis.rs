//! The AnalysisService routes (#1224, #1360).

use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::HeaderMap;
use axum::response::Response;
use axum::routing::{get, post};
use serde::Deserialize;

use super::{answer, body, forward, negotiate};
use crate::analysis::analyze_stream_request::Kind;
use crate::analysis::{
    AnalyzeDocument, AnalyzePipelineRequest, AnalyzeStreamHeader, AnalyzeStreamRequest, CapabilitiesRequest,
    DocumentWindowRequest, FormatDocumentRequest, InspectDocumentGraphRequest, ListOutputFormatsRequest,
    LoadBundleRequest, ProjectChunksRequest, UnloadBundleRequest,
};
use crate::{App, failure};

pub(super) fn routes(r: Router<App>) -> Router<App> {
    r.route("/api/capabilities", get(capabilities))
        .route("/api/bundles/load", post(load))
        .route("/api/bundles/unload", post(unload))
        .route("/api/analyze/stream", post(analyze_stream))
        .route("/api/analyze/progressive", post(progressive))
        .route("/api/analyze/window", post(window))
        .route("/api/analyze/typed", post(typed))
        .route("/api/analyze/formats", get(formats))
        .route("/api/analyze/format", post(format))
        .route("/api/chunks/project", post(project))
}

/// `GET /api/capabilities`: Capabilities.
async fn capabilities(State(app): State<App>) -> Response {
    answer(app.analysis.clone().capabilities(CapabilitiesRequest {}).await)
}

unary!(
    /// `POST /api/bundles/load`: LoadBundle, an admin call the server
    /// refuses unless it serves them.
    load: analysis.load_bundle(LoadBundleRequest)
);
unary!(
    /// `POST /api/bundles/unload`: UnloadBundle, an admin call too.
    unload: analysis.unload_bundle(UnloadBundleRequest)
);
streamed!(
    /// `POST /api/analyze/progressive`: AnalyzeDocumentProgressive, each
    /// AnalyzeProgressiveEvent as it comes.
    progressive: analysis.analyze_document_progressive(AnalyzePipelineRequest)
);
unary!(
    /// `POST /api/analyze/window`: InspectDocumentWindow, one page of a
    /// graph's view.
    window: analysis.inspect_document_window(DocumentWindowRequest)
);
unary!(
    /// `POST /api/analyze/typed`: InspectDocumentGraph, a graph's typed
    /// annotations.
    typed: analysis.inspect_document_graph(InspectDocumentGraphRequest)
);
listing!(
    /// `GET /api/analyze/formats`: ListOutputFormats.
    formats: analysis.list_output_formats(ListOutputFormatsRequest)
);
unary!(
    /// `POST /api/analyze/format`: FormatDocument, a graph written in one
    /// output format.
    format: analysis.format_document(FormatDocumentRequest)
);
unary!(
    /// `POST /api/chunks/project`: ProjectChunks, a graph's chunk vectors
    /// projected for plotting.
    project: analysis.project_chunks(ProjectChunksRequest)
);

/// The body of `POST /api/analyze/stream`: the stream's header and its
/// documents.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Analyze {
    #[serde(default)]
    header: AnalyzeStreamHeader,
    documents: Vec<AnalyzeDocument>,
}

/// `POST /api/analyze/stream`: one AnalyzeStream call with the header and
/// the documents, its answers passed on as they come.
async fn analyze_stream(
    State(app): State<App>,
    headers: HeaderMap,
    req: Result<Json<Analyze>, JsonRejection>,
) -> Response {
    let req = match body(req) {
        Ok(b) => b,
        Err(refusal) => return *refusal,
    };
    let sse = match negotiate(&headers) {
        Ok(sse) => sse,
        Err(refusal) => return *refusal,
    };
    let frames: Vec<AnalyzeStreamRequest> = std::iter::once(Kind::Header(req.header))
        .chain(req.documents.into_iter().map(Kind::Document))
        .map(|k| AnalyzeStreamRequest { kind: Some(k) })
        .collect();
    match app.analysis.clone().analyze_stream(tokio_stream::iter(frames)).await {
        Ok(r) => forward(sse, r.into_inner()),
        Err(e) => failure(e),
    }
}
