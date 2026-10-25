//! The page's AnalysisService calls (#1224). Each route is one gRPC call,
//! and its JSON is the proto message itself (build.rs), so the demo holds
//! no state and no analysis of its own. A failed call answers with the
//! status and its `turbo-code` and `turbo-field` trailers, as every route
//! does (`failure`).

use std::convert::Infallible;

use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::analysis::analyze_stream_request::Kind;
use crate::analysis::{
    AnalyzeDocument, AnalyzeStreamHeader, AnalyzeStreamRequest, CapabilitiesRequest, LoadBundleRequest,
    UnloadBundleRequest,
};
use crate::{App, error_body, failure, refused};

/// Answers to one stream the demo holds before the page reads them.
const STREAM_BUFFER: usize = 16;

fn answer<T: Serialize>(r: Result<tonic::Response<T>, tonic::Status>) -> Response {
    match r {
        Ok(r) => Json(r.into_inner()).into_response(),
        Err(e) => failure(e),
    }
}

/// `GET /api/capabilities`: Capabilities.
pub(crate) async fn capabilities(State(app): State<App>) -> Response {
    answer(app.analysis.clone().capabilities(CapabilitiesRequest {}).await)
}

/// `POST /api/bundles/load`: LoadBundle, an admin call the server refuses
/// unless it serves them.
pub(crate) async fn load(State(app): State<App>, body: Result<Json<LoadBundleRequest>, JsonRejection>) -> Response {
    match body {
        Ok(Json(req)) => answer(app.analysis.clone().load_bundle(req).await),
        Err(e) => refused(e.body_text()),
    }
}

/// `POST /api/bundles/unload`: UnloadBundle, an admin call too.
pub(crate) async fn unload(State(app): State<App>, body: Result<Json<UnloadBundleRequest>, JsonRejection>) -> Response {
    match body {
        Ok(Json(req)) => answer(app.analysis.clone().unload_bundle(req).await),
        Err(e) => refused(e.body_text()),
    }
}

/// The body of `POST /api/analyze/stream`: the stream's header and its
/// documents.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Analyze {
    #[serde(default)]
    header: AnalyzeStreamHeader,
    documents: Vec<AnalyzeDocument>,
}

/// `POST /api/analyze/stream`: one AnalyzeStream call with the header and
/// the documents, its answers passed on as they come, one JSON line each
/// (`application/x-ndjson`), or one server-sent event each when the page
/// accepts `text/event-stream`. A status that ends the stream after it
/// started is the last line, `{"error": ...}`, or an `error` event.
pub(crate) async fn analyze_stream(
    State(app): State<App>,
    headers: HeaderMap,
    body: Result<Json<Analyze>, JsonRejection>,
) -> Response {
    let req = match body {
        Ok(Json(b)) => b,
        Err(e) => return refused(e.body_text()),
    };
    let Some(sse) = crate::accept::stream_sse(&headers) else {
        return (
            StatusCode::NOT_ACCEPTABLE,
            Json(serde_json::json!({ "error": {
                "grpc": null,
                "message": "Accept must allow application/x-ndjson or text/event-stream"
            }})),
        )
            .into_response();
    };
    let frames: Vec<AnalyzeStreamRequest> = std::iter::once(Kind::Header(req.header))
        .chain(req.documents.into_iter().map(Kind::Document))
        .map(|k| AnalyzeStreamRequest { kind: Some(k) })
        .collect();
    let mut answers = match app.analysis.clone().analyze_stream(tokio_stream::iter(frames)).await {
        Ok(r) => r.into_inner(),
        Err(e) => return failure(e),
    };
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(STREAM_BUFFER);
    tokio::spawn(async move {
        loop {
            let line = match answers.message().await {
                Ok(Some(a)) => match serde_json::to_vec(&a) {
                    Ok(json) => frame(sse, false, json),
                    Err(e) => frame(sse, true, local_error(e.to_string())),
                },
                Ok(None) => return,
                Err(e) => {
                    let _ = tx.send(Ok(frame(sse, true, error_json(&e)))).await;
                    return;
                }
            };
            // A page that went away drops the receiver, and returning
            // drops the call.
            if tx.send(Ok(line)).await.is_err() {
                return;
            }
        }
    });
    let kind = if sse { "text/event-stream" } else { "application/x-ndjson" };
    ([(header::CONTENT_TYPE, kind), (header::CACHE_CONTROL, "no-cache")], Body::from_stream(ReceiverStream::new(rx)))
        .into_response()
}

fn error_json(s: &tonic::Status) -> Vec<u8> {
    serde_json::to_vec(&error_body(s)).unwrap_or_default()
}

fn local_error(message: String) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({ "error": { "grpc": null, "message": message } })).unwrap_or_default()
}

/// One answer as a line or an event. Compact JSON has no raw newline, so
/// one `data:` line holds it.
fn frame(sse: bool, error: bool, json: Vec<u8>) -> Bytes {
    let mut out = Vec::with_capacity(json.len() + 24);
    if sse {
        if error {
            out.extend_from_slice(b"event: error\n");
        }
        out.extend_from_slice(b"data: ");
        out.extend_from_slice(&json);
        out.extend_from_slice(b"\n\n");
    } else {
        out.extend_from_slice(&json);
        out.push(b'\n');
    }
    Bytes::from(out)
}
