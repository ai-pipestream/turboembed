//! The page's calls to the server's gRPC services (#1224, #1360). Each
//! route is one gRPC call, and its JSON is the proto message itself
//! (build.rs), so the demo holds no state and does no analysis of its own.
//! A failed call answers with the status and its `turbo-code` and
//! `turbo-field` trailers (`crate::failure`). A call the server answers
//! with a stream is passed on as it comes, one JSON line each
//! (`application/x-ndjson`), or one server-sent event each when the page
//! accepts `text/event-stream` (`forward`).

use std::convert::Infallible;

use axum::Json;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::rejection::JsonRejection;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::{App, error_body, failure, refused};

/// A `POST` route whose body is the request message of a unary call, and
/// whose answer is the response message.
macro_rules! unary {
    ($(#[$doc:meta])* $name:ident: $client:ident.$rpc:ident($req:ty)) => {
        $(#[$doc])*
        pub(super) async fn $name(
            axum::extract::State(app): axum::extract::State<crate::App>,
            body: Result<axum::Json<$req>, axum::extract::rejection::JsonRejection>,
        ) -> axum::response::Response {
            match crate::gateway::body(body) {
                Ok(req) => crate::gateway::answer(app.$client.clone().$rpc(req).await),
                Err(refusal) => *refusal,
            }
        }
    };
}

/// A `GET` route for a unary call that is sent its default request.
macro_rules! listing {
    ($(#[$doc:meta])* $name:ident: $client:ident.$rpc:ident($req:ty)) => {
        $(#[$doc])*
        pub(super) async fn $name(
            axum::extract::State(app): axum::extract::State<crate::App>,
        ) -> axum::response::Response {
            crate::gateway::answer(app.$client.clone().$rpc(<$req>::default()).await)
        }
    };
}

/// A `POST` route whose body is the request message of a call the server
/// answers with a stream, passed on by `forward`.
macro_rules! streamed {
    ($(#[$doc:meta])* $name:ident: $client:ident.$rpc:ident($req:ty)) => {
        $(#[$doc])*
        pub(super) async fn $name(
            axum::extract::State(app): axum::extract::State<crate::App>,
            headers: axum::http::HeaderMap,
            body: Result<axum::Json<$req>, axum::extract::rejection::JsonRejection>,
        ) -> axum::response::Response {
            let req = match crate::gateway::body(body) {
                Ok(req) => req,
                Err(refusal) => return *refusal,
            };
            let sse = match crate::gateway::negotiate(&headers) {
                Ok(sse) => sse,
                Err(refusal) => return *refusal,
            };
            match app.$client.clone().$rpc(req).await {
                Ok(r) => crate::gateway::forward(sse, r.into_inner()),
                Err(e) => crate::failure(e),
            }
        }
    };
}

mod analysis;
mod catalog;
mod search;
mod vocabulary;

/// Answers to one stream the demo holds before the page reads them.
const STREAM_BUFFER: usize = 16;

/// The largest body of a route that sends a client stream: the server's
/// 256 MiB dictionary upload cap. Every other route keeps axum's 2 MB
/// default.
pub(super) const CLIENT_STREAM_BODY_BYTES: usize = 256 << 20;

/// A client-stream route with the larger body limit.
pub(super) fn large<H, T>(handler: H, limit: usize) -> axum::routing::MethodRouter<App>
where
    H: axum::handler::Handler<T, App>,
    T: 'static,
{
    axum::routing::post(handler).layer(axum::extract::DefaultBodyLimit::max(limit))
}

/// Every gateway route.
pub(crate) fn routes(r: Router<App>) -> Router<App> {
    vocabulary::routes(catalog::routes(search::routes(analysis::routes(r))))
}

pub(super) fn answer<T: Serialize>(r: Result<tonic::Response<T>, tonic::Status>) -> Response {
    match r {
        Ok(r) => Json(r.into_inner()).into_response(),
        Err(e) => failure(e),
    }
}

/// A body the demo could read, or its HTTP 400, or 413 when it is over
/// the route's limit; boxed as a response is large beside a request.
pub(super) fn body<T>(body: Result<Json<T>, JsonRejection>) -> Result<T, Box<Response>> {
    body.map(|Json(b)| b).map_err(|e| {
        let mut refusal = refused(e.body_text());
        if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
            *refusal.status_mut() = StatusCode::PAYLOAD_TOO_LARGE;
        }
        Box::new(refusal)
    })
}

/// Whether the answers go out as events, or the HTTP 406 when the page
/// accepts neither form; checked before any call is made.
pub(super) fn negotiate(headers: &HeaderMap) -> Result<bool, Box<Response>> {
    crate::accept::stream_sse(headers).ok_or_else(|| {
        let error = serde_json::json!({ "error": {
            "grpc": null,
            "message": "Accept must allow application/x-ndjson or text/event-stream"
        }});
        Box::new((StatusCode::NOT_ACCEPTABLE, Json(error)).into_response())
    })
}

/// Passes a stream's answers on as they come. A status that ends the
/// stream after it started is the last line, `{"error": ...}`, or an
/// `error` event. A page that goes away drops the call.
pub(super) fn forward<T>(sse: bool, mut answers: tonic::Streaming<T>) -> Response
where
    T: Serialize + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(STREAM_BUFFER);
    tokio::spawn(async move {
        loop {
            // A page that went away closes the receiver; an endless stream
            // such as a collection watch must not outlive it.
            let next = tokio::select! {
                next = answers.message() => next,
                () = tx.closed() => return,
            };
            let line = match next {
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
            // Returning drops the call.
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
