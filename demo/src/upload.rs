//! Uploads: `PUT /upload/<package>.<Service>/<Method>` with an
//! `application/octet-stream` body of frames (frames.rs), fed as it
//! arrives to the client-streaming RPC of that path, for a browser, which
//! cannot make a client-streaming call. Through the same gate as a Connect
//! call, and only when the method is allowed, which as a write it is not
//! by default. The answer is 200 and the RPC's reply message as
//! `application/proto`; or the error's HTTP status by the Connect table
//! (413 for a body over the streamed ceiling) with `grpc-status`,
//! `grpc-message` and the server's `turbo-*` metadata as headers and no
//! body.

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use tokio::sync::oneshot;
use tonic::Code;
use tonic::client::Grpc;
use tonic::codegen::http::uri::PathAndQuery;
use tonic::metadata::MetadataMap;

use crate::App;
use crate::codec::Raw;
use crate::connect::{UNARY, http_status, turbo};
use crate::frames::{Frames, Kind, Refusal};
use crate::limits::{MAX_REPLY_BYTES, MAX_REQUEST_BYTES, MAX_STREAM_BYTES};

/// The one content type an upload takes.
pub const OCTET_STREAM: &str = "application/octet-stream";

/// One upload, with CORS headers for a listed origin.
pub(crate) async fn call(State(app): State<App>, req: Request) -> Response {
    let origin = req.headers().get(header::ORIGIN).cloned();
    let mut reply = match forward(&app, req).await {
        Ok(r) => r,
        Err(f) => refuse(&f.0, &f.1),
    };
    app.gate.grant(origin.as_ref(), reply.headers_mut());
    reply
}

/// A refusal and the server's metadata, boxed: a `MetadataMap` is large.
type Failure = Box<(Refusal, MetadataMap)>;

fn fail(code: Code, message: impl Into<String>) -> Failure {
    Box::new((Refusal::new(code, message), MetadataMap::new()))
}

async fn forward(app: &App, req: Request) -> Result<Response, Failure> {
    let (parts, body) = req.into_parts();
    if !octet_stream(&parts.headers) {
        let accept = [(header::ACCEPT, HeaderValue::from_static(OCTET_STREAM))];
        return Ok((StatusCode::UNSUPPORTED_MEDIA_TYPE, accept).into_response());
    }
    app.gate.check(&parts.headers).map_err(|(code, msg)| fail(code, msg))?;
    let path = parts.uri.path().strip_prefix("/upload").unwrap_or_default().to_string();
    match app.allow.get(&path) {
        Ok(m) if m.upload => {}
        Err(true) if crate::allow::is_upload(&path) => {
            return Err(fail(Code::PermissionDenied, format!("{path} is not on the demo's allowlist")));
        }
        _ => return Err(fail(Code::Unimplemented, format!("{path} is not an upload the demo forwards"))),
    }
    let length = parts.headers.get(header::CONTENT_LENGTH).and_then(|v| v.to_str().ok()?.parse::<u64>().ok());
    if length.is_some_and(|n| n > MAX_STREAM_BYTES) {
        return Err(Box::new((Refusal::too_large(), MetadataMap::new())));
    }
    let mut grpc = Grpc::new(app.channel.clone())
        .max_decoding_message_size(MAX_REPLY_BYTES)
        .max_encoding_message_size(MAX_REQUEST_BYTES);
    if let Err(e) = grpc.ready().await {
        return Err(fail(Code::Unavailable, format!("the server: {e}")));
    }
    let (refused, refusal) = oneshot::channel();
    let messages = messages(Frames::new(body.into_data_stream(), Kind::of(&path)), refused);
    let rpc = PathAndQuery::try_from(path).map_err(|e| fail(Code::Internal, e.to_string()))?;
    // A refusal of the body ends the call without ending its stream, so
    // the server never takes a cut-short upload as whole.
    let call = grpc.client_streaming(tonic::Request::new(messages), rpc, Raw);
    tokio::select! {
        reply = call => match reply {
            Ok(r) => Ok(([(header::CONTENT_TYPE, UNARY)], r.into_inner()).into_response()),
            Err(s) => Err(Box::new((Refusal::new(s.code(), s.message()), s.metadata().clone()))),
        },
        Ok(r) = refusal => Err(Box::new((r, MetadataMap::new()))),
    }
}

/// The body's messages; on a refusal, the refusal sent to `refused` and
/// the stream left open until the call is dropped.
fn messages<B, E>(
    frames: Frames<B>,
    refused: oneshot::Sender<Refusal>,
) -> impl futures_util::Stream<Item = Bytes> + Send + 'static
where
    B: futures_util::Stream<Item = Result<Bytes, E>> + Unpin + Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    futures_util::stream::unfold((frames, Some(refused)), |(mut frames, mut refused)| async move {
        match frames.next().await {
            Ok(Some(m)) => Some((m, (frames, refused))),
            Ok(None) => None,
            Err(r) => {
                if let Some(tx) = refused.take() {
                    let _ = tx.send(r);
                }
                std::future::pending().await
            }
        }
    })
}

fn octet_stream(headers: &HeaderMap) -> bool {
    let kind = headers.get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("");
    kind.split(';').next().is_some_and(|k| k.trim().eq_ignore_ascii_case(OCTET_STREAM))
}

/// An upload's error: its status, `grpc-status`, `grpc-message` and the
/// server's `turbo-*` metadata, and no body.
fn refuse(r: &Refusal, meta: &MetadataMap) -> Response {
    let mut headers = HeaderMap::new();
    headers.insert("grpc-status", HeaderValue::from(r.code as i32));
    if let Ok(v) = HeaderValue::from_str(&percent(&r.message)) {
        headers.insert("grpc-message", v);
    }
    for (k, v) in turbo(meta) {
        if let (Ok(k), Ok(v)) = (HeaderName::try_from(k), HeaderValue::try_from(v)) {
            headers.append(k, v);
        }
    }
    let status = if r.too_large { StatusCode::PAYLOAD_TOO_LARGE } else { http_status(r.code) };
    (status, headers, Body::empty()).into_response()
}

/// `grpc-message` as gRPC sends it: printable ASCII but `%`, the rest
/// percent-encoded.
fn percent(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if (0x20..0x7f).contains(&b) && b != b'%' {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_go_in_headers_with_the_mapped_status() {
        assert_eq!(percent("a 100% b\n\u{e9}"), "a 100%25 b%0A%C3%A9");
        let r = refuse(&Refusal::too_large(), &MetadataMap::new());
        assert_eq!(r.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(r.headers()["grpc-status"], "8");
        let mut meta = MetadataMap::new();
        meta.insert("turbo-code", "256".parse().unwrap());
        let r = refuse(&Refusal::new(Code::InvalidArgument, "bad"), &meta);
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
        assert_eq!(r.headers()["turbo-code"], "256");
        let mut h = HeaderMap::new();
        h.insert(header::CONTENT_TYPE, HeaderValue::from_static("Application/Octet-Stream; x=y"));
        assert!(octet_stream(&h));
        h.insert(header::CONTENT_TYPE, HeaderValue::from_static("multipart/form-data; boundary=x"));
        assert!(!octet_stream(&h));
        assert_eq!(Kind::of("/turbo.vocabulary.v1.TurboVocabularyService/ImportDictionary"), Kind::HeaderThenRaw);
    }
}
