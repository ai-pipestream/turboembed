//! The Connect protocol with the binary codec only: a unary call is a
//! POST of `application/proto` answered with the message or an error
//! body, a server-streaming call a POST of `application/connect+proto`
//! answered with enveloped messages and an end-of-stream envelope. The
//! messages go to the server and back as bytes (codec.rs); only the
//! protocol's own error body and end-of-stream envelope are JSON, by the
//! protocol's definition, and they carry a code, a message and the
//! server's `turbo-*` metadata, never a message.

use std::convert::Infallible;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use bytes::{BufMut, Bytes, BytesMut};
use futures_util::StreamExt;
use tonic::Code;
use tonic::client::Grpc;
use tonic::codec::Streaming;
use tonic::codegen::http::uri::PathAndQuery;
use tonic::metadata::{KeyAndValueRef, MetadataMap};

use crate::budget::Budget;
use crate::codec::Raw;
use crate::{App, MAX_MESSAGE_BYTES};

pub const UNARY: &str = "application/proto";
pub const STREAM: &str = "application/connect+proto";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Protocol {
    Unary,
    Stream,
}

/// A refusal before or instead of the server's answer.
struct Refusal(Code, String);

/// One Connect call, forwarded to the server when its method is allowed.
pub(crate) async fn call(State(app): State<App>, req: Request) -> Response {
    let started = tokio::time::Instant::now();
    let (parts, body) = req.into_parts();
    let Some(protocol) = protocol(&parts.headers) else { return unsupported() };
    let path = parts.uri.path();
    let method = match app.allow.get(path) {
        Ok(m) => m,
        Err(true) => {
            let msg = format!("{path} is not on the demo's allowlist");
            return refuse(protocol, Refusal(Code::PermissionDenied, msg));
        }
        Err(false) => {
            let msg = format!("{path} is not a method the demo forwards");
            return refuse(protocol, Refusal(Code::Unimplemented, msg));
        }
    };
    // A server-streaming method answers in envelopes, which the unary
    // protocol has no room for.
    if protocol == Protocol::Unary && method.streaming {
        return unsupported();
    }
    let budget = match Budget::new(&parts.headers, started) {
        Ok(b) => b,
        Err(s) => return refuse(protocol, Refusal(s.code(), s.message().into())),
    };
    let response = async {
        let mut request = match request(&parts.headers, protocol, body).await {
            Ok(r) => r,
            Err(r) => return refuse(protocol, r),
        };
        let mut grpc = Grpc::new(app.channel.clone())
            .max_decoding_message_size(MAX_MESSAGE_BYTES)
            .max_encoding_message_size(MAX_MESSAGE_BYTES);
        if let Err(e) = grpc.ready().await {
            return refuse(protocol, Refusal(Code::Unavailable, format!("the server: {e}")));
        }
        match budget.remaining() {
            Ok(Some(time)) => request.set_timeout(time),
            Ok(None) => {}
            Err(s) => return refuse(protocol, Refusal(s.code(), s.message().into())),
        }
        let path = PathAndQuery::from_static(method.path);
        match protocol {
            Protocol::Unary => match grpc.unary(request, path, Raw).await {
                Ok(r) => ([(header::CONTENT_TYPE, UNARY)], r.into_inner()).into_response(),
                Err(s) => unary_error(s.code(), s.message(), s.metadata()),
            },
            Protocol::Stream => match grpc.server_streaming(request, path, Raw).await {
                Ok(r) => stream(r.into_inner(), budget),
                Err(s) => stream_reply(Body::from(end(Some((s.code(), s.message())), s.metadata()))),
            },
        }
    };
    match budget.wait(response).await {
        Ok(r) => r,
        Err(s) => refuse(protocol, Refusal(s.code(), s.message().into())),
    }
}

fn protocol(headers: &HeaderMap) -> Option<Protocol> {
    let kind = headers.get(header::CONTENT_TYPE)?.to_str().ok()?;
    let kind = kind.split(';').next()?.trim();
    if kind.eq_ignore_ascii_case(UNARY) {
        Some(Protocol::Unary)
    } else if kind.eq_ignore_ascii_case(STREAM) {
        Some(Protocol::Stream)
    } else {
        None
    }
}

/// 415 for JSON and every other content type, naming the two accepted.
fn unsupported() -> Response {
    let accept = HeaderValue::from_static("application/proto, application/connect+proto");
    (StatusCode::UNSUPPORTED_MEDIA_TYPE, [(HeaderName::from_static("accept-post"), accept)]).into_response()
}

/// The request message, read whole up to the message limit, with the
/// call's deadline. Compression is not offered, so a compressed request is
/// refused.
async fn request(headers: &HeaderMap, protocol: Protocol, body: Body) -> Result<tonic::Request<Bytes>, Refusal> {
    let encoding = if protocol == Protocol::Unary { "content-encoding" } else { "connect-content-encoding" };
    if headers.get(encoding).is_some_and(|v| v.as_bytes() != b"identity") {
        return Err(Refusal(Code::Unimplemented, "the demo accepts no compressed requests".into()));
    }
    let framing = if protocol == Protocol::Stream { 5 } else { 0 };
    let body = read(body, MAX_MESSAGE_BYTES + framing).await?;
    let message = if protocol == Protocol::Stream { unenvelope(body)? } else { body };
    Ok(tonic::Request::new(message))
}

async fn read(body: Body, limit: usize) -> Result<Bytes, Refusal> {
    let mut out = BytesMut::new();
    let mut data = body.into_data_stream();
    while let Some(chunk) = data.next().await {
        let chunk = chunk.map_err(|e| Refusal(Code::InvalidArgument, format!("the request body: {e}")))?;
        if out.len() + chunk.len() > limit {
            return Err(Refusal(Code::ResourceExhausted, format!("the request is over {limit} bytes")));
        }
        out.extend_from_slice(&chunk);
    }
    Ok(out.freeze())
}

/// The one message of a server-streaming request.
fn unenvelope(body: Bytes) -> Result<Bytes, Refusal> {
    let bad = |m: &str| Err(Refusal(Code::InvalidArgument, m.into()));
    if body.len() < 5 {
        return bad("a streaming request is one enveloped message");
    }
    let len = u32::from_be_bytes([body[1], body[2], body[3], body[4]]) as usize;
    match body[0] {
        0 => {}
        1 => return Err(Refusal(Code::Unimplemented, "the demo accepts no compressed messages".into())),
        _ => return bad("a streaming request's envelope has flags other than 0"),
    }
    if body.len() - 5 != len {
        return bad("a streaming request is one enveloped message");
    }
    Ok(body.slice(5..))
}

fn refuse(protocol: Protocol, Refusal(code, message): Refusal) -> Response {
    match protocol {
        Protocol::Unary => unary_error(code, &message, &MetadataMap::new()),
        Protocol::Stream => stream_reply(Body::from(end(Some((code, &message)), &MetadataMap::new()))),
    }
}

/// A unary error: the HTTP status for the code, the error body, and the
/// server's `turbo-*` metadata as headers, where Connect clients read a
/// unary error's metadata.
fn unary_error(code: Code, message: &str, meta: &MetadataMap) -> Response {
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    for (k, v) in turbo(meta) {
        if let (Ok(k), Ok(v)) = (HeaderName::try_from(k), HeaderValue::try_from(v)) {
            headers.append(k, v);
        }
    }
    (http_status(code), headers, error(code, message)).into_response()
}

fn stream_reply(body: Body) -> Response {
    ([(header::CONTENT_TYPE, STREAM)], body).into_response()
}

/// The server's messages as they come, each in an envelope, then the
/// end-of-stream envelope with the call's status and trailers. When the
/// page goes away the body is dropped, and with it the call to the server.
fn stream(messages: Streaming<Bytes>, budget: Budget) -> Response {
    let frames = futures_util::stream::unfold(Some(messages), move |state| async move {
        let mut messages = state?;
        let frame = match budget.wait(messages.message()).await.and_then(|r| r) {
            Ok(Some(m)) => return Some((Ok::<Bytes, Infallible>(envelope(0, &m)), Some(messages))),
            Ok(None) => match budget.wait(messages.trailers()).await.and_then(|r| r) {
                Ok(t) => end(None, &t.unwrap_or_default()),
                Err(s) => end(Some((s.code(), s.message())), s.metadata()),
            },
            Err(s) => end(Some((s.code(), s.message())), s.metadata()),
        };
        Some((Ok(frame), None))
    });
    stream_reply(Body::from_stream(frames))
}

fn envelope(flags: u8, payload: &[u8]) -> Bytes {
    let mut b = BytesMut::with_capacity(5 + payload.len());
    b.put_u8(flags);
    b.put_u32(payload.len() as u32);
    b.put_slice(payload);
    b.freeze()
}

/// The end-of-stream envelope: `{"error": ..., "metadata": ...}`, each
/// part present when there is one.
fn end(status: Option<(Code, &str)>, meta: &MetadataMap) -> Bytes {
    let mut parts = Vec::new();
    if let Some((code, message)) = status {
        parts.push(format!("\"error\":{}", error(code, message)));
    }
    let mut keys: Vec<(String, Vec<String>)> = Vec::new();
    for (k, v) in turbo(meta) {
        match keys.iter_mut().find(|(key, _)| *key == k) {
            Some((_, values)) => values.push(quote(&v)),
            None => keys.push((k, vec![quote(&v)])),
        }
    }
    if !keys.is_empty() {
        let fields: Vec<String> = keys.iter().map(|(k, v)| format!("{}:[{}]", quote(k), v.join(","))).collect();
        parts.push(format!("\"metadata\":{{{}}}", fields.join(",")));
    }
    envelope(2, format!("{{{}}}", parts.join(",")).as_bytes())
}

/// The server's `turbo-*` metadata (`turbo-code`, `turbo-field`), in order.
fn turbo(meta: &MetadataMap) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for kv in meta.iter() {
        if let KeyAndValueRef::Ascii(k, v) = kv
            && k.as_str().starts_with("turbo-")
            && let Ok(v) = v.to_str()
        {
            out.push((k.as_str().to_owned(), v.to_owned()));
        }
    }
    out
}

/// The protocol's error object, `{"code": ..., "message": ...}`.
fn error(code: Code, message: &str) -> String {
    if message.is_empty() {
        format!("{{\"code\":\"{}\"}}", name(code))
    } else {
        format!("{{\"code\":\"{}\",\"message\":{}}}", name(code), quote(message))
    }
}

fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The Connect name of a gRPC code.
pub fn name(code: Code) -> &'static str {
    match code {
        Code::Cancelled => "canceled",
        Code::InvalidArgument => "invalid_argument",
        Code::DeadlineExceeded => "deadline_exceeded",
        Code::NotFound => "not_found",
        Code::AlreadyExists => "already_exists",
        Code::PermissionDenied => "permission_denied",
        Code::ResourceExhausted => "resource_exhausted",
        Code::FailedPrecondition => "failed_precondition",
        Code::Aborted => "aborted",
        Code::OutOfRange => "out_of_range",
        Code::Unimplemented => "unimplemented",
        Code::Internal => "internal",
        Code::Unavailable => "unavailable",
        Code::DataLoss => "data_loss",
        Code::Unauthenticated => "unauthenticated",
        Code::Ok | Code::Unknown => "unknown",
    }
}

/// The HTTP status of a unary error, by the Connect protocol's table.
fn http_status(code: Code) -> StatusCode {
    let n = match code {
        Code::Cancelled => 499,
        Code::InvalidArgument | Code::FailedPrecondition | Code::OutOfRange => 400,
        Code::DeadlineExceeded => 504,
        Code::NotFound => 404,
        Code::AlreadyExists | Code::Aborted => 409,
        Code::PermissionDenied => 403,
        Code::ResourceExhausted => 429,
        Code::Unimplemented => 501,
        Code::Unavailable => 503,
        Code::Unauthenticated => 401,
        Code::Ok | Code::Unknown | Code::Internal | Code::DataLoss => 500,
    };
    StatusCode::from_u16(n).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_end_envelope_quotes_its_text_and_carries_the_turbo_metadata() {
        let mut meta = MetadataMap::new();
        meta.insert("turbo-code", "1280".parse().unwrap());
        meta.insert("turbo-field", "2".parse().unwrap());
        meta.insert("other", "x".parse().unwrap());
        let e = end(Some((Code::NotFound, "no \"model\"\n")), &meta);
        assert_eq!(e[0], 2);
        assert_eq!(u32::from_be_bytes(e[1..5].try_into().unwrap()) as usize, e.len() - 5);
        assert_eq!(
            std::str::from_utf8(&e[5..]).unwrap(),
            r#"{"error":{"code":"not_found","message":"no \"model\"\u000a"},"metadata":{"turbo-code":["1280"],"turbo-field":["2"]}}"#
        );
        assert_eq!(&end(None, &MetadataMap::new())[5..], b"{}");
        assert_eq!(http_status(Code::ResourceExhausted), StatusCode::TOO_MANY_REQUESTS);
    }
}
