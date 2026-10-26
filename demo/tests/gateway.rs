//! The demo's gateway routes in front of the real server: the tiny
//! bundle on the CPU, the demo calling it over gRPC, and these tests
//! calling the demo over HTTP as a page does. The AnalysisService stream
//! and bundle routes are here; the other routes are in gateway/.

#[path = "gateway/analysis.rs"]
mod analysis;
#[path = "gateway/catalog.rs"]
mod catalog;
#[path = "gateway/search.rs"]
mod search;
#[path = "gateway/vocabulary.rs"]
mod vocabulary;

use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use turbo_kserve::api::Runtime;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES, ModelConfig};
use turbo_kserve::{Serve, Server};
use turbo_kserve_demo::Demo;

const NAME: &str = "tiny-bert-bundle";

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle").canonicalize().unwrap()
}

fn cpu() -> u32 {
    let rt = Runtime::create().unwrap();
    (0..16).find(|&i| rt.device_info(i).map(|d| d.kind == turbo::TURBO_DEVICE_CPU).unwrap_or(false)).expect("a CPU")
}

async fn start(admin: bool) -> (Server, Demo) {
    start_with(Serve { admin, ..Serve::default() }).await
}

/// The server with the tiny bundle loaded and `serve`'s services, and the
/// demo in front of it.
async fn start_with(serve: Serve) -> (Server, Demo) {
    let model = ModelConfig::parse(&format!("bundle={},device={},sessions=1", tiny().display(), cpu())).unwrap();
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model], DEFAULT_MAX_MESSAGE_BYTES, serve).await.unwrap();
    s.load().await.unwrap();
    let d = Demo::start("127.0.0.1:0".parse().unwrap(), &format!("http://{}", s.local_addr()), vec![]).await.unwrap();
    (s, d)
}

/// One HTTP/1.1 request: the status, the content type and the body, a
/// chunked body joined.
async fn http(d: &Demo, method: &str, path: &str, accept: &str, body: &str) -> (u16, String, String) {
    let mut c = TcpStream::connect(d.local_addr()).await.unwrap();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: demo\r\nConnection: close\r\nAccept: {accept}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    c.write_all(req.as_bytes()).await.unwrap();
    let mut out = Vec::new();
    c.read_to_end(&mut out).await.unwrap();
    let out = String::from_utf8(out).unwrap();
    let (head, mut rest) = out.split_once("\r\n\r\n").unwrap();
    let head = head.to_ascii_lowercase();
    let status = head.split(' ').nth(1).unwrap().parse().unwrap();
    let kind = head.lines().find_map(|l| l.strip_prefix("content-type: ").map(str::to_string)).unwrap_or_default();
    if !head.contains("transfer-encoding: chunked") {
        return (status, kind, rest.to_string());
    }
    let mut body = String::new();
    loop {
        let (size, after) = rest.split_once("\r\n").unwrap();
        let n = usize::from_str_radix(size.trim(), 16).unwrap();
        if n == 0 {
            return (status, kind, body);
        }
        body.push_str(&after[..n]);
        rest = &after[n + 2..];
    }
}

async fn call(d: &Demo, method: &str, path: &str, body: Value) -> (u16, Value) {
    let (status, kind, body) = http(d, method, path, "application/json", &body.to_string()).await;
    assert_eq!(kind, "application/json", "{path}: {body}");
    (status, serde_json::from_str(&body).unwrap())
}

/// A call that must answer 200 with JSON.
async fn ok(d: &Demo, method: &str, path: &str, body: Value) -> Value {
    let (status, v) = call(d, method, path, body).await;
    assert_eq!(status, 200, "{method} {path}: {v}");
    v
}

/// A call that must fail with the gRPC `code` and carry a `turbo-code`.
async fn refused(d: &Demo, method: &str, path: &str, body: Value, code: &str) -> Value {
    let (status, e) = call(d, method, path, body).await;
    assert_eq!((status, &e["error"]["grpc"]), (502, &json!(code)), "{method} {path}: {e}");
    assert!(e["error"]["turbo_code"].as_str().unwrap().parse::<u32>().unwrap() > 0, "{e}");
    e
}

/// A streamed call refused before its first message: a JSON error with
/// the gRPC `code` and a `turbo-code`.
async fn stream_refused(d: &Demo, path: &str, body: Value, code: &str) -> Value {
    let (status, kind, text) = http(d, "POST", path, "application/x-ndjson", &body.to_string()).await;
    assert_eq!((status, kind.as_str()), (502, "application/json"), "{path}: {text}");
    let e: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(e["error"]["grpc"], code, "{path}: {e}");
    assert!(e["error"]["turbo_code"].as_str().unwrap().parse::<u32>().unwrap() > 0, "{e}");
    e
}

/// A streamed answer as NDJSON lines, each parsed.
async fn lines(d: &Demo, path: &str, body: Value) -> Vec<Value> {
    let (status, kind, text) = http(d, "POST", path, "application/x-ndjson", &body.to_string()).await;
    assert_eq!((status, kind.as_str()), (200, "application/x-ndjson"), "{path}: {text}");
    text.lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

/// A scratch directory for one test.
fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("demo-gateway-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Standard base64 with padding, as the protobuf JSON mapping writes bytes.
fn base64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for g in bytes.chunks(3) {
        let n = g.iter().enumerate().fold(0u32, |n, (i, &v)| n | (v as u32) << (16 - 8 * i));
        for i in 0..4 {
            out.push(if i <= g.len() { A[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

fn unbase64(text: &str) -> Vec<u8> {
    let mut out = Vec::new();
    let (mut n, mut bits) = (0u32, 0);
    for c in text.trim_end_matches('=').bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            _ => 63,
        };
        n = n << 6 | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((n >> bits) as u8);
            n &= (1 << bits) - 1;
        }
    }
    out
}

fn stream_body() -> Value {
    json!({ "header": { "order": 0 }, "documents": [
        { "id": "a", "model": NAME, "text": "a short text to embed" },
        { "id": "b", "model": "no-such-model", "text": "never analyzed" },
    ]})
}

#[tokio::test]
async fn capabilities_and_a_stream_come_back_as_the_proto_messages() {
    let (s, d) = start(false).await;
    let (status, caps) = call(&d, "GET", "/api/capabilities", Value::Null).await;
    assert_eq!(status, 200, "{caps}");
    assert_eq!(caps["bundles"][0]["name"], NAME, "{caps}");
    assert!(caps["server"]["stream"]["max_inflight_items_per_stream"].as_u64().unwrap() > 0, "{caps}");

    let (status, kind, body) = http(&d, "POST", "/api/analyze/stream", "*/*", &stream_body().to_string()).await;
    assert_eq!((status, kind.as_str()), (200, "application/x-ndjson"), "{body}");
    let lines: Vec<Value> = body.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines.len(), 2, "{body}");
    assert_eq!((&lines[0]["id"], &lines[1]["id"]), (&json!("a"), &json!("b")));
    let vectors = &lines[0]["outcome"]["vectors"];
    let data = vectors["data"].as_array().unwrap();
    assert_eq!(data.len() as u64, vectors["dim"].as_u64().unwrap());
    assert!(lines[1]["outcome"]["failure"]["turbo_code"].as_i64().unwrap() > 0, "{body}");

    // The same vector as /api/infer answers for the text.
    let (status, inferred) =
        call(&d, "POST", "/api/infer", json!({ "model": NAME, "texts": ["a short text to embed"] })).await;
    assert_eq!(status, 200);
    let f32s = |v: &Value| v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<_>>();
    assert_eq!(f32s(&inferred["vectors"][0]), f32s(&vectors["data"]));

    let (status, kind, body) =
        http(&d, "POST", "/api/analyze/stream", "text/event-stream", &stream_body().to_string()).await;
    assert_eq!((status, kind.as_str()), (200, "text/event-stream"), "{body}");
    let events: Vec<&str> = body.split("\n\n").filter(|e| !e.is_empty()).collect();
    assert_eq!(events.len(), 2, "{body}");
    let first: Value = serde_json::from_str(events[0].strip_prefix("data: ").unwrap()).unwrap();
    assert_eq!(first, lines[0]);
    d.stop().await;
    s.stop().await;
}

#[tokio::test]
async fn refusals_carry_the_turbo_trailers() {
    let (s, d) = start(false).await;
    // A header over the server's cap fails the call before it streams.
    let over = json!({ "header": { "max_inflight": u32::MAX }, "documents": [] });
    // Accept a stream representation so this exercises the gRPC refusal,
    // rather than the gateway's earlier HTTP format negotiation refusal.
    let (status, kind, body) = http(&d, "POST", "/api/analyze/stream", "*/*", &over.to_string()).await;
    assert_eq!(kind, "application/json");
    let e: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(status, 502, "{e}");
    assert_eq!(e["error"]["grpc"], "InvalidArgument", "{e}");
    assert!(e["error"]["message"].as_str().unwrap().contains("max_inflight"), "{e}");
    assert!(e["error"]["turbo_code"].as_str().unwrap().parse::<u32>().unwrap() > 0, "{e}");
    assert!(e["error"]["turbo_field"].is_string(), "{e}");
    // The admin calls on a server without them.
    let (status, e) = call(&d, "POST", "/api/bundles/unload", json!({ "name": NAME })).await;
    assert_eq!((status, &e["error"]["grpc"]), (502, &json!("Unimplemented")), "{e}");
    assert!(e["error"]["turbo_code"].is_string(), "{e}");
    // A body that is not the message is refused by the demo itself.
    let (status, e) = call(&d, "POST", "/api/analyze/stream", json!({ "documents": [], "extra": 1 })).await;
    assert_eq!((status, &e["error"]["grpc"]), (400, &Value::Null), "{e}");
    d.stop().await;
    s.stop().await;
}

#[tokio::test]
async fn bundles_unload_and_load_through_the_admin_calls() {
    let (s, d) = start(true).await;
    let (status, r) = call(&d, "POST", "/api/bundles/unload", json!({ "name": NAME })).await;
    assert_eq!((status, r), (200, json!({})));
    let (status, e) = call(&d, "POST", "/api/bundles/unload", json!({ "name": NAME })).await;
    assert_eq!((status, &e["error"]["grpc"]), (502, &json!("NotFound")), "{e}");
    let load = json!({ "name": NAME, "bundle": tiny().to_str().unwrap(), "device": cpu(), "sessions": 1 });
    let (status, r) = call(&d, "POST", "/api/bundles/load", load).await;
    assert_eq!(status, 200, "{r}");
    assert_eq!(r["capabilities"]["name"], NAME, "{r}");
    d.stop().await;
    s.stop().await;
}

#[tokio::test]
async fn stream_format_respects_accept_ranges_and_quality() {
    let (s, d) = start(false).await;
    let cases = [
        ("text/event-stream;q=0, application/x-ndjson", 200, "application/x-ndjson"),
        ("text/event-stream;q=0.1, application/x-ndjson;q=0.9", 200, "application/x-ndjson"),
        ("text/event-stream-extra, application/x-ndjson", 200, "application/x-ndjson"),
        ("TEXT/EVENT-STREAM", 200, "text/event-stream"),
        ("text/event-stream;q=0, */*;q=1", 200, "application/x-ndjson"),
        ("text/*;q=0.9, application/*;q=0.1", 200, "text/event-stream"),
        ("text/event-stream;q=1, application/x-ndjson;q=0.1", 200, "text/event-stream"),
        ("text/event-stream;q=0, application/x-ndjson;q=0", 406, "application/json"),
        ("text/event-stream;", 200, "text/event-stream"),
        ("text/event-stream;;q=1", 200, "text/event-stream"),
        ("application/x-ndjson;", 200, "application/x-ndjson"),
        (r#"text/event-stream;charset="utf\-8""#, 200, "text/event-stream"),
        (r#"application/x-ndjson;charset="UTF\-8";"#, 200, "application/x-ndjson"),
        ("application/x-ndjson;profile=\"é\", text/event-stream", 200, "text/event-stream"),
        ("application/x-ndjson;q=0\r\nAccept: text/event-stream;;", 200, "text/event-stream"),
        (
            r#"application/x-ndjson;profile="a,text/event-stream,b", application/x-ndjson;q=0.4"#,
            200,
            "application/x-ndjson",
        ),
    ];
    let mut failures = Vec::new();
    for (accept, expected_status, expected_kind) in cases {
        let (status, kind, body) = http(&d, "POST", "/api/analyze/stream", accept, &stream_body().to_string()).await;
        if status != expected_status || kind != expected_kind {
            failures.push(format!("{accept}: got {status} {kind}, expected {expected_status} {expected_kind}"));
            continue;
        }
        if status == 200 {
            let first = if kind == "text/event-stream" {
                body.lines().next().unwrap().strip_prefix("data: ").unwrap()
            } else {
                body.lines().next().unwrap()
            };
            let answer: Value = serde_json::from_str(first).unwrap();
            assert_eq!(answer["id"], "a");
            assert!(answer["outcome"]["vectors"]["dim"].as_u64().unwrap() > 0);
        }
    }
    d.stop().await;
    s.stop().await;
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[tokio::test]
async fn unacceptable_stream_format_is_refused_without_an_upstream_server() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let d = Demo::start("127.0.0.1:0".parse().unwrap(), &format!("http://{addr}"), vec![]).await.unwrap();
    for accept in [
        "*/*;q=0",
        r#"text/event-stream;charset="u\tf-8";q=0"#,
        r#"text/event-stream;charset="utf\,8""#,
        r#"application/x-ndjson;profile="é", text/event-stream;q=0"#,
    ] {
        let (status, kind, body) = http(&d, "POST", "/api/analyze/stream", accept, r#"{"documents":[]}"#).await;
        assert_eq!((status, kind.as_str()), (406, "application/json"), "{accept}: {body}");
        assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["error"]["grpc"], Value::Null);
    }
    d.stop().await;
}
