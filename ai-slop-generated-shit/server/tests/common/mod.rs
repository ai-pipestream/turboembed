//! Shared harness: one engine over the six mock bundles, the axum router
//! driven in process, and the generated gRPC client over an ephemeral port.
//!
//! Mock ordinal 1 is the accelerator (ordinal 0 is the CPU), which is what
//! the device policy says a server should ask for by name.

#![allow(dead_code)]

use std::sync::{Arc, OnceLock};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use serde_json::Value;
use tonic::transport::{Channel, Server};
use tower::ServiceExt;

use turbo_inferstream::config::{Config, ModelSpec};
use turbo_inferstream::engine::Engine;
use turbo_inferstream::grpc;
use turbo_inferstream::grpc::ext::inferstream_extension_client::InferstreamExtensionClient;
use turbo_inferstream::grpc::inference::grpc_inference_service_client::GrpcInferenceServiceClient;

/// The extension service's generated client over a connected channel.
pub type ExtClient = InferstreamExtensionClient<Channel>;

/// Path of a mock bundle directory.
pub fn bundle(kind: &str) -> String {
    format!("{}/../testdata/bundles/mock/{kind}", env!("CARGO_MANIFEST_DIR"))
}

/// One `--model` flag, parsed, or a panic naming the flag.
pub fn spec(flag: &str) -> ModelSpec {
    Config::parse_model_flag(flag).unwrap_or_else(|e| panic!("--model {flag}: {e}"))
}

/// An engine over the given `--model` flags.
pub fn engine_of(flags: &[&str]) -> Arc<Engine> {
    let config = Config { provider_libs: Vec::new(), models: flags.iter().map(|f| spec(f)).collect(), pages: None };
    Engine::load(&config).unwrap_or_else(|e| panic!("engine over {flags:?}: {e}"))
}

/// The `--model` flags of the six mock bundles, under the names the tests use.
pub fn all_model_flags() -> Vec<String> {
    [
        ("embed", "embedding"),
        ("rerank", "reranker"),
        ("classify", "classifier"),
        ("tag", "token-classifier"),
        ("chat", "generative"),
        ("run", "generic"),
    ]
    .iter()
    .map(|(name, dir)| format!("name={name},bundle={},provider=mock,ordinal=1", bundle(dir)))
    .collect()
}

/// The shared engine over all six mock bundles. Built once per test binary;
/// every model keeps its default buckets and one session per bucket.
pub fn engine() -> Arc<Engine> {
    static ENGINE: OnceLock<Arc<Engine>> = OnceLock::new();
    ENGINE
        .get_or_init(|| {
            let flags = all_model_flags();
            engine_of(&flags.iter().map(String::as_str).collect::<Vec<_>>())
        })
        .clone()
}

/// The HTTP router over the shared engine.
pub fn app() -> Router {
    turbo_inferstream::http::router(engine())
}

/// A response reduced to what the assertions read.
pub struct Res {
    /// HTTP status.
    pub status: StatusCode,
    /// The `content-type` header, or an empty string.
    pub content_type: String,
    /// The whole body as text.
    pub text: String,
}

impl Res {
    /// The body parsed as JSON, or a panic naming the body.
    pub fn json(&self) -> Value {
        serde_json::from_str(&self.text)
            .unwrap_or_else(|e| panic!("body is not JSON ({e}); status {}, body: {}", self.status, self.text))
    }
}

async fn send(router: Router, request: Request<Body>) -> Res {
    let uri = request.uri().to_string();
    let response = router.oneshot(request).await.unwrap_or_else(|e| panic!("{uri}: {e}"));
    let status = response.status();
    let content_type = response.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    let bytes = response.into_body().collect().await.unwrap_or_else(|e| panic!("{uri}: body: {e}")).to_bytes();
    Res { status, content_type, text: String::from_utf8_lossy(&bytes).into_owned() }
}

/// `GET uri` against the shared engine's router.
pub async fn get(uri: &str) -> Res {
    get_on(app(), uri).await
}

/// `GET uri` against a given router.
pub async fn get_on(router: Router, uri: &str) -> Res {
    send(router, Request::builder().uri(uri).body(Body::empty()).expect("request")).await
}

/// `POST uri` with a JSON body against the shared engine's router.
pub async fn post(uri: &str, body: &Value) -> Res {
    post_on(app(), uri, body).await
}

/// `POST uri` with a JSON body against a given router.
pub async fn post_on(router: Router, uri: &str, body: &Value) -> Res {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    send(router, request).await
}

/// Serve `engine` over HTTP on an ephemeral port, for the cases that need a
/// real socket (a client that hangs up mid-stream). The server lives as long
/// as the calling test's runtime.
pub async fn serve_http(engine: Arc<Engine>) -> std::net::SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind an ephemeral port for HTTP");
    let addr = listener.local_addr().expect("the bound address");
    let app = turbo_inferstream::http::router(engine);
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("HTTP server");
    });
    addr
}

/// One HTTP/1.1 POST written by hand, so the test owns the socket and can
/// close it whenever it likes.
pub fn http_post_bytes(addr: std::net::SocketAddr, path: &str, body: &Value) -> Vec<u8> {
    let body = body.to_string();
    format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

/// Serve `engine` over gRPC on an ephemeral port: the Open Inference
/// Protocol service, Inferstream's extension service and reflection over
/// both, the same three the binary serves. The server lives as long as the
/// calling test's runtime.
pub async fn serve_grpc(engine: Arc<Engine>) -> std::net::SocketAddr {
    let incoming = tonic::transport::server::TcpIncoming::bind("127.0.0.1:0".parse().expect("addr"))
        .expect("bind an ephemeral port for the gRPC server");
    let addr = incoming.local_addr().expect("the bound address");
    let service = grpc::GrpcInferenceServiceServer::new(grpc::Service { engine: engine.clone() });
    let ext = grpc::InferstreamExtensionServer::new(grpc::ExtService { engine });
    let reflection = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(grpc::FILE_DESCRIPTOR_SET)
        .build_v1()
        .expect("the reflection service over the file descriptor set");
    tokio::spawn(async move {
        Server::builder()
            .add_service(service)
            .add_service(ext)
            .add_service(reflection)
            .serve_with_incoming(incoming)
            .await
            .expect("gRPC server");
    });
    addr
}

/// A channel to a gRPC server that may still be coming up.
pub async fn grpc_channel(addr: std::net::SocketAddr) -> Channel {
    let endpoint = format!("http://{addr}");
    for attempt in 0..50 {
        match Channel::from_shared(endpoint.clone()).expect("endpoint").connect().await {
            Ok(c) => return c,
            Err(e) if attempt == 49 => panic!("cannot connect to {endpoint}: {e}"),
            Err(_) => tokio::time::sleep(std::time::Duration::from_millis(10)).await,
        }
    }
    unreachable!("the loop returns or panics")
}

/// Serve `engine` over gRPC on an ephemeral port and connect the generated
/// client to it. The server lives as long as the calling test's runtime.
pub async fn grpc_client(engine: Arc<Engine>) -> GrpcInferenceServiceClient<Channel> {
    let addr = serve_grpc(engine).await;
    GrpcInferenceServiceClient::new(grpc_channel(addr).await)
}

/// The extension service's client over its own server.
pub async fn ext_client(engine: Arc<Engine>) -> ExtClient {
    let addr = serve_grpc(engine).await;
    ExtClient::new(grpc_channel(addr).await)
}

/// The SSE `data:` payloads of a stream body, in order.
pub fn sse_data(text: &str) -> Vec<String> {
    text.lines().filter_map(|l| l.strip_prefix("data: ")).map(str::to_string).collect()
}

/// The SSE event names of a stream body, in order.
pub fn sse_events(text: &str) -> Vec<String> {
    text.lines().filter_map(|l| l.strip_prefix("event: ")).map(str::to_string).collect()
}

/// Decode the base64 `/v1/embeddings` returns back to its floats.
pub fn base64_to_f32(text: &str) -> Vec<f32> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let value = |c: u8| TABLE.iter().position(|&t| t == c).unwrap_or_else(|| panic!("`{}` is not base64", c as char));
    let mut bytes = Vec::new();
    for chunk in text.as_bytes().chunks(4) {
        assert_eq!(chunk.len(), 4, "base64 `{text}` is not a whole number of quads");
        let pad = chunk.iter().filter(|&&c| c == b'=').count();
        let n = (0..4).fold(0u32, |acc, i| acc << 6 | if chunk[i] == b'=' { 0 } else { value(chunk[i]) as u32 });
        for (i, b) in [(n >> 16) as u8, (n >> 8) as u8, n as u8].into_iter().enumerate() {
            if i < 3 - pad {
                bytes.push(b);
            }
        }
    }
    assert_eq!(bytes.len() % 4, 0, "base64 `{text}` does not hold whole f32 values");
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// The `data` array of an output tensor of that name, or a panic listing
/// the outputs the response did carry.
pub fn output<'a>(response: &'a Value, name: &str) -> &'a Value {
    let outputs = response["outputs"].as_array().unwrap_or_else(|| panic!("no outputs in {response}"));
    outputs.iter().find(|o| o["name"] == name).unwrap_or_else(|| {
        let names: Vec<&str> = outputs.iter().filter_map(|o| o["name"].as_str()).collect();
        panic!("no output `{name}`; outputs: {names:?}")
    })
}

/// The floats of an FP32 output tensor.
pub fn floats(tensor: &Value) -> Vec<f32> {
    tensor["data"]
        .as_array()
        .unwrap_or_else(|| panic!("output `{}` has no data array", tensor["name"]))
        .iter()
        .map(|v| v.as_f64().unwrap_or_else(|| panic!("output `{}` holds a non-number", tensor["name"])) as f32)
        .collect()
}

/// The strings of a BYTES output tensor.
pub fn strings(tensor: &Value) -> Vec<String> {
    tensor["data"]
        .as_array()
        .unwrap_or_else(|| panic!("output `{}` has no data array", tensor["name"]))
        .iter()
        .map(|v| v.as_str().unwrap_or_else(|| panic!("output `{}` holds a non-string", tensor["name"])).to_string())
        .collect()
}

/// The shape of a tensor in a JSON response.
pub fn shape(tensor: &Value) -> Vec<i64> {
    tensor["shape"]
        .as_array()
        .unwrap_or_else(|| panic!("output `{}` has no shape", tensor["name"]))
        .iter()
        .map(|v| v.as_i64().unwrap_or_else(|| panic!("output `{}` has a non-integer extent", tensor["name"])))
        .collect()
}
