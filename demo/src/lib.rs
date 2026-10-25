//! A web page for trying a running turbo-kserve from a browser
//! (docs/demo.md). Browsers do not speak gRPC, so this serves the page and
//! answers its JSON calls with gRPC calls to the server; it holds no model
//! and links no backend.

use std::collections::HashMap;
use std::net::SocketAddr;

use axum::extract::{Path, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tonic::transport::{Channel, Endpoint};

/// The messages and client of open_inference_grpc.proto.
pub mod proto {
    #![allow(clippy::all)]
    tonic::include_proto!("inference");
}

/// The messages and client of turbo_analysis.proto, with serde for the
/// page's JSON (build.rs).
pub mod analysis {
    #![allow(clippy::all)]
    tonic::include_proto!("turbo.result");
}

mod gateway;
mod wire;

use analysis::CapabilitiesRequest;
use analysis::analysis_service_client::AnalysisServiceClient;
use proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use proto::infer_parameter::ParameterChoice;
use proto::model_infer_request::InferInputTensor;
use proto::*;

/// The largest reply read from the server: the server's own default limit.
pub const MAX_REPLY_BYTES: usize = 64 << 20;

const INDEX: &str = include_str!("../static/index.html");
const SCRIPT: &str = include_str!("../static/app.js");
const STYLE: &str = include_str!("../static/style.css");

#[derive(Clone)]
struct App {
    client: GrpcInferenceServiceClient<Channel>,
    analysis: AnalysisServiceClient<Channel>,
    server: String,
    models: Vec<String>,
}

/// The demo, answering HTTP until stopped or dropped.
pub struct Demo {
    addr: SocketAddr,
    stop: Option<oneshot::Sender<()>>,
    task: JoinHandle<std::io::Result<()>>,
}

impl Demo {
    /// Listens on `listen` and calls the server at `server`
    /// (`http://HOST:PORT`). `models` are the names the page offers first;
    /// when empty, the page offers the bundles the server's
    /// `AnalysisService/Capabilities` lists. The server need not be up: each
    /// call connects as needed.
    pub async fn start(listen: SocketAddr, server: &str, models: Vec<String>) -> Result<Demo, String> {
        let endpoint = Endpoint::from_shared(server.to_string()).map_err(|e| format!("server {server}: {e}"))?;
        if !matches!(endpoint.uri().scheme_str(), Some("http" | "https")) || endpoint.uri().authority().is_none() {
            return Err(format!("server {server}: must be an http:// or https:// URL"));
        }
        let channel = endpoint.connect_lazy();
        let client = GrpcInferenceServiceClient::new(channel.clone())
            .max_decoding_message_size(MAX_REPLY_BYTES)
            .max_encoding_message_size(MAX_REPLY_BYTES);
        let analysis = AnalysisServiceClient::new(channel)
            .max_decoding_message_size(MAX_REPLY_BYTES)
            .max_encoding_message_size(MAX_REPLY_BYTES);
        let app = App { client, analysis, server: server.to_string(), models };
        let router = Router::new()
            .route("/", get(|| async { asset("text/html; charset=utf-8", INDEX) }))
            .route("/app.js", get(|| async { asset("text/javascript; charset=utf-8", SCRIPT) }))
            .route("/style.css", get(|| async { asset("text/css; charset=utf-8", STYLE) }))
            .route("/api/status", get(status))
            .route("/api/models/{name}", get(metadata))
            .route("/api/infer", post(infer))
            .route("/api/capabilities", get(gateway::capabilities))
            .route("/api/bundles/load", post(gateway::load))
            .route("/api/bundles/unload", post(gateway::unload))
            .route("/api/analyze/stream", post(gateway::analyze_stream))
            .with_state(app);
        let listener = TcpListener::bind(listen).await.map_err(|e| format!("listen on {listen}: {e}"))?;
        let addr = listener.local_addr().map_err(|e| format!("listen on {listen}: {e}"))?;
        let (stop, stopped) = oneshot::channel::<()>();
        let task = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
        });
        Ok(Demo { addr, stop: Some(stop), task })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Stops listening and waits for calls in flight.
    pub async fn stop(mut self) {
        if let Some(s) = self.stop.take() {
            let _ = s.send(());
        }
        let _ = (&mut self.task).await;
    }
}

impl Drop for Demo {
    fn drop(&mut self) {
        if let Some(s) = self.stop.take() {
            let _ = s.send(());
        }
    }
}

fn asset(kind: &'static str, body: &'static str) -> Response {
    ([(header::CONTENT_TYPE, kind), (header::CACHE_CONTROL, "no-cache")], body).into_response()
}

/// A gRPC status as the page shows it: the status, the library's message,
/// and the `turbo-code` and `turbo-field` trailers when the server sent them.
fn failure(s: tonic::Status) -> Response {
    (StatusCode::BAD_GATEWAY, Json(error_body(&s))).into_response()
}

fn error_body(s: &tonic::Status) -> Value {
    let meta = |k: &str| s.metadata().get(k).and_then(|v| v.to_str().ok()).map(str::to_string);
    json!({ "error": {
        "grpc": format!("{:?}", s.code()),
        "message": s.message(),
        "turbo_code": meta("turbo-code"),
        "turbo_field": meta("turbo-field"),
    }})
}

fn bad_gateway(message: String) -> Response {
    (StatusCode::BAD_GATEWAY, Json(json!({ "error": { "grpc": null, "message": message } }))).into_response()
}

fn refused(message: String) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({ "error": { "grpc": null, "message": message } }))).into_response()
}

/// The names the page offers and where they came from: the names the demo
/// was started with when there are any, else the bundles the server's
/// Capabilities lists, in the order it serves them. A server without the
/// analysis service, or one that fails the call, offers none; the page can
/// still add a name by hand.
async fn offered(app: &App) -> (Vec<String>, &'static str) {
    if !app.models.is_empty() {
        return (app.models.clone(), "flags");
    }
    match app.analysis.clone().capabilities(CapabilitiesRequest {}).await {
        Ok(r) => {
            let mut names: Vec<String> = Vec::new();
            for b in r.into_inner().bundles {
                if !b.name.is_empty() && !names.contains(&b.name) {
                    names.push(b.name);
                }
            }
            (names, "capabilities")
        }
        Err(_) => (Vec::new(), "none"),
    }
}

/// The server's liveness, readiness and metadata, and each offered model's
/// readiness. A call that fails is reported in place of its answer.
async fn status(State(app): State<App>) -> Response {
    let mut c = app.client.clone();
    let live = c.server_live(ServerLiveRequest {}).await.map(|r| r.into_inner().live);
    let Ok(live) = live else {
        let e = live.unwrap_err();
        return Json(json!({ "server": app.server, "reachable": false, "message": e.message(),
                            "models_from": if app.models.is_empty() { "none" } else { "flags" },
                            "models": app.models.iter().map(|m| json!({ "name": m })).collect::<Vec<_>>() }))
        .into_response();
    };
    let (names, models_from) = offered(&app).await;
    let ready = c.server_ready(ServerReadyRequest {}).await.map(|r| r.into_inner().ready).unwrap_or(false);
    let meta = c.server_metadata(ServerMetadataRequest {}).await.ok().map(|r| r.into_inner());
    let mut models = Vec::new();
    for m in &names {
        let r = c.model_ready(ModelReadyRequest { name: m.clone(), version: String::new() }).await;
        models.push(match r {
            Ok(r) => json!({ "name": m, "ready": r.into_inner().ready }),
            Err(e) => json!({ "name": m, "ready": false, "message": e.message() }),
        });
    }
    Json(json!({
        "server": app.server,
        "reachable": true,
        "live": live,
        "ready": ready,
        "name": meta.as_ref().map(|m| m.name.clone()),
        "version": meta.as_ref().map(|m| m.version.clone()),
        "extensions": meta.map(|m| m.extensions).unwrap_or_default(),
        "models_from": models_from,
        "models": models,
    }))
    .into_response()
}

/// ModelMetadata, as JSON.
async fn metadata(State(app): State<App>, Path(name): Path<String>) -> Response {
    let r = app.client.clone().model_metadata(ModelMetadataRequest { name, version: String::new() }).await;
    let m = match r {
        Ok(r) => r.into_inner(),
        Err(e) => return failure(e),
    };
    let tensors = |t: &[model_metadata_response::TensorMetadata]| {
        t.iter().map(|t| json!({ "name": t.name, "datatype": t.datatype, "shape": t.shape })).collect::<Vec<_>>()
    };
    Json(json!({
        "name": m.name,
        "versions": m.versions,
        "inputs": tensors(&m.inputs),
        "outputs": tensors(&m.outputs),
        "properties": m.properties,
    }))
    .into_response()
}

#[derive(Deserialize)]
struct Infer {
    model: String,
    texts: Vec<String>,
    /// Request parameters by name: a string is a `string_param`, an integer
    /// an `int64_param` (docs/kserve.md, Parameters).
    #[serde(default)]
    parameters: HashMap<String, Value>,
}

/// One ModelInfer with the texts, answered with the vectors as rows of
/// numbers and the run's summary.
async fn infer(State(app): State<App>, body: Result<Json<Infer>, axum::extract::rejection::JsonRejection>) -> Response {
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => return refused(e.body_text()),
    };
    let mut parameters = HashMap::new();
    for (k, v) in req.parameters {
        let choice = match &v {
            Value::String(s) => ParameterChoice::StringParam(s.clone()),
            Value::Number(n) if n.is_i64() => ParameterChoice::Int64Param(n.as_i64().unwrap()),
            _ => return refused(format!("parameter {k}: {v} is neither a string nor an integer")),
        };
        parameters.insert(k, InferParameter { parameter_choice: Some(choice) });
    }
    let batch = req.texts.len() as i64;
    let request = ModelInferRequest {
        model_name: req.model,
        parameters,
        inputs: vec![InferInputTensor {
            name: "texts".into(),
            datatype: "BYTES".into(),
            shape: vec![batch],
            contents: Some(InferTensorContents {
                bytes_contents: req.texts.into_iter().map(String::into_bytes).collect(),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let r = match app.client.clone().model_infer(request).await {
        Ok(r) => r.into_inner(),
        Err(e) => return failure(e),
    };
    match decode_infer_reply(batch, r) {
        Ok(v) => Json(v).into_response(),
        Err(msg) => bad_gateway(msg),
    }
}

fn decode_infer_reply(batch: i64, r: ModelInferResponse) -> Result<Value, String> {
    let Some(out) = r.outputs.first() else {
        return Err("the reply has no output".into());
    };
    if out.datatype != "FP32" {
        return Err(format!("output datatype {:?} is not FP32", out.datatype));
    }
    let shape = out.shape.clone();
    let &[rows, cols] = shape.as_slice() else {
        return Err(format!("output shape {shape:?} is not [batch, dim]"));
    };
    if rows != batch || cols < 0 {
        return Err(format!("output shape {shape:?} does not match batch {batch}"));
    }
    let dim = cols as usize;
    let Some(expected_len) = (batch as usize).checked_mul(dim) else {
        return Err(format!("output shape {shape:?} overflows"));
    };
    let floats: Vec<f32> = if let Some(raw) = r.raw_output_contents.first() {
        let (chunks, rem) = raw.as_chunks::<4>();
        if !rem.is_empty() {
            return Err(format!("raw output length {} is not a multiple of 4", raw.len()));
        }
        chunks.iter().map(|b| f32::from_le_bytes(*b)).collect()
    } else {
        out.contents.as_ref().map(|c| c.fp32_contents.clone()).unwrap_or_default()
    };
    if floats.len() != expected_len {
        return Err(format!("output length {} does not match shape {shape:?}", floats.len()));
    }
    let vectors: Vec<&[f32]> = if dim == 0 { vec![&[][..]; batch as usize] } else { floats.chunks(dim).collect() };
    let summary: serde_json::Map<String, Value> = r
        .parameters
        .into_iter()
        .filter_map(|(k, p)| {
            let v = match p.parameter_choice? {
                ParameterChoice::StringParam(s) => Value::from(s),
                ParameterChoice::Int64Param(i) => Value::from(i),
                ParameterChoice::BoolParam(b) => Value::from(b),
                ParameterChoice::Uint64Param(u) => Value::from(u),
                ParameterChoice::DoubleParam(d) => Value::from(d),
            };
            Some((k, v))
        })
        .collect();
    Ok(json!({
        "model": r.model_name,
        "version": r.model_version,
        "shape": shape,
        "vectors": vectors,
        "summary": summary,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proto::model_infer_response::InferOutputTensor;

    #[tokio::test]
    async fn start_rejects_server_without_http_or_https_scheme() {
        let listen: SocketAddr = "127.0.0.1:0".parse().unwrap();
        for bad in ["127.0.0.1:8081", "ftp://127.0.0.1:8081", "ws://127.0.0.1:8081", "/ just-a-path"] {
            assert!(Demo::start(listen, bad, vec![]).await.is_err(), "{bad}");
        }
    }

    #[test]
    fn decode_infer_reply_validates_outputs_shape_and_float_count() {
        assert_eq!(decode_infer_reply(1, ModelInferResponse::default()), Err("the reply has no output".into()));
        assert_eq!(bad_gateway("the reply has no output".into()).status(), StatusCode::BAD_GATEWAY);

        let wrong_datatype = ModelInferResponse {
            outputs: vec![InferOutputTensor { datatype: "INT32".into(), shape: vec![1, 1], ..Default::default() }],
            raw_output_contents: vec![42i32.to_le_bytes().to_vec()],
            ..Default::default()
        };
        assert!(decode_infer_reply(1, wrong_datatype).unwrap_err().contains("not FP32"));

        let missing_datatype = ModelInferResponse {
            outputs: vec![InferOutputTensor { shape: vec![1, 1], ..Default::default() }],
            raw_output_contents: vec![1.0f32.to_le_bytes().to_vec()],
            ..Default::default()
        };
        assert!(decode_infer_reply(1, missing_datatype).unwrap_err().contains("not FP32"));

        let wrong_rank = ModelInferResponse {
            outputs: vec![InferOutputTensor { datatype: "FP32".into(), shape: vec![2], ..Default::default() }],
            ..Default::default()
        };
        assert!(decode_infer_reply(1, wrong_rank).unwrap_err().contains("not [batch, dim]"));

        let wrong_batch = ModelInferResponse {
            outputs: vec![InferOutputTensor { datatype: "FP32".into(), shape: vec![2, 2], ..Default::default() }],
            ..Default::default()
        };
        assert!(decode_infer_reply(1, wrong_batch).unwrap_err().contains("does not match batch 1"));

        let unaligned_raw = ModelInferResponse {
            outputs: vec![InferOutputTensor { datatype: "FP32".into(), shape: vec![1, 1], ..Default::default() }],
            raw_output_contents: vec![vec![1, 2, 3]],
            ..Default::default()
        };
        assert!(decode_infer_reply(1, unaligned_raw).unwrap_err().contains("not a multiple of 4"));

        let short_raw = ModelInferResponse {
            outputs: vec![InferOutputTensor { datatype: "FP32".into(), shape: vec![1, 2], ..Default::default() }],
            raw_output_contents: vec![1.0f32.to_le_bytes().to_vec()],
            ..Default::default()
        };
        assert!(decode_infer_reply(1, short_raw).unwrap_err().contains("output length 1 does not match shape [1, 2]"));

        let fp32_fallback = ModelInferResponse {
            model_name: "m".into(),
            model_version: "1".into(),
            outputs: vec![InferOutputTensor {
                datatype: "FP32".into(),
                shape: vec![2, 2],
                contents: Some(InferTensorContents { fp32_contents: vec![1.0, 2.0, 3.0, 4.0], ..Default::default() }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let v = decode_infer_reply(2, fp32_fallback).unwrap();
        assert_eq!(v["vectors"], json!([[1.0, 2.0], [3.0, 4.0]]));
    }
}
