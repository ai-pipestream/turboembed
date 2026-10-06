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
    /// the protocol has no call that lists them. The server need not be up:
    /// each call connects as needed.
    pub async fn start(listen: SocketAddr, server: &str, models: Vec<String>) -> Result<Demo, String> {
        let endpoint = Endpoint::from_shared(server.to_string()).map_err(|e| format!("server {server}: {e}"))?;
        let client = GrpcInferenceServiceClient::new(endpoint.connect_lazy())
            .max_decoding_message_size(MAX_REPLY_BYTES)
            .max_encoding_message_size(MAX_REPLY_BYTES);
        let app = App { client, server: server.to_string(), models };
        let router = Router::new()
            .route("/", get(|| async { asset("text/html; charset=utf-8", INDEX) }))
            .route("/app.js", get(|| async { asset("text/javascript; charset=utf-8", SCRIPT) }))
            .route("/style.css", get(|| async { asset("text/css; charset=utf-8", STYLE) }))
            .route("/api/status", get(status))
            .route("/api/models/{name}", get(metadata))
            .route("/api/infer", post(infer))
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
    let meta = |k: &str| s.metadata().get(k).and_then(|v| v.to_str().ok()).map(str::to_string);
    let body = json!({ "error": {
        "grpc": format!("{:?}", s.code()),
        "message": s.message(),
        "turbo_code": meta("turbo-code"),
        "turbo_field": meta("turbo-field"),
    }});
    (StatusCode::BAD_GATEWAY, Json(body)).into_response()
}

fn refused(message: String) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({ "error": { "grpc": null, "message": message } }))).into_response()
}

/// The server's liveness, readiness and metadata, and each offered model's
/// readiness. A call that fails is reported in place of its answer.
async fn status(State(app): State<App>) -> Response {
    let mut c = app.client.clone();
    let live = c.server_live(ServerLiveRequest {}).await.map(|r| r.into_inner().live);
    let Ok(live) = live else {
        let e = live.unwrap_err();
        return Json(json!({ "server": app.server, "reachable": false, "message": e.message(),
                            "models": app.models.iter().map(|m| json!({ "name": m })).collect::<Vec<_>>() }))
        .into_response();
    };
    let ready = c.server_ready(ServerReadyRequest {}).await.map(|r| r.into_inner().ready).unwrap_or(false);
    let meta = c.server_metadata(ServerMetadataRequest {}).await.ok().map(|r| r.into_inner());
    let mut models = Vec::new();
    for m in &app.models {
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
    let Some(out) = r.outputs.first() else {
        return refused("the reply has no output".into());
    };
    let shape = out.shape.clone();
    let dim = shape.get(1).copied().unwrap_or(0).max(0) as usize;
    let raw = r.raw_output_contents.first().map(Vec::as_slice).unwrap_or(&[]);
    let floats: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let vectors: Vec<&[f32]> = if dim == 0 { vec![] } else { floats.chunks(dim).collect() };
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
    Json(json!({
        "model": r.model_name,
        "version": r.model_version,
        "shape": shape,
        "vectors": vectors,
        "summary": summary,
    }))
    .into_response()
}
