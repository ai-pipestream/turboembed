//! The HTTP surface: the Open Inference Protocol v2 REST binding under
//! `/v2`, the OpenAI-shaped routes under `/v1`, and `/info` (the fields
//! text-embeddings-inference clients read).

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{FromRequest, Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use turbo::provider::{ClassifyOptions, EmbedOptions, GenerateDesc, RerankOptions};
use turbo::types::{FinishReason, ModelKind, Truncate};

use crate::engine::{self, Engine, Served};
use crate::error::{Result, ServeError};
use crate::oip::{self, Data, InferRequest, Param, Tensor};

type App = Arc<Engine>;

/// The router over the engine.
pub fn router(engine: Arc<Engine>) -> Router {
    Router::new()
        // Open Inference Protocol v2, REST binding.
        .route("/v2", get(server_metadata))
        .route("/v2/health/live", get(live))
        .route("/v2/health/ready", get(ready))
        .route("/v2/models", get(list_models))
        .route("/v2/models/{name}", get(model_metadata))
        .route("/v2/models/{name}/versions/{version}", get(model_metadata_versioned))
        .route("/v2/models/{name}/ready", get(model_ready))
        .route("/v2/models/{name}/versions/{version}/ready", get(model_ready_versioned))
        .route("/v2/models/{name}/infer", post(infer))
        .route("/v2/models/{name}/versions/{version}/infer", post(infer_versioned))
        // The model repository extension (index, load, unload at run time).
        .route("/v2/repository/index", post(repository_index))
        .route("/v2/repository/models/{name}/load", post(repository_load))
        .route("/v2/repository/models/{name}/unload", post(repository_unload))
        // OpenAI-shaped routes.
        .route("/v1/models", get(v1_models))
        .route("/v1/embeddings", post(v1_embeddings))
        .route("/v1/rerank", post(v1_rerank))
        .route("/v1/classify", post(v1_classify))
        .route("/v1/chat/completions", post(v1_chat))
        // text-embeddings-inference style info.
        .route("/info", get(info))
        .route("/health", get(live))
        .with_state(engine)
}

// ---------------------------------------------------------------------------
// Errors as JSON
// ---------------------------------------------------------------------------

/// The OIP error object, with the Turbo status and field added.
struct HttpError(ServeError);

impl IntoResponse for HttpError {
    fn into_response(self) -> Response {
        let e = self.0;
        let body = json!({
            "error": e.to_string(),
            "status": e.status,
            "field": e.field,
            "message": e.message,
        });
        (e.http_status(), Json(body)).into_response()
    }
}

impl From<ServeError> for HttpError {
    fn from(e: ServeError) -> Self {
        Self(e)
    }
}

type Reply<T> = std::result::Result<T, HttpError>;

/// A JSON request body; a body that is missing, malformed or of the
/// wrong shape is refused with the same error object as every other
/// failure (400, `BAD_REQUEST`), not axum's plain-text rejection.
struct Body<T>(T);

impl<S, T> FromRequest<S> for Body<T>
where
    S: Send + Sync,
    T: serde::de::DeserializeOwned,
{
    type Rejection = HttpError;

    async fn from_request(req: axum::extract::Request, state: &S) -> std::result::Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(v)) => Ok(Body(v)),
            Err(rejection) => {
                Err(HttpError(ServeError::bad_request(format!("request body: {}", rejection.body_text()))))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// OIP v2 REST
// ---------------------------------------------------------------------------

async fn server_metadata() -> Json<Value> {
    Json(json!({ "name": oip::SERVER_NAME, "version": oip::SERVER_VERSION, "extensions": oip::extensions() }))
}

async fn live() -> StatusCode {
    StatusCode::OK
}

async fn ready(State(app): State<App>) -> StatusCode {
    if app.is_empty() {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    }
}

async fn list_models(State(app): State<App>) -> Json<Value> {
    Json(json!({ "models": app.snapshot().iter().map(|s| oip::model_meta(s)).collect::<Vec<_>>() }))
}

async fn model_metadata(State(app): State<App>, Path(name): Path<String>) -> Reply<Json<oip::ModelMeta>> {
    Ok(Json(oip::model_meta(&*app.model(&name)?)))
}

async fn model_metadata_versioned(
    State(app): State<App>,
    Path((name, version)): Path<(String, String)>,
) -> Reply<Json<oip::ModelMeta>> {
    check_version(&version)?;
    model_metadata(State(app), Path(name)).await
}

async fn model_ready(State(app): State<App>, Path(name): Path<String>) -> StatusCode {
    if app.model(&name).is_ok() {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    }
}

async fn model_ready_versioned(State(app): State<App>, Path((name, version)): Path<(String, String)>) -> StatusCode {
    if check_version(&version).is_err() {
        return StatusCode::NOT_FOUND;
    }
    model_ready(State(app), Path(name)).await
}

fn check_version(v: &str) -> Result<()> {
    if v == oip::MODEL_VERSION {
        Ok(())
    } else {
        Err(ServeError::not_found(format!(
            "model version `{v}` does not exist; this server serves version {}",
            oip::MODEL_VERSION
        )))
    }
}

/// One tensor of the JSON binding.
#[derive(Deserialize)]
struct JsonTensor {
    name: String,
    #[serde(default)]
    shape: Vec<i64>,
    datatype: String,
    #[serde(default)]
    data: Vec<Value>,
    #[serde(default)]
    parameters: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct JsonRequestedOutput {
    name: String,
}

/// The Inference Request JSON Object.
#[derive(Deserialize)]
struct JsonInferRequest {
    #[serde(default)]
    id: String,
    #[serde(default)]
    parameters: BTreeMap<String, Value>,
    #[serde(default)]
    inputs: Vec<JsonTensor>,
    #[serde(default)]
    outputs: Vec<JsonRequestedOutput>,
}

fn params_from_json(map: &BTreeMap<String, Value>) -> Result<BTreeMap<String, Param>> {
    let mut out = BTreeMap::new();
    for (k, v) in map {
        let p = match v {
            Value::Bool(b) => Param::Bool(*b),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Param::Int(i)
                } else if let Some(u) = n.as_u64() {
                    Param::Uint(u)
                } else {
                    Param::Double(n.as_f64().unwrap_or(f64::NAN))
                }
            }
            Value::String(s) => Param::Str(s.clone()),
            other => {
                return Err(ServeError::bad_request(format!(
                    "parameter `{k}` must be a boolean, number or string, not {other}"
                )))
            }
        };
        out.insert(k.clone(), p);
    }
    Ok(out)
}

fn params_to_json(map: &BTreeMap<String, Param>) -> Value {
    Value::Object(
        map.iter()
            .map(|(k, v)| {
                let j = match v {
                    Param::Bool(b) => json!(b),
                    Param::Int(i) => json!(i),
                    Param::Str(s) => json!(s),
                    Param::Double(d) => json!(d),
                    Param::Uint(u) => json!(u),
                };
                (k.clone(), j)
            })
            .collect(),
    )
}

/// JSON `data` as typed contents of the declared datatype. A BYTES element
/// is a JSON string.
fn data_from_json(name: &str, datatype: &str, data: &[Value]) -> Result<Data> {
    let bad = |i: usize, what: &str| ServeError::bad_request(format!("input `{name}`: data[{i}] is not {what}"));
    Ok(match datatype {
        "BYTES" => Data::Bytes(
            data.iter()
                .enumerate()
                .map(|(i, v)| v.as_str().map(|s| s.as_bytes().to_vec()).ok_or_else(|| bad(i, "a string")))
                .collect::<Result<_>>()?,
        ),
        "FP32" => Data::Fp32(
            data.iter()
                .enumerate()
                .map(|(i, v)| v.as_f64().map(|f| f as f32).ok_or_else(|| bad(i, "a number")))
                .collect::<Result<_>>()?,
        ),
        "FP64" => Data::Fp64(
            data.iter()
                .enumerate()
                .map(|(i, v)| v.as_f64().ok_or_else(|| bad(i, "a number")))
                .collect::<Result<_>>()?,
        ),
        "INT32" | "INT16" | "INT8" => Data::Int32(
            data.iter()
                .enumerate()
                .map(|(i, v)| v.as_i64().and_then(|n| i32::try_from(n).ok()).ok_or_else(|| bad(i, "a 32-bit integer")))
                .collect::<Result<_>>()?,
        ),
        "INT64" => Data::Int64(
            data.iter()
                .enumerate()
                .map(|(i, v)| v.as_i64().ok_or_else(|| bad(i, "an integer")))
                .collect::<Result<_>>()?,
        ),
        "UINT32" | "UINT16" | "UINT8" => Data::Uint32(
            data.iter()
                .enumerate()
                .map(|(i, v)| {
                    v.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| bad(i, "a 32-bit unsigned integer"))
                })
                .collect::<Result<_>>()?,
        ),
        "UINT64" => Data::Uint64(
            data.iter()
                .enumerate()
                .map(|(i, v)| v.as_u64().ok_or_else(|| bad(i, "an unsigned integer")))
                .collect::<Result<_>>()?,
        ),
        "BOOL" => Data::Bool(
            data.iter()
                .enumerate()
                .map(|(i, v)| v.as_bool().ok_or_else(|| bad(i, "a boolean")))
                .collect::<Result<_>>()?,
        ),
        other => {
            return Err(ServeError::bad_request(format!(
                "input `{name}`: datatype `{other}` is not carried by the JSON binding"
            )))
        }
    })
}

fn data_to_json(data: &Data) -> Vec<Value> {
    match data {
        Data::Bytes(v) => v.iter().map(|b| json!(String::from_utf8_lossy(b))).collect(),
        Data::Fp32(v) => v.iter().map(|x| json!(x)).collect(),
        Data::Fp64(v) => v.iter().map(|x| json!(x)).collect(),
        Data::Int32(v) => v.iter().map(|x| json!(x)).collect(),
        Data::Int64(v) => v.iter().map(|x| json!(x)).collect(),
        Data::Uint32(v) => v.iter().map(|x| json!(x)).collect(),
        Data::Uint64(v) => v.iter().map(|x| json!(x)).collect(),
        Data::Bool(v) => v.iter().map(|x| json!(x)).collect(),
    }
}

async fn infer(
    State(app): State<App>,
    Path(name): Path<String>,
    Body(req): Body<JsonInferRequest>,
) -> Reply<Json<Value>> {
    let served = app.model(&name)?;
    let mut inputs = Vec::new();
    for t in &req.inputs {
        let shape = if t.shape.is_empty() { vec![t.data.len() as i64] } else { t.shape.clone() };
        inputs.push(Tensor {
            name: t.name.clone(),
            datatype: t.datatype.clone(),
            shape,
            data: data_from_json(&t.name, &t.datatype, &t.data)?,
            parameters: params_from_json(&t.parameters)?,
        });
    }
    let request = InferRequest {
        id: req.id,
        parameters: params_from_json(&req.parameters)?,
        inputs,
        outputs: req.outputs.iter().map(|o| o.name.clone()).collect(),
    };
    let resp = oip::infer(served, request).await?;
    Ok(Json(json!({
        "model_name": resp.model_name,
        "model_version": resp.model_version,
        "id": resp.id,
        "parameters": params_to_json(&resp.parameters),
        "outputs": resp.outputs.iter().map(|t| json!({
            "name": t.name,
            "shape": t.shape,
            "datatype": t.datatype,
            "data": data_to_json(&t.data),
        })).collect::<Vec<_>>(),
    })))
}

async fn infer_versioned(
    State(app): State<App>,
    Path((name, version)): Path<(String, String)>,
    Body(req): Body<JsonInferRequest>,
) -> Reply<Json<Value>> {
    check_version(&version)?;
    infer(State(app), Path(name), Body(req)).await
}

// ---------------------------------------------------------------------------
// Model repository extension
// ---------------------------------------------------------------------------

async fn repository_index(State(app): State<App>) -> Json<Value> {
    Json(json!(app
        .snapshot()
        .iter()
        .map(|s| json!({
            "name": s.name,
            "version": oip::MODEL_VERSION,
            "state": "READY",
            "reason": "",
            "bundle": s.spec.bundle.display().to_string(),
            "provider": s.spec.provider,
            "device": s.device.name,
            "kind": format!("{:?}", s.info().kind),
        }))
        .collect::<Vec<_>>()))
}

/// The body of a load: the same keys as a `--model` flag.
#[derive(Deserialize)]
struct LoadRequest {
    bundle: String,
    provider: String,
    #[serde(default)]
    ordinal: u32,
    #[serde(default)]
    buckets: Vec<String>,
    #[serde(default)]
    sessions: u32,
    #[serde(default)]
    generations: u32,
}

async fn repository_load(
    State(app): State<App>,
    Path(name): Path<String>,
    Body(req): Body<LoadRequest>,
) -> Reply<Json<Value>> {
    let spec = crate::config::ModelSpec::from_request(
        Some(name),
        &req.bundle,
        &req.provider,
        req.ordinal,
        &req.buckets,
        req.sessions,
        req.generations,
    )?;
    let engine = app.clone();
    let loaded = tokio::task::spawn_blocking(move || engine.load_model(&spec))
        .await
        .map_err(|e| ServeError::internal(format!("load task failed: {e}")))??;
    Ok(Json(json!({ "name": loaded })))
}

async fn repository_unload(State(app): State<App>, Path(name): Path<String>) -> Reply<Json<Value>> {
    app.unload(&name)?;
    Ok(Json(json!({ "name": name, "state": "UNAVAILABLE" })))
}

// ---------------------------------------------------------------------------
// OpenAI-shaped routes
// ---------------------------------------------------------------------------

async fn v1_models(State(app): State<App>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": app.snapshot().iter().map(|s| json!({
            "id": s.name,
            "object": "model",
            "owned_by": s.info().provider_id,
            "created": 0,
            "kind": format!("{:?}", s.info().kind),
            "model_id": s.info().model_id,
        })).collect::<Vec<_>>(),
    }))
}

/// `input` is a string or a list of strings.
#[derive(Deserialize)]
#[serde(untagged)]
enum Input {
    One(String),
    Many(Vec<String>),
}

impl Input {
    fn into_vec(self) -> Vec<String> {
        match self {
            Input::One(s) => vec![s],
            Input::Many(v) => v,
        }
    }
}

#[derive(Deserialize)]
struct EmbeddingsRequest {
    model: String,
    input: Input,
    #[serde(default)]
    encoding_format: Option<String>,
    #[serde(default)]
    dimensions: Option<u32>,
    /// Turbo extension: `none`, `right`, `left` (default: the model's).
    #[serde(default)]
    truncate: Option<String>,
    /// Turbo extension: `query` or `document`, applying the bundle's prefix.
    #[serde(default)]
    prompt_role: Option<String>,
    /// Turbo extension: `l2` or `none`.
    #[serde(default)]
    normalize: Option<String>,
}

fn find_model(app: &Engine, name: &str, kind: ModelKind) -> Result<Arc<Served>> {
    let served = app.model(name)?;
    if served.info().kind != kind {
        return Err(ServeError::bad_request(format!(
            "model `{name}` is {:?}; this route needs {kind:?}",
            served.info().kind
        )));
    }
    Ok(served)
}

async fn v1_embeddings(State(app): State<App>, Body(req): Body<EmbeddingsRequest>) -> Reply<Json<Value>> {
    let served = find_model(&app, &req.model, ModelKind::Embedding)?;
    let format = req.encoding_format.as_deref().unwrap_or("float");
    if format != "float" && format != "base64" {
        return Err(ServeError::bad_request(format!("encoding_format `{format}` is not float or base64")).into());
    }
    let opts = EmbedOptions {
        truncate: engine::parse_truncate(req.truncate.as_deref())?,
        prompt_role: engine::parse_prompt_role(req.prompt_role.as_deref())?,
        normalize: engine::parse_normalize(req.normalize.as_deref())?,
        output_dim: req.dimensions.unwrap_or(0),
        ..Default::default()
    };
    let e = engine::embed(served.clone(), req.input.into_vec(), opts).await?;
    let data: Vec<Value> = e
        .vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let embedding = if format == "base64" {
                json!(base64_of(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()))
            } else {
                json!(v)
            };
            json!({ "object": "embedding", "index": i, "embedding": embedding })
        })
        .collect();
    Ok(Json(json!({
        "object": "list",
        "data": data,
        "model": served.name,
        // Token counts are reported only when the bundle's tokenizer counted
        // them; a zero would be a claim, not a measurement.
        "usage": e.tokens.map(|t| json!({ "prompt_tokens": t, "total_tokens": t })),
        "turbo": { "device": served.device.name, "placement": format!("{:?}", e.placement), "device_ms": e.device_ms, "dim": e.dim },
    })))
}

#[derive(Deserialize)]
struct RerankRequest {
    model: String,
    query: String,
    documents: Vec<String>,
    #[serde(default)]
    top_n: Option<u32>,
    #[serde(default)]
    return_documents: bool,
    #[serde(default)]
    raw_scores: bool,
    #[serde(default)]
    truncate: Option<String>,
}

async fn v1_rerank(State(app): State<App>, Body(req): Body<RerankRequest>) -> Reply<Json<Value>> {
    let served = find_model(&app, &req.model, ModelKind::Reranker)?;
    let opts = RerankOptions {
        truncate: engine::parse_truncate(req.truncate.as_deref())?,
        top_n: req.top_n.unwrap_or(0),
        return_sorted: true,
        raw_scores: req.raw_scores,
        ..Default::default()
    };
    let docs = req.documents.clone();
    let r = engine::rerank(served.clone(), req.query, req.documents, opts).await?;
    let results: Vec<Value> = r
        .sorted
        .iter()
        .map(|&i| {
            let mut item = json!({ "index": i, "relevance_score": r.scores[i as usize] });
            if req.return_documents {
                item["document"] = json!({ "text": docs[i as usize] });
            }
            item
        })
        .collect();
    Ok(Json(json!({
        "model": served.name,
        "results": results,
        "turbo": { "device": served.device.name, "device_ms": r.device_ms, "scores": r.scores },
    })))
}

#[derive(Deserialize)]
struct ClassifyRequest {
    model: String,
    inputs: Input,
    #[serde(default)]
    raw_scores: bool,
    #[serde(default)]
    truncate: Option<String>,
    #[serde(default)]
    aggregation: Option<String>,
}

/// `/v1/classify`: sequence classification returns, per input, labels
/// with scores best first; token classification returns, per input, the
/// entity spans.
async fn v1_classify(State(app): State<App>, Body(req): Body<ClassifyRequest>) -> Reply<Json<Value>> {
    let served = app.model(&req.model)?;
    let opts = ClassifyOptions {
        truncate: engine::parse_truncate(req.truncate.as_deref())?,
        aggregation: engine::parse_aggregation(req.aggregation.as_deref())?,
        raw_scores: req.raw_scores,
        ..Default::default()
    };
    let texts = req.inputs.into_vec();
    match served.info().kind {
        ModelKind::Classifier => {
            let c = engine::classify(served.clone(), texts, opts).await?;
            let rows: Vec<Value> = c
                .scores
                .iter()
                .map(|row| {
                    let mut items: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
                    items.sort_by(|a, b| b.1.total_cmp(&a.1));
                    json!(items.iter().map(|(i, s)| json!({ "label": c.labels[*i], "score": s })).collect::<Vec<_>>())
                })
                .collect();
            Ok(Json(
                json!({ "model": served.name, "results": rows, "turbo": { "device": served.device.name, "device_ms": c.device_ms } }),
            ))
        }
        ModelKind::TokenClassifier => {
            let t = engine::token_classify(served.clone(), texts.clone(), opts).await?;
            let rows: Vec<Value> = t
                .spans
                .iter()
                .enumerate()
                .map(|(row, spans)| {
                    json!(spans
                        .iter()
                        .map(|s| {
                            let text = &texts[row];
                            let (b, e) = (s.byte_start as usize, s.byte_end as usize);
                            let word = text.get(b..e).unwrap_or("");
                            json!({
                                "entity_group": t.labels.get(s.label as usize).cloned().unwrap_or_else(|| s.label.to_string()),
                                "score": s.score,
                                "word": word,
                                "start": b,
                                "end": e,
                            })
                        })
                        .collect::<Vec<_>>())
                })
                .collect();
            Ok(Json(
                json!({ "model": served.name, "results": rows, "turbo": { "device": served.device.name, "device_ms": t.device_ms } }),
            ))
        }
        other => Err(ServeError::bad_request(format!(
            "model `{}` is {other:?}; /v1/classify needs a classifier or token classifier",
            served.name
        ))
        .into()),
    }
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<Stop>,
    #[serde(default)]
    n: Option<u32>,
}

fn chat_desc(req: &ChatRequest) -> Result<GenerateDesc> {
    if let Some(n) = req.n {
        if n != 1 {
            return Err(ServeError::field(
                GenerateDesc::FIELD_N_SEQUENCES,
                format!("n {n} is not served; one choice per request"),
            ));
        }
    }
    let mut d = GenerateDesc::default();
    if let Some(m) = req.max_completion_tokens.or(req.max_tokens) {
        d.max_new_tokens = m;
    }
    if let Some(t) = req.temperature {
        d.temperature = t;
    }
    if let Some(p) = req.top_p {
        d.top_p = p;
    }
    if let Some(k) = req.top_k {
        d.top_k = k;
    }
    d.seed = req.seed;
    d.stop = match &req.stop {
        None => Vec::new(),
        Some(Stop::One(s)) => vec![s.clone()],
        Some(Stop::Many(v)) => v.clone(),
    };
    Ok(d)
}

fn finish_name(r: FinishReason) -> &'static str {
    match r {
        FinishReason::Eos | FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::Cancelled => "cancelled",
        FinishReason::None => "none",
    }
}

/// Distinguishes completions started in the same second.
static NEXT_CHAT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

async fn v1_chat(State(app): State<App>, Body(req): Body<ChatRequest>) -> Reply<Response> {
    let served = find_model(&app, &req.model, ModelKind::Generative)?;
    let desc = chat_desc(&req)?;
    let messages: Vec<(String, String)> = req.messages.iter().map(|m| (m.role.clone(), m.content.clone())).collect();
    let id = format!("chatcmpl-{:x}-{:x}", now_secs(), NEXT_CHAT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
    let created = now_secs();
    let model_name = served.name.clone();
    let mut rx = engine::generate(served, messages, desc).await?;
    if !req.stream {
        let mut text = String::new();
        let mut prompt_tokens = 0;
        let mut generated = 0;
        let mut finish = "none";
        while let Some(piece) = rx.recv().await {
            let piece = piece?;
            text.push_str(&piece.text);
            prompt_tokens = piece.prompt_tokens;
            generated = piece.generated_tokens;
            if piece.done {
                finish = finish_name(piece.finish_reason);
            }
        }
        return Ok(Json(json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": text }, "finish_reason": finish }],
            "usage": { "prompt_tokens": prompt_tokens, "completion_tokens": generated, "total_tokens": prompt_tokens + generated },
        }))
        .into_response());
    }
    let stream = async_stream(move |yield_| async move {
        let mut sent_role = false;
        while let Some(piece) = rx.recv().await {
            match piece {
                Ok(p) => {
                    let mut delta = json!({});
                    if !sent_role {
                        delta["role"] = json!("assistant");
                        sent_role = true;
                    }
                    if !p.text.is_empty() {
                        delta["content"] = json!(p.text);
                    }
                    let chunk = json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{ "index": 0, "delta": delta, "finish_reason": if p.done { Value::from(finish_name(p.finish_reason)) } else { Value::Null } }],
                        "usage": if p.done { json!({ "prompt_tokens": p.prompt_tokens, "completion_tokens": p.generated_tokens, "total_tokens": p.prompt_tokens + p.generated_tokens }) } else { Value::Null },
                    });
                    if !yield_.send(Event::default().data(chunk.to_string())).await {
                        // The client hung up: dropping `rx` cancels the
                        // generation on the device.
                        return;
                    }
                    if p.done {
                        break;
                    }
                }
                Err(e) => {
                    // The stream ends with an error event a client can read;
                    // the HTTP status was 200 by the time it started.
                    let err = json!({ "error": { "message": e.to_string(), "type": e.status, "field": e.field } });
                    let _ = yield_.send(Event::default().event("error").data(err.to_string())).await;
                    break;
                }
            }
        }
        let _ = yield_.send(Event::default().data("[DONE]")).await;
    });
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
}

/// A stream fed by an async block through a channel: the block runs on
/// its own task and each event it sends becomes a stream item.
fn async_stream<F, Fut>(f: F) -> EventStream
where
    F: FnOnce(Yield) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(16);
    tokio::spawn(f(Yield { tx }));
    EventStream { rx }
}

/// The receiving side as a `Stream` for `Sse`.
struct EventStream {
    rx: tokio::sync::mpsc::Receiver<Event>,
}

impl futures_core::Stream for EventStream {
    type Item = std::result::Result<Event, Infallible>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx).map(|e| e.map(Ok))
    }
}

/// The sending side, called as a function by the producing block.
#[derive(Clone)]
struct Yield {
    tx: tokio::sync::mpsc::Sender<Event>,
}

impl Yield {
    /// False when the client has gone (the stream body was dropped).
    async fn send(&self, e: Event) -> bool {
        self.tx.send(e).await.is_ok()
    }
}

fn base64_of(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { TABLE[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[n as usize & 63] as char } else { '=' });
    }
    out
}

// ---------------------------------------------------------------------------
// /info
// ---------------------------------------------------------------------------

/// The fields text-embeddings-inference publishes on `/info`, for the
/// first embedding model (or the first model), plus every served model.
#[derive(Serialize)]
struct Info {
    model_id: String,
    model_sha: String,
    model_dtype: String,
    model_type: Value,
    max_input_length: u32,
    max_batch_tokens: u32,
    max_batch_requests: u32,
    max_client_batch_size: u32,
    tokenization_workers: u32,
    version: String,
    sha: String,
    docker_label: Option<String>,
    models: Vec<Value>,
}

async fn info(State(app): State<App>) -> Json<Info> {
    let models = app.snapshot();
    let primary = models
        .iter()
        .find(|s| s.info().kind == ModelKind::Embedding)
        .or_else(|| models.first())
        .expect("the engine serves at least one model");
    let i = primary.info();
    let model_type = match i.kind {
        ModelKind::Embedding => {
            json!({ "embedding": { "pooling": i.pooling.map(|p| format!("{p:?}").to_lowercase()) } })
        }
        ModelKind::Reranker => {
            json!({ "reranker": { "id2label": i.labels.iter().enumerate().map(|(k, v)| (k.to_string(), v.clone())).collect::<BTreeMap<_, _>>() } })
        }
        ModelKind::Classifier | ModelKind::TokenClassifier => {
            json!({ "classifier": { "id2label": i.labels.iter().enumerate().map(|(k, v)| (k.to_string(), v.clone())).collect::<BTreeMap<_, _>>() } })
        }
        other => json!({ format!("{other:?}").to_lowercase(): {} }),
    };
    let models: Vec<Value> = models
        .iter()
        .map(|s| {
            json!({
                "name": s.name,
                "model_id": s.info().model_id,
                "kind": format!("{:?}", s.info().kind),
                "provider": s.info().provider_id,
                "device": s.device.name,
                "max_seq": s.max_seq(),
                "max_batch": s.max_batch(),
                "truncate_default": format!("{:?}", Truncate::Model).to_lowercase(),
            })
        })
        .collect();
    Json(Info {
        model_id: i.model_id.clone(),
        model_sha: i.revision.clone(),
        model_dtype: i.dtype_used.map(|d| format!("{d:?}").to_lowercase()).unwrap_or_else(|| "float32".into()),
        model_type,
        max_input_length: primary.max_seq(),
        max_batch_tokens: primary.max_seq() * primary.max_batch(),
        max_batch_requests: primary.max_batch(),
        max_client_batch_size: primary.max_batch(),
        tokenization_workers: 1,
        version: oip::SERVER_VERSION.to_string(),
        sha: option_env!("TURBO_INFERSTREAM_GIT_COMMIT").unwrap_or("").to_string(),
        docker_label: None,
        models,
    })
}
