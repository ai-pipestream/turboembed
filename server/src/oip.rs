//! The Open Inference Protocol v2 mapping, shared by the gRPC and the REST
//! binding: how each served model kind is described as tensors and how an
//! inference request's tensors become a Turbo call and back.
//!
//! Tensor names per kind (the same as the Spring demo, so one client works
//! against both):
//!
//! | kind | inputs | outputs |
//! |---|---|---|
//! | embedding | `text` BYTES `[n]` | `embeddings` FP32 `[n, dim]` |
//! | reranker | `query` BYTES `[1]`, `documents` BYTES `[n]` | `scores` FP32 `[n]`, `sorted` INT32 `[k]` |
//! | classifier | `text` BYTES `[n]` | `scores` FP32 `[n, labels]`, `labels` BYTES `[labels]` |
//! | token classifier | `text` BYTES `[n]` | `spans` BYTES `[m]` (JSON per span), `labels` BYTES `[labels]` |
//! | generative | `prompt` BYTES `[1]` or `messages` BYTES `[t]` (JSON per turn) | `text` BYTES `[1]`; finish reason and token counts in the response parameters |
//! | generic | the model's own inputs | the model's own outputs |
//!
//! Turbo options travel in the request `parameters` map under their
//! `turbo_` names (`truncate`, `max_tokens`, `prompt_role`, `normalize`,
//! `pooling`, `output_dim`, `top_n`, `raw_scores`, `aggregation`,
//! `max_new_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stop`).

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use turbo::provider::{ClassifyOptions, EmbedOptions, GenerateDesc, RerankOptions};
use turbo::types::{DType, ModelKind};

use crate::engine::{self, RunInput, Served};
use crate::error::{Result, ServeError};

/// A parameter value, as OIP defines it (bool, int64, string, double, uint64).
#[derive(Debug, Clone, PartialEq)]
pub enum Param {
    /// `bool_param`.
    Bool(bool),
    /// `int64_param`.
    Int(i64),
    /// `string_param`.
    Str(String),
    /// `double_param`.
    Double(f64),
    /// `uint64_param`.
    Uint(u64),
}

impl Param {
    /// The value as the string the option parsers read.
    pub fn as_str(&self) -> String {
        match self {
            Param::Bool(b) => b.to_string(),
            Param::Int(i) => i.to_string(),
            Param::Str(s) => s.clone(),
            Param::Double(d) => d.to_string(),
            Param::Uint(u) => u.to_string(),
        }
    }
}

/// Tensor contents, typed.
#[derive(Debug, Clone)]
pub enum Data {
    /// `BYTES`: one byte string per element.
    Bytes(Vec<Vec<u8>>),
    /// `FP32`.
    Fp32(Vec<f32>),
    /// `FP64`.
    Fp64(Vec<f64>),
    /// `INT32`, and the narrower signed types.
    Int32(Vec<i32>),
    /// `INT64`.
    Int64(Vec<i64>),
    /// `UINT32`, and the narrower unsigned types.
    Uint32(Vec<u32>),
    /// `UINT64`.
    Uint64(Vec<u64>),
    /// `BOOL`.
    Bool(Vec<bool>),
}

impl Data {
    /// The OIP datatype string of the contents.
    pub fn datatype(&self) -> &'static str {
        match self {
            Data::Bytes(_) => "BYTES",
            Data::Fp32(_) => "FP32",
            Data::Fp64(_) => "FP64",
            Data::Int32(_) => "INT32",
            Data::Int64(_) => "INT64",
            Data::Uint32(_) => "UINT32",
            Data::Uint64(_) => "UINT64",
            Data::Bool(_) => "BOOL",
        }
    }
}

/// An input or output tensor.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// The tensor's name in the model's metadata.
    pub name: String,
    /// The OIP datatype string, which must match `data`.
    pub datatype: String,
    /// Extents, outermost first.
    pub shape: Vec<i64>,
    /// The elements.
    pub data: Data,
    /// Per-tensor parameters.
    pub parameters: BTreeMap<String, Param>,
}

/// An inference request, binding-independent.
#[derive(Debug, Clone, Default)]
pub struct InferRequest {
    /// The client's correlation id, echoed in the response.
    pub id: String,
    /// Request parameters; this server's Turbo options live here.
    pub parameters: BTreeMap<String, Param>,
    /// The input tensors.
    pub inputs: Vec<Tensor>,
    /// Names the client wants back; empty means all.
    pub outputs: Vec<String>,
}

/// An inference response, binding-independent.
#[derive(Debug, Clone)]
pub struct InferResponse {
    /// The model that ran.
    pub model_name: String,
    /// The version that ran; this server has one.
    pub model_version: String,
    /// The request's correlation id.
    pub id: String,
    /// Response parameters (`device_ms`, `placement`, and the generation
    /// counts).
    pub parameters: BTreeMap<String, Param>,
    /// The output tensors.
    pub outputs: Vec<Tensor>,
}

/// Tensor metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    /// The tensor's name.
    pub name: String,
    /// Its OIP datatype string.
    pub datatype: String,
    /// Its extents; `-1` is a dimension the request decides.
    pub shape: Vec<i64>,
}

/// Model metadata.
#[derive(Debug, Clone, Serialize)]
pub struct ModelMeta {
    /// The name clients use.
    pub name: String,
    /// The versions served; this server has one.
    pub versions: Vec<String>,
    /// The provider the model runs on.
    pub platform: String,
    /// The input tensors the model takes.
    pub inputs: Vec<TensorMeta>,
    /// The output tensors it produces.
    pub outputs: Vec<TensorMeta>,
    /// What the bundle and the device say about the model.
    pub properties: BTreeMap<String, String>,
}

/// `ServerMetadata.name`.
pub const SERVER_NAME: &str = "turbo-inferstream";
/// `ServerMetadata.version`.
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");
/// The one model version this server has.
pub const MODEL_VERSION: &str = "1";

/// Extensions this server implements beyond the core protocol.
pub fn extensions() -> Vec<String> {
    vec!["turbo_parameters".to_string()]
}

fn meta(name: &str, datatype: &str, shape: &[i64]) -> TensorMeta {
    TensorMeta { name: name.to_string(), datatype: datatype.to_string(), shape: shape.to_vec() }
}

/// The OIP datatype name of a Turbo dtype.
pub fn datatype_of(d: DType) -> &'static str {
    match d {
        DType::Bool => "BOOL",
        DType::U8 => "UINT8",
        DType::U16 => "UINT16",
        DType::U32 => "UINT32",
        DType::U64 => "UINT64",
        DType::I8 => "INT8",
        DType::I16 => "INT16",
        DType::I32 => "INT32",
        DType::I64 => "INT64",
        DType::F16 => "FP16",
        DType::BF16 => "BF16",
        DType::F32 => "FP32",
        DType::F64 => "FP64",
        DType::Bytes => "BYTES",
    }
}

/// The Turbo dtype of an OIP datatype name, for the types this server
/// carries as typed contents.
pub fn dtype_of(name: &str) -> Result<DType> {
    Ok(match name {
        "BOOL" => DType::Bool,
        "UINT32" => DType::U32,
        "UINT64" => DType::U64,
        "INT32" => DType::I32,
        "INT64" => DType::I64,
        "FP32" => DType::F32,
        "FP64" => DType::F64,
        "BYTES" => DType::Bytes,
        other => {
            return Err(ServeError::bad_request(format!(
            "datatype `{other}` is not carried by this server (BOOL, UINT32, UINT64, INT32, INT64, FP32, FP64, BYTES)"
        )))
        }
    })
}

/// Model metadata for a served model.
pub fn model_meta(served: &Served) -> ModelMeta {
    let info = served.info();
    let (inputs, outputs) = match info.kind {
        ModelKind::Embedding => {
            (vec![meta("text", "BYTES", &[-1])], vec![meta("embeddings", "FP32", &[-1, info.dim as i64])])
        }
        ModelKind::Reranker => (
            vec![meta("query", "BYTES", &[1]), meta("documents", "BYTES", &[-1])],
            vec![meta("scores", "FP32", &[-1]), meta("sorted", "INT32", &[-1])],
        ),
        ModelKind::Classifier => (
            vec![meta("text", "BYTES", &[-1])],
            vec![
                meta("scores", "FP32", &[-1, info.labels.len() as i64]),
                meta("labels", "BYTES", &[info.labels.len() as i64]),
            ],
        ),
        ModelKind::TokenClassifier => (
            vec![meta("text", "BYTES", &[-1])],
            vec![meta("spans", "BYTES", &[-1]), meta("labels", "BYTES", &[info.labels.len() as i64])],
        ),
        ModelKind::Generative => {
            (vec![meta("prompt", "BYTES", &[1]), meta("messages", "BYTES", &[-1])], vec![meta("text", "BYTES", &[1])])
        }
        ModelKind::Generic => (
            info.inputs.iter().map(|t| meta(&t.name, datatype_of(t.dtype), &t.shape)).collect(),
            info.outputs.iter().map(|t| meta(&t.name, datatype_of(t.dtype), &t.shape)).collect(),
        ),
    };
    let mut properties = BTreeMap::new();
    properties.insert("model_id".into(), info.model_id.clone());
    properties.insert("revision".into(), info.revision.clone());
    properties.insert("task".into(), format!("{:?}", info.task).to_lowercase());
    properties.insert("kind".into(), format!("{:?}", info.kind));
    properties.insert("max_seq".into(), info.max_seq.to_string());
    properties.insert("max_batch".into(), info.max_batch.to_string());
    properties.insert("served_max_seq".into(), served.max_seq().to_string());
    properties.insert("served_max_batch".into(), served.max_batch().to_string());
    properties.insert("device".into(), served.device.name.clone());
    properties.insert("provider".into(), info.provider_id.clone());
    if let Some(p) = info.pooling {
        properties.insert("pooling".into(), format!("{p:?}").to_lowercase());
    }
    if let Some(n) = info.normalize {
        properties.insert("normalize".into(), format!("{n:?}").to_lowercase());
    }
    ModelMeta {
        name: served.name.clone(),
        versions: vec![MODEL_VERSION.to_string()],
        platform: info.provider_id.clone(),
        inputs,
        outputs,
        properties,
    }
}

// ---------------------------------------------------------------------------
// Request helpers
// ---------------------------------------------------------------------------

fn input<'a>(req: &'a InferRequest, name: &str) -> Result<&'a Tensor> {
    req.inputs.iter().find(|t| t.name == name).ok_or_else(|| {
        ServeError::bad_request(format!(
            "input `{name}` is missing; inputs given: {}",
            req.inputs.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", ")
        ))
    })
}

/// The strings of a BYTES tensor.
fn texts(t: &Tensor) -> Result<Vec<String>> {
    if t.datatype != "BYTES" {
        return Err(ServeError::bad_request(format!("input `{}` must be datatype BYTES, not {}", t.name, t.datatype)));
    }
    let Data::Bytes(items) = &t.data else {
        return Err(ServeError::bad_request(format!(
            "input `{}` is declared BYTES but carries {}",
            t.name,
            t.data.datatype()
        )));
    };
    let declared: i64 = t.shape.iter().product();
    if !t.shape.is_empty() && declared != items.len() as i64 {
        return Err(ServeError::bad_request(format!(
            "input `{}` declares shape {:?} ({declared} elements) but carries {} strings",
            t.name,
            t.shape,
            items.len()
        )));
    }
    items
        .iter()
        .enumerate()
        .map(|(i, b)| {
            String::from_utf8(b.clone())
                .map_err(|e| ServeError::bad_request(format!("input `{}`[{i}] is not UTF-8: {e}", t.name)))
        })
        .collect()
}

fn param<'a>(params: &'a BTreeMap<String, Param>, name: &str) -> Option<&'a Param> {
    params.get(name)
}

fn param_str(params: &BTreeMap<String, Param>, name: &str) -> Option<String> {
    param(params, name).map(Param::as_str)
}

fn param_u32(params: &BTreeMap<String, Param>, name: &str, field: u32) -> Result<Option<u32>> {
    match param(params, name) {
        None => Ok(None),
        Some(Param::Int(i)) if *i >= 0 && *i <= u32::MAX as i64 => Ok(Some(*i as u32)),
        Some(Param::Uint(u)) if *u <= u32::MAX as u64 => Ok(Some(*u as u32)),
        Some(Param::Str(s)) => {
            s.parse::<u32>().map(Some).map_err(|e| ServeError::field(field, format!("parameter `{name}` `{s}`: {e}")))
        }
        Some(other) => {
            Err(ServeError::field(field, format!("parameter `{name}` must be a non-negative integer, not {other:?}")))
        }
    }
}

fn param_f32(params: &BTreeMap<String, Param>, name: &str, field: u32) -> Result<Option<f32>> {
    match param(params, name) {
        None => Ok(None),
        Some(Param::Double(d)) => Ok(Some(*d as f32)),
        Some(Param::Int(i)) => Ok(Some(*i as f32)),
        Some(Param::Uint(u)) => Ok(Some(*u as f32)),
        Some(Param::Str(s)) => {
            s.parse::<f32>().map(Some).map_err(|e| ServeError::field(field, format!("parameter `{name}` `{s}`: {e}")))
        }
        Some(other) => Err(ServeError::field(field, format!("parameter `{name}` must be a number, not {other:?}"))),
    }
}

fn param_bool(params: &BTreeMap<String, Param>, name: &str, field: u32) -> Result<Option<bool>> {
    match param(params, name) {
        None => Ok(None),
        Some(Param::Bool(b)) => Ok(Some(*b)),
        Some(Param::Int(i)) => Ok(Some(*i != 0)),
        Some(Param::Str(s)) => match s.as_str() {
            "true" | "1" => Ok(Some(true)),
            "false" | "0" => Ok(Some(false)),
            other => Err(ServeError::field(field, format!("parameter `{name}` `{other}` is not true or false"))),
        },
        Some(other) => Err(ServeError::field(field, format!("parameter `{name}` must be a boolean, not {other:?}"))),
    }
}

/// Embedding options from a parameters map.
pub fn embed_options(params: &BTreeMap<String, Param>) -> Result<EmbedOptions> {
    Ok(EmbedOptions {
        truncate: engine::parse_truncate(param_str(params, "truncate").as_deref())?,
        max_tokens: param_u32(params, "max_tokens", EmbedOptions::FIELD_MAX_TOKENS)?.unwrap_or(0),
        prompt_role: engine::parse_prompt_role(param_str(params, "prompt_role").as_deref())?,
        normalize: engine::parse_normalize(param_str(params, "normalize").as_deref())?,
        pooling: engine::parse_pooling(param_str(params, "pooling").as_deref())?,
        output_dim: param_u32(params, "output_dim", EmbedOptions::FIELD_OUTPUT_DIM)?.unwrap_or(0),
        ..Default::default()
    })
}

/// Rerank options from a parameters map.
pub fn rerank_options(params: &BTreeMap<String, Param>) -> Result<RerankOptions> {
    Ok(RerankOptions {
        truncate: engine::parse_truncate(param_str(params, "truncate").as_deref())?,
        max_tokens: param_u32(params, "max_tokens", RerankOptions::FIELD_MAX_TOKENS)?.unwrap_or(0),
        top_n: param_u32(params, "top_n", RerankOptions::FIELD_TOP_N)?.unwrap_or(0),
        return_sorted: true,
        raw_scores: param_bool(params, "raw_scores", RerankOptions::FIELD_RAW_SCORES)?.unwrap_or(false),
    })
}

/// Classification options from a parameters map.
pub fn classify_options(params: &BTreeMap<String, Param>) -> Result<ClassifyOptions> {
    Ok(ClassifyOptions {
        truncate: engine::parse_truncate(param_str(params, "truncate").as_deref())?,
        max_tokens: param_u32(params, "max_tokens", ClassifyOptions::FIELD_MAX_TOKENS)?.unwrap_or(0),
        aggregation: engine::parse_aggregation(param_str(params, "aggregation").as_deref())?,
        raw_scores: param_bool(params, "raw_scores", ClassifyOptions::FIELD_RAW_SCORES)?.unwrap_or(false),
    })
}

/// Generation descriptor from a parameters map.
pub fn generate_desc(params: &BTreeMap<String, Param>) -> Result<GenerateDesc> {
    let mut d = GenerateDesc::default();
    if let Some(v) = param_u32(params, "max_new_tokens", GenerateDesc::FIELD_MAX_NEW_TOKENS)? {
        d.max_new_tokens = v;
    }
    if let Some(v) = param_u32(params, "min_new_tokens", GenerateDesc::FIELD_MIN_NEW_TOKENS)? {
        d.min_new_tokens = v;
    }
    if let Some(v) = param_f32(params, "temperature", GenerateDesc::FIELD_TEMPERATURE)? {
        d.temperature = v;
    }
    if let Some(v) = param_f32(params, "top_p", GenerateDesc::FIELD_TOP_P)? {
        d.top_p = v;
    }
    if let Some(v) = param_u32(params, "top_k", GenerateDesc::FIELD_TOP_K)? {
        d.top_k = v;
    }
    if let Some(v) = param_f32(params, "repeat_penalty", GenerateDesc::FIELD_REPEAT_PENALTY)? {
        d.repeat_penalty = v;
    }
    match param(params, "seed") {
        None => {}
        Some(Param::Int(i)) if *i >= 0 => d.seed = Some(*i as u64),
        Some(Param::Uint(u)) => d.seed = Some(*u),
        Some(Param::Str(s)) => {
            d.seed = Some(
                s.parse::<u64>()
                    .map_err(|e| ServeError::field(GenerateDesc::FIELD_SEED, format!("seed `{s}`: {e}")))?,
            )
        }
        Some(other) => {
            return Err(ServeError::field(
                GenerateDesc::FIELD_SEED,
                format!("seed must be a non-negative integer, not {other:?}"),
            ))
        }
    }
    if let Some(s) = param_str(params, "stop") {
        d.stop = s.split('\u{1f}').map(str::to_string).filter(|s| !s.is_empty()).collect();
        if d.stop.is_empty() {
            d.stop = vec![s];
        }
    }
    Ok(d)
}

// ---------------------------------------------------------------------------
// Infer
// ---------------------------------------------------------------------------

/// Run an inference request against a served model.
pub async fn infer(served: Arc<Served>, req: InferRequest) -> Result<InferResponse> {
    let mut parameters = BTreeMap::new();
    let outputs = match served.info().kind {
        ModelKind::Embedding => {
            let t = texts(input(&req, "text")?)?;
            let n = t.len() as i64;
            let e = engine::embed(served.clone(), t, embed_options(&req.parameters)?).await?;
            parameters.insert("device_ms".to_string(), Param::Double(e.device_ms));
            parameters.insert("placement".to_string(), Param::Str(format!("{:?}", e.placement)));
            if let Some(tokens) = e.tokens {
                parameters.insert("prompt_tokens".to_string(), Param::Int(tokens as i64));
            }
            vec![Tensor {
                name: "embeddings".into(),
                datatype: "FP32".into(),
                shape: vec![n, e.dim as i64],
                data: Data::Fp32(e.vectors.into_iter().flatten().collect()),
                parameters: BTreeMap::new(),
            }]
        }
        ModelKind::Reranker => {
            let q = texts(input(&req, "query")?)?;
            if q.len() != 1 {
                return Err(ServeError::bad_request(format!("input `query` must hold one string, not {}", q.len())));
            }
            let docs = texts(input(&req, "documents")?)?;
            let r = engine::rerank(
                served.clone(),
                q.into_iter().next().expect("one"),
                docs,
                rerank_options(&req.parameters)?,
            )
            .await?;
            parameters.insert("device_ms".to_string(), Param::Double(r.device_ms));
            vec![
                Tensor {
                    name: "scores".into(),
                    datatype: "FP32".into(),
                    shape: vec![r.scores.len() as i64],
                    data: Data::Fp32(r.scores),
                    parameters: BTreeMap::new(),
                },
                Tensor {
                    name: "sorted".into(),
                    datatype: "INT32".into(),
                    shape: vec![r.sorted.len() as i64],
                    data: Data::Int32(r.sorted),
                    parameters: BTreeMap::new(),
                },
            ]
        }
        ModelKind::Classifier => {
            let t = texts(input(&req, "text")?)?;
            let c = engine::classify(served.clone(), t, classify_options(&req.parameters)?).await?;
            parameters.insert("device_ms".to_string(), Param::Double(c.device_ms));
            let rows = c.scores.len() as i64;
            let width = c.labels.len() as i64;
            vec![
                Tensor {
                    name: "scores".into(),
                    datatype: "FP32".into(),
                    shape: vec![rows, width],
                    data: Data::Fp32(c.scores.into_iter().flatten().collect()),
                    parameters: BTreeMap::new(),
                },
                labels_tensor(&c.labels),
            ]
        }
        ModelKind::TokenClassifier => {
            let t = texts(input(&req, "text")?)?;
            let tagged = engine::token_classify(served.clone(), t, classify_options(&req.parameters)?).await?;
            parameters.insert("device_ms".to_string(), Param::Double(tagged.device_ms));
            let mut spans = Vec::new();
            for (row, list) in tagged.spans.iter().enumerate() {
                for s in list {
                    let label = tagged.labels.get(s.label as usize).cloned().unwrap_or_else(|| s.label.to_string());
                    let json = serde_json::json!({
                        "row": row,
                        "byte_start": s.byte_start,
                        "byte_end": s.byte_end,
                        "label": label,
                        "score": s.score,
                    });
                    spans.push(json.to_string().into_bytes());
                }
            }
            vec![
                Tensor {
                    name: "spans".into(),
                    datatype: "BYTES".into(),
                    shape: vec![spans.len() as i64],
                    data: Data::Bytes(spans),
                    parameters: BTreeMap::new(),
                },
                labels_tensor(&tagged.labels),
            ]
        }
        ModelKind::Generative => {
            let messages = if let Some(t) = req.inputs.iter().find(|t| t.name == "messages") {
                let turns = texts(t)?;
                let mut msgs = Vec::new();
                for (i, turn) in turns.iter().enumerate() {
                    #[derive(Deserialize)]
                    struct Turn {
                        role: String,
                        content: String,
                    }
                    let t: Turn = serde_json::from_str(turn).map_err(|e| {
                        ServeError::bad_request(format!(
                            "input `messages`[{i}] is not a {{\"role\", \"content\"}} object: {e}"
                        ))
                    })?;
                    msgs.push((t.role, t.content));
                }
                msgs
            } else {
                let p = texts(input(&req, "prompt")?)?;
                if p.len() != 1 {
                    return Err(ServeError::bad_request(format!(
                        "input `prompt` must hold one string, not {}",
                        p.len()
                    )));
                }
                vec![("user".to_string(), p.into_iter().next().expect("one"))]
            };
            let mut rx = engine::generate(served.clone(), messages, generate_desc(&req.parameters)?).await?;
            let mut text = String::new();
            let mut generated = 0;
            let mut prompt_tokens = 0;
            let mut finish = String::new();
            while let Some(piece) = rx.recv().await {
                let piece = piece?;
                text.push_str(&piece.text);
                generated = piece.generated_tokens;
                prompt_tokens = piece.prompt_tokens;
                if piece.done {
                    finish = format!("{:?}", piece.finish_reason).to_lowercase();
                }
            }
            parameters.insert("finish_reason".to_string(), Param::Str(finish));
            parameters.insert("prompt_tokens".to_string(), Param::Int(prompt_tokens as i64));
            parameters.insert("generated_tokens".to_string(), Param::Int(generated as i64));
            vec![Tensor {
                name: "text".into(),
                datatype: "BYTES".into(),
                shape: vec![1],
                data: Data::Bytes(vec![text.into_bytes()]),
                parameters: BTreeMap::new(),
            }]
        }
        ModelKind::Generic => {
            let mut inputs = Vec::new();
            for t in &req.inputs {
                let dtype = dtype_of(&t.datatype)?;
                let shape: Vec<u64> = t
                    .shape
                    .iter()
                    .map(|&d| {
                        u64::try_from(d)
                            .map_err(|_| ServeError::bad_request(format!("input `{}` has a negative extent", t.name)))
                    })
                    .collect::<Result<_>>()?;
                inputs.push(RunInput { name: t.name.clone(), dtype, shape, bytes: pack(&t.data, dtype, &t.name)? });
            }
            let params: Vec<(String, String)> = req
                .parameters
                .iter()
                .filter(|(k, _)| !k.starts_with("turbo_internal"))
                .map(|(k, v)| (k.clone(), v.as_str()))
                .collect();
            let (outs, ms) = engine::run_generic(served.clone(), inputs, params).await?;
            parameters.insert("device_ms".to_string(), Param::Double(ms));
            outs.into_iter()
                .map(|o| {
                    Ok(Tensor {
                        name: o.name,
                        datatype: datatype_of(o.dtype).to_string(),
                        shape: o.shape.iter().map(|&d| d as i64).collect(),
                        data: unpack(&o.bytes, o.dtype)?,
                        parameters: BTreeMap::new(),
                    })
                })
                .collect::<Result<Vec<_>>>()?
        }
    };
    let outputs = if req.outputs.is_empty() {
        outputs
    } else {
        for want in &req.outputs {
            if !outputs.iter().any(|o| &o.name == want) {
                return Err(ServeError::bad_request(format!(
                    "requested output `{want}` is not produced; outputs: {}",
                    outputs.iter().map(|o| o.name.as_str()).collect::<Vec<_>>().join(", ")
                )));
            }
        }
        outputs.into_iter().filter(|o| req.outputs.contains(&o.name)).collect()
    };
    Ok(InferResponse {
        model_name: served.name.clone(),
        model_version: MODEL_VERSION.to_string(),
        id: req.id,
        parameters,
        outputs,
    })
}

fn labels_tensor(labels: &[String]) -> Tensor {
    Tensor {
        name: "labels".into(),
        datatype: "BYTES".into(),
        shape: vec![labels.len() as i64],
        data: Data::Bytes(labels.iter().map(|l| l.as_bytes().to_vec()).collect()),
        parameters: BTreeMap::new(),
    }
}

/// Typed contents packed as little-endian host bytes of `dtype`.
fn pack(data: &Data, dtype: DType, name: &str) -> Result<Vec<u8>> {
    let mismatch = || {
        ServeError::bad_request(format!(
            "input `{name}` is declared {} but carries {}",
            datatype_of(dtype),
            data.datatype()
        ))
    };
    Ok(match (data, dtype) {
        (Data::Fp32(v), DType::F32) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        (Data::Fp64(v), DType::F64) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        (Data::Int32(v), DType::I32) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        (Data::Int64(v), DType::I64) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        (Data::Uint32(v), DType::U32) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        (Data::Uint64(v), DType::U64) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        (Data::Bool(v), DType::Bool) => v.iter().map(|&b| u8::from(b)).collect(),
        (Data::Bytes(_), DType::Bytes) => {
            return Err(ServeError::bad_request(format!(
                "input `{name}`: BYTES inputs are not bound to generic models"
            )))
        }
        _ => return Err(mismatch()),
    })
}

/// Host bytes of `dtype` as typed contents.
fn unpack(bytes: &[u8], dtype: DType) -> Result<Data> {
    Ok(match dtype {
        DType::F32 => {
            Data::Fp32(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes"))).collect())
        }
        DType::F64 => {
            Data::Fp64(bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes"))).collect())
        }
        DType::I32 => {
            Data::Int32(bytes.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().expect("4 bytes"))).collect())
        }
        DType::I64 => {
            Data::Int64(bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes"))).collect())
        }
        DType::U32 => {
            Data::Uint32(bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes"))).collect())
        }
        DType::U64 => {
            Data::Uint64(bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes"))).collect())
        }
        DType::Bool => Data::Bool(bytes.iter().map(|&b| b != 0).collect()),
        other => {
            return Err(ServeError::from(turbo::Error::new(
                turbo::abi::TURBO_E_UNSUPPORTED_DTYPE,
                format!("output dtype {other:?} is not carried as typed contents by this server"),
            )))
        }
    })
}
