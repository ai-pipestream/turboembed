//! llama.cpp (GGUF) backend for inferstream.
//!
//! One crate serves every llama.cpp build flavor; the *device* is decided by
//! how the native library was compiled plus the per-model `device` config:
//!
//! | `device` | build requirement | used by |
//! |---|---|---|
//! | `cuda`   | llama.cpp built with `GGML_CUDA`   | `inferstream-nvidia` (fallback/GGUF path under TRT-LLM) |
//! | `sycl`   | llama.cpp built with `GGML_SYCL` against Intel oneAPI | `inferstream-intel` (Arc/Battlemage via Level Zero) |
//! | `metal`  | llama.cpp built with Metal (default on macOS) | `inferstream-apple` (alternative to MLX for GGUF) |
//! | `vulkan` | llama.cpp built with `GGML_VULKAN` | portability escape hatch |
//! | `cpu`    | any build | everywhere |
//!
//! ## Two integration modes
//!
//! **Server-client mode (live today).** When the model config sets
//! `endpoint`, this backend forwards to a running `llama-server` over its
//! native HTTP API instead of linking the engine in-process. The device
//! question is then answered by how *that* server binary was built (e.g. the
//! `ghcr.io/ggml-org/llama.cpp:server-intel` image is a SYCL build). Wire
//! shapes match the mock backend exactly:
//! * generation → unary `infer`: one `BYTES` input `text` (alias `prompt`),
//!   one `BYTES` output `text` with the full completion; `tokens_predicted` /
//!   `tokens_evaluated` come back as response parameters.
//! * token streaming → `infer_stream`: `POST /completion` with `stream:true`,
//!   one `BYTES` output `token` chunk per SSE event, `final` bool parameter
//!   on the last chunk.
//! * `tokenize` / `detokenize` → the server's `/tokenize` (`with_pieces`) and
//!   `/detokenize` endpoints. llama.cpp reports no byte offsets, so
//!   `with_offsets` yields empty offset lists; `skip_special_tokens` is
//!   delegated to the server's own detokenizer rendering.
//!
//! Generation request parameters (all optional): `max_tokens` (int64 →
//! `n_predict`, default 128), `temperature` (double), `top_p` (double),
//! `seed` (int64), `stop` (string, single stop sequence).
//!
//! **In-process FFI mode (crate features `runtime` / `cuda` / `metal`).**
//! With `path` (a GGUF file) and no `endpoint`, the engine links llama.cpp
//! in-process through the maintained `llama-cpp-2` crate: eager model load at
//! startup, per-request context in `spawn_blocking`, live token streaming
//! with the same wire shapes as server-client mode, and Tokenize/Detokenize
//! from the GGUF vocabulary (no byte offsets; `pad_to_longest` unsupported).
//! The `stop` parameter is server-client-only today. Without the `runtime`
//! feature, path-only models construct but report `Unavailable` at request
//! time, naming the feature to enable. SYCL/Vulkan in-process flavors are
//! not compiled by this crate yet — use server-client mode for those (e.g.
//! krick-1's `server-intel` llama-server container).

use std::collections::{HashMap, VecDeque};

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use inferstream_backend::{
    Backend, BackendError, ModelMetadata, ResponseStream, TokenizeOptions,
};
use inferstream_protocol::extension::Encoding;
use inferstream_protocol::inference::{
    infer_parameter::ParameterChoice, model_infer_response::InferOutputTensor,
    model_metadata_response::TensorMetadata, InferParameter, ModelInferRequest,
    ModelInferResponse,
};
use inferstream_protocol::tensor::{pack_bytes, unpack_bytes, DataType};
use serde::Deserialize;
use serde_json::json;

/// Device a llama.cpp model should run on. Must match a capability the
/// linked llama.cpp library was actually built with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LlamaDevice {
    Cuda,
    Sycl,
    Metal,
    Vulkan,
    #[default]
    Cpu,
}

impl LlamaDevice {
    pub fn from_config(s: &str) -> Result<Self, BackendError> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "cuda" => Self::Cuda,
            "sycl" => Self::Sycl,
            "metal" => Self::Metal,
            "vulkan" => Self::Vulkan,
            "cpu" => Self::Cpu,
            other => {
                return Err(BackendError::InvalidRequest(format!(
                    "unknown llama.cpp device {other:?} (expected cuda|sycl|metal|vulkan|cpu)"
                )))
            }
        })
    }
}

/// Configuration for one llama.cpp-served model.
#[derive(Debug, Clone, Default)]
pub struct LlamaCppConfig {
    /// Path to the GGUF file (in-process FFI mode; stub today).
    pub model_path: String,
    /// Base URL of a running `llama-server` (server-client mode), e.g.
    /// `"http://127.0.0.1:8085"`. When set, `model_path` is not required.
    pub endpoint: Option<String>,
    /// Target device (see [`LlamaDevice`]). In server-client mode this is
    /// informational: the server binary's build decides the real device.
    pub device: LlamaDevice,
    /// Layers to offload to the accelerator; `None` = offload everything.
    pub n_gpu_layers: Option<u32>,
    /// Concurrent sequences the context schedules (`n_parallel`).
    pub max_batch_size: Option<u32>,
    /// In-process mode: context window (`n_ctx`); `None` = 4096 capped to
    /// the model's training context.
    pub n_ctx: Option<u32>,
}

#[cfg(feature = "runtime")]
mod engine;

/// HTTP client onto one running `llama-server`.
#[derive(Debug, Clone)]
struct ServerClient {
    http: reqwest::Client,
    base: String,
}

impl ServerClient {
    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }
}

/// llama.cpp backend: server-client mode when `endpoint` is configured,
/// in-process engine (feature `runtime`) or stub otherwise.
#[derive(Debug, Default, Clone)]
pub struct LlamaCppBackend {
    config: LlamaCppConfig,
    server: Option<ServerClient>,
    #[cfg(feature = "runtime")]
    engine: Option<engine::LlamaEngine>,
}

/// Default `n_predict` when the request carries no `max_tokens` parameter.
const DEFAULT_MAX_TOKENS: i64 = 128;

/// Generation knobs decoded from OIP request parameters.
#[derive(Debug, Clone, PartialEq)]
struct GenParams {
    n_predict: i64,
    temperature: Option<f64>,
    top_p: Option<f64>,
    seed: Option<i64>,
    stop: Option<String>,
}

fn gen_params(parameters: &HashMap<String, InferParameter>) -> GenParams {
    let int = |key: &str| match parameters.get(key).and_then(|p| p.parameter_choice.as_ref()) {
        Some(ParameterChoice::Int64Param(v)) => Some(*v),
        _ => None,
    };
    let double = |key: &str| match parameters.get(key).and_then(|p| p.parameter_choice.as_ref()) {
        Some(ParameterChoice::DoubleParam(v)) => Some(*v),
        _ => None,
    };
    let string = |key: &str| match parameters.get(key).and_then(|p| p.parameter_choice.as_ref()) {
        Some(ParameterChoice::StringParam(v)) => Some(v.clone()),
        _ => None,
    };
    GenParams {
        n_predict: int("max_tokens").unwrap_or(DEFAULT_MAX_TOKENS),
        temperature: double("temperature"),
        top_p: double("top_p"),
        seed: int("seed"),
        stop: string("stop"),
    }
}

fn completion_body(prompt: &str, params: &GenParams, stream: bool) -> serde_json::Value {
    let mut body = json!({
        "prompt": prompt,
        "n_predict": params.n_predict,
        "stream": stream,
        "cache_prompt": true,
    });
    let map = body.as_object_mut().expect("body is an object");
    if let Some(t) = params.temperature {
        map.insert("temperature".into(), json!(t));
    }
    if let Some(p) = params.top_p {
        map.insert("top_p".into(), json!(p));
    }
    if let Some(s) = params.seed {
        map.insert("seed".into(), json!(s));
    }
    if let Some(stop) = &params.stop {
        map.insert("stop".into(), json!([stop]));
    }
    body
}

/// Extract the single `BYTES` prompt input, named `text` or `prompt`
/// (same façade convention as the mock backend's generation surface).
fn prompt_from(request: &ModelInferRequest) -> Result<String, BackendError> {
    if request.inputs.len() != 1 {
        return Err(BackendError::InvalidRequest(format!(
            "generation expects exactly one input tensor, got {}",
            request.inputs.len()
        )));
    }
    let input = &request.inputs[0];
    if input.name != "text" && input.name != "prompt" {
        return Err(BackendError::InvalidRequest(format!(
            "generation input must be named \"text\" or \"prompt\", got {:?}",
            input.name
        )));
    }
    if input.datatype != DataType::Bytes.as_oip() {
        return Err(BackendError::InvalidRequest(format!(
            "input {:?} must be BYTES, got {:?}",
            input.name, input.datatype
        )));
    }
    let element = if let Some(raw) = request.raw_input_contents.first() {
        unpack_bytes(raw)
            .map_err(|e| BackendError::InvalidRequest(format!("bad BYTES payload: {e}")))?
            .into_iter()
            .next()
    } else {
        input
            .contents
            .as_ref()
            .and_then(|c| c.bytes_contents.first().cloned())
    };
    let element = element.ok_or_else(|| {
        BackendError::InvalidRequest("prompt tensor carries no elements".to_string())
    })?;
    String::from_utf8(element)
        .map_err(|e| BackendError::InvalidRequest(format!("prompt is not UTF-8: {e}")))
}

/// One `/completion` result or SSE chunk from llama-server.
#[derive(Debug, Deserialize)]
struct CompletionChunk {
    #[serde(default)]
    content: String,
    #[serde(default)]
    stop: bool,
    #[serde(default)]
    tokens_predicted: Option<i64>,
    #[serde(default)]
    tokens_evaluated: Option<i64>,
}

/// Split complete SSE events out of `buf` (which keeps any trailing partial
/// event) and return their `data:` payloads in arrival order.
fn drain_sse_events(buf: &mut Vec<u8>) -> Vec<String> {
    let mut payloads = Vec::new();
    while let Some(boundary) = buf.windows(2).position(|w| w == b"\n\n") {
        let event: Vec<u8> = buf.drain(..boundary + 2).collect();
        for line in event.split(|&b| b == b'\n') {
            let line = String::from_utf8_lossy(line);
            let line = line.trim_end_matches('\r');
            if let Some(data) = line.strip_prefix("data:") {
                payloads.push(data.trim_start().to_string());
            }
        }
    }
    payloads
}

fn bool_param(value: bool) -> InferParameter {
    InferParameter {
        parameter_choice: Some(ParameterChoice::BoolParam(value)),
    }
}

fn int_param(value: i64) -> InferParameter {
    InferParameter {
        parameter_choice: Some(ParameterChoice::Int64Param(value)),
    }
}

/// Request identity echoed on every streamed chunk.
#[derive(Debug, Clone)]
struct ChunkMeta {
    model_name: String,
    model_version: String,
    id: String,
}

/// Build one streamed `token` chunk (mock-backend wire shape: `BYTES` output
/// `token`, `final` bool parameter, token counters on the last chunk).
fn token_chunk(meta: &ChunkMeta, chunk: &CompletionChunk) -> ModelInferResponse {
    let mut parameters = HashMap::from([("final".to_string(), bool_param(chunk.stop))]);
    if chunk.stop {
        if let Some(n) = chunk.tokens_predicted {
            parameters.insert("tokens_predicted".to_string(), int_param(n));
        }
        if let Some(n) = chunk.tokens_evaluated {
            parameters.insert("tokens_evaluated".to_string(), int_param(n));
        }
    }
    ModelInferResponse {
        model_name: meta.model_name.clone(),
        model_version: meta.model_version.clone(),
        id: meta.id.clone(),
        parameters,
        outputs: vec![InferOutputTensor {
            name: "token".to_string(),
            datatype: DataType::Bytes.as_oip().to_string(),
            shape: vec![1],
            parameters: HashMap::new(),
            contents: None,
        }],
        raw_output_contents: vec![pack_bytes(&[chunk.content.as_bytes()])],
    }
}

/// `/tokenize?with_pieces=true` element: llama-server returns the piece as a
/// string, or as raw bytes when it is not valid UTF-8 on its own.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PieceRepr {
    Text(String),
    Bytes(Vec<u8>),
}

impl PieceRepr {
    fn into_string(self) -> String {
        match self {
            Self::Text(s) => s,
            Self::Bytes(b) => String::from_utf8_lossy(&b).into_owned(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct TokenizePiece {
    id: u32,
    piece: PieceRepr,
}

#[derive(Debug, Deserialize)]
struct TokenizeResponseBody {
    tokens: Vec<TokenizePiece>,
}

#[derive(Debug, Deserialize)]
struct DetokenizeResponseBody {
    content: String,
}

fn connect_err(base: &str, e: reqwest::Error) -> BackendError {
    BackendError::Unavailable(format!("llama-server at {base} is unreachable: {e}"))
}

async fn check_http_error(response: reqwest::Response) -> Result<reqwest::Response, BackendError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response.text().await.unwrap_or_default();
    let detail = body.chars().take(300).collect::<String>();
    Err(BackendError::Internal(format!(
        "llama-server returned HTTP {status}: {detail}"
    )))
}

impl LlamaCppBackend {
    pub fn new(config: LlamaCppConfig) -> Result<Self, BackendError> {
        let server = match config.endpoint.as_deref() {
            Some(endpoint) if !endpoint.is_empty() => Some(ServerClient {
                http: reqwest::Client::new(),
                base: endpoint.trim_end_matches('/').to_string(),
            }),
            _ => None,
        };
        if server.is_none() && config.model_path.is_empty() {
            return Err(BackendError::InvalidRequest(
                "llama-cpp models require `path` (a GGUF file, in-process mode) or \
                 `endpoint` (a running llama-server, server-client mode)"
                    .into(),
            ));
        }
        // In-process engine: only when no endpoint routes the model to a
        // llama-server. Loads eagerly so a bad GGUF fails at startup.
        #[cfg(feature = "runtime")]
        let engine = if server.is_none() {
            Some(engine::LlamaEngine::new(&config)?)
        } else {
            None
        };
        Ok(Self {
            config,
            server,
            #[cfg(feature = "runtime")]
            engine,
        })
    }

    pub fn config(&self) -> &LlamaCppConfig {
        &self.config
    }

    fn server(&self) -> Result<&ServerClient, BackendError> {
        self.server.as_ref().ok_or_else(Self::unavailable)
    }

    /// The in-process engine, when this model runs in FFI mode.
    #[cfg(feature = "runtime")]
    fn engine(&self) -> Option<&engine::LlamaEngine> {
        self.engine.as_ref()
    }

    fn unavailable() -> BackendError {
        BackendError::Unavailable(
            "llama.cpp in-process mode is not compiled into this binary; rebuild with \
             the crate's `runtime` (CPU) or `cuda` feature (nvidia: llamacpp-runtime / \
             llamacpp-cuda / full-cuda), or configure `endpoint` to forward to a \
             running llama-server"
                .into(),
        )
    }
}

#[async_trait]
impl Backend for LlamaCppBackend {
    fn id(&self) -> &str {
        "llama-cpp"
    }

    async fn model_ready(&self, _model_name: &str, _model_version: &str) -> bool {
        #[cfg(feature = "runtime")]
        if self.engine().is_some() {
            return true;
        }
        let Some(server) = &self.server else {
            return false;
        };
        match server.http.get(server.url("/health")).send().await {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        }
    }

    async fn model_metadata(
        &self,
        model_name: &str,
        _model_version: &str,
    ) -> Result<ModelMetadata, BackendError> {
        #[cfg(feature = "runtime")]
        if let Some(engine) = self.engine() {
            return Ok(engine.metadata(model_name));
        }
        let server = self.server()?;
        let mut properties = HashMap::from([
            ("endpoint".to_string(), server.base.clone()),
            ("device".to_string(), format!("{:?}", self.config.device)),
        ]);
        // Best-effort: surface the GGUF the server actually loaded.
        if let Ok(response) = server.http.get(server.url("/props")).send().await {
            if let Ok(props) = response.json::<serde_json::Value>().await {
                if let Some(path) = props.get("model_path").and_then(|v| v.as_str()) {
                    properties.insert("model_path".to_string(), path.to_string());
                }
            }
        }
        let bytes = DataType::Bytes.as_oip().to_string();
        Ok(ModelMetadata {
            name: model_name.to_string(),
            versions: vec!["1".to_string()],
            platform: "llama_cpp".to_string(),
            inputs: vec![TensorMetadata {
                name: "text".to_string(),
                datatype: bytes.clone(),
                shape: vec![-1],
            }],
            outputs: vec![
                TensorMetadata {
                    name: "text".to_string(),
                    datatype: bytes.clone(),
                    shape: vec![1],
                },
                TensorMetadata {
                    name: "token".to_string(),
                    datatype: bytes,
                    shape: vec![1],
                },
            ],
            properties,
        })
    }

    async fn infer(&self, request: ModelInferRequest) -> Result<ModelInferResponse, BackendError> {
        #[cfg(feature = "runtime")]
        if let Some(engine) = self.engine() {
            return engine.infer(request).await;
        }
        let server = self.server()?;
        let prompt = prompt_from(&request)?;
        let params = gen_params(&request.parameters);
        let response = server
            .http
            .post(server.url("/completion"))
            .json(&completion_body(&prompt, &params, false))
            .send()
            .await
            .map_err(|e| connect_err(&server.base, e))?;
        let completion: CompletionChunk = check_http_error(response)
            .await?
            .json()
            .await
            .map_err(|e| BackendError::Internal(format!("bad /completion response: {e}")))?;

        let mut parameters = HashMap::new();
        if let Some(n) = completion.tokens_predicted {
            parameters.insert("tokens_predicted".to_string(), int_param(n));
        }
        if let Some(n) = completion.tokens_evaluated {
            parameters.insert("tokens_evaluated".to_string(), int_param(n));
        }
        Ok(ModelInferResponse {
            model_name: request.model_name,
            model_version: request.model_version,
            id: request.id,
            parameters,
            outputs: vec![InferOutputTensor {
                name: "text".to_string(),
                datatype: DataType::Bytes.as_oip().to_string(),
                shape: vec![1],
                parameters: HashMap::new(),
                contents: None,
            }],
            raw_output_contents: vec![pack_bytes(&[completion.content.as_bytes()])],
        })
    }

    async fn infer_stream(
        &self,
        request: ModelInferRequest,
    ) -> Result<ResponseStream, BackendError> {
        #[cfg(feature = "runtime")]
        if let Some(engine) = self.engine() {
            return engine.infer_stream(request);
        }
        let server = self.server()?.clone();
        let prompt = prompt_from(&request)?;
        let params = gen_params(&request.parameters);
        let response = server
            .http
            .post(server.url("/completion"))
            .json(&completion_body(&prompt, &params, true))
            .send()
            .await
            .map_err(|e| connect_err(&server.base, e))?;
        let response = check_http_error(response).await?;

        struct SseState {
            bytes: BoxStream<'static, reqwest::Result<bytes::Bytes>>,
            buf: Vec<u8>,
            pending: VecDeque<Result<ModelInferResponse, BackendError>>,
            meta: ChunkMeta,
            done: bool,
        }

        let state = SseState {
            bytes: response.bytes_stream().boxed(),
            buf: Vec::new(),
            pending: VecDeque::new(),
            meta: ChunkMeta {
                model_name: request.model_name,
                model_version: request.model_version,
                id: request.id,
            },
            done: false,
        };

        let stream = futures::stream::unfold(state, |mut s| async move {
            loop {
                if let Some(item) = s.pending.pop_front() {
                    return Some((item, s));
                }
                if s.done {
                    return None;
                }
                match s.bytes.next().await {
                    Some(Ok(chunk)) => {
                        s.buf.extend_from_slice(&chunk);
                        for payload in drain_sse_events(&mut s.buf) {
                            // The OpenAI-compatible endpoints terminate with
                            // `data: [DONE]`; the native /completion endpoint
                            // flags the last JSON chunk with `stop:true`.
                            if payload == "[DONE]" {
                                s.done = true;
                                break;
                            }
                            match serde_json::from_str::<CompletionChunk>(&payload) {
                                Ok(completion) => {
                                    let is_final = completion.stop;
                                    s.pending.push_back(Ok(token_chunk(&s.meta, &completion)));
                                    if is_final {
                                        s.done = true;
                                        break;
                                    }
                                }
                                Err(e) => {
                                    s.pending.push_back(Err(BackendError::Internal(format!(
                                        "bad SSE chunk from llama-server: {e}"
                                    ))));
                                    s.done = true;
                                    break;
                                }
                            }
                        }
                    }
                    Some(Err(e)) => {
                        s.done = true;
                        return Some((
                            Err(BackendError::Internal(format!(
                                "llama-server stream failed mid-generation: {e}"
                            ))),
                            s,
                        ));
                    }
                    // Upstream closed without a stop flag: end the stream.
                    None => return None,
                }
            }
        });
        Ok(Box::pin(stream))
    }

    async fn tokenize(
        &self,
        _model_name: &str,
        texts: &[String],
        options: &TokenizeOptions,
    ) -> Result<Vec<Encoding>, BackendError> {
        #[cfg(feature = "runtime")]
        if let Some(engine) = self.engine() {
            return engine.tokenize(texts, options);
        }
        let server = self.server()?;
        let mut encodings = Vec::with_capacity(texts.len());
        for text in texts {
            let response = server
                .http
                .post(server.url("/tokenize"))
                .json(&json!({
                    "content": text,
                    "add_special": options.add_special_tokens,
                    "with_pieces": true,
                }))
                .send()
                .await
                .map_err(|e| connect_err(&server.base, e))?;
            let body: TokenizeResponseBody = check_http_error(response)
                .await?
                .json()
                .await
                .map_err(|e| BackendError::Internal(format!("bad /tokenize response: {e}")))?;
            let mut encoding = Encoding::default();
            for piece in body.tokens {
                encoding.input_ids.push(piece.id);
                encoding.tokens.push(piece.piece.into_string());
            }
            if let Some(limit) = options.truncate_to {
                encoding.input_ids.truncate(limit);
                encoding.tokens.truncate(limit);
            }
            encoding.attention_mask = vec![1; encoding.input_ids.len()];
            // llama-server reports no byte offsets; `with_offsets` requests
            // get empty offset lists (documented in the crate docs).
            encodings.push(encoding);
        }
        if options.pad_to_longest {
            let longest = encodings
                .iter()
                .map(|e| e.input_ids.len())
                .max()
                .unwrap_or(0);
            for encoding in &mut encodings {
                while encoding.input_ids.len() < longest {
                    encoding.input_ids.push(0);
                    encoding.attention_mask.push(0);
                    encoding.tokens.push(String::new());
                }
            }
        }
        Ok(encodings)
    }

    async fn detokenize(
        &self,
        _model_name: &str,
        sequences: &[Vec<u32>],
        _skip_special_tokens: bool,
    ) -> Result<Vec<String>, BackendError> {
        #[cfg(feature = "runtime")]
        if let Some(engine) = self.engine() {
            return engine.detokenize(sequences, _skip_special_tokens);
        }
        let server = self.server()?;
        let mut texts = Vec::with_capacity(sequences.len());
        for ids in sequences {
            let response = server
                .http
                .post(server.url("/detokenize"))
                .json(&json!({ "tokens": ids }))
                .send()
                .await
                .map_err(|e| connect_err(&server.base, e))?;
            let body: DetokenizeResponseBody = check_http_error(response)
                .await?
                .json()
                .await
                .map_err(|e| BackendError::Internal(format!("bad /detokenize response: {e}")))?;
            texts.push(body.content);
        }
        Ok(texts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_parsing() {
        assert_eq!(LlamaDevice::from_config("CUDA").unwrap(), LlamaDevice::Cuda);
        assert_eq!(LlamaDevice::from_config("sycl").unwrap(), LlamaDevice::Sycl);
        assert!(LlamaDevice::from_config("tpu").is_err());
    }

    #[test]
    fn requires_model_path_or_endpoint() {
        assert!(LlamaCppBackend::new(LlamaCppConfig::default()).is_err());
        // Path-only construction succeeds in the stub build; with `runtime`
        // the engine loads eagerly, so a nonexistent GGUF fails at startup.
        let path_only = LlamaCppBackend::new(LlamaCppConfig {
            model_path: "/models/x.gguf".into(),
            ..Default::default()
        });
        #[cfg(not(feature = "runtime"))]
        assert!(path_only.is_ok());
        #[cfg(feature = "runtime")]
        assert!(matches!(path_only, Err(BackendError::Unavailable(_))));
        let server_mode = LlamaCppBackend::new(LlamaCppConfig {
            endpoint: Some("http://127.0.0.1:8085/".into()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            server_mode.server.as_ref().unwrap().base,
            "http://127.0.0.1:8085",
            "trailing slash trimmed"
        );
        // Empty endpoint string does not count as server mode.
        assert!(LlamaCppBackend::new(LlamaCppConfig {
            endpoint: Some(String::new()),
            ..Default::default()
        })
        .is_err());
    }

    #[cfg(not(feature = "runtime"))]
    #[tokio::test]
    async fn path_only_mode_is_a_stub() {
        let backend = LlamaCppBackend::new(LlamaCppConfig {
            model_path: "/models/x.gguf".into(),
            ..Default::default()
        })
        .unwrap();
        assert!(!backend.model_ready("m", "").await);
        assert!(matches!(
            backend.infer(ModelInferRequest::default()).await,
            Err(BackendError::Unavailable(_))
        ));
    }

    fn text_request(prompt: &str) -> ModelInferRequest {
        use inferstream_protocol::inference::model_infer_request::InferInputTensor;
        ModelInferRequest {
            model_name: "m".into(),
            id: "req-1".into(),
            inputs: vec![InferInputTensor {
                name: "text".into(),
                datatype: DataType::Bytes.as_oip().into(),
                shape: vec![1],
                ..Default::default()
            }],
            raw_input_contents: vec![pack_bytes(&[prompt.as_bytes()])],
            ..Default::default()
        }
    }

    #[test]
    fn prompt_extraction_accepts_text_and_prompt_names() {
        assert_eq!(prompt_from(&text_request("hello")).unwrap(), "hello");
        let mut req = text_request("hi");
        req.inputs[0].name = "prompt".into();
        assert_eq!(prompt_from(&req).unwrap(), "hi");
        req.inputs[0].name = "embedding".into();
        assert!(matches!(
            prompt_from(&req),
            Err(BackendError::InvalidRequest(_))
        ));
        let mut req = text_request("hi");
        req.inputs[0].datatype = "FP32".into();
        assert!(matches!(
            prompt_from(&req),
            Err(BackendError::InvalidRequest(_))
        ));
    }

    #[test]
    fn gen_params_map_oip_parameters_to_llama_knobs() {
        let mut parameters = HashMap::new();
        parameters.insert("max_tokens".to_string(), int_param(42));
        parameters.insert(
            "temperature".to_string(),
            InferParameter {
                parameter_choice: Some(ParameterChoice::DoubleParam(0.2)),
            },
        );
        parameters.insert(
            "stop".to_string(),
            InferParameter {
                parameter_choice: Some(ParameterChoice::StringParam("\n\n".into())),
            },
        );
        let params = gen_params(&parameters);
        assert_eq!(params.n_predict, 42);
        assert_eq!(params.temperature, Some(0.2));
        assert_eq!(params.stop.as_deref(), Some("\n\n"));
        assert_eq!(gen_params(&HashMap::new()).n_predict, DEFAULT_MAX_TOKENS);

        let body = completion_body("p", &params, true);
        assert_eq!(body["n_predict"], 42);
        assert_eq!(body["stream"], true);
        assert_eq!(body["stop"][0], "\n\n");
    }

    #[test]
    fn sse_events_drain_incrementally() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"data: {\"content\":\"a\"}\n\ndata: {\"cont");
        let events = drain_sse_events(&mut buf);
        assert_eq!(events, vec!["{\"content\":\"a\"}"]);
        assert_eq!(buf, b"data: {\"cont", "partial event stays buffered");
        buf.extend_from_slice(b"ent\":\"b\"}\n\n");
        let events = drain_sse_events(&mut buf);
        assert_eq!(events, vec!["{\"content\":\"b\"}"]);
        assert!(buf.is_empty());
    }

    #[test]
    fn sse_events_handle_crlf_and_done_sentinel() {
        // llama.cpp uses \n\n; CRLF line ends must still parse (payloads
        // surface once an LF-LF boundary lands in the buffer).
        let mut buf = b"data: {\"x\":1}\r\n\r\ndata: [DONE]\n\n".to_vec();
        let events = drain_sse_events(&mut buf);
        assert!(events.contains(&"[DONE]".to_string()));
        assert!(events.iter().any(|e| e == "{\"x\":1}"));
    }

    #[test]
    fn token_chunks_match_mock_wire_shape() {
        let meta = ChunkMeta {
            model_name: "m".into(),
            model_version: String::new(),
            id: "req-9".into(),
        };
        let mid = token_chunk(
            &meta,
            &CompletionChunk {
                content: "hel".into(),
                stop: false,
                tokens_predicted: None,
                tokens_evaluated: None,
            },
        );
        assert_eq!(mid.id, "req-9");
        assert_eq!(mid.outputs[0].name, "token");
        assert!(matches!(
            mid.parameters.get("final").and_then(|p| p.parameter_choice.as_ref()),
            Some(ParameterChoice::BoolParam(false))
        ));
        let last = token_chunk(
            &meta,
            &CompletionChunk {
                content: String::new(),
                stop: true,
                tokens_predicted: Some(8),
                tokens_evaluated: Some(5),
            },
        );
        assert!(matches!(
            last.parameters.get("final").and_then(|p| p.parameter_choice.as_ref()),
            Some(ParameterChoice::BoolParam(true))
        ));
        assert!(matches!(
            last.parameters
                .get("tokens_predicted")
                .and_then(|p| p.parameter_choice.as_ref()),
            Some(ParameterChoice::Int64Param(8))
        ));
        let elements = unpack_bytes(&mid.raw_output_contents[0]).unwrap();
        assert_eq!(elements[0], b"hel");
    }

    #[test]
    fn tokenize_piece_repr_decodes_strings_and_bytes() {
        let body: TokenizeResponseBody = serde_json::from_str(
            r#"{"tokens":[{"id":9707,"piece":"Hello"},{"id":11,"piece":[240,159]}]}"#,
        )
        .unwrap();
        assert_eq!(body.tokens[0].id, 9707);
        assert_eq!(
            match &body.tokens[0].piece {
                PieceRepr::Text(s) => s.clone(),
                _ => panic!("expected text"),
            },
            "Hello"
        );
        // Invalid-UTF-8 byte pieces decode lossily instead of failing.
        let lossy = match body.tokens[1].piece {
            PieceRepr::Bytes(ref b) => String::from_utf8_lossy(b).into_owned(),
            _ => panic!("expected bytes"),
        };
        assert!(!lossy.is_empty());
    }
}
