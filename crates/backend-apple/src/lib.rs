//! Apple MLX backend for inferstream — native macOS hosts only.
//!
//! # Deployment model
//!
//! Apple GPU (Metal) is not a passthrough device. `inferstream-apple` runs
//! **natively on a macOS host**. Do not serve this backend from a Linux
//! container.
//!
//! # How it works
//!
//! The engine is **in-process native MLX** — Apple's MLX linked through
//! Swift (`native/mlx-engine`, mlx-swift + mlx-swift-lm) via a C ABI, the
//! same shape as NVIDIA in-process CUDA. There is **no Python interpreter**
//! on Embed / Tokenize / StreamInfer.
//!
//! * **Embeddings** (`MLXEmbedders`): BERT-family MiniLM / BGE / E5 / GTE
//!   on Metal. Unary `infer` with BYTES `text` → FP32 `embedding`.
//! * **Generation** (`MLXLLM`): `infer_stream` streams one BYTES `token`
//!   chunk per decoded piece. The final chunk carries
//!   `decode_tokens_per_second` (engine-side, not gRPC wall-clock).
//! * **Tokenize/Detokenize** are served by the Rust `tokenizers` crate from
//!   `tokenizer_dir` (`tokenizer.json`) — never Python, never MLX.
//!
//! Weights are local directories produced by `cargo xtask fetch --mlx`
//! (`models/mlx/<alias>/`). A Hugging Face repo id in config is resolved to
//! that directory; the runtime does not shell out to download.
//!
//! The crate compiles on Linux as a stub so CI type-checks the wiring.

mod ffi;
mod native;

pub use native::{EmbedResult, GenerateStats, MlxEngine, PingResult};

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use inferstream_backend::{Backend, BackendError, ModelMetadata, ResponseStream};
use inferstream_protocol::inference::{
    infer_parameter::ParameterChoice, model_infer_response::InferOutputTensor,
    model_metadata_response::TensorMetadata, InferParameter, ModelInferRequest, ModelInferResponse,
};
use inferstream_protocol::tensor::{pack_bytes, pack_fp32, unpack_bytes, DataType};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

/// Configuration for one MLX-served model.
#[derive(Debug, Clone)]
pub struct MlxConfig {
    /// Local MLX model directory, or an HF repo id that resolves to
    /// `models/mlx/<alias>` after `cargo xtask fetch --mlx`.
    pub model: String,
    pub max_batch: usize,
    pub normalize: bool,
    pub max_output_tokens: u32,
}

impl Default for MlxConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            max_batch: 32,
            normalize: true,
            max_output_tokens: 256,
        }
    }
}

/// Apple MLX backend. One instance per configured model; all instances share
/// one in-process [`MlxEngine`].
pub struct MlxBackend {
    config: MlxConfig,
    engine: Arc<MlxEngine>,
    resolved: PathBuf,
    dim: AtomicUsize,
}

impl MlxBackend {
    pub fn new(config: MlxConfig, engine: Arc<MlxEngine>) -> Result<Self, BackendError> {
        if config.model.is_empty() {
            return Err(BackendError::InvalidRequest(
                "mlx models require path (a local MLX directory from `cargo xtask fetch --mlx`)"
                    .into(),
            ));
        }
        if config.max_batch == 0 {
            return Err(BackendError::InvalidRequest(
                "mlx max_batch must be at least 1".into(),
            ));
        }
        let resolved = resolve_model_dir(&config.model);
        Ok(Self {
            config,
            engine,
            resolved,
            dim: AtomicUsize::new(0),
        })
    }

    pub fn config(&self) -> &MlxConfig {
        &self.config
    }

    fn utf8_batch(request: &ModelInferRequest, name: &str) -> Result<Vec<String>, BackendError> {
        let (index, tensor) = request
            .inputs
            .iter()
            .enumerate()
            .find(|(_, t)| t.name == name)
            .ok_or_else(|| {
                BackendError::InvalidRequest(format!("expected an input tensor named {name:?}"))
            })?;
        if tensor.datatype != DataType::Bytes.as_oip() {
            return Err(BackendError::InvalidRequest(format!(
                "input {name:?} must be BYTES, got {:?}",
                tensor.datatype
            )));
        }
        let raw = request.raw_input_contents.get(index).ok_or_else(|| {
            BackendError::InvalidRequest(format!(
                "input {name:?} must be sent via raw_input_contents"
            ))
        })?;
        let elements = unpack_bytes(raw)
            .map_err(|e| BackendError::InvalidRequest(format!("malformed BYTES payload: {e}")))?;
        if elements.is_empty() {
            return Err(BackendError::InvalidRequest(format!(
                "input {name:?} contained no elements"
            )));
        }
        elements
            .into_iter()
            .map(|e| {
                String::from_utf8(e).map_err(|e| {
                    BackendError::InvalidRequest(format!("input {name:?} is not UTF-8: {e}"))
                })
            })
            .collect()
    }

    fn bool_param(request: &ModelInferRequest, name: &str) -> Option<bool> {
        match request.parameters.get(name)?.parameter_choice.as_ref()? {
            ParameterChoice::BoolParam(v) => Some(*v),
            _ => None,
        }
    }

    fn int_param(request: &ModelInferRequest, name: &str) -> Option<i64> {
        match request.parameters.get(name)?.parameter_choice.as_ref()? {
            ParameterChoice::Int64Param(v) => Some(*v),
            _ => None,
        }
    }

    fn is_generation_request(request: &ModelInferRequest) -> bool {
        request.parameters.contains_key("max_tokens")
            || request.inputs.iter().any(|t| t.name == "prompt")
    }

    fn final_chunk_params(is_final: bool, decode_tps: Option<f64>) -> HashMap<String, InferParameter> {
        let mut map = HashMap::from([(
            "final".to_string(),
            InferParameter {
                parameter_choice: Some(ParameterChoice::BoolParam(is_final)),
            },
        )]);
        if let Some(tps) = decode_tps {
            map.insert(
                "decode_tokens_per_second".into(),
                InferParameter {
                    parameter_choice: Some(ParameterChoice::DoubleParam(tps)),
                },
            );
        }
        map
    }

    fn token_chunk(
        request: &ModelInferRequest,
        token: &str,
        is_final: bool,
        decode_tps: Option<f64>,
    ) -> ModelInferResponse {
        ModelInferResponse {
            model_name: request.model_name.clone(),
            model_version: request.model_version.clone(),
            id: request.id.clone(),
            parameters: Self::final_chunk_params(is_final, decode_tps),
            outputs: vec![InferOutputTensor {
                name: "token".to_string(),
                datatype: DataType::Bytes.as_oip().to_string(),
                shape: vec![1],
                parameters: HashMap::new(),
                contents: None,
            }],
            raw_output_contents: vec![pack_bytes(&[token.as_bytes()])],
        }
    }
}

/// Resolve a catalog `path` to a local MLX directory.
///
/// Accepts an existing directory, or an HF repo id / alias that was fetched
/// into `models/mlx/<alias>` / `models/mlx/<last-path-component>`.
pub fn resolve_model_dir(model: &str) -> PathBuf {
    let as_path = PathBuf::from(model);
    if as_path.is_dir() {
        return as_path;
    }
    if let Some(name) = alias_from_model(model) {
        let local = PathBuf::from("models/mlx").join(&name);
        if local.is_dir() {
            return local;
        }
    }
    as_path
}

fn alias_from_model(model: &str) -> Option<String> {
    if !model.contains('/') {
        return Some(model.to_string());
    }
    model.rsplit('/').next().map(|s| {
        s.trim_end_matches("-4bit")
            .trim_end_matches("-Instruct")
            .to_string()
    })
}

#[async_trait]
impl Backend for MlxBackend {
    fn id(&self) -> &str {
        "mlx"
    }

    async fn model_ready(&self, _model_name: &str, _model_version: &str) -> bool {
        let engine = Arc::clone(&self.engine);
        tokio::task::spawn_blocking(move || engine.ping().is_ok())
            .await
            .unwrap_or(false)
    }

    async fn model_metadata(
        &self,
        model_name: &str,
        _model_version: &str,
    ) -> Result<ModelMetadata, BackendError> {
        let dim = self.dim.load(Ordering::Relaxed);
        Ok(ModelMetadata {
            name: model_name.to_string(),
            versions: vec!["1".to_string()],
            platform: "mlx".to_string(),
            inputs: vec![TensorMetadata {
                name: "text".to_string(),
                datatype: DataType::Bytes.as_oip().to_string(),
                shape: vec![-1],
            }],
            outputs: vec![TensorMetadata {
                name: "embedding".to_string(),
                datatype: DataType::Fp32.as_oip().to_string(),
                shape: vec![if dim > 0 { dim as i64 } else { -1 }],
            }],
            properties: HashMap::from([
                ("backend".to_string(), "mlx".to_string()),
                ("model".to_string(), self.config.model.clone()),
                ("resolved".to_string(), self.resolved.display().to_string()),
                ("engine".to_string(), "mlx-swift".to_string()),
            ]),
        })
    }

    async fn infer(&self, request: ModelInferRequest) -> Result<ModelInferResponse, BackendError> {
        let texts = Self::utf8_batch(&request, "text")?;
        if texts.len() > self.config.max_batch {
            return Err(BackendError::InvalidRequest(format!(
                "batch of {} exceeds max_batch {}; chunk upstream",
                texts.len(),
                self.config.max_batch
            )));
        }
        let normalize = Self::bool_param(&request, "normalize").unwrap_or(self.config.normalize);
        let batch = texts.len();
        let engine = Arc::clone(&self.engine);
        let model = self.resolved.clone();
        let embed = tokio::task::spawn_blocking(move || {
            engine.embed(&model.to_string_lossy(), &texts, normalize)
        })
        .await
        .map_err(|e| BackendError::Internal(format!("embed task: {e}")))??;
        if embed.vectors.len() != batch {
            return Err(BackendError::Internal(format!(
                "engine returned {} vectors for {batch} texts",
                embed.vectors.len()
            )));
        }
        if embed.dimensions == 0 || embed.vectors.iter().any(|v| v.len() != embed.dimensions) {
            return Err(BackendError::Internal(
                "engine returned inconsistent embedding dimensions".into(),
            ));
        }
        self.dim.store(embed.dimensions, Ordering::Relaxed);

        let mut values = Vec::with_capacity(batch * embed.dimensions);
        for vector in &embed.vectors {
            values.extend_from_slice(vector);
        }
        let shape = if batch == 1 {
            vec![embed.dimensions as i64]
        } else {
            vec![batch as i64, embed.dimensions as i64]
        };
        Ok(ModelInferResponse {
            model_name: request.model_name,
            model_version: request.model_version,
            id: request.id,
            parameters: HashMap::new(),
            outputs: vec![InferOutputTensor {
                name: "embedding".to_string(),
                datatype: DataType::Fp32.as_oip().to_string(),
                shape,
                parameters: HashMap::new(),
                contents: None,
            }],
            raw_output_contents: vec![pack_fp32(&values)],
        })
    }

    async fn infer_stream(
        &self,
        request: ModelInferRequest,
    ) -> Result<ResponseStream, BackendError> {
        if !Self::is_generation_request(&request) {
            let response = self.infer(request).await;
            return Ok(Box::pin(futures::stream::once(async move { response })));
        }

        let prompt = Self::utf8_batch(&request, "prompt")
            .or_else(|_| Self::utf8_batch(&request, "text"))?
            .remove(0);
        let max_tokens = Self::int_param(&request, "max_tokens")
            .filter(|&v| v > 0)
            .map(|v| v as u32)
            .unwrap_or(self.config.max_output_tokens);

        let engine = Arc::clone(&self.engine);
        let model = self.resolved.clone();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::task::spawn_blocking(move || {
            let tx_tok = tx.clone();
            let result = engine.generate(
                &model.to_string_lossy(),
                &prompt,
                max_tokens,
                move |token| {
                    let _ = tx_tok.send(Ok(token));
                },
            );
            match result {
                Ok(stats) => {
                    let _ = tx.send(Err(format!("__done__:{:.6}", stats.decode_tps)));
                }
                Err(e) => {
                    let _ = tx.send(Err(e.to_string()));
                }
            }
        });

        let stream = UnboundedReceiverStream::new(rx).map(move |item| match item {
            Ok(token) => Ok(Self::token_chunk(&request, &token, false, None)),
            Err(msg) if msg.starts_with("__done__:") => {
                let tps = msg.trim_start_matches("__done__:").parse::<f64>().ok();
                Ok(Self::token_chunk(&request, "", true, tps))
            }
            Err(msg) => Err(BackendError::Internal(msg)),
        });
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_request(texts: &[&str]) -> ModelInferRequest {
        use inferstream_protocol::inference::model_infer_request::InferInputTensor;
        let bytes: Vec<&[u8]> = texts.iter().map(|t| t.as_bytes()).collect();
        ModelInferRequest {
            model_name: "minilm".into(),
            id: "req-1".into(),
            inputs: vec![InferInputTensor {
                name: "text".into(),
                datatype: DataType::Bytes.as_oip().into(),
                shape: vec![texts.len() as i64],
                parameters: HashMap::new(),
                contents: None,
            }],
            raw_input_contents: vec![pack_bytes(&bytes)],
            ..Default::default()
        }
    }

    #[test]
    fn requires_model() {
        let engine = Arc::new(MlxEngine::new());
        assert!(MlxBackend::new(MlxConfig::default(), Arc::clone(&engine)).is_err());
        assert!(MlxBackend::new(
            MlxConfig {
                model: "models/mlx/minilm".into(),
                ..Default::default()
            },
            engine,
        )
        .is_ok());
    }

    #[test]
    fn utf8_batch_parses_and_validates() {
        let request = text_request(&["hello", "wörld"]);
        assert_eq!(
            MlxBackend::utf8_batch(&request, "text").unwrap(),
            vec!["hello", "wörld"]
        );
        assert!(matches!(
            MlxBackend::utf8_batch(&request, "prompt"),
            Err(BackendError::InvalidRequest(_))
        ));
    }

    #[tokio::test]
    async fn oversized_batch_rejected_before_engine() {
        let backend = MlxBackend::new(
            MlxConfig {
                model: "m".into(),
                max_batch: 1,
                ..Default::default()
            },
            Arc::new(MlxEngine::new()),
        )
        .unwrap();
        let result = backend.infer(text_request(&["a", "b"])).await;
        assert!(matches!(result, Err(BackendError::InvalidRequest(_))));
    }

    #[test]
    fn generation_detection() {
        let embed = text_request(&["hi"]);
        assert!(!MlxBackend::is_generation_request(&embed));
        let mut generate = text_request(&["hi"]);
        generate.parameters.insert(
            "max_tokens".into(),
            InferParameter {
                parameter_choice: Some(ParameterChoice::Int64Param(8)),
            },
        );
        assert!(MlxBackend::is_generation_request(&generate));
    }

    #[test]
    fn token_chunk_shape() {
        let request = text_request(&["hi"]);
        let chunk = MlxBackend::token_chunk(&request, "tok", true, Some(42.5));
        assert_eq!(chunk.id, "req-1");
        assert_eq!(chunk.outputs[0].name, "token");
        assert_eq!(
            unpack_bytes(&chunk.raw_output_contents[0]).unwrap(),
            vec![b"tok".to_vec()]
        );
        assert!(matches!(
            chunk.parameters.get("final").unwrap().parameter_choice,
            Some(ParameterChoice::BoolParam(true))
        ));
        assert!(matches!(
            chunk
                .parameters
                .get("decode_tokens_per_second")
                .unwrap()
                .parameter_choice,
            Some(ParameterChoice::DoubleParam(_))
        ));
    }

    #[test]
    fn resolve_prefers_existing_dir() {
        let dir = std::env::temp_dir().join(format!("inferstream-mlx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(resolve_model_dir(dir.to_str().unwrap()), dir);
        std::fs::remove_dir_all(&dir).ok();
    }
}
