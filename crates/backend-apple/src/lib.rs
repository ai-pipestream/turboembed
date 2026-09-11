//! Apple MLX backend for inferstream — native macOS hosts only.
//!
//! # Deployment model — read this first
//!
//! Apple GPU (Metal) and the Apple Neural Engine are **not** passthrough
//! devices the way NVIDIA CUDA is on Linux: there is no equivalent of the
//! NVIDIA container toolkit for macOS, and Linux containers on a Mac run in a
//! VM without Metal access. `inferstream-apple` therefore runs **natively on
//! a macOS host** (Mac server / Mac worker node) — same gRPC contract as the
//! Linux arch binaries, different host. Do not attempt to serve this backend
//! from a container.
//!
//! # How it works
//!
//! MLX has first-class Python bindings and moving targets elsewhere, so this
//! backend talks to a **persistent Python worker** (`python/mlx_bridge.py`)
//! over newline-delimited JSON on stdin/stdout — see [`bridge`]. The worker
//! is spawned once and caches loaded models, so after the first request the
//! model is hot in unified memory and per-call latency is dominated by the
//! actual Metal compute, not process startup or model load.
//!
//! * **Embeddings** (`mlx-embeddings`): OIP unary `infer` with a BYTES
//!   `text` input tensor → FP32 `embedding` output, `[d]` for one text and
//!   `[n, d]` for a batch — the same convention as the mock and ORT
//!   backends, so `inferstream.v1.Embed` works unchanged.
//! * **Generation** (`mlx-lm`): `infer_stream` with a `max_tokens`
//!   parameter (or a `prompt` input tensor) streams one BYTES `token` chunk
//!   per decoded token, then an empty final chunk with `final = true`.
//!   Requests without generation markers fall back to single-chunk embed.
//! * **Tokenize/Detokenize** are served by the server layer from the
//!   model's configured `tokenizer_dir` (HF `tokenizer.json`) — no Python
//!   round-trip.
//!
//! The crate is pure Rust and compiles on every platform so Linux CI can
//! type-check the arch-apple wiring; off-macOS (or without the venv) every
//! call reports `Unavailable` with setup instructions.

mod bridge;

pub use bridge::{BridgeReply, MlxWorker, MlxWorkerConfig};

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use inferstream_backend::{Backend, BackendError, ModelMetadata, ResponseStream};
use inferstream_protocol::inference::{
    infer_parameter::ParameterChoice, model_infer_response::InferOutputTensor,
    model_metadata_response::TensorMetadata, InferParameter, ModelInferRequest, ModelInferResponse,
};
use inferstream_protocol::tensor::{pack_bytes, pack_fp32, unpack_bytes, DataType};
use serde::Deserialize;
use serde_json::json;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

/// Configuration for one MLX-served model.
#[derive(Debug, Clone)]
pub struct MlxConfig {
    /// MLX model: a HuggingFace repo id (e.g.
    /// `"mlx-community/all-MiniLM-L6-v2-4bit"`) or a local MLX model
    /// directory (safetensors + config).
    pub model: String,
    /// Max texts per embed call. Chunk upstream instead of raising this:
    /// passively cooled Macs thermal-throttle on sustained big batches.
    pub max_batch: usize,
    /// L2-normalize embeddings unless the request overrides it.
    pub normalize: bool,
    /// Default cap on generated tokens when the request omits `max_tokens`.
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
/// one persistent [`MlxWorker`] so every model lives in the same hot worker.
pub struct MlxBackend {
    config: MlxConfig,
    worker: Arc<MlxWorker>,
    /// Embedding dimension observed from the first successful embed
    /// (0 = not yet known). Reported through `ModelMetadata`.
    dim: AtomicUsize,
}

#[derive(Deserialize)]
struct EmbedResult {
    dimensions: usize,
    vectors: Vec<Vec<f32>>,
}

impl MlxBackend {
    pub fn new(config: MlxConfig, worker: Arc<MlxWorker>) -> Result<Self, BackendError> {
        if config.model.is_empty() {
            return Err(BackendError::InvalidRequest(
                "mlx models require path (a HF repo id or MLX model directory)".into(),
            ));
        }
        if config.max_batch == 0 {
            return Err(BackendError::InvalidRequest(
                "mlx max_batch must be at least 1".into(),
            ));
        }
        Ok(Self {
            config,
            worker,
            dim: AtomicUsize::new(0),
        })
    }

    pub fn config(&self) -> &MlxConfig {
        &self.config
    }

    /// All elements of a named BYTES input tensor, decoded as UTF-8.
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

    /// A request is a generation request when it carries generation markers;
    /// everything else on this backend is an embedding request.
    fn is_generation_request(request: &ModelInferRequest) -> bool {
        request.parameters.contains_key("max_tokens")
            || request.inputs.iter().any(|t| t.name == "prompt")
    }

    fn final_chunk_params(is_final: bool) -> HashMap<String, InferParameter> {
        HashMap::from([(
            "final".to_string(),
            InferParameter {
                parameter_choice: Some(ParameterChoice::BoolParam(is_final)),
            },
        )])
    }

    fn token_chunk(request: &ModelInferRequest, token: &str, is_final: bool) -> ModelInferResponse {
        ModelInferResponse {
            model_name: request.model_name.clone(),
            model_version: request.model_version.clone(),
            id: request.id.clone(),
            parameters: Self::final_chunk_params(is_final),
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

#[async_trait]
impl Backend for MlxBackend {
    fn id(&self) -> &str {
        "mlx"
    }

    async fn model_ready(&self, _model_name: &str, _model_version: &str) -> bool {
        // Ready = the persistent worker is alive and MLX can run a Metal
        // kernel. Deliberately does NOT force a model download; the first
        // embed/generate warms the model into the worker's cache.
        self.worker.call(json!({"op": "ping"})).await.is_ok()
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
                // -1 until the first embed reveals the model's dimension.
                shape: vec![if dim > 0 { dim as i64 } else { -1 }],
            }],
            properties: HashMap::from([
                ("backend".to_string(), "mlx".to_string()),
                ("model".to_string(), self.config.model.clone()),
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
        let result = self
            .worker
            .call(json!({
                "op": "embed",
                "model": self.config.model,
                "texts": texts,
                "normalize": normalize,
            }))
            .await?;
        let embed: EmbedResult = serde_json::from_value(result)
            .map_err(|e| BackendError::Internal(format!("malformed embed result: {e}")))?;
        if embed.vectors.len() != batch {
            return Err(BackendError::Internal(format!(
                "bridge returned {} vectors for {batch} texts",
                embed.vectors.len()
            )));
        }
        if embed.dimensions == 0 || embed.vectors.iter().any(|v| v.len() != embed.dimensions) {
            return Err(BackendError::Internal(
                "bridge returned inconsistent embedding dimensions".into(),
            ));
        }
        self.dim.store(embed.dimensions, Ordering::Relaxed);

        let mut values = Vec::with_capacity(batch * embed.dimensions);
        for vector in &embed.vectors {
            values.extend_from_slice(vector);
        }
        // Same shape convention as mock/ORT: [d] for one text, [n, d] batch.
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
            // Embedding request on the streaming RPC: one embed chunk.
            let response = self.infer(request).await;
            return Ok(Box::pin(futures::stream::once(async move { response })));
        }

        // Generation via mlx-lm. The prompt is either a dedicated "prompt"
        // tensor or the first element of "text".
        let prompt = Self::utf8_batch(&request, "prompt")
            .or_else(|_| Self::utf8_batch(&request, "text"))?
            .remove(0);
        let max_tokens = Self::int_param(&request, "max_tokens")
            .filter(|&v| v > 0)
            .map(|v| v as u32)
            .unwrap_or(self.config.max_output_tokens);
        let chunks = self
            .worker
            .call_stream(json!({
                "op": "generate",
                "model": self.config.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
            }))
            .await?;

        let stream = ReceiverStream::new(chunks).map(move |reply| match reply {
            Ok(BridgeReply::Chunk(value)) => {
                let token = value
                    .get("token")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                Ok(Self::token_chunk(&request, &token, false))
            }
            // The worker signals completion after the last token, so the
            // final flag rides on an empty trailing chunk.
            Ok(BridgeReply::Done(_)) => Ok(Self::token_chunk(&request, "", true)),
            Ok(BridgeReply::Ok(value)) => Err(BackendError::Internal(format!(
                "unexpected unary reply during generation: {value}"
            ))),
            Err(error) => Err(error),
        });
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_worker() -> Arc<MlxWorker> {
        Arc::new(MlxWorker::new(MlxWorkerConfig {
            python: "/definitely/not/python".into(),
            script: "/definitely/not/bridge.py".into(),
        }))
    }

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
        assert!(MlxBackend::new(MlxConfig::default(), test_worker()).is_err());
        assert!(MlxBackend::new(
            MlxConfig {
                model: "mlx-community/all-MiniLM-L6-v2-4bit".into(),
                ..Default::default()
            },
            test_worker(),
        )
        .is_ok());
        assert!(MlxBackend::new(
            MlxConfig {
                model: "m".into(),
                max_batch: 0,
                ..Default::default()
            },
            test_worker(),
        )
        .is_err());
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

        let mut bad_dtype = text_request(&["x"]);
        bad_dtype.inputs[0].datatype = DataType::Fp32.as_oip().into();
        assert!(matches!(
            MlxBackend::utf8_batch(&bad_dtype, "text"),
            Err(BackendError::InvalidRequest(_))
        ));

        let mut not_utf8 = text_request(&["x"]);
        not_utf8.raw_input_contents = vec![pack_bytes(&[&[0xff, 0xfe][..]])];
        assert!(matches!(
            MlxBackend::utf8_batch(&not_utf8, "text"),
            Err(BackendError::InvalidRequest(_))
        ));
    }

    #[tokio::test]
    async fn oversized_batch_rejected_before_bridge() {
        let backend = MlxBackend::new(
            MlxConfig {
                model: "m".into(),
                max_batch: 1,
                ..Default::default()
            },
            test_worker(),
        )
        .unwrap();
        let result = backend.infer(text_request(&["a", "b"])).await;
        assert!(matches!(result, Err(BackendError::InvalidRequest(_))));
    }

    #[tokio::test]
    async fn missing_venv_is_unavailable_not_panic() {
        let backend = MlxBackend::new(
            MlxConfig {
                model: "m".into(),
                ..Default::default()
            },
            test_worker(),
        )
        .unwrap();
        assert!(matches!(
            backend.infer(text_request(&["hi"])).await,
            Err(BackendError::Unavailable(_))
        ));
        assert!(!backend.model_ready("m", "").await);
        // Metadata never touches the worker.
        let metadata = backend.model_metadata("m", "").await.unwrap();
        assert_eq!(metadata.platform, "mlx");
        assert_eq!(metadata.outputs[0].shape, vec![-1]);
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

        let mut prompted = text_request(&["hi"]);
        prompted.inputs[0].name = "prompt".into();
        assert!(MlxBackend::is_generation_request(&prompted));
    }

    #[tokio::test]
    async fn stream_without_generation_markers_falls_back_to_embed_error_path() {
        // With a dead worker the embed fallback surfaces Unavailable as a
        // single stream item (per the default-adaptation contract).
        let backend = MlxBackend::new(
            MlxConfig {
                model: "m".into(),
                ..Default::default()
            },
            test_worker(),
        )
        .unwrap();
        let mut stream = backend.infer_stream(text_request(&["hi"])).await.unwrap();
        assert!(matches!(
            stream.next().await,
            Some(Err(BackendError::Unavailable(_)))
        ));
    }

    #[test]
    fn token_chunk_shape() {
        let request = text_request(&["hi"]);
        let chunk = MlxBackend::token_chunk(&request, "tok", true);
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
    }
}
