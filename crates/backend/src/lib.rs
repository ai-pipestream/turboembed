//! Backend abstraction for inferstream.
//!
//! A [`Backend`] owns one or more loaded models and executes inference for
//! them. The server routes each request by model name through the
//! [`Registry`], so backends never see traffic for models they do not serve.
//!
//! Backends speak the OIP protobuf types directly ([`ModelInferRequest`] /
//! [`ModelInferResponse`]) rather than an intermediate representation: the
//! façade's job is routing, auth, and protocol correctness, not data-model
//! translation. Raw tensor payload helpers live in `inferstream_protocol`.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use inferstream_protocol::extension::Encoding;
use inferstream_protocol::inference::{
    model_metadata_response::TensorMetadata, ModelInferRequest, ModelInferResponse,
};

/// Errors a backend can surface to the server layer.
///
/// The server maps these onto gRPC status codes (`NotFound`,
/// `InvalidArgument`, `Unavailable`, `Internal`).
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("model {0:?} is not served by this backend")]
    ModelNotFound(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("backend not available: {0}")]
    Unavailable(String),
    #[error("inference failed: {0}")]
    Internal(String),
}

/// Static metadata a backend reports for one of its models
/// (surfaced through the OIP `ModelMetadata` RPC).
#[derive(Debug, Clone, Default)]
pub struct ModelMetadata {
    pub name: String,
    pub versions: Vec<String>,
    /// OIP platform string, e.g. `"mock"`, `"llama_cpp"`, `"onnxruntime_onnx"`.
    pub platform: String,
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
    pub properties: HashMap<String, String>,
}

/// A boxed stream of response chunks produced for one streaming request.
pub type ResponseStream =
    Pin<Box<dyn Stream<Item = Result<ModelInferResponse, BackendError>> + Send>>;

/// Options for [`Backend::tokenize`], mirroring the
/// `inferstream.v1.TokenizeRequest` fields.
#[derive(Debug, Clone)]
pub struct TokenizeOptions {
    /// Add the tokenizer's special tokens (CLS/SEP/BOS/…). Defaults to true —
    /// what embedding inference feeds the model.
    pub add_special_tokens: bool,
    /// Also return per-token byte offsets into the original text.
    pub with_offsets: bool,
    /// Truncate each sequence to at most this many tokens.
    pub truncate_to: Option<usize>,
    /// Pad every sequence in the batch to the longest sequence's length.
    pub pad_to_longest: bool,
}

impl Default for TokenizeOptions {
    fn default() -> Self {
        Self {
            add_special_tokens: true,
            with_offsets: false,
            truncate_to: None,
            pad_to_longest: false,
        }
    }
}

/// A pluggable inference backend.
///
/// Implementations must be cheap to share (`Arc<dyn Backend>`) and safe to
/// call concurrently; any internal batching or serialization is the backend's
/// concern.
#[async_trait]
pub trait Backend: Send + Sync + 'static {
    /// Stable identifier for this backend kind, e.g. `"mock"`, `"llama-cpp"`.
    fn id(&self) -> &str;

    /// Whether the named model is loaded and ready to serve.
    async fn model_ready(&self, model_name: &str, model_version: &str) -> bool;

    /// Metadata for the named model.
    async fn model_metadata(
        &self,
        model_name: &str,
        model_version: &str,
    ) -> Result<ModelMetadata, BackendError>;

    /// Unary inference. The returned response must echo the request `id`.
    async fn infer(&self, request: ModelInferRequest) -> Result<ModelInferResponse, BackendError>;

    /// Streaming inference: zero or more response chunks for one request
    /// (e.g. one chunk per generated token). Every chunk must echo the
    /// request `id` so clients can correlate multiplexed responses.
    ///
    /// The default implementation adapts [`Backend::infer`] into a
    /// single-chunk stream, so unary-only backends work on the streaming RPC
    /// without extra code.
    async fn infer_stream(
        &self,
        request: ModelInferRequest,
    ) -> Result<ResponseStream, BackendError> {
        let response = self.infer(request).await;
        Ok(Box::pin(futures::stream::once(async move { response })))
    }

    // ------------------------------------------------------------------
    // INFERSTREAM EXTENSION surface (`inferstream.v1.InferstreamService`).
    // Every method defaults to `Unavailable`, so existing backends compile
    // unchanged and the server reports an honest gRPC status until the
    // engine wires a real implementation. The server layer may also satisfy
    // Tokenize/Detokenize from a locally configured `tokenizer.json`
    // without consulting the backend at all.
    // ------------------------------------------------------------------

    /// Tokenize a batch of texts with the model's tokenizer.
    async fn tokenize(
        &self,
        model_name: &str,
        _texts: &[String],
        _options: &TokenizeOptions,
    ) -> Result<Vec<Encoding>, BackendError> {
        Err(BackendError::Unavailable(format!(
            "backend {:?} has no server-side tokenizer for model {model_name:?}; \
             configure tokenizer_dir (tokenizer.json) for the model",
            self.id()
        )))
    }

    /// Decode batches of token ids back into text (inverse of
    /// [`Backend::tokenize`]).
    async fn detokenize(
        &self,
        model_name: &str,
        _sequences: &[Vec<u32>],
        _skip_special_tokens: bool,
    ) -> Result<Vec<String>, BackendError> {
        Err(BackendError::Unavailable(format!(
            "backend {:?} has no server-side tokenizer for model {model_name:?}; \
             configure tokenizer_dir (tokenizer.json) for the model",
            self.id()
        )))
    }

    /// Score `documents` against `query`; returns one relevance score per
    /// document, in input order (higher = more relevant).
    ///
    /// Default: `Unavailable`. The mock implements word-overlap for
    /// explicit mock models only. Catalog cross-encoder aliases
    /// (`ms-marco-minilm-l6`) go through the TurboRerank C ABI
    /// (`include/turborerank.h`) via `TurboRerankBackend` — never
    /// word-overlap.
    async fn rerank(
        &self,
        model_name: &str,
        _query: &str,
        _documents: &[String],
    ) -> Result<Vec<f32>, BackendError> {
        Err(BackendError::Unavailable(format!(
            "backend {:?} has no reranker for model {model_name:?}",
            self.id()
        )))
    }

    /// Write a row-major LE FP32 embedding blob into `dest`, reusing
    /// `dest.capacity` (SOLIDIFY 6 output scratch).
    ///
    /// Default: pack a BYTES `text` tensor, [`Backend::infer`], copy
    /// `raw_output_contents`. Mock stays on that path (explicit 8-d FNV).
    /// TurboEmbed overrides and copies engine-arena floats straight into
    /// the rented slab — no intermediate `pack_fp32` heap Vec.
    async fn embed_packed_into(
        &self,
        model_name: &str,
        texts: &[String],
        pooling: &str,
        normalize: Option<bool>,
        truncate_to: u32,
        dest: &mut Vec<u8>,
    ) -> Result<PackedEmbed, BackendError> {
        let request = infer_from_embed(model_name, texts, pooling, normalize, truncate_to);
        let response = self.infer(request).await?;
        packed_from_infer(response, dest)
    }

    /// Write scores into `dest` (input order), reusing `dest.capacity`.
    ///
    /// Default: [`Backend::rerank`] then `extend` into `dest`. Mock
    /// word-overlap is unchanged. TurboRerank overrides with
    /// `score_into` so the C ABI writes the rented row.
    async fn rerank_into(
        &self,
        model_name: &str,
        query: &str,
        documents: &[String],
        dest: &mut Vec<f32>,
    ) -> Result<(), BackendError> {
        let scores = self.rerank(model_name, query, documents).await?;
        dest.clear();
        dest.extend_from_slice(&scores);
        Ok(())
    }
}

/// Metadata for a blob written by [`Backend::embed_packed_into`].
#[derive(Debug, Clone)]
pub struct PackedEmbed {
    pub dim: u32,
    pub count: u32,
    pub model_name: String,
    pub model_version: String,
}

fn infer_from_embed(
    model_name: &str,
    texts: &[String],
    pooling: &str,
    normalize: Option<bool>,
    truncate_to: u32,
) -> ModelInferRequest {
    use inferstream_protocol::inference::{
        infer_parameter::ParameterChoice, model_infer_request::InferInputTensor, InferParameter,
    };
    use inferstream_protocol::tensor::{pack_bytes, DataType};

    let mut parameters = HashMap::new();
    if !pooling.is_empty() {
        parameters.insert(
            "pooling".to_string(),
            InferParameter {
                parameter_choice: Some(ParameterChoice::StringParam(pooling.to_string())),
            },
        );
    }
    if let Some(normalize) = normalize {
        parameters.insert(
            "normalize".to_string(),
            InferParameter {
                parameter_choice: Some(ParameterChoice::BoolParam(normalize)),
            },
        );
    }
    if truncate_to > 0 {
        parameters.insert(
            "truncate".to_string(),
            InferParameter {
                parameter_choice: Some(ParameterChoice::Int64Param(i64::from(truncate_to))),
            },
        );
    }
    let text_bytes: Vec<&[u8]> = texts.iter().map(|t| t.as_bytes()).collect();
    ModelInferRequest {
        model_name: model_name.to_string(),
        parameters,
        inputs: vec![InferInputTensor {
            name: "text".to_string(),
            datatype: DataType::Bytes.as_oip().to_string(),
            shape: vec![texts.len() as i64],
            parameters: HashMap::new(),
            contents: None,
        }],
        raw_input_contents: vec![pack_bytes(&text_bytes)],
        ..Default::default()
    }
}

fn packed_from_infer(
    response: ModelInferResponse,
    dest: &mut Vec<u8>,
) -> Result<PackedEmbed, BackendError> {
    use inferstream_protocol::tensor::DataType;

    let (index, output) = response
        .outputs
        .iter()
        .enumerate()
        .find(|(_, o)| o.name == "embedding")
        .ok_or_else(|| {
            BackendError::Internal("backend returned no output tensor named \"embedding\"".into())
        })?;
    if output.datatype != DataType::Fp32.as_oip() {
        return Err(BackendError::Internal(format!(
            "output \"embedding\" must be FP32, backend returned {:?}",
            output.datatype
        )));
    }
    let raw = response.raw_output_contents.get(index).ok_or_else(|| {
        BackendError::Internal("backend returned no raw content for \"embedding\"".into())
    })?;
    let dim = output
        .shape
        .last()
        .copied()
        .filter(|&d| d > 0)
        .ok_or_else(|| BackendError::Internal("embedding output reported an empty shape".into()))?
        as usize;
    if raw.len() % 4 != 0 {
        return Err(BackendError::Internal(format!(
            "embedding blob length {} is not a multiple of 4",
            raw.len()
        )));
    }
    let n_floats = raw.len() / 4;
    if n_floats % dim != 0 {
        return Err(BackendError::Internal(format!(
            "embedding blob length {} is not a multiple of dim {dim}",
            n_floats
        )));
    }
    dest.clear();
    inferstream_protocol::output_scratch::ensure_bytes(dest, raw.len());
    dest.extend_from_slice(raw);
    Ok(PackedEmbed {
        dim: dim as u32,
        count: (n_floats / dim) as u32,
        model_name: response.model_name,
        model_version: response.model_version,
    })
}

/// Catalog MiniLM-L6 cross-encoder aliases. Mock word-overlap must not
/// answer these names — they are TurboRerank ABI models.
pub fn is_catalog_cross_encoder_alias(name: &str) -> bool {
    let n = name.trim();
    n.eq_ignore_ascii_case("ms-marco-minilm-l6")
        || n.eq_ignore_ascii_case("ms-marco-minilm-l6-v2")
}

/// Routes model names to the backend that serves them.
///
/// Built once at startup from the routing config; lookups are lock-free.
#[derive(Default)]
pub struct Registry {
    by_model: HashMap<String, Arc<dyn Backend>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `backend` as the server for `model_name`.
    /// Returns the previously registered backend if the name was taken.
    pub fn register(
        &mut self,
        model_name: impl Into<String>,
        backend: Arc<dyn Backend>,
    ) -> Option<Arc<dyn Backend>> {
        self.by_model.insert(model_name.into(), backend)
    }

    /// The backend serving `model_name`, if any.
    pub fn lookup(&self, model_name: &str) -> Option<Arc<dyn Backend>> {
        self.by_model.get(model_name).cloned()
    }

    /// All registered model names (unordered).
    pub fn model_names(&self) -> impl Iterator<Item = &str> {
        self.by_model.keys().map(String::as_str)
    }

    pub fn is_empty(&self) -> bool {
        self.by_model.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NullBackend;

    #[async_trait]
    impl Backend for NullBackend {
        fn id(&self) -> &str {
            "null"
        }
        async fn model_ready(&self, _: &str, _: &str) -> bool {
            true
        }
        async fn model_metadata(&self, name: &str, _: &str) -> Result<ModelMetadata, BackendError> {
            Ok(ModelMetadata {
                name: name.to_string(),
                ..Default::default()
            })
        }
        async fn infer(
            &self,
            request: ModelInferRequest,
        ) -> Result<ModelInferResponse, BackendError> {
            Ok(ModelInferResponse {
                model_name: request.model_name,
                id: request.id,
                ..Default::default()
            })
        }
    }

    #[tokio::test]
    async fn registry_routes_by_model_name() {
        let mut registry = Registry::new();
        registry.register("m1", Arc::new(NullBackend));
        assert!(registry.lookup("m1").is_some());
        assert!(registry.lookup("m2").is_none());
    }

    #[tokio::test]
    async fn default_extension_surface_is_unavailable() {
        let backend = NullBackend;
        assert!(matches!(
            backend
                .tokenize("m", &["x".to_string()], &TokenizeOptions::default())
                .await,
            Err(BackendError::Unavailable(_))
        ));
        assert!(matches!(
            backend.detokenize("m", &[vec![1, 2]], false).await,
            Err(BackendError::Unavailable(_))
        ));
        assert!(matches!(
            backend.rerank("m", "q", &["d".to_string()]).await,
            Err(BackendError::Unavailable(_))
        ));
    }

    #[test]
    fn catalog_ce_alias_is_minilm_l6() {
        assert!(is_catalog_cross_encoder_alias("ms-marco-minilm-l6"));
        assert!(is_catalog_cross_encoder_alias("MS-MARCO-MiniLM-L6"));
        assert!(!is_catalog_cross_encoder_alias("minilm"));
        assert!(!is_catalog_cross_encoder_alias("mock-embed"));
    }

    #[test]
    fn tokenize_options_default_adds_special_tokens() {
        let options = TokenizeOptions::default();
        assert!(options.add_special_tokens);
        assert!(!options.with_offsets);
        assert!(options.truncate_to.is_none());
        assert!(!options.pad_to_longest);
    }

    #[tokio::test]
    async fn default_infer_stream_adapts_unary() {
        use futures::StreamExt;
        let backend = NullBackend;
        let request = ModelInferRequest {
            model_name: "m1".into(),
            id: "req-1".into(),
            ..Default::default()
        };
        let mut stream = backend.infer_stream(request).await.unwrap();
        let chunk = stream.next().await.unwrap().unwrap();
        assert_eq!(chunk.id, "req-1");
        assert!(stream.next().await.is_none());
    }
}
