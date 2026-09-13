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
    /// Default: `Unavailable`. The mock implements word-overlap for the
    /// wire path. Real MiniLM CE scores live in the TurboRerank C ABI
    /// (`include/turborerank.h`); a later façade should call that ABI
    /// instead of inventing scores here.
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
