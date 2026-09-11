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
