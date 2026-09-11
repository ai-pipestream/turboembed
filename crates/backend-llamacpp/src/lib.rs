//! llama.cpp backend stub for inferstream.
//!
//! Planned as the primary MVP engine: GGUF models for embeddings and token
//! streaming, portable across CUDA / Metal / Vulkan / CPU builds of
//! llama.cpp. The integration will bind through a maintained Rust wrapper
//! (e.g. `llama-cpp-2`) or direct FFI, mapping:
//!
//! * embeddings → unary `infer` returning an `FP32` `embedding` tensor
//! * generation → `infer_stream` emitting one `BYTES` `token` chunk per
//!   decoded token, with a `final` bool parameter on the last chunk
//!   (same shapes the mock backend already produces).
//!
//! This crate currently compiles without any native dependency and reports
//! `Unavailable` at runtime, keeping the workspace buildable everywhere while
//! reserving the crate boundary and config surface.

use async_trait::async_trait;
use inferstream_backend::{Backend, BackendError, ModelMetadata};
use inferstream_protocol::inference::{ModelInferRequest, ModelInferResponse};

/// Stub llama.cpp backend. Construction succeeds; every operation reports
/// [`BackendError::Unavailable`] until the engine is wired up.
#[derive(Debug, Default, Clone)]
pub struct LlamaCppBackend {
    /// Filesystem path to the GGUF model this backend would load.
    pub model_path: Option<String>,
}

impl LlamaCppBackend {
    pub fn new(model_path: Option<String>) -> Self {
        Self { model_path }
    }

    fn unavailable() -> BackendError {
        BackendError::Unavailable(
            "llama.cpp backend is a stub in v0.1; build with a wired engine to serve GGUF models"
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
        false
    }

    async fn model_metadata(
        &self,
        _model_name: &str,
        _model_version: &str,
    ) -> Result<ModelMetadata, BackendError> {
        Err(Self::unavailable())
    }

    async fn infer(&self, _request: ModelInferRequest) -> Result<ModelInferResponse, BackendError> {
        Err(Self::unavailable())
    }
}
