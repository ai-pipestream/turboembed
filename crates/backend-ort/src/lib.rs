//! ONNX Runtime backend stub for inferstream.
//!
//! Planned integration through the maintained `ort` crate (pykeio/ort):
//! session-per-model with execution providers selected by config (CPU
//! default; CUDA / DirectML / CoreML where available). ONNX models are
//! naturally unary tensor-in/tensor-out, so this backend will implement
//! `infer` and rely on the default single-chunk `infer_stream` adapter.
//!
//! This crate currently compiles without ONNX Runtime and reports
//! `Unavailable` at runtime.

use async_trait::async_trait;
use inferstream_backend::{Backend, BackendError, ModelMetadata};
use inferstream_protocol::inference::{ModelInferRequest, ModelInferResponse};

/// Stub ONNX Runtime backend.
#[derive(Debug, Default, Clone)]
pub struct OrtBackend {
    /// Filesystem path to the `.onnx` model this backend would load.
    pub model_path: Option<String>,
}

impl OrtBackend {
    pub fn new(model_path: Option<String>) -> Self {
        Self { model_path }
    }

    fn unavailable() -> BackendError {
        BackendError::Unavailable(
            "ONNX Runtime backend is a stub in v0.1; no session is loaded".into(),
        )
    }
}

#[async_trait]
impl Backend for OrtBackend {
    fn id(&self) -> &str {
        "onnxruntime"
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

    async fn infer(
        &self,
        _request: ModelInferRequest,
    ) -> Result<ModelInferResponse, BackendError> {
        Err(Self::unavailable())
    }
}
