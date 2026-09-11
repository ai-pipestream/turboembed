//! Apple MLX / Metal backend stub for inferstream.
//!
//! # Deployment model — read this first
//!
//! Apple GPU (Metal) and the Apple Neural Engine are **not** passthrough
//! devices the way NVIDIA CUDA is on Linux: there is no equivalent of the
//! NVIDIA container toolkit for macOS, and Linux containers on a Mac run in a
//! VM without Metal access. Consequently this backend only makes sense when
//! inferstream runs **natively on a macOS host** (a Mac server or Mac worker
//! node). A typical fleet runs Linux containers for llama.cpp / ONNX Runtime /
//! OpenVINO / TensorRT-LLM, plus bare-metal macOS inferstream instances for
//! Apple silicon — same gRPC surface, different hosts.
//!
//! # Planned integration
//!
//! Candidates, in rough priority order:
//! 1. **MLX** via `mlx-rs` — embeddings and token streaming on Apple silicon.
//! 2. **llama.cpp with Metal** — already covered by the llama.cpp backend
//!    when built on macOS; this crate then focuses on MLX-only model formats.
//! 3. **Core ML / Foundation Models** — ANE offload for supported models.
//!
//! The crate compiles on every platform (it is pure Rust until an engine is
//! wired in) but is only expected to become functional behind
//! `cfg(target_os = "macos")`. Linux CI never needs Apple frameworks.

use async_trait::async_trait;
use inferstream_backend::{Backend, BackendError, ModelMetadata};
use inferstream_protocol::inference::{ModelInferRequest, ModelInferResponse};

/// Stub Apple MLX / Metal backend. macOS-host-only by design.
#[derive(Debug, Default, Clone)]
pub struct AppleBackend {
    /// Path to the MLX / Core ML model this backend would load.
    pub model_path: Option<String>,
}

impl AppleBackend {
    pub fn new(model_path: Option<String>) -> Self {
        Self { model_path }
    }

    fn unavailable() -> BackendError {
        if cfg!(target_os = "macos") {
            BackendError::Unavailable(
                "Apple backend is a stub in v0.1; MLX/Metal engine not yet wired".into(),
            )
        } else {
            BackendError::Unavailable(
                "Apple backend requires a native macOS host; Metal/ANE do not pass through \
                 Linux containers"
                    .into(),
            )
        }
    }
}

#[async_trait]
impl Backend for AppleBackend {
    fn id(&self) -> &str {
        "apple"
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
