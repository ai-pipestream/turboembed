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
//! # Integration plan (MLX-first)
//!
//! 1. **MLX** via the `mlx-rs` bindings — primary path for embeddings and
//!    token streaming on Apple silicon (unified memory, Metal kernels):
//!    * embeddings → unary `infer`, output `embedding` FP32 `[d]`
//!    * generation → `infer_stream`, one `token` BYTES chunk per decoded
//!      token, `final` bool parameter on the last chunk
//!    * model artifacts: MLX-format directories (safetensors + config), set
//!      via the model's `path`.
//! 2. **llama.cpp-Metal** for GGUF models on the same host — covered by the
//!    shared `backend-llamacpp` crate with `device = "metal"`, not by this
//!    crate.
//! 3. **Core ML / Foundation Models** for ANE offload — investigation item,
//!    after MLX.
//!
//! The MLX runtime only exists on macOS, so the functional path is gated
//! `cfg(target_os = "macos")` once `mlx-rs` lands. The crate itself stays
//! pure Rust and compiles on every platform so Linux CI can type-check the
//! arch-apple wiring; on non-macOS it always reports `Unavailable`.

use async_trait::async_trait;
use inferstream_backend::{Backend, BackendError, ModelMetadata};
use inferstream_protocol::inference::{ModelInferRequest, ModelInferResponse};

/// Configuration for one MLX-served model.
#[derive(Debug, Clone, Default)]
pub struct MlxConfig {
    /// MLX model directory (safetensors + config + tokenizer).
    pub model_path: String,
    /// Default cap on generated tokens when the request omits `max_tokens`.
    pub max_output_tokens: Option<u32>,
}

/// Apple MLX backend. macOS-host-only by design; stub until `mlx-rs` lands.
#[derive(Debug, Default, Clone)]
pub struct MlxBackend {
    config: MlxConfig,
}

impl MlxBackend {
    pub fn new(config: MlxConfig) -> Result<Self, BackendError> {
        if config.model_path.is_empty() {
            return Err(BackendError::InvalidRequest(
                "mlx models require path (an MLX model directory)".into(),
            ));
        }
        Ok(Self { config })
    }

    pub fn config(&self) -> &MlxConfig {
        &self.config
    }

    fn unavailable() -> BackendError {
        if cfg!(target_os = "macos") {
            BackendError::Unavailable(
                "MLX backend is a stub; the mlx-rs engine has not landed yet".into(),
            )
        } else {
            BackendError::Unavailable(
                "MLX requires a native macOS host; Metal/ANE do not pass through Linux \
                 containers — run inferstream-apple on the Mac itself"
                    .into(),
            )
        }
    }
}

#[async_trait]
impl Backend for MlxBackend {
    fn id(&self) -> &str {
        "mlx"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_model_path() {
        assert!(MlxBackend::new(MlxConfig::default()).is_err());
        assert!(MlxBackend::new(MlxConfig {
            model_path: "/models/mlx-llama".into(),
            ..Default::default()
        })
        .is_ok());
    }
}
