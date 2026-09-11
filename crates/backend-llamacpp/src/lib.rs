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
//! **SYCL note:** building and *running* the SYCL flavor requires the Intel
//! oneAPI environment (`source /opt/intel/oneapi/setvars.sh`) so that
//! `libsycl`, Level Zero, and oneMKL resolve — both in the build shell and in
//! the service unit that launches `inferstream-intel`.
//!
//! ## Planned integration
//!
//! Binding through a maintained wrapper (`llama-cpp-2`) or direct FFI:
//! * embeddings → unary `infer`, output `embedding` FP32 `[d]`
//! * generation → `infer_stream`, one `token` BYTES chunk per decoded token
//!   with the `final` bool parameter on the last chunk
//!   (identical wire shapes to the mock backend).
//!
//! The engine link is stubbed: this crate compiles everywhere as pure Rust
//! and reports `Unavailable` at runtime.

use async_trait::async_trait;
use inferstream_backend::{Backend, BackendError, ModelMetadata};
use inferstream_protocol::inference::{ModelInferRequest, ModelInferResponse};

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
    /// Path to the GGUF file.
    pub model_path: String,
    /// Target device (see [`LlamaDevice`]).
    pub device: LlamaDevice,
    /// Layers to offload to the accelerator; `None` = offload everything.
    pub n_gpu_layers: Option<u32>,
    /// Concurrent sequences the context schedules (`n_parallel`).
    pub max_batch_size: Option<u32>,
}

/// llama.cpp backend. Skeleton in v0.x: full config surface, engine link
/// stubbed.
#[derive(Debug, Default, Clone)]
pub struct LlamaCppBackend {
    config: LlamaCppConfig,
}

impl LlamaCppBackend {
    pub fn new(config: LlamaCppConfig) -> Result<Self, BackendError> {
        if config.model_path.is_empty() {
            return Err(BackendError::InvalidRequest(
                "llama-cpp models require path (a GGUF file)".into(),
            ));
        }
        Ok(Self { config })
    }

    pub fn config(&self) -> &LlamaCppConfig {
        &self.config
    }

    fn unavailable() -> BackendError {
        BackendError::Unavailable(
            "llama.cpp backend is a stub; the engine FFI has not landed yet".into(),
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
    fn requires_model_path() {
        assert!(LlamaCppBackend::new(LlamaCppConfig::default()).is_err());
        assert!(LlamaCppBackend::new(LlamaCppConfig {
            model_path: "/models/x.gguf".into(),
            ..Default::default()
        })
        .is_ok());
    }
}
