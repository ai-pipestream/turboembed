//! `inferstream-apple`: the Apple arch binary — **native macOS hosts only**.
//!
//! Serves MLX models (`backend = "mlx"`) and GGUF models via llama.cpp-Metal
//! (`backend = "llama-cpp"`, `device = "metal"`). Metal and the Neural
//! Engine do not pass through Linux containers, so this binary is deployed
//! directly on the Mac (launchd service or plain process), not
//! containerized. It still compiles on Linux as a stub so CI can type-check
//! the wiring; at runtime on non-macOS every engine reports Unavailable.

use std::sync::Arc;

use inferstream_backend::Backend;
use inferstream_backend_mock::MockBackend;
use inferstream_server::config::{BackendKind, ModelConfig};
use inferstream_server::{unsupported, ServerError};

fn invalid(model: &ModelConfig, message: String) -> ServerError {
    ServerError::InvalidModelConfig {
        model: model.name.clone(),
        backend: model.backend.as_str(),
        message,
    }
}

fn factory() -> impl inferstream_server::BackendFactory {
    let mock: Arc<MockBackend> = Arc::new(MockBackend::default());
    move |model: &ModelConfig| -> Result<Arc<dyn Backend>, ServerError> {
        match model.backend {
            BackendKind::Mock => Ok(mock.clone()),
            BackendKind::Mlx => {
                let backend = inferstream_backend_apple::MlxBackend::new(
                    inferstream_backend_apple::MlxConfig {
                        model_path: model.path.clone().unwrap_or_default(),
                        max_output_tokens: None,
                    },
                )
                .map_err(|e| invalid(model, e.to_string()))?;
                Ok(Arc::new(backend))
            }
            BackendKind::LlamaCpp => {
                use inferstream_backend_llamacpp::{LlamaCppBackend, LlamaCppConfig, LlamaDevice};
                let device = model
                    .device
                    .as_deref()
                    .map(LlamaDevice::from_config)
                    .transpose()
                    .map_err(|e| invalid(model, e.to_string()))?
                    .unwrap_or(LlamaDevice::Metal);
                let endpoint = model.endpoint.clone().or_else(|| {
                    model
                        .path
                        .is_none()
                        .then(|| std::env::var("INFERSTREAM_LLAMACPP_ENDPOINT").ok())
                        .flatten()
                });
                let backend = LlamaCppBackend::new(LlamaCppConfig {
                    model_path: model.path.clone().unwrap_or_default(),
                    endpoint,
                    device,
                    n_gpu_layers: model.n_gpu_layers,
                    max_batch_size: model.max_batch_size,
                    n_ctx: model.n_ctx,
                })
                .map_err(|e| invalid(model, e.to_string()))?;
                Ok(Arc::new(backend))
            }
            BackendKind::TrtLlm | BackendKind::Ort | BackendKind::Openvino | BackendKind::Ovms => {
                Err(unsupported(
                    model,
                    "not an Apple-arch engine; use inferstream-nvidia (trt-llm, ort) or \
                     inferstream-intel (openvino, ovms, ort)",
                ))
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    inferstream_server::run_cli("inferstream-apple", &factory()).await
}
