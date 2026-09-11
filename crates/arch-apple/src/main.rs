//! `inferstream-apple`: the Apple arch binary — **native macOS hosts only**.
//!
//! Serves MLX models (`backend = "mlx"`, via the persistent Python bridge
//! worker in `backend-apple`) and GGUF models via llama.cpp-Metal
//! (`backend = "llama-cpp"`, `device = "metal"`). Metal and the Neural
//! Engine do not pass through Linux containers, so this binary is deployed
//! directly on the Mac (launchd service or plain process), not
//! containerized. It still compiles on Linux so CI can type-check the
//! wiring; at runtime on non-macOS the MLX bridge spawn fails and every MLX
//! call reports Unavailable.

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
    // One persistent Python worker shared by every MLX model, so all models
    // stay hot in the same process (see backend-apple's bridge module).
    // Location comes from INFERSTREAM_MLX_PYTHON / INFERSTREAM_MLX_BRIDGE
    // (defaults: .venv/bin/python, python/mlx_bridge.py — scripts/setup-mlx.sh
    // creates the venv).
    let mlx_worker: Arc<inferstream_backend_apple::MlxWorker> =
        Arc::new(inferstream_backend_apple::MlxWorker::new(
            inferstream_backend_apple::MlxWorkerConfig::from_env(),
        ));
    move |model: &ModelConfig| -> Result<Arc<dyn Backend>, ServerError> {
        match model.backend {
            BackendKind::Mock => Ok(mock.clone()),
            BackendKind::Mlx => {
                let defaults = inferstream_backend_apple::MlxConfig::default();
                let backend = inferstream_backend_apple::MlxBackend::new(
                    inferstream_backend_apple::MlxConfig {
                        model: model.path.clone().unwrap_or_default(),
                        max_batch: model
                            .max_batch_size
                            .map(|n| n as usize)
                            .unwrap_or(defaults.max_batch),
                        normalize: model.normalize.unwrap_or(defaults.normalize),
                        max_output_tokens: defaults.max_output_tokens,
                    },
                    Arc::clone(&mlx_worker),
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
    inferstream_server::run_cli(
        "inferstream-apple",
        Some(inferstream_server::Arch::Apple),
        &factory(),
    )
    .await
}
