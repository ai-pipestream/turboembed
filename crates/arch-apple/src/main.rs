//! LEGACY Rust `inferstream-apple` façade.
//!
//! The supported Mac serve path is the all-Swift gRPC server in `swift/`
//! (`make apple`, `scripts/smoke-apple.sh`). Catalog embeds there call
//! `libTurboEmbed.dylib`. This Rust binary still compiles so Linux CI can
//! type-check the old FFI wiring plus the TurboEmbed façade. Do not use it
//! as the Apple production server.

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
    let mlx_engine: Arc<inferstream_backend_apple::MlxEngine> =
        Arc::new(inferstream_backend_apple::MlxEngine::from_env());
    move |model: &ModelConfig| -> Result<Arc<dyn Backend>, ServerError> {
        match model.backend {
            BackendKind::Mock => Ok(mock.clone()),
            BackendKind::Mlx => {
                // Catalog embeds (pooling set) go through TurboEmbed C ABI.
                // LLM aliases stay on the legacy MLX generation path so
                // Tokenize / StreamInfer are not rewritten.
                if model.pooling.is_some() {
                    let backend =
                        inferstream_backend_turboembed::TurboEmbedBackend::open_for_model(
                            &model.name,
                            model.backend.as_str(),
                            model.device.as_deref(),
                        )
                        .map_err(|e| invalid(model, e.to_string()))?;
                    return Ok(Arc::new(backend));
                }
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
                    Arc::clone(&mlx_engine),
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
            BackendKind::TrtLlm | BackendKind::Ort | BackendKind::Openvino => Err(unsupported(
                model,
                "not an Apple-arch engine; use inferstream-nvidia (trt-llm, ort) or \
                     inferstream-intel (openvino, ort)",
            )),
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
