//! `inferstream-intel`: the Intel arch binary.
//!
//! Two engine candidates for Arc/Battlemage-class hardware, kept side by
//! side for the planned bake-off (see README): llama.cpp built with
//! `GGML_SYCL` (oneAPI / Level Zero) for GGUF models, and OpenVINO for
//! CPU/GPU/NPU graphs. Both need the oneAPI environment sourced
//! (`source /opt/intel/oneapi/setvars.sh`) in the build shell and in the
//! service unit that launches this binary. ONNX Runtime covers plain ONNX
//! models; the mock backend is always available for wire-path smoke tests.

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
            BackendKind::LlamaCpp => {
                #[cfg(feature = "llamacpp")]
                {
                    use inferstream_backend_llamacpp::{
                        LlamaCppBackend, LlamaCppConfig, LlamaDevice,
                    };
                    let device = model
                        .device
                        .as_deref()
                        .map(LlamaDevice::from_config)
                        .transpose()
                        .map_err(|e| invalid(model, e.to_string()))?
                        .unwrap_or(LlamaDevice::Sycl);
                    let backend = LlamaCppBackend::new(LlamaCppConfig {
                        model_path: model.path.clone().unwrap_or_default(),
                        device,
                        n_gpu_layers: model.n_gpu_layers,
                        max_batch_size: model.max_batch_size,
                    })
                    .map_err(|e| invalid(model, e.to_string()))?;
                    Ok(Arc::new(backend))
                }
                #[cfg(not(feature = "llamacpp"))]
                Err(unsupported(model, "rebuild with --features llamacpp"))
            }
            BackendKind::Openvino => {
                #[cfg(feature = "openvino")]
                {
                    Ok(Arc::new(
                        inferstream_backend_openvino::OpenVinoBackend::new(
                            model.path.clone(),
                            model.device.clone(),
                        ),
                    ))
                }
                #[cfg(not(feature = "openvino"))]
                Err(unsupported(model, "rebuild with --features openvino"))
            }
            BackendKind::Ort => {
                #[cfg(feature = "ort")]
                {
                    Ok(Arc::new(inferstream_backend_ort::OrtBackend::new(
                        model.path.clone(),
                    )))
                }
                #[cfg(not(feature = "ort"))]
                Err(unsupported(model, "rebuild with --features ort"))
            }
            BackendKind::TrtLlm | BackendKind::Mlx => Err(unsupported(
                model,
                "not an Intel-arch engine; use inferstream-nvidia (trt-llm) or \
                 inferstream-apple (mlx)",
            )),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    inferstream_server::run_cli("inferstream-intel", &factory()).await
}
