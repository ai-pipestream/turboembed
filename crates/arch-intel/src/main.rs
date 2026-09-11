//! `inferstream-intel`: the Intel arch binary.
//!
//! Primary engine path: OpenVINO, reached two ways. `backend = "ovms"`
//! forwards to a running OpenVINO Model Server over gRPC (same KServe OIP V2
//! family; no host OpenVINO install needed — this is the first working path
//! on OVMS-in-Docker hosts like krick-1). `backend = "openvino"` is the
//! in-process runtime (stub until the OpenVINO runtime is linked). Secondary:
//! llama.cpp built with `GGML_SYCL` (oneAPI / Level Zero, Docker-proven) for
//! GGUF models — the other side of the planned Battlemage bake-off (see
//! README). In-process engines need the oneAPI environment sourced
//! (`source /opt/intel/oneapi/setvars.sh`) in the build shell and in the
//! service unit that launches this binary; the ovms client backend does not.
//! ONNX Runtime covers plain ONNX models; the mock backend is always
//! available for wire-path smoke tests.

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
                    // Server-client mode: `endpoint` (or the environment
                    // fallback) points at a running llama-server; without it
                    // the in-process FFI path needs `path` (a GGUF file).
                    let endpoint = model.endpoint.clone().or_else(|| {
                        // Only models without a GGUF `path` fall back to the
                        // environment; path-mode entries stay in-process.
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
                #[cfg(not(feature = "llamacpp"))]
                Err(unsupported(model, "rebuild with --features llamacpp"))
            }
            BackendKind::Ovms => {
                #[cfg(feature = "ovms")]
                {
                    let endpoint = model
                        .endpoint
                        .clone()
                        .or_else(|| std::env::var("INFERSTREAM_OVMS_ENDPOINT").ok())
                        .ok_or_else(|| {
                            invalid(
                                model,
                                "backend \"ovms\" needs an upstream gRPC endpoint: set \
                                 `endpoint` in the model entry or INFERSTREAM_OVMS_ENDPOINT \
                                 in the environment"
                                    .to_string(),
                            )
                        })?;
                    let backend = inferstream_backend_ovms::OvmsBackend::new(endpoint)
                        .map_err(|e| invalid(model, e.to_string()))?;
                    Ok(Arc::new(backend))
                }
                #[cfg(not(feature = "ovms"))]
                Err(unsupported(model, "rebuild with --features ovms"))
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
                    use inferstream_backend_ort::{OrtBackend, OrtConfig, OrtDevice, Pooling};
                    let device = model
                        .device
                        .as_deref()
                        .map(OrtDevice::from_config)
                        .transpose()
                        .map_err(|e| invalid(model, e.to_string()))?
                        .unwrap_or_default();
                    let pooling = model
                        .pooling
                        .as_deref()
                        .map(Pooling::from_config)
                        .transpose()
                        .map_err(|e| invalid(model, e.to_string()))?
                        .unwrap_or_default();
                    let backend = OrtBackend::new(OrtConfig {
                        model_path: model.path.clone().unwrap_or_default(),
                        tokenizer_path: model.tokenizer_dir.clone(),
                        device,
                        max_seq_len: model.max_seq_len.map(|v| v as usize),
                        pooling,
                        normalize: model.normalize,
                    })
                    .map_err(|e| invalid(model, e.to_string()))?;
                    Ok(Arc::new(backend))
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

#[cfg(test)]
mod tests {
    use super::*;
    use inferstream_server::{build_registry, config::Config};

    #[test]
    fn routes_mock_and_rejects_foreign_arch_backends() {
        let config = Config::from_toml(
            r#"
            [[models]]
            name = "smoke"
            backend = "mock"
            "#,
        )
        .unwrap();
        let registry = build_registry(&config, &factory()).unwrap();
        assert!(registry.lookup("smoke").is_some());

        for (backend, snippet) in [
            ("trt-llm", "engine_dir = \"/engines/x\""),
            ("mlx", "path = \"/models/x\""),
        ] {
            let config = Config::from_toml(&format!(
                "[[models]]\nname = \"m\"\nbackend = \"{backend}\"\n{snippet}\n"
            ))
            .unwrap();
            let result = build_registry(&config, &factory());
            assert!(
                matches!(result, Err(ServerError::UnsupportedBackend { .. })),
                "backend {backend} must be rejected by the Intel binary"
            );
        }
    }

    #[cfg(feature = "ovms")]
    #[test]
    fn ovms_without_endpoint_fails_at_startup() {
        // No `endpoint` and no INFERSTREAM_OVMS_ENDPOINT: actionable error.
        std::env::remove_var("INFERSTREAM_OVMS_ENDPOINT");
        let config = Config::from_toml(
            r#"
            [[models]]
            name = "minilm_pipeline"
            backend = "ovms"
            "#,
        )
        .unwrap();
        let result = build_registry(&config, &factory());
        assert!(matches!(
            result,
            Err(ServerError::InvalidModelConfig { .. })
        ));
    }
}
