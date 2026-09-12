//! `inferstream-nvidia`: the NVIDIA arch binary.
//!
//! The **primary path today is ONNX Runtime on GPU** (`backend = "ort"`,
//! `device = "cuda"`; real link behind feature `ort-cuda`, CPU-only link
//! behind `ort-runtime`) serving encoder embedding models (BGE, MiniLM).
//! TensorRT-LLM (`backend = "trt-llm"`, routing surface behind `trtllm`,
//! real runtime link behind `trtllm-sys`) is the optional later peak path
//! for generative models; llama.cpp-CUDA serves GGUF models as the
//! secondary/fallback path. The mock backend is always available for
//! wire-path smoke tests before engines are loaded.

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
            BackendKind::TrtLlm => {
                #[cfg(feature = "trtllm")]
                {
                    let backend = inferstream_backend_trtllm::TrtLlmBackend::new(
                        inferstream_backend_trtllm::TrtLlmConfig {
                            engine_dir: model.engine_dir.clone().unwrap_or_default(),
                            tokenizer_dir: model.tokenizer_dir.clone(),
                            max_batch_size: model.max_batch_size,
                            dtype: model.dtype.clone(),
                            kv_cache_free_gpu_mem_fraction: None,
                            max_output_tokens: None,
                        },
                    )
                    .map_err(|e| invalid(model, e.to_string()))?;
                    Ok(Arc::new(backend))
                }
                #[cfg(not(feature = "trtllm"))]
                Err(unsupported(model, "rebuild with --features trtllm"))
            }
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
                        .unwrap_or(LlamaDevice::Cuda);
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
                #[cfg(not(feature = "llamacpp"))]
                Err(unsupported(model, "rebuild with --features llamacpp"))
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
            BackendKind::Openvino | BackendKind::Mlx => Err(unsupported(
                model,
                "not an NVIDIA-arch engine; use inferstream-intel (openvino) or \
                 inferstream-apple (mlx)",
            )),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    inferstream_server::run_cli(
        "inferstream-nvidia",
        Some(inferstream_server::Arch::Nvidia),
        &factory(),
    )
    .await
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

        for backend in ["openvino", "mlx"] {
            let config = Config::from_toml(&format!(
                "[[models]]\nname = \"m\"\nbackend = \"{backend}\"\npath = \"/models/x\"\n"
            ))
            .unwrap();
            let result = build_registry(&config, &factory());
            assert!(
                matches!(result, Err(ServerError::UnsupportedBackend { .. })),
                "backend {backend} must be rejected by the NVIDIA binary"
            );
        }
    }

    #[cfg(all(feature = "ort", not(feature = "ort-runtime")))]
    #[test]
    fn ort_stub_builds_but_reports_not_ready() {
        // Without the runtime feature the ORT surface still routes; readiness
        // is false and inference reports Unavailable naming the feature.
        let config = Config::from_toml(
            r#"
            [[models]]
            name = "minilm"
            backend = "ort"
            path = "/models/minilm.onnx"
            pooling = "mean"
            "#,
        )
        .unwrap();
        let registry = build_registry(&config, &factory()).unwrap();
        let backend = registry.lookup("minilm").unwrap();
        let ready = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(backend.model_ready("minilm", ""));
        assert!(!ready);
    }

    #[cfg(feature = "ort")]
    #[test]
    fn ort_invalid_device_fails_at_startup() {
        let config = Config::from_toml(
            r#"
            [[models]]
            name = "minilm"
            backend = "ort"
            device = "npu"
            path = "/models/minilm.onnx"
            "#,
        )
        .unwrap();
        assert!(matches!(
            build_registry(&config, &factory()),
            Err(ServerError::InvalidModelConfig { .. })
        ));
    }
}
