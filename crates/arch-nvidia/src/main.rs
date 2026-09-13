//! `inferstream-nvidia`: the NVIDIA arch binary.
//!
//! Catalog embeds (`backend = "ort"`, typically `minilm`) are a thin gRPC
//! façade over the TurboEmbed C ABI (`include/turboembed.h`). Real vectors
//! need `--features ort-cuda` (ORT CUDA EP + IoBinding). Missing GPU or
//! missing feature fails at startup — never mock, never silent CPU.
//! TensorRT-LLM (`backend = "trt-llm"`) is the optional later peak path
//! for generative models; llama.cpp-CUDA serves GGUF models. The mock
//! backend is always available for wire-path smoke tests.

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
            BackendKind::Mock => inferstream_server::serve_mock(model, mock.clone()),
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
                    let backend =
                        inferstream_backend_turboembed::TurboEmbedBackend::open_for_model(
                            &model.name,
                            model.backend.as_str(),
                            model.device.as_deref(),
                        )
                        .map_err(|e| invalid(model, e.to_string()))?;
                    Ok(Arc::new(backend))
                }
                #[cfg(not(feature = "ort"))]
                Err(unsupported(model, "rebuild with --features ort"))
            }
            BackendKind::TurboRerank => {
                #[cfg(feature = "turborerank")]
                {
                    let backend =
                        inferstream_backend_turborerank::TurboRerankBackend::open_for_model(
                            &model.name,
                            model.device.as_deref(),
                            model.path.as_deref(),
                            model.max_batch_size,
                        )
                        .map_err(|e| invalid(model, e.to_string()))?;
                    Ok(Arc::new(backend))
                }
                #[cfg(not(feature = "turborerank"))]
                Err(unsupported(
                    model,
                    "rebuild with --features turborerank (catalog CE aliases \
                     never fall back to word-overlap)",
                ))
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

    #[cfg(all(feature = "ort", not(feature = "ort-cuda")))]
    #[test]
    fn catalog_ort_alias_fails_without_turboembed_provider() {
        // Catalog aliases must not sit behind a stub that only fails at
        // request time. Construction names TurboEmbed + ort-cuda.
        let config = Config::from_toml(
            r#"
            [[models]]
            name = "minilm"
            backend = "ort"
            device = "cuda"
            path = "/models/minilm.onnx"
            pooling = "mean"
            "#,
        )
        .unwrap();
        let err = match build_registry(&config, &factory()) {
            Ok(_) => panic!("minilm must fail at startup without ort-cuda"),
            Err(e) => e,
        };
        assert!(
            matches!(err, ServerError::InvalidModelConfig { .. }),
            "minilm must fail at startup without ort-cuda, got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("TurboEmbed") || msg.contains("ort-cuda") || msg.contains("refusing"),
            "startup error must name TurboEmbed / ort-cuda, got {msg}"
        );
    }

    #[cfg(not(feature = "turborerank"))]
    #[test]
    fn catalog_ce_alias_fails_without_turborerank_feature() {
        let config = Config::from_toml(
            r#"
            [[models]]
            name = "ms-marco-minilm-l6"
            backend = "turborerank"
            device = "cuda"
            path = "models/rerank/ms-marco-minilm-l6"
            "#,
        )
        .unwrap();
        let err = match build_registry(&config, &factory()) {
            Ok(_) => panic!("CE alias must fail at startup without --features turborerank"),
            Err(e) => e,
        };
        assert!(
            matches!(err, ServerError::UnsupportedBackend { .. }),
            "expected UnsupportedBackend, got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("turborerank"),
            "startup error must name the turborerank feature, got {msg}"
        );
    }

    #[test]
    fn mock_cannot_satisfy_catalog_ce_alias() {
        let config = Config::from_toml(
            r#"
            [[models]]
            name = "ms-marco-minilm-l6"
            backend = "mock"
            "#,
        )
        .unwrap();
        let err = match build_registry(&config, &factory()) {
            Ok(_) => panic!("CE alias on mock must fail at startup"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("mock") || msg.contains("word-overlap") || msg.contains("MiniLM"),
            "must refuse mock-as-MiniLM, got {msg}"
        );
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
