//! `inferstream-intel`: the Intel arch binary.
//!
//! Primary catalog embed path: TurboEmbed C ABI (`backend = "openvino"`)
//! over in-process OpenVINO GenAI `TextEmbeddingPipeline` (feature
//! `openvino-genai`). Clients send plain strings; the ABI owns
//! openvino-tokenizers + CLS/MEAN/LAST + L2. There is **no OVMS client**
//! and no silent CPU fallback for AUTO/GPU. Secondary: llama.cpp
//! `GGML_SYCL` in-process (`--features llamacpp-sycl`). Tokenize for GGUF
//! is the llama.cpp vocab; embed Tokenize uses `tokenizer.json` next to
//! the OV model dir. GenAI and SYCL need oneAPI / OpenVINO sourced
//! (`source /opt/intel/oneapi/setvars.sh` or OpenVINO `setupvars.sh`).

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
            BackendKind::Openvino => {
                #[cfg(feature = "openvino")]
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
                #[cfg(not(feature = "openvino"))]
                Err(unsupported(model, "rebuild with --features openvino"))
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
    inferstream_server::run_cli(
        "inferstream-intel",
        Some(inferstream_server::Arch::Intel),
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

    #[test]
    fn ovms_backend_is_rejected_at_parse() {
        let result = Config::from_toml(
            r#"
            [[models]]
            name = "minilm_pipeline"
            backend = "ovms"
            endpoint = "http://127.0.0.1:8000"
            "#,
        );
        assert!(
            result.is_err(),
            "backend = \"ovms\" must not parse after OVMS removal"
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
            device = "GPU"
            path = "models/ov-rerank/ms-marco-minilm-l6"
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
        assert!(err.to_string().contains("turborerank"));
    }

    #[cfg(all(feature = "openvino", not(feature = "openvino-genai")))]
    #[test]
    fn openvino_without_genai_fails_at_startup() {
        let config = Config::from_toml(
            r#"
            [[models]]
            name = "minilm"
            backend = "openvino"
            path = "models/ov/minilm"
            device = "GPU"
            pooling = "mean"
            "#,
        )
        .unwrap();
        let result = build_registry(&config, &factory());
        let err = match result {
            Ok(_) => panic!("default GenAI path must fail at startup without openvino-genai"),
            Err(e) => e,
        };
        assert!(
            matches!(err, ServerError::InvalidModelConfig { .. }),
            "expected InvalidModelConfig, got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("TurboEmbed")
                || msg.contains("openvino-genai")
                || msg.contains("refusing"),
            "startup error must name TurboEmbed / openvino-genai, got {msg}"
        );
    }
}
