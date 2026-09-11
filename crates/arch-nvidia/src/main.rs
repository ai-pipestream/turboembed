//! `inferstream-nvidia`: the NVIDIA arch binary.
//!
//! Primary engine: ONNX Runtime with the CUDA / TensorRT execution provider
//! for embeddings (`backend = "ort"`). Secondary: llama.cpp-CUDA for GGUF
//! models. The in-process TensorRT-LLM Executor (`backend = "trt-llm"`,
//! routing surface behind feature `trtllm`, real runtime link behind
//! `trtllm-sys`) is deferred until generative LLMs are mandated; the
//! skeleton stays wired so enabling it is a build flag plus config entry.
//! The mock backend is always available for wire-path smoke tests before
//! engines are loaded.

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
    inferstream_server::run_cli("inferstream-nvidia", &factory()).await
}
