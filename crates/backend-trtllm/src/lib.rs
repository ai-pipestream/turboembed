//! TensorRT-LLM backend for inferstream — the NVIDIA peak-latency /
//! best-$-per-token path.
//!
//! # Integration shape (decided)
//!
//! This backend embeds the **TensorRT-LLM Executor API in-process** (C++
//! `tensorrt_llm::executor::Executor` behind a thin FFI layer). It is *not* a
//! NIM / OpenAI-HTTP wrapper and does not shell out to Triton: requests go
//! gRPC → registry → Executor with zero extra hops, which is the point of
//! the façade. NIM is used only as a benchmark oracle (see README).
//!
//! ## OIP tensor contract
//!
//! **Generation** (`ModelStreamInfer`, one gRPC stream chunk per Executor
//! response):
//!
//! | direction | tensor | dtype | shape | meaning |
//! |---|---|---|---|---|
//! | in  | `text`            | BYTES | `[1]`   | prompt (server-side tokenize via `tokenizer_dir`) |
//! | in  | `input_ids`       | INT32 | `[n]`   | alternative: pre-tokenized prompt (skips tokenizer) |
//! | in  | `max_tokens`      | INT32 | `[1]`   | optional cap, default from config |
//! | in  | `sampling.*`      | —     | —       | temperature/top_p/top_k as `parameters`, not tensors |
//! | out | `token`           | BYTES | `[1]`   | detokenized text piece for this chunk |
//! | out | `output_ids`      | INT32 | `[m]`   | token ids in this chunk (usually `m == 1` streaming) |
//! | out | param `final`     | bool  | —       | set on the terminal chunk (maps `Result::isFinal`) |
//!
//! Executor mapping: each OIP request becomes one
//! `executor::Request{input_token_ids, sampling_config, streaming=true}`;
//! `Executor::awaitResponses` results are fanned back onto the gRPC stream
//! keyed by the Executor request id ↔ OIP `request.id` correlation table.
//! In-flight batching is Executor-internal; the façade never batches.
//!
//! **Embeddings** (unary `ModelInfer`): TRT engine built from the embedding
//! model (via `trtllm-build` or plain TensorRT for BERT-likes); input `text`
//! (BYTES) or `input_ids` (INT32), output `embedding` FP32 `[d]` — identical
//! wire shape to the mock backend, so clients don't change between dev and
//! GPU serving.
//!
//! ## What ships today
//!
//! The FFI link is stubbed. This crate compiles as pure Rust everywhere and
//! provides the full config surface ([`TrtLlmConfig`]) plus request
//! validation, returning `Unavailable` from runtime entry points unless
//! built with `--features trtllm-sys` on a CUDA + TRT-LLM host (in which
//! case it currently still returns `Unavailable`; the cxx/bindgen layer is
//! the next unit of work and slots in behind [`TrtLlmBackend::infer`] /
//! [`TrtLlmBackend::infer_stream`] without touching the server).

use async_trait::async_trait;
use serde::Deserialize;

use inferstream_backend::{Backend, BackendError, ModelMetadata};
use inferstream_protocol::inference::{ModelInferRequest, ModelInferResponse};

/// Configuration for one TRT-LLM-served model, mirroring the knobs the
/// Executor actually exposes (subset; extend as the FFI lands).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TrtLlmConfig {
    /// Directory containing the compiled engine (`rank0.engine`,
    /// `config.json`) produced by `trtllm-build`.
    pub engine_dir: String,

    /// Tokenizer artifacts (HF `tokenizer.json` etc.) for server-side
    /// tokenize/detokenize when clients send/receive `text` tensors.
    pub tokenizer_dir: Option<String>,

    /// Executor `max_batch_size`; must not exceed what the engine was built
    /// with. Default: engine's build-time value.
    pub max_batch_size: Option<u32>,

    /// Compute dtype the engine was built for (`fp16` / `bf16` / `fp8` /
    /// `int8`). Informational + validated against engine config at load.
    pub dtype: Option<String>,

    /// Fraction of free GPU memory handed to the KV cache
    /// (`kv_cache_free_gpu_mem_fraction`). Default 0.9.
    pub kv_cache_free_gpu_mem_fraction: Option<f32>,

    /// Default cap on generated tokens when the request omits `max_tokens`.
    pub max_output_tokens: Option<u32>,
}

/// TensorRT-LLM Executor backend. Skeleton in v0.x: full config + contract,
/// runtime stubbed until the FFI layer lands behind `trtllm-sys`.
#[derive(Debug, Default, Clone)]
pub struct TrtLlmBackend {
    config: TrtLlmConfig,
}

impl TrtLlmBackend {
    /// Validate config shape (paths present, sane fractions). Does not touch
    /// the GPU; engine load happens on first readiness once FFI lands.
    pub fn new(config: TrtLlmConfig) -> Result<Self, BackendError> {
        if config.engine_dir.is_empty() {
            return Err(BackendError::InvalidRequest(
                "trt-llm models require engine_dir (directory from trtllm-build)".into(),
            ));
        }
        if let Some(fraction) = config.kv_cache_free_gpu_mem_fraction {
            if !(0.0..=1.0).contains(&fraction) {
                return Err(BackendError::InvalidRequest(format!(
                    "kv_cache_free_gpu_mem_fraction must be in [0, 1], got {fraction}"
                )));
            }
        }
        Ok(Self { config })
    }

    pub fn config(&self) -> &TrtLlmConfig {
        &self.config
    }

    fn unavailable() -> BackendError {
        if cfg!(feature = "trtllm-sys") {
            BackendError::Unavailable(
                "TRT-LLM Executor FFI layer not yet implemented (trtllm-sys is a link-surface \
                 placeholder)"
                    .into(),
            )
        } else {
            BackendError::Unavailable(
                "TRT-LLM backend built without the Executor runtime; rebuild inferstream-nvidia \
                 with --features trtllm on a CUDA + TensorRT-LLM host"
                    .into(),
            )
        }
    }
}

#[async_trait]
impl Backend for TrtLlmBackend {
    fn id(&self) -> &str {
        "trt-llm"
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
    fn requires_engine_dir() {
        assert!(TrtLlmBackend::new(TrtLlmConfig::default()).is_err());
        assert!(TrtLlmBackend::new(TrtLlmConfig {
            engine_dir: "/engines/llama".into(),
            ..Default::default()
        })
        .is_ok());
    }

    #[test]
    fn validates_kv_cache_fraction() {
        let result = TrtLlmBackend::new(TrtLlmConfig {
            engine_dir: "/engines/llama".into(),
            kv_cache_free_gpu_mem_fraction: Some(1.5),
            ..Default::default()
        });
        assert!(result.is_err());
    }
}
