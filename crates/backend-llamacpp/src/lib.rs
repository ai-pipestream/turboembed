//! llama.cpp (GGUF) backend for inferstream.
//!
//! One crate serves every llama.cpp build flavor; the *device* is decided by
//! how the native library was compiled plus the per-model `device` config:
//!
//! | `device` | build requirement | used by |
//! |---|---|---|
//! | `cuda`   | crate feature `cuda` (llama.cpp built with `GGML_CUDA`)   | `inferstream-nvidia` (GGUF generation under ORT embeddings) |
//! | `metal`  | crate feature `metal` (default on macOS) | `inferstream-apple` (alternative to MLX for GGUF) |
//! | `cpu`    | crate feature `runtime` | everywhere |
//! | `sycl` / `vulkan` | not built by this crate yet | reported `Unavailable` at startup |
//!
//! ## Wire contract
//!
//! * generation → `infer_stream`: one `token` BYTES chunk per decoded token
//!   piece, echoing the request `id`, with a `final` bool parameter on the
//!   last chunk (identical shapes to the mock backend). Generation
//!   parameters ride on the OIP request `parameters`: `max_tokens` (int,
//!   default 128), `temperature` (double, default 0 = greedy), `seed` (int).
//! * unary `infer` → the same generation collected into one `text` BYTES
//!   output (the full completion).
//! * `tokenize` / `detokenize` → served from the GGUF's own vocabulary.
//!
//! Without the `runtime` feature this crate compiles everywhere as pure Rust
//! and reports `Unavailable` at runtime.

#[cfg(not(feature = "runtime"))]
use async_trait::async_trait;
use inferstream_backend::BackendError;
#[cfg(not(feature = "runtime"))]
use inferstream_backend::{Backend, ModelMetadata};
#[cfg(not(feature = "runtime"))]
use inferstream_protocol::inference::ModelInferResponse;
use inferstream_protocol::inference::{infer_parameter::ParameterChoice, ModelInferRequest};

/// Device a llama.cpp model should run on. Must match a capability the
/// linked llama.cpp library was actually built with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LlamaDevice {
    Cuda,
    Sycl,
    Metal,
    Vulkan,
    #[default]
    Cpu,
}

impl LlamaDevice {
    pub fn from_config(s: &str) -> Result<Self, BackendError> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "cuda" => Self::Cuda,
            "sycl" => Self::Sycl,
            "metal" => Self::Metal,
            "vulkan" => Self::Vulkan,
            "cpu" => Self::Cpu,
            other => {
                return Err(BackendError::InvalidRequest(format!(
                    "unknown llama.cpp device {other:?} (expected cuda|sycl|metal|vulkan|cpu)"
                )))
            }
        })
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Sycl => "sycl",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
            Self::Cpu => "cpu",
        }
    }
}

/// Configuration for one llama.cpp-served model.
#[derive(Debug, Clone, Default)]
pub struct LlamaCppConfig {
    /// Path to the GGUF file.
    pub model_path: String,
    /// Target device (see [`LlamaDevice`]).
    pub device: LlamaDevice,
    /// Layers to offload to the accelerator; `None` = offload everything.
    pub n_gpu_layers: Option<u32>,
    /// Concurrent sequences the context schedules (`n_parallel`).
    pub max_batch_size: Option<u32>,
    /// Context window (`n_ctx`); `None` = 4096 capped to the model's
    /// training context.
    pub n_ctx: Option<u32>,
}

/// Per-request generation parameters, parsed from the OIP request
/// `parameters` map. Pure logic, so the default (stub) build tests it.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateParams {
    /// Maximum new tokens to generate (`max_tokens`, default 128).
    pub max_tokens: usize,
    /// Sampling temperature (`temperature`); `0` = greedy (default).
    pub temperature: f32,
    /// Sampling seed (`seed`, default 42; only used when temperature > 0).
    pub seed: u32,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.0,
            seed: 42,
        }
    }
}

impl GenerateParams {
    pub fn from_request(request: &ModelInferRequest) -> Result<Self, BackendError> {
        let mut params = Self::default();
        if let Some(choice) = parameter(request, "max_tokens") {
            match choice {
                ParameterChoice::Int64Param(v) if *v > 0 => params.max_tokens = *v as usize,
                other => {
                    return Err(BackendError::InvalidRequest(format!(
                        "parameter \"max_tokens\" must be a positive int64, got {other:?}"
                    )))
                }
            }
        }
        if let Some(choice) = parameter(request, "temperature") {
            match choice {
                ParameterChoice::DoubleParam(v) if *v >= 0.0 => params.temperature = *v as f32,
                other => {
                    return Err(BackendError::InvalidRequest(format!(
                        "parameter \"temperature\" must be a non-negative double, got {other:?}"
                    )))
                }
            }
        }
        if let Some(choice) = parameter(request, "seed") {
            match choice {
                ParameterChoice::Int64Param(v) if *v >= 0 => params.seed = *v as u32,
                other => {
                    return Err(BackendError::InvalidRequest(format!(
                        "parameter \"seed\" must be a non-negative int64, got {other:?}"
                    )))
                }
            }
        }
        Ok(params)
    }
}

fn parameter<'a>(request: &'a ModelInferRequest, name: &str) -> Option<&'a ParameterChoice> {
    request
        .parameters
        .get(name)
        .and_then(|p| p.parameter_choice.as_ref())
}

/// Extract the single UTF-8 prompt from the OIP request's `text` tensor.
#[cfg_attr(not(feature = "runtime"), allow(dead_code))]
fn text_prompt(request: &ModelInferRequest) -> Result<String, BackendError> {
    use inferstream_protocol::tensor::{unpack_bytes, DataType};
    let (index, tensor) = request
        .inputs
        .iter()
        .enumerate()
        .find(|(_, t)| t.name == "text")
        .ok_or_else(|| {
            BackendError::InvalidRequest("expected an input tensor named \"text\"".into())
        })?;
    if tensor.datatype != DataType::Bytes.as_oip() {
        return Err(BackendError::InvalidRequest(format!(
            "input \"text\" must be BYTES, got {:?}",
            tensor.datatype
        )));
    }
    let raw = request.raw_input_contents.get(index).ok_or_else(|| {
        BackendError::InvalidRequest("input \"text\" must be sent via raw_input_contents".into())
    })?;
    let mut elements = unpack_bytes(raw)
        .map_err(|e| BackendError::InvalidRequest(format!("malformed BYTES payload: {e}")))?;
    if elements.len() != 1 {
        return Err(BackendError::InvalidRequest(format!(
            "generation takes exactly one text element per request, got {}",
            elements.len()
        )));
    }
    String::from_utf8(elements.remove(0))
        .map_err(|e| BackendError::InvalidRequest(format!("prompt is not UTF-8: {e}")))
}

#[cfg(feature = "runtime")]
mod engine;
#[cfg(feature = "runtime")]
pub use engine::LlamaCppBackend;

/// Stub used when the crate is compiled without the `runtime` feature: the
/// routing surface still type-checks, and requests fail with a clear
/// `Unavailable` naming the feature to enable.
#[cfg(not(feature = "runtime"))]
#[derive(Debug, Default, Clone)]
pub struct LlamaCppBackend {
    config: LlamaCppConfig,
}

#[cfg(not(feature = "runtime"))]
impl LlamaCppBackend {
    pub fn new(config: LlamaCppConfig) -> Result<Self, BackendError> {
        if config.model_path.is_empty() {
            return Err(BackendError::InvalidRequest(
                "llama-cpp models require path (a GGUF file)".into(),
            ));
        }
        Ok(Self { config })
    }

    pub fn config(&self) -> &LlamaCppConfig {
        &self.config
    }

    fn unavailable() -> BackendError {
        BackendError::Unavailable(
            "llama.cpp is not compiled into this binary; rebuild with \
             --features llamacpp-runtime (CPU) or llamacpp-cuda (CUDA)"
                .into(),
        )
    }
}

#[cfg(not(feature = "runtime"))]
#[async_trait]
impl Backend for LlamaCppBackend {
    fn id(&self) -> &str {
        "llama-cpp"
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
    use inferstream_protocol::inference::InferParameter;
    use inferstream_protocol::tensor::pack_bytes;
    use std::collections::HashMap;

    #[test]
    fn device_parsing() {
        assert_eq!(LlamaDevice::from_config("CUDA").unwrap(), LlamaDevice::Cuda);
        assert_eq!(LlamaDevice::from_config("sycl").unwrap(), LlamaDevice::Sycl);
        assert!(LlamaDevice::from_config("tpu").is_err());
    }

    #[cfg(not(feature = "runtime"))]
    #[test]
    fn requires_model_path() {
        assert!(LlamaCppBackend::new(LlamaCppConfig::default()).is_err());
        assert!(LlamaCppBackend::new(LlamaCppConfig {
            model_path: "/models/x.gguf".into(),
            ..Default::default()
        })
        .is_ok());
    }

    fn request_with_params(params: HashMap<String, InferParameter>) -> ModelInferRequest {
        ModelInferRequest {
            parameters: params,
            ..Default::default()
        }
    }

    fn int_param(v: i64) -> InferParameter {
        InferParameter {
            parameter_choice: Some(ParameterChoice::Int64Param(v)),
        }
    }

    fn double_param(v: f64) -> InferParameter {
        InferParameter {
            parameter_choice: Some(ParameterChoice::DoubleParam(v)),
        }
    }

    #[test]
    fn generate_params_defaults() {
        let params = GenerateParams::from_request(&ModelInferRequest::default()).unwrap();
        assert_eq!(params, GenerateParams::default());
        assert_eq!(params.max_tokens, 128);
        assert_eq!(params.temperature, 0.0);
    }

    #[test]
    fn generate_params_parse_and_validate() {
        let request = request_with_params(HashMap::from([
            ("max_tokens".to_string(), int_param(32)),
            ("temperature".to_string(), double_param(0.7)),
            ("seed".to_string(), int_param(7)),
        ]));
        let params = GenerateParams::from_request(&request).unwrap();
        assert_eq!(params.max_tokens, 32);
        assert!((params.temperature - 0.7).abs() < 1e-6);
        assert_eq!(params.seed, 7);

        let bad = request_with_params(HashMap::from([("max_tokens".to_string(), int_param(0))]));
        assert!(matches!(
            GenerateParams::from_request(&bad),
            Err(BackendError::InvalidRequest(_))
        ));
        let bad = request_with_params(HashMap::from([(
            "temperature".to_string(),
            double_param(-1.0),
        )]));
        assert!(matches!(
            GenerateParams::from_request(&bad),
            Err(BackendError::InvalidRequest(_))
        ));
    }

    #[test]
    fn text_prompt_takes_exactly_one_element() {
        use inferstream_protocol::inference::model_infer_request::InferInputTensor;
        let make = |elements: &[&[u8]]| ModelInferRequest {
            inputs: vec![InferInputTensor {
                name: "text".into(),
                datatype: "BYTES".into(),
                shape: vec![elements.len() as i64],
                parameters: HashMap::new(),
                contents: None,
            }],
            raw_input_contents: vec![pack_bytes(elements)],
            ..Default::default()
        };
        assert_eq!(text_prompt(&make(&[b"hello"])).unwrap(), "hello");
        assert!(matches!(
            text_prompt(&make(&[b"a", b"b"])),
            Err(BackendError::InvalidRequest(_))
        ));
        assert!(matches!(
            text_prompt(&ModelInferRequest::default()),
            Err(BackendError::InvalidRequest(_))
        ));
    }
}
