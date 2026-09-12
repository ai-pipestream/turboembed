//! ONNX Runtime embedding backend for inferstream.
//!
//! Serves transformer embedding models exported to ONNX (BGE, MiniLM, …)
//! through the maintained [`ort`](https://github.com/pykeio/ort) crate:
//! one session per model, tokenization server-side via HuggingFace
//! `tokenizers`, mean/CLS pooling and L2 normalization in Rust.
//!
//! Execution providers are selected by config `device`:
//!
//! * `"cpu"` — always available once the `runtime` feature is on.
//! * `"cuda"` — CUDA EP; crate feature `cuda`, needs NVIDIA driver +
//!   CUDA 12 + cuDNN 9 on the host.
//! * `"tensorrt"` — TensorRT EP; crate feature `tensorrt`, additionally
//!   needs TensorRT 10 libraries on the host.
//!
//! EP registration uses `error_on_failure`, so a missing GPU stack fails
//! **at startup** with the real ONNX Runtime error instead of silently
//! falling back to CPU.
//!
//! Without the `runtime` feature this crate still compiles everywhere and
//! reports `Unavailable` at request time (the pre-engine stub behavior), so
//! CI and non-GPU hosts keep type-checking the full routing surface.
//!
//! Wire contract (same as the mock, so clients do not change): unary
//! `ModelInfer` with a `BYTES` input tensor named `text` (one or more
//! elements), output `embedding` FP32 `[d]` for a single text or `[n, d]`
//! for a batch.

#[cfg(not(feature = "runtime"))]
use async_trait::async_trait;
use inferstream_backend::BackendError;
#[cfg(not(feature = "runtime"))]
use inferstream_backend::{Backend, ModelMetadata};
use inferstream_protocol::inference::ModelInferRequest;
#[cfg(not(feature = "runtime"))]
use inferstream_protocol::inference::ModelInferResponse;

/// Which ONNX Runtime execution provider to register for a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OrtDevice {
    #[default]
    Cpu,
    Cuda,
    TensorRt,
}

impl OrtDevice {
    /// Parse the config `device` string (`"cpu"` / `"cuda"` / `"tensorrt"`).
    pub fn from_config(s: &str) -> Result<Self, BackendError> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "tensorrt" | "trt" => Ok(Self::TensorRt),
            other => Err(BackendError::InvalidRequest(format!(
                "unknown ort device {other:?}; expected \"cpu\", \"cuda\" or \"tensorrt\""
            ))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::TensorRt => "tensorrt",
        }
    }
}

/// How token-level hidden states are reduced to one sentence embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Pooling {
    /// Attention-mask-weighted mean over tokens (sentence-transformers
    /// default; correct for BGE-style models exported without a pooling head).
    #[default]
    Mean,
    /// First token (`[CLS]`) hidden state.
    Cls,
}

impl Pooling {
    /// Parse the config `pooling` string (`"mean"` / `"cls"`).
    pub fn from_config(s: &str) -> Result<Self, BackendError> {
        match s.to_ascii_lowercase().as_str() {
            "mean" => Ok(Self::Mean),
            "cls" => Ok(Self::Cls),
            other => Err(BackendError::InvalidRequest(format!(
                "unknown pooling {other:?}; expected \"mean\" or \"cls\""
            ))),
        }
    }
}

/// Configuration for one ONNX Runtime embedding session.
#[derive(Debug, Clone, Default)]
pub struct OrtConfig {
    /// Path to the `.onnx` model file.
    pub model_path: String,
    /// Path to `tokenizer.json` (HuggingFace fast tokenizer). When omitted,
    /// `tokenizer.json` is searched next to the model file and one directory
    /// up (the HF snapshot layout, where the model lives in `onnx/`).
    pub tokenizer_path: Option<String>,
    pub device: OrtDevice,
    /// Truncation length for tokenization; defaults to 512.
    pub max_seq_len: Option<usize>,
    pub pooling: Pooling,
    /// L2-normalize embeddings (default true — what BGE/MiniLM clients expect).
    pub normalize: Option<bool>,
}

/// Pooling / normalization math shared by the real engine and its tests.
/// Kept engine-free so `cargo test` exercises it without ONNX Runtime.
pub mod pool {
    /// Attention-mask-weighted mean over the token axis.
    ///
    /// `hidden` is row-major `[batch, seq, dim]`, `mask` is `[batch, seq]`
    /// (1 = real token). Returns `[batch, dim]`.
    pub fn mean_pool(
        hidden: &[f32],
        mask: &[i64],
        batch: usize,
        seq: usize,
        dim: usize,
    ) -> Vec<f32> {
        debug_assert_eq!(hidden.len(), batch * seq * dim);
        debug_assert_eq!(mask.len(), batch * seq);
        let mut out = vec![0.0f32; batch * dim];
        for b in 0..batch {
            let mut count = 0f32;
            for s in 0..seq {
                if mask[b * seq + s] == 0 {
                    continue;
                }
                count += 1.0;
                let row = &hidden[(b * seq + s) * dim..(b * seq + s + 1) * dim];
                let acc = &mut out[b * dim..(b + 1) * dim];
                for (a, v) in acc.iter_mut().zip(row) {
                    *a += v;
                }
            }
            if count > 0.0 {
                for a in &mut out[b * dim..(b + 1) * dim] {
                    *a /= count;
                }
            }
        }
        out
    }

    /// First-token (`[CLS]`) pooling: `[batch, seq, dim]` → `[batch, dim]`.
    pub fn cls_pool(hidden: &[f32], batch: usize, seq: usize, dim: usize) -> Vec<f32> {
        debug_assert_eq!(hidden.len(), batch * seq * dim);
        let mut out = Vec::with_capacity(batch * dim);
        for b in 0..batch {
            out.extend_from_slice(&hidden[b * seq * dim..b * seq * dim + dim]);
        }
        out
    }

    /// In-place L2 normalization of each `dim`-length row.
    pub fn l2_normalize(rows: &mut [f32], dim: usize) {
        for row in rows.chunks_exact_mut(dim) {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in row {
                    *v /= norm;
                }
            }
        }
    }
}

/// Extract the batch of UTF-8 texts from the OIP request's `text` tensor.
/// (Only the real engine calls this at runtime; the stub build keeps it for
/// its unit tests.)
#[cfg_attr(not(feature = "runtime"), allow(dead_code))]
fn text_inputs(request: &ModelInferRequest) -> Result<Vec<String>, BackendError> {
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
    let elements = unpack_bytes(raw)
        .map_err(|e| BackendError::InvalidRequest(format!("malformed BYTES payload: {e}")))?;
    if elements.is_empty() {
        return Err(BackendError::InvalidRequest(
            "input \"text\" contained no elements".into(),
        ));
    }
    elements
        .into_iter()
        .map(|e| {
            String::from_utf8(e)
                .map_err(|e| BackendError::InvalidRequest(format!("text is not UTF-8: {e}")))
        })
        .collect()
}

#[cfg(feature = "runtime")]
mod engine;
#[cfg(feature = "runtime")]
pub use engine::OrtBackend;

/// Stub used when the crate is compiled without the `runtime` feature:
/// the routing surface still type-checks, and requests fail with a clear
/// `Unavailable` naming the feature to enable.
#[cfg(not(feature = "runtime"))]
#[derive(Debug, Default)]
pub struct OrtBackend {}

#[cfg(not(feature = "runtime"))]
impl OrtBackend {
    pub fn new(config: OrtConfig) -> Result<Self, BackendError> {
        let _ = config;
        // Fail at construction so catalog aliases never sit in a mock-shaped
        // registry and only blow up at request time.
        Err(Self::unavailable())
    }

    fn unavailable() -> BackendError {
        BackendError::Unavailable(
            "ONNX Runtime is not compiled into this binary; rebuild with \
             --features ort-runtime (CPU) or ort-cuda (CUDA EP)"
                .into(),
        )
    }
}

#[cfg(not(feature = "runtime"))]
#[async_trait]
impl Backend for OrtBackend {
    fn id(&self) -> &str {
        "onnxruntime"
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
    use super::pool::{cls_pool, l2_normalize, mean_pool};
    use super::*;
    use inferstream_protocol::inference::model_infer_request::InferInputTensor;
    use inferstream_protocol::tensor::pack_bytes;
    use std::collections::HashMap;

    #[test]
    fn mean_pool_respects_attention_mask() {
        // batch=1, seq=3, dim=2; third token is padding.
        let hidden = [1.0, 2.0, 3.0, 4.0, 100.0, 100.0];
        let mask = [1, 1, 0];
        let out = mean_pool(&hidden, &mask, 1, 3, 2);
        assert_eq!(out, vec![2.0, 3.0]);
    }

    #[test]
    fn mean_pool_batches_independently() {
        // batch=2, seq=2, dim=1.
        let hidden = [1.0, 3.0, 10.0, 30.0];
        let mask = [1, 1, 1, 0];
        let out = mean_pool(&hidden, &mask, 2, 2, 1);
        assert_eq!(out, vec![2.0, 10.0]);
    }

    #[test]
    fn cls_pool_takes_first_token() {
        let hidden = [1.0, 2.0, 9.0, 9.0, 5.0, 6.0, 9.0, 9.0];
        let out = cls_pool(&hidden, 2, 2, 2);
        assert_eq!(out, vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn l2_normalize_unit_norm() {
        let mut rows = vec![3.0, 4.0, 0.0, 0.0];
        l2_normalize(&mut rows, 2);
        assert!((rows[0] - 0.6).abs() < 1e-6);
        assert!((rows[1] - 0.8).abs() < 1e-6);
        // Zero row stays zero instead of dividing by zero.
        assert_eq!(&rows[2..], &[0.0, 0.0]);
    }

    #[test]
    fn device_and_pooling_parse() {
        assert_eq!(OrtDevice::from_config("CUDA").unwrap(), OrtDevice::Cuda);
        assert_eq!(OrtDevice::from_config("trt").unwrap(), OrtDevice::TensorRt);
        assert!(OrtDevice::from_config("npu").is_err());
        assert_eq!(Pooling::from_config("mean").unwrap(), Pooling::Mean);
        assert!(Pooling::from_config("max").is_err());
    }

    #[test]
    fn text_inputs_extracts_batch() {
        let request = ModelInferRequest {
            model_name: "embed".into(),
            inputs: vec![InferInputTensor {
                name: "text".into(),
                datatype: "BYTES".into(),
                shape: vec![2],
                parameters: HashMap::new(),
                contents: None,
            }],
            raw_input_contents: vec![pack_bytes(&[b"hello".as_slice(), b"world"])],
            ..Default::default()
        };
        assert_eq!(text_inputs(&request).unwrap(), vec!["hello", "world"]);
    }

    #[test]
    fn text_inputs_rejects_missing_tensor() {
        let request = ModelInferRequest::default();
        assert!(matches!(
            text_inputs(&request),
            Err(BackendError::InvalidRequest(_))
        ));
    }

    #[cfg(not(feature = "runtime"))]
    #[test]
    fn stub_fails_at_construction() {
        assert!(matches!(
            OrtBackend::new(OrtConfig::default()),
            Err(BackendError::Unavailable(_))
        ));
    }
}
