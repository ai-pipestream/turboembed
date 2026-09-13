//! OpenVINO GenAI embedding backend for inferstream.
//!
//! Default Intel embed path: in-process C++
//! [`ov::genai::TextEmbeddingPipeline`](https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.TextEmbeddingPipeline.html)
//! via a cxx bridge. Clients send **plain strings**; openvino-tokenizers
//! runs inside the pipeline (no OVMS gRPC, no Python).
//!
//! Pooling: CLS / MEAN / LAST_TOKEN. L2 normalize is the GenAI default.
//! Device: `CPU` / `GPU` / `NPU` / `AUTO` (GPU when available if unset).
//!
//! Feature `genai` links OpenVINO + OpenVINO GenAI. Without it this crate
//! still type-checks; [`OpenVinoBackend::new`] fails at startup naming the
//! feature so a default `intel.toml` cannot silently serve stubs.
//!
//! Environment: `source /opt/intel/oneapi/setvars.sh` (or standalone
//! OpenVINO `setupvars.sh`) in the build shell and the service unit. GPU/NPU
//! need Level Zero + compute-runtime. See `docs/intel-genai-embed.md`.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use inferstream_backend::{Backend, BackendError, ModelMetadata};
use inferstream_protocol::inference::{
    model_infer_response::InferOutputTensor, model_metadata_response::TensorMetadata,
    ModelInferRequest, ModelInferResponse,
};
use inferstream_protocol::tensor::{pack_fp32, DataType};

#[cfg(feature = "genai")]
mod ffi;

#[cfg(feature = "genai")]
mod engine;

/// OpenVINO device string the GenAI pipeline compiles for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OvDevice {
    /// Pick GPU if the runtime lists one, otherwise CPU.
    #[default]
    Auto,
    Cpu,
    Gpu,
    Npu,
}

impl OvDevice {
    /// Parse config `device` (`"CPU"` / `"GPU"` / `"NPU"` / `"AUTO"`, any case).
    pub fn from_config(s: &str) -> Result<Self, BackendError> {
        match s.to_ascii_uppercase().as_str() {
            "CPU" => Ok(Self::Cpu),
            "GPU" => Ok(Self::Gpu),
            "NPU" => Ok(Self::Npu),
            "AUTO" => Ok(Self::Auto),
            other => Err(BackendError::InvalidRequest(format!(
                "unknown openvino device {other:?}; expected \"CPU\", \"GPU\", \"NPU\" or \"AUTO\""
            ))),
        }
    }

    /// OpenVINO device name (`"CPU"` / `"GPU"` / `"NPU"` / `"AUTO"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "AUTO",
            Self::Cpu => "CPU",
            Self::Gpu => "GPU",
            Self::Npu => "NPU",
        }
    }
}

/// How token-level hidden states are reduced to one sentence embedding.
///
/// Passed through to `ov::genai::TextEmbeddingPipeline::Config::pooling_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Pooling {
    /// First token (`[CLS]`) — BGE family.
    Cls,
    /// Attention-weighted mean — MiniLM / MPNet / E5 / GTE / Nomic.
    #[default]
    Mean,
    /// Last non-padding token. Pair with left padding on decoder embedders.
    Last,
}

impl Pooling {
    /// Parse config `pooling` (`"cls"` / `"mean"` / `"last"` / `"last_token"`).
    pub fn from_config(s: &str) -> Result<Self, BackendError> {
        match s.to_ascii_lowercase().as_str() {
            "cls" => Ok(Self::Cls),
            "mean" => Ok(Self::Mean),
            "last" | "last_token" | "last-token" => Ok(Self::Last),
            other => Err(BackendError::InvalidRequest(format!(
                "unknown pooling {other:?}; expected \"cls\", \"mean\" or \"last\""
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cls => "cls",
            Self::Mean => "mean",
            Self::Last => "last",
        }
    }

    /// Discriminant for the cxx bridge (`0=CLS`, `1=MEAN`, `2=LAST_TOKEN`).
    pub fn as_genai_u8(self) -> u8 {
        match self {
            Self::Cls => 0,
            Self::Mean => 1,
            Self::Last => 2,
        }
    }
}

/// Configuration for one in-process GenAI embedding pipeline.
#[derive(Debug, Clone, Default)]
pub struct OpenVinoConfig {
    /// Directory in OpenVINO GenAI layout: `openvino_model.xml` +
    /// `openvino_tokenizer.xml` (and their `.bin` weights) plus, for the
    /// inferstream Tokenize RPC, `tokenizer.json`.
    pub models_path: String,
    /// `None` / `Auto` → GPU when the runtime lists one, else CPU.
    pub device: Option<OvDevice>,
    pub pooling: Pooling,
    /// L2-normalize embeddings (GenAI default true).
    pub normalize: Option<bool>,
    /// Optional `max_length` forwarded to the GenAI tokenizer.
    pub max_seq_len: Option<usize>,
}

/// Pooling / normalization math shared by tests and the mock embedder.
/// The live GenAI pipeline applies the same ops in C++; these helpers keep
/// CI honest without linking OpenVINO.
pub mod pool {
    /// Attention-mask-weighted mean over the token axis.
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

    /// First-token (`[CLS]`) pooling.
    pub fn cls_pool(hidden: &[f32], batch: usize, seq: usize, dim: usize) -> Vec<f32> {
        debug_assert_eq!(hidden.len(), batch * seq * dim);
        let mut out = Vec::with_capacity(batch * dim);
        for b in 0..batch {
            out.extend_from_slice(&hidden[b * seq * dim..b * seq * dim + dim]);
        }
        out
    }

    /// Last *unmasked* token pooling (`LAST_TOKEN`).
    pub fn last_pool(
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
            let mut last = 0usize;
            for s in 0..seq {
                if mask[b * seq + s] != 0 {
                    last = s;
                }
            }
            let src = &hidden[(b * seq + last) * dim..(b * seq + last + 1) * dim];
            out[b * dim..(b + 1) * dim].copy_from_slice(src);
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
pub fn text_inputs(request: &ModelInferRequest) -> Result<Vec<String>, BackendError> {
    use inferstream_protocol::tensor::unpack_bytes;
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

/// Read `hidden_size` / `d_model` from a GenAI model dir's `config.json`.
pub fn embedding_dim_from_config(models_path: &Path) -> Option<usize> {
    let text = std::fs::read_to_string(models_path.join("config.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    value
        .get("hidden_size")
        .or_else(|| value.get("d_model"))
        .or_else(|| value.get("sentence_embedding_dimension"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
}

/// Files TextEmbeddingPipeline needs in `models_path`.
pub const REQUIRED_IR_FILES: &[&str] = &[
    "openvino_model.xml",
    "openvino_model.bin",
    "openvino_tokenizer.xml",
    "openvino_tokenizer.bin",
];

/// Check a GenAI-layout directory; returns missing filenames.
pub fn missing_ir_files(models_path: &Path) -> Vec<&'static str> {
    REQUIRED_IR_FILES
        .iter()
        .copied()
        .filter(|name| !models_path.join(name).is_file())
        .collect()
}

/// Pick a concrete OpenVINO device string from config + runtime listing.
///
/// `AUTO` / unset → GPU when any listed device starts with `GPU`, else CPU.
/// Explicit `GPU` / `NPU` fail if the plugin is missing (no silent fallback).
pub fn resolve_device(
    requested: Option<OvDevice>,
    available: &[String],
) -> Result<String, BackendError> {
    let has_gpu = available.iter().any(|d| d.starts_with("GPU"));
    let has_npu = available.iter().any(|d| d.starts_with("NPU"));
    let has_cpu = available.iter().any(|d| d == "CPU" || d.starts_with("CPU"));
    match requested.unwrap_or(OvDevice::Auto) {
        OvDevice::Auto => {
            if has_gpu {
                Ok("GPU".into())
            } else if has_cpu || available.is_empty() {
                Ok("CPU".into())
            } else {
                Ok(available[0].clone())
            }
        }
        OvDevice::Gpu => {
            if has_gpu {
                Ok("GPU".into())
            } else {
                Err(BackendError::Unavailable(format!(
                    "OpenVINO GPU plugin unavailable: device = \"GPU\" but runtime \
                     listed {available:?}; source setupvars.sh / oneAPI and check Level Zero"
                )))
            }
        }
        OvDevice::Npu => {
            if has_npu {
                Ok("NPU".into())
            } else {
                Err(BackendError::Unavailable(format!(
                    "OpenVINO NPU plugin unavailable: device = \"NPU\" but runtime listed {available:?}"
                )))
            }
        }
        OvDevice::Cpu => Ok("CPU".into()),
    }
}

/// Something that turns a batch of strings into a flat `[n * dim]` FP32 blob.
pub(crate) trait Embedder: Send + Sync {
    fn embed(&self, texts: &[String]) -> Result<(usize, Vec<f32>), BackendError>;
    fn device(&self) -> &str;
    fn pooling(&self) -> Pooling;
    fn normalize(&self) -> bool;
    fn models_path(&self) -> &str;
    fn embedding_dim(&self) -> Option<usize>;
}

/// Deterministic embedder for unit/integration tests (no OpenVINO).
pub struct MockEmbedder {
    models_path: String,
    device: String,
    pooling: Pooling,
    normalize: bool,
    dim: usize,
}

impl MockEmbedder {
    pub fn new(config: OpenVinoConfig, dim: usize) -> Self {
        Self {
            models_path: config.models_path,
            device: config.device.unwrap_or(OvDevice::Cpu).as_str().to_string(),
            pooling: config.pooling,
            normalize: config.normalize.unwrap_or(true),
            dim,
        }
    }
}

impl Embedder for MockEmbedder {
    fn embed(&self, texts: &[String]) -> Result<(usize, Vec<f32>), BackendError> {
        if texts.is_empty() {
            return Err(BackendError::InvalidRequest(
                "input \"text\" contained no elements".into(),
            ));
        }
        let mut out = Vec::with_capacity(texts.len() * self.dim);
        for (i, text) in texts.iter().enumerate() {
            let mut row = vec![0.0f32; self.dim];
            // Stable, non-zero vector from the UTF-8 bytes + index so
            // identical texts match and distinct texts differ.
            for (j, byte) in text.bytes().enumerate() {
                row[j % self.dim] += (byte as f32) * 0.01 + (i as f32) * 0.001;
            }
            if row.iter().all(|v| *v == 0.0) {
                row[0] = 1.0;
            }
            if self.normalize {
                pool::l2_normalize(&mut row, self.dim);
            }
            out.extend_from_slice(&row);
        }
        Ok((self.dim, out))
    }

    fn device(&self) -> &str {
        &self.device
    }

    fn pooling(&self) -> Pooling {
        self.pooling
    }

    fn normalize(&self) -> bool {
        self.normalize
    }

    fn models_path(&self) -> &str {
        &self.models_path
    }

    fn embedding_dim(&self) -> Option<usize> {
        Some(self.dim)
    }
}

/// In-process OpenVINO GenAI embedding backend.
pub struct OpenVinoBackend {
    inner: Arc<dyn Embedder>,
}

impl std::fmt::Debug for OpenVinoBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenVinoBackend")
            .field("models_path", &self.inner.models_path())
            .field("device", &self.inner.device())
            .field("pooling", &self.inner.pooling())
            .finish()
    }
}

impl OpenVinoBackend {
    /// Load a GenAI `TextEmbeddingPipeline` from `config.models_path`.
    ///
    /// Without the `genai` feature this fails immediately so a default
    /// Intel config cannot start with a silent stub.
    pub fn new(config: OpenVinoConfig) -> Result<Self, BackendError> {
        #[cfg(feature = "genai")]
        {
            let embedder = engine::GenaiEmbedder::load(config)?;
            return Ok(Self {
                inner: Arc::new(embedder),
            });
        }
        #[cfg(not(feature = "genai"))]
        {
            let _ = config;
            Err(BackendError::Unavailable(
                "OpenVINO GenAI TextEmbeddingPipeline is not compiled into this binary; \
                 rebuild inferstream-intel with --features openvino-genai \
                 (needs OpenVINO + OpenVINO GenAI; see docs/intel-genai-embed.md)"
                    .into(),
            ))
        }
    }

    /// Test/integration constructor: deterministic vectors, no OpenVINO.
    pub fn mock(config: OpenVinoConfig, dim: usize) -> Self {
        Self {
            inner: Arc::new(MockEmbedder::new(config, dim)),
        }
    }
}

fn pack_response(
    request: ModelInferRequest,
    dim: usize,
    embeddings: Vec<f32>,
) -> Result<ModelInferResponse, BackendError> {
    let batch = embeddings.len().checked_div(dim).unwrap_or(0);
    if dim == 0 || embeddings.len() != batch * dim {
        return Err(BackendError::Internal(format!(
            "ragged embedding blob (len={}, dim={dim})",
            embeddings.len()
        )));
    }
    let shape = if batch == 1 {
        vec![dim as i64]
    } else {
        vec![batch as i64, dim as i64]
    };
    Ok(ModelInferResponse {
        model_name: request.model_name,
        model_version: request.model_version,
        id: request.id,
        parameters: Default::default(),
        outputs: vec![InferOutputTensor {
            name: "embedding".to_string(),
            datatype: DataType::Fp32.as_oip().to_string(),
            shape,
            parameters: Default::default(),
            contents: None,
        }],
        raw_output_contents: vec![pack_fp32(&embeddings).into()],
    })
}

#[async_trait]
impl Backend for OpenVinoBackend {
    fn id(&self) -> &str {
        "openvino"
    }

    async fn model_ready(&self, _model_name: &str, _model_version: &str) -> bool {
        true
    }

    async fn model_metadata(
        &self,
        model_name: &str,
        _model_version: &str,
    ) -> Result<ModelMetadata, BackendError> {
        let mut properties = std::collections::HashMap::from([
            (
                "model_path".to_string(),
                self.inner.models_path().to_string(),
            ),
            ("device".to_string(), self.inner.device().to_string()),
            (
                "pooling".to_string(),
                self.inner.pooling().as_str().to_string(),
            ),
            ("normalize".to_string(), self.inner.normalize().to_string()),
            ("engine".to_string(), "text_embedding_pipeline".to_string()),
        ]);
        if let Some(dim) = self.inner.embedding_dim() {
            properties.insert("embedding_dim".to_string(), dim.to_string());
        }
        Ok(ModelMetadata {
            name: model_name.to_string(),
            versions: vec!["1".to_string()],
            platform: "openvino_genai".to_string(),
            inputs: vec![TensorMetadata {
                name: "text".to_string(),
                datatype: DataType::Bytes.as_oip().to_string(),
                shape: vec![-1],
            }],
            outputs: vec![TensorMetadata {
                name: "embedding".to_string(),
                datatype: DataType::Fp32.as_oip().to_string(),
                shape: vec![-1],
            }],
            properties,
        })
    }

    async fn infer(&self, request: ModelInferRequest) -> Result<ModelInferResponse, BackendError> {
        let texts = text_inputs(&request)?;
        let inner = Arc::clone(&self.inner);
        let (dim, embeddings) = tokio::task::spawn_blocking(move || inner.embed(&texts))
            .await
            .map_err(|e| BackendError::Internal(format!("inference task panicked: {e}")))??;
        pack_response(request, dim, embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::pool::{cls_pool, l2_normalize, last_pool, mean_pool};
    use super::*;
    use inferstream_backend::Backend;
    use inferstream_protocol::inference::model_infer_request::InferInputTensor;
    use inferstream_protocol::tensor::{pack_bytes, unpack_fp32};
    use std::collections::HashMap;

    #[test]
    fn mean_pool_respects_attention_mask() {
        let hidden = [1.0, 2.0, 3.0, 4.0, 100.0, 100.0];
        let mask = [1, 1, 0];
        assert_eq!(mean_pool(&hidden, &mask, 1, 3, 2), vec![2.0, 3.0]);
    }

    #[test]
    fn cls_pool_takes_first_token() {
        let hidden = [1.0, 2.0, 9.0, 9.0, 5.0, 6.0, 9.0, 9.0];
        assert_eq!(cls_pool(&hidden, 2, 2, 2), vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn last_pool_skips_padding() {
        // batch=1, seq=3, dim=2; last real token is index 1.
        let hidden = [1.0, 1.0, 3.0, 4.0, 9.0, 9.0];
        let mask = [1, 1, 0];
        assert_eq!(last_pool(&hidden, &mask, 1, 3, 2), vec![3.0, 4.0]);
    }

    #[test]
    fn l2_normalize_unit_norm() {
        let mut rows = vec![3.0, 4.0, 0.0, 0.0];
        l2_normalize(&mut rows, 2);
        assert!((rows[0] - 0.6).abs() < 1e-6);
        assert!((rows[1] - 0.8).abs() < 1e-6);
        assert_eq!(&rows[2..], &[0.0, 0.0]);
    }

    #[test]
    fn resolve_prefers_gpu_on_auto() {
        let d = resolve_device(Some(OvDevice::Auto), &["CPU".into(), "GPU.0".into()]).unwrap();
        assert_eq!(d, "GPU");
        let err = resolve_device(Some(OvDevice::Gpu), &["CPU".into()]).unwrap_err();
        assert!(matches!(err, BackendError::Unavailable(_)));
        assert_eq!(
            resolve_device(Some(OvDevice::Cpu), &["GPU".into()]).unwrap(),
            "CPU"
        );
    }

    #[test]
    fn device_and_pooling_parse() {
        assert_eq!(OvDevice::from_config("gpu").unwrap(), OvDevice::Gpu);
        assert_eq!(OvDevice::from_config("NPU").unwrap(), OvDevice::Npu);
        assert_eq!(OvDevice::from_config("auto").unwrap(), OvDevice::Auto);
        assert!(OvDevice::from_config("cuda").is_err());
        assert_eq!(Pooling::from_config("LAST_TOKEN").unwrap(), Pooling::Last);
        assert_eq!(Pooling::from_config("cls").unwrap(), Pooling::Cls);
        assert!(Pooling::from_config("max").is_err());
        assert_eq!(Pooling::Last.as_genai_u8(), 2);
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

    #[test]
    fn missing_ir_files_lists_absentees() {
        let tmp = std::env::temp_dir().join("inferstream-ov-empty");
        let _ = std::fs::create_dir_all(&tmp);
        let missing = missing_ir_files(&tmp);
        assert_eq!(missing, REQUIRED_IR_FILES);
    }

    fn infer_request(texts: &[&str]) -> ModelInferRequest {
        let bytes: Vec<&[u8]> = texts.iter().map(|t| t.as_bytes()).collect();
        ModelInferRequest {
            model_name: "minilm".into(),
            id: "r1".into(),
            inputs: vec![InferInputTensor {
                name: "text".into(),
                datatype: "BYTES".into(),
                shape: vec![texts.len() as i64],
                parameters: HashMap::new(),
                contents: None,
            }],
            raw_input_contents: vec![pack_bytes(&bytes)],
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn mock_embed_packs_single_and_batch() {
        let backend = OpenVinoBackend::mock(
            OpenVinoConfig {
                models_path: "models/ov/minilm".into(),
                device: Some(OvDevice::Gpu),
                pooling: Pooling::Mean,
                normalize: Some(true),
                max_seq_len: Some(256),
            },
            4,
        );
        assert!(backend.model_ready("minilm", "1").await);
        let meta = backend.model_metadata("minilm", "1").await.unwrap();
        assert_eq!(meta.platform, "openvino_genai");
        assert_eq!(
            meta.properties.get("device").map(String::as_str),
            Some("GPU")
        );
        assert_eq!(
            meta.properties.get("engine").map(String::as_str),
            Some("text_embedding_pipeline")
        );

        let single = backend.infer(infer_request(&["hello"])).await.unwrap();
        assert_eq!(single.outputs[0].shape, vec![4]);
        let v = unpack_fp32(&single.raw_output_contents[0]).unwrap();
        assert_eq!(v.len(), 4);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "mock L2 {norm}");

        let batch = backend
            .infer(infer_request(&["hello", "world"]))
            .await
            .unwrap();
        assert_eq!(batch.outputs[0].shape, vec![2, 4]);
        assert_eq!(unpack_fp32(&batch.raw_output_contents[0]).unwrap().len(), 8);
    }

    #[cfg(not(feature = "genai"))]
    #[test]
    fn new_without_genai_is_unavailable() {
        let err = OpenVinoBackend::new(OpenVinoConfig {
            models_path: "models/ov/minilm".into(),
            device: Some(OvDevice::Gpu),
            ..Default::default()
        })
        .unwrap_err();
        assert!(matches!(err, BackendError::Unavailable(_)));
        assert!(err.to_string().contains("openvino-genai"));
    }
}
