//! Thin [`Backend`] over the shared TurboRerank C ABI.
//!
//! Catalog cross-encoder aliases (`ms-marco-minilm-l6`) go through
//! [`include/turborerank.h`](../../../include/turborerank.h) — the same
//! symbols `crates/turborerank` wraps. This crate does **not** invent
//! relevance scores. Word-overlap stays on the explicit mock backend
//! for wire-path tests only.
//!
//! Product scores are **sigmoid(CLS logit)** (TEI `raw_scores=false`).
//! The library returns floats in **input order**; `extension.rs` still
//! owns sort + `top_n`.
//!
//! Device policy matches the ABI:
//! * `AUTO` is host-default GPU (CUDA / OpenVINO GPU / Metal).
//! * Missing accelerator or missing weights → construction fails.
//! * [`Device::Mock`] is refused for every catalog CE alias.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use inferstream_backend::{is_catalog_cross_encoder_alias, Backend, BackendError, ModelMetadata};
use inferstream_protocol::inference::{
    model_metadata_response::TensorMetadata, ModelInferRequest, ModelInferResponse,
};
use inferstream_protocol::tensor::DataType;
use turborerank::{
    Activation, Device, Engine, Error as TrError, Truncation,
};

/// TEI `/rerank` default `--max-client-batch-size`.
pub const DEFAULT_MAX_DOCUMENTS: u32 = 32;

/// Parse a catalog / `[[models]]` `device` into an ABI device.
///
/// Missing / `"auto"` → [`Device::Auto`] (host GPU). `"mock"` is always an
/// error — catalog CE aliases never sit on the word-overlap smoke path.
pub fn device_from_config(device: Option<&str>) -> Result<Device, BackendError> {
    let raw = device.map(str::trim).filter(|s| !s.is_empty());
    if raw.is_some_and(|s| s.eq_ignore_ascii_case("mock")) {
        return Err(BackendError::Unavailable(
            "catalog cross-encoder aliases refuse device=mock; \
             TurboRerank mock is ABI smoke only and never scores MiniLM"
                .into(),
        ));
    }
    match raw.map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("auto") => Ok(Device::Auto),
        Some("cpu") => Ok(Device::Cpu),
        Some("cuda") => Ok(Device::Cuda),
        Some("tensorrt" | "trt") => Ok(Device::TensorRt),
        Some("gpu" | "openvino-gpu" | "openvino_gpu") => Ok(Device::OpenVinoGpu),
        Some("openvino-cpu" | "openvino_cpu") => Ok(Device::OpenVinoCpu),
        Some("npu" | "openvino-npu" | "openvino_npu") => Ok(Device::OpenVinoNpu),
        Some("metal") => Ok(Device::Metal),
        Some(other) => Err(BackendError::InvalidRequest(format!(
            "unknown turborerank device {other:?}; expected \
             \"cuda\", \"cpu\", \"tensorrt\", \"GPU\", \"metal\" or \"auto\""
        ))),
    }
}

/// One catalog CE alias served through a dedicated TurboRerank engine.
pub struct TurboRerankBackend {
    inner: Arc<Inner>,
}

struct Inner {
    alias: String,
    device: Device,
    max_documents: u32,
    engine: Mutex<Engine>,
}

impl TurboRerankBackend {
    /// Create an engine, `load_model(alias)`, refuse mock.
    pub fn open(
        alias: &str,
        device: Device,
        config_path: Option<&Path>,
        max_documents: Option<u32>,
    ) -> Result<Self, BackendError> {
        if alias.is_empty() {
            return Err(BackendError::InvalidRequest(
                "TurboRerank catalog alias must not be empty".into(),
            ));
        }
        if matches!(device, Device::Mock) {
            return Err(BackendError::Unavailable(format!(
                "catalog alias {alias:?} refuses Device::Mock; \
                 TurboRerank mock is ABI smoke only and never scores MiniLM"
            )));
        }
        if !is_catalog_cross_encoder_alias(alias) && alias.eq_ignore_ascii_case("mock") {
            return Err(BackendError::Unavailable(
                "TurboRerank catalog façade refuses the ABI-smoke mock alias".into(),
            ));
        }

        let engine = match config_path {
            Some(path) => Engine::create_with_config(device, Some(path)),
            None => Engine::create(device),
        }
        .map_err(|e| map_open_error(alias, device, e))?;
        engine
            .load_model(alias)
            .map_err(|e| map_open_error(alias, device, e))?;

        Ok(Self {
            inner: Arc::new(Inner {
                alias: alias.to_string(),
                device,
                max_documents: max_documents.unwrap_or(DEFAULT_MAX_DOCUMENTS),
                engine: Mutex::new(engine),
            }),
        })
    }

    /// Factory helper: map config `device` / `path` then [`Self::open`].
    pub fn open_for_model(
        name: &str,
        device: Option<&str>,
        path: Option<&str>,
        max_batch_size: Option<u32>,
    ) -> Result<Self, BackendError> {
        let device = device_from_config(device)?;
        let path = path.map(PathBuf::from);
        Self::open(name, device, path.as_deref(), max_batch_size)
    }

    pub fn alias(&self) -> &str {
        &self.inner.alias
    }

    pub fn device(&self) -> Device {
        self.inner.device
    }

    pub fn max_documents(&self) -> u32 {
        self.inner.max_documents
    }
}

fn map_open_error(alias: &str, device: Device, err: TrError) -> BackendError {
    let hint = match device {
        Device::Cuda | Device::Auto | Device::TensorRt => {
            "rebuild inferstream-nvidia with --features turborerank \
             (and fetch MiniLM CE weights: make fetch-rerankers). \
             CUDA/AUTO/TensorRT never fall back to CPU or word-overlap"
        }
        Device::OpenVinoGpu | Device::OpenVinoNpu => {
            "rebuild inferstream-intel with --features turborerank \
             (OpenVINO IR: make convert-rerank-ov). GPU/NPU never \
             fall back to CPU or word-overlap"
        }
        Device::OpenVinoCpu | Device::Cpu => {
            "explicit CPU still needs the turborerank feature and \
             weights (make fetch-rerankers); catalog AUTO never selects CPU"
        }
        Device::Metal => {
            "Apple catalog rerank needs the C++/Metal ABI on Machine C \
             (make apple; docs/turborerank-swift.md)"
        }
        Device::Mock => "catalog aliases refuse mock",
    };
    BackendError::Unavailable(format!(
        "TurboRerank C ABI failed to load catalog alias {alias:?} on device {}: {err}. \
         {hint}. Missing weights or accelerator never fall back to the word-overlap mock",
        device.as_str()
    ))
}

fn map_tr(err: TrError) -> BackendError {
    match err {
        TrError::InvalidArgument(m) => BackendError::InvalidRequest(m),
        TrError::NotFound(m) => BackendError::ModelNotFound(m),
        TrError::NotImplemented(m)
        | TrError::Unavailable(m)
        | TrError::UnsupportedDevice(m) => BackendError::Unavailable(m),
        TrError::Internal(m) | TrError::OutOfMemory(m) => BackendError::Internal(m),
    }
}

fn reject_mock_shaped(scores: &[f32]) -> Result<(), BackendError> {
    if scores.is_empty() {
        return Err(BackendError::Internal(
            "TurboRerank returned no scores".into(),
        ));
    }
    let all_equal = scores.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-8);
    let only_unit = scores
        .iter()
        .all(|s| (*s - 0.0).abs() < 1e-8 || (*s - 1.0).abs() < 1e-8 || (*s - 0.5).abs() < 1e-8);
    if all_equal && only_unit {
        return Err(BackendError::Internal(format!(
            "FAKE: catalog CE scores look like the word-overlap mock {scores:?}"
        )));
    }
    Ok(())
}

#[async_trait]
impl Backend for TurboRerankBackend {
    fn id(&self) -> &str {
        "turborerank"
    }

    async fn model_ready(&self, _model_name: &str, _model_version: &str) -> bool {
        true
    }

    async fn model_metadata(
        &self,
        model_name: &str,
        _model_version: &str,
    ) -> Result<ModelMetadata, BackendError> {
        Ok(ModelMetadata {
            name: model_name.to_string(),
            versions: vec!["1".to_string()],
            platform: "turborerank".to_string(),
            inputs: vec![TensorMetadata {
                name: "query".to_string(),
                datatype: DataType::Bytes.as_oip().to_string(),
                shape: vec![1],
            }],
            outputs: vec![TensorMetadata {
                name: "score".to_string(),
                datatype: DataType::Fp32.as_oip().to_string(),
                shape: vec![1],
            }],
            properties: HashMap::from([
                ("alias".to_string(), self.inner.alias.clone()),
                ("device".to_string(), self.inner.device.as_str().to_string()),
                ("abi".to_string(), turborerank::abi_version().to_string()),
                ("engine".to_string(), "turborerank".to_string()),
                ("activation".to_string(), "sigmoid".to_string()),
            ]),
        })
    }

    async fn infer(&self, _request: ModelInferRequest) -> Result<ModelInferResponse, BackendError> {
        Err(BackendError::Unavailable(format!(
            "catalog alias {:?} is a TurboRerank cross-encoder; use the \
             inferstream.v1 Rerank RPC (not ModelInfer / Embed)",
            self.inner.alias
        )))
    }

    async fn rerank(
        &self,
        model_name: &str,
        query: &str,
        documents: &[String],
    ) -> Result<Vec<f32>, BackendError> {
        if documents.is_empty() {
            return Err(BackendError::InvalidRequest(
                "documents must not be empty".into(),
            ));
        }
        let max = self.inner.max_documents as usize;
        if documents.len() > max {
            return Err(BackendError::InvalidRequest(format!(
                "rerank batch of {} exceeds max_client_batch_size {max} \
                 (TEI default); chunk upstream",
                documents.len()
            )));
        }
        let inner = Arc::clone(&self.inner);
        let alias = if model_name.is_empty() {
            inner.alias.clone()
        } else {
            model_name.to_string()
        };
        let query = query.to_string();
        let documents = documents.to_vec();
        let n_docs = documents.len();
        let scores = tokio::task::spawn_blocking(move || {
            let views: Vec<&str> = documents.iter().map(String::as_str).collect();
            let engine = inner.engine.lock().map_err(|_| {
                BackendError::Internal("TurboRerank engine mutex poisoned".into())
            })?;
            engine
                .score(
                    Some(&alias),
                    &query,
                    &views,
                    Truncation::LongestFirst,
                    Activation::Sigmoid,
                    0,
                )
                .map_err(map_tr)
        })
        .await
        .map_err(|e| BackendError::Internal(format!("TurboRerank score task panicked: {e}")))??;
        if scores.len() != n_docs {
            return Err(BackendError::Internal(format!(
                "TurboRerank returned {} scores for {n_docs} documents",
                scores.len()
            )));
        }
        reject_mock_shaped(&scores)?;
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_map_host_gpu_defaults() {
        assert_eq!(device_from_config(None).unwrap(), Device::Auto);
        assert_eq!(device_from_config(Some("cuda")).unwrap(), Device::Cuda);
        assert_eq!(device_from_config(Some("CPU")).unwrap(), Device::Cpu);
        assert_eq!(device_from_config(Some("GPU")).unwrap(), Device::OpenVinoGpu);
        assert_eq!(
            device_from_config(Some("openvino-cpu")).unwrap(),
            Device::OpenVinoCpu
        );
        assert_eq!(device_from_config(Some("metal")).unwrap(), Device::Metal);
        assert_eq!(device_from_config(Some("auto")).unwrap(), Device::Auto);
    }

    #[test]
    fn device_map_rejects_mock_and_unknown() {
        assert!(matches!(
            device_from_config(Some("mock")),
            Err(BackendError::Unavailable(_))
        ));
        assert!(matches!(
            device_from_config(Some("sycl")),
            Err(BackendError::InvalidRequest(_))
        ));
    }

    fn open_err(alias: &str, device: Device) -> BackendError {
        match TurboRerankBackend::open(alias, device, None, None) {
            Ok(_) => panic!("expected {alias} on {device:?} to fail"),
            Err(e) => e,
        }
    }

    #[test]
    fn catalog_alias_refuses_mock_device() {
        let err = open_err("ms-marco-minilm-l6", Device::Mock);
        let msg = err.to_string();
        assert!(
            msg.contains("Mock") || msg.contains("mock"),
            "expected mock refusal, got {msg}"
        );
        assert!(
            !msg.to_ascii_lowercase().contains("serving word-overlap")
                && !msg.to_ascii_lowercase().contains("word overlap as minilm"),
            "must not mention serving mock MiniLM: {msg}"
        );
    }

    #[test]
    fn missing_weights_or_device_fail_loud() {
        // Without a fetched checkpoint, CPU create+load must fail — never a
        // constructed backend that would later serve word-overlap.
        if turborerank::weights_present() {
            return;
        }
        let err = open_err("ms-marco-minilm-l6", Device::Cpu);
        assert!(
            matches!(err, BackendError::Unavailable(_)),
            "expected Unavailable, got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("TurboRerank") || msg.contains("UNAVAILABLE") || msg.contains("unavail"),
            "startup error must name TurboRerank / unavailable, got {msg}"
        );
        assert!(
            msg.contains("weights") || msg.contains("UNAVAILABLE") || msg.contains("missing"),
            "must name missing weights, got {msg}"
        );
        assert!(
            !msg.to_ascii_lowercase().contains("serving word-overlap"),
            "must not serve word-overlap: {msg}"
        );
    }
}
