//! Thin [`Backend`] over the shared TurboEmbed C ABI.
//!
//! Catalog embed aliases (`minilm`, …) go through
//! [`include/turboembed.h`](../../../include/turboembed.h) — the same
//! symbols `crates/turboembed` wraps. This crate does **not** tokenize,
//! generate, or reimplement ORT / GenAI / MLX. Those live behind the ABI.
//!
//! Device policy matches the ABI:
//! * `AUTO` is host-default GPU (CUDA / OpenVINO GPU / Metal).
//! * Missing accelerator → construction fails. Never mock, never silent CPU.
//! * `Device::Mock` is refused for every catalog alias.
//!
//! Tokenize / Detokenize stay on the server `tokenizer_dir` map (or the
//! LLM backends). This façade only implements unary embed `infer`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use inferstream_backend::{Backend, BackendError, ModelMetadata, PackedEmbed};
use inferstream_protocol::inference::{
    infer_parameter::ParameterChoice, model_infer_response::InferOutputTensor,
    model_metadata_response::TensorMetadata, InferParameter, ModelInferRequest, ModelInferResponse,
};
use inferstream_protocol::output_scratch;
use inferstream_protocol::tensor::{unpack_bytes, DataType};
use turboembed::{
    Device, EmbedOptions, Embeddings, Engine, Error as TeError, OutputFormat, Pooling,
};

/// Parse a catalog / `[[models]]` `backend` + `device` into an ABI device.
///
/// Missing / `"auto"` → [`Device::Auto`] (host GPU). `"mock"` is always an
/// error — catalog embeds never sit on the 8-d FNV smoke path.
pub fn device_from_config(backend: &str, device: Option<&str>) -> Result<Device, BackendError> {
    let backend = backend.to_ascii_lowercase();
    let raw = device.map(str::trim).filter(|s| !s.is_empty());
    if raw.is_some_and(|s| s.eq_ignore_ascii_case("mock")) {
        return Err(BackendError::Unavailable(
            "catalog embed aliases refuse device=mock; TurboEmbed mock is ABI smoke only".into(),
        ));
    }
    match backend.as_str() {
        "ort" | "onnxruntime" => match raw.map(|s| s.to_ascii_lowercase()).as_deref() {
            None | Some("auto") => Ok(Device::Auto),
            Some("cuda") => Ok(Device::Cuda),
            Some("cpu") => Ok(Device::Cpu),
            Some("tensorrt" | "trt") => Ok(Device::TensorRt),
            Some(other) => Err(BackendError::InvalidRequest(format!(
                "unknown ort device {other:?}; expected \"cuda\", \"cpu\", \"tensorrt\" or \"auto\""
            ))),
        },
        "openvino" => match raw.map(|s| s.to_ascii_uppercase()).as_deref() {
            None | Some("AUTO") => Ok(Device::Auto),
            Some("GPU") | Some("OPENVINO-GPU") | Some("OPENVINO_GPU") => Ok(Device::OpenVinoGpu),
            Some("CPU") | Some("OPENVINO-CPU") | Some("OPENVINO_CPU") => Ok(Device::OpenVinoCpu),
            Some("NPU") | Some("OPENVINO-NPU") | Some("OPENVINO_NPU") => Ok(Device::OpenVinoNpu),
            Some(other) => Err(BackendError::InvalidRequest(format!(
                "unknown openvino device {other:?}; expected \"GPU\", \"CPU\", \"NPU\" or \"AUTO\""
            ))),
        },
        "mlx" => match raw.map(|s| s.to_ascii_lowercase()).as_deref() {
            None | Some("auto") | Some("metal") | Some("gpu") => Ok(Device::Auto),
            Some("cpu") => Err(BackendError::Unavailable(
                "Apple catalog embeds refuse device=cpu; AUTO/METAL is host GPU, never a CPU swap"
                    .into(),
            )),
            Some(other) => Err(BackendError::InvalidRequest(format!(
                "unknown mlx device {other:?}; expected \"metal\", \"auto\" or omit"
            ))),
        },
        other => Err(BackendError::InvalidRequest(format!(
            "backend {other:?} is not a TurboEmbed catalog embed path \
             (expected ort / openvino / mlx)"
        ))),
    }
}

/// One catalog alias served through a dedicated TurboEmbed engine.
///
/// The C ABI holds one loaded session per engine on nvidia/intel, so each
/// alias gets its own [`Engine`]. Apple's dylib can hold several; we still
/// isolate so a failed load cannot clobber another alias.
pub struct TurboEmbedBackend {
    inner: Arc<Inner>,
}

struct Inner {
    alias: String,
    device: Device,
    dim: u32,
    engine: Mutex<Engine>,
}

impl TurboEmbedBackend {
    /// Create an engine, `load_model(alias)`, refuse mock / dim-8 leaks.
    pub fn open(alias: &str, device: Device) -> Result<Self, BackendError> {
        if alias.is_empty() {
            return Err(BackendError::InvalidRequest(
                "TurboEmbed catalog alias must not be empty".into(),
            ));
        }
        if matches!(device, Device::Mock) {
            return Err(BackendError::Unavailable(format!(
                "catalog alias {alias:?} refuses Device::Mock; TurboEmbed mock is ABI smoke only"
            )));
        }
        if is_mock_alias(alias) {
            return Err(BackendError::Unavailable(format!(
                "{alias:?} is the ABI-smoke mock; catalog Embed never serves it"
            )));
        }

        let engine = Engine::create(device).map_err(|e| map_open_error(alias, device, e))?;
        engine
            .load_model(alias)
            .map_err(|e| map_open_error(alias, device, e))?;

        let dim = dim_after_load(&engine, alias)?;
        if dim == 8 {
            return Err(BackendError::Internal(format!(
                "FAKE: catalog alias {alias:?} loaded as dim=8 (FNV mock). \
                 MiniLM is 384-d. Mock is never a substitute for a missing provider"
            )));
        }

        Ok(Self {
            inner: Arc::new(Inner {
                alias: alias.to_string(),
                device,
                dim,
                engine: Mutex::new(engine),
            }),
        })
    }

    /// Factory helper: map config `backend`/`device` then [`Self::open`].
    pub fn open_for_model(
        name: &str,
        backend: &str,
        device: Option<&str>,
    ) -> Result<Self, BackendError> {
        let device = device_from_config(backend, device)?;
        Self::open(name, device)
    }

    pub fn alias(&self) -> &str {
        &self.inner.alias
    }

    pub fn device(&self) -> Device {
        self.inner.device
    }

    pub fn embedding_dim(&self) -> u32 {
        self.inner.dim
    }
}

fn is_mock_alias(alias: &str) -> bool {
    alias.eq_ignore_ascii_case("mock") || alias.eq_ignore_ascii_case("mock-embed")
}

fn dim_after_load(engine: &Engine, alias: &str) -> Result<u32, BackendError> {
    let list = engine.list_models().map_err(map_te)?;
    for info in list.iter() {
        if info.alias == alias && info.dim > 0 {
            return Ok(info.dim);
        }
    }
    // Provider loaded but list omitted the row (should not happen). Warmup.
    let emb = engine
        .embed_one(alias, "hello world", &EmbedOptions::default())
        .map_err(map_te)?;
    let dim = emb.dim() as u32;
    if dim == 0 {
        return Err(BackendError::Internal(format!(
            "TurboEmbed loaded {alias:?} but reported dim=0"
        )));
    }
    Ok(dim)
}

fn map_open_error(alias: &str, device: Device, err: TeError) -> BackendError {
    let hint = match device {
        Device::Cuda | Device::Auto | Device::TensorRt => {
            "rebuild inferstream-nvidia with --features ort-cuda \
             (TurboEmbed ORT CUDA IoBinding; see docs/turboembed.md)"
        }
        Device::OpenVinoGpu | Device::OpenVinoNpu => {
            "rebuild inferstream-intel with --features openvino-genai \
             (TurboEmbed GenAI; see docs/intel-genai-embed.md)"
        }
        Device::OpenVinoCpu | Device::Cpu => {
            "explicit CPU still needs a real provider feature \
             (ort-cuda or openvino-genai); catalog AUTO never selects CPU"
        }
        Device::Metal => {
            "Apple catalog embeds need libTurboEmbed.dylib on Metal \
             (make apple; docs/turboembed-swift.md)"
        }
        Device::Hailo => {
            "Raspberry Pi AI HAT+ embeds need --features hailo on a host \
             with HailoRT (docs/hailo-embed.md)"
        }
        Device::Mock => "catalog aliases refuse mock",
    };
    BackendError::Unavailable(format!(
        "TurboEmbed C ABI failed to load catalog alias {alias:?} on device {}: {err}. \
         {hint}. Missing accelerator never falls back to mock or CPU",
        device.as_str()
    ))
}

fn map_te(err: TeError) -> BackendError {
    match err {
        TeError::InvalidArgument(m) => BackendError::InvalidRequest(m),
        TeError::NotFound(m) => BackendError::ModelNotFound(m),
        TeError::NotImplemented(m) | TeError::Unavailable(m) | TeError::UnsupportedDevice(m) => {
            BackendError::Unavailable(m)
        }
        TeError::Internal(m) | TeError::OutOfMemory(m) | TeError::Other { message: m, .. } => {
            BackendError::Internal(m)
        }
    }
}

fn text_inputs(request: &ModelInferRequest) -> Result<Vec<String>, BackendError> {
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

fn options_from_request(request: &ModelInferRequest) -> EmbedOptions {
    let mut opts = EmbedOptions {
        pooling: Pooling::Default,
        normalize: None,
        truncate_to: None,
        output_format: OutputFormat::Typed,
    };
    if let Some(s) = string_param(request.parameters.get("pooling")) {
        opts.pooling = match s.to_ascii_lowercase().as_str() {
            "mean" => Pooling::Mean,
            "cls" => Pooling::Cls,
            "last" => Pooling::Last,
            _ => Pooling::Default,
        };
    }
    if let Some(v) = bool_param(request.parameters.get("normalize")) {
        opts.normalize = Some(v);
    }
    if let Some(n) = int_param(request.parameters.get("truncate")) {
        if n > 0 {
            opts.truncate_to = Some(n as u32);
        }
    }
    opts
}

fn string_param(p: Option<&InferParameter>) -> Option<&str> {
    match p.and_then(|p| p.parameter_choice.as_ref()) {
        Some(ParameterChoice::StringParam(s)) if !s.is_empty() => Some(s.as_str()),
        _ => None,
    }
}

fn bool_param(p: Option<&InferParameter>) -> Option<bool> {
    match p.and_then(|p| p.parameter_choice.as_ref()) {
        Some(ParameterChoice::BoolParam(v)) => Some(*v),
        _ => None,
    }
}

fn int_param(p: Option<&InferParameter>) -> Option<i64> {
    match p.and_then(|p| p.parameter_choice.as_ref()) {
        Some(ParameterChoice::Int64Param(v)) => Some(*v),
        _ => None,
    }
}

fn pack_response(
    request: ModelInferRequest,
    embeddings: Embeddings,
) -> Result<ModelInferResponse, BackendError> {
    let dim = embeddings.dim();
    let count = embeddings.count();
    if dim == 0 || count == 0 {
        return Err(BackendError::Internal(
            "TurboEmbed returned an empty embedding batch".into(),
        ));
    }
    if dim == 8 {
        return Err(BackendError::Internal(
            "FAKE: TurboEmbed returned dim=8 (FNV mock) for a catalog embed".into(),
        ));
    }
    let values = embeddings.values();
    if values.len() != count * dim {
        return Err(BackendError::Internal(format!(
            "TurboEmbed ragged blob (len={}, count={count}, dim={dim})",
            values.len()
        )));
    }
    let shape = if count == 1 {
        vec![dim as i64]
    } else {
        vec![count as i64, dim as i64]
    };
    Ok(ModelInferResponse {
        model_name: request.model_name,
        model_version: request.model_version,
        id: request.id,
        parameters: HashMap::new(),
        outputs: vec![InferOutputTensor {
            name: "embedding".to_string(),
            datatype: DataType::Fp32.as_oip().to_string(),
            shape,
            parameters: HashMap::new(),
            contents: None,
        }],
        raw_output_contents: vec![output_scratch::pack_le_f32(values)],
    })
}

#[async_trait]
impl Backend for TurboEmbedBackend {
    fn id(&self) -> &str {
        "turboembed"
    }

    async fn model_ready(&self, _model_name: &str, _model_version: &str) -> bool {
        true
    }

    async fn model_metadata(
        &self,
        model_name: &str,
        _model_version: &str,
    ) -> Result<ModelMetadata, BackendError> {
        let dim = self.inner.dim as i64;
        Ok(ModelMetadata {
            name: model_name.to_string(),
            versions: vec!["1".to_string()],
            platform: "turboembed".to_string(),
            inputs: vec![TensorMetadata {
                name: "text".to_string(),
                datatype: DataType::Bytes.as_oip().to_string(),
                shape: vec![-1],
            }],
            outputs: vec![TensorMetadata {
                name: "embedding".to_string(),
                datatype: DataType::Fp32.as_oip().to_string(),
                shape: vec![dim],
            }],
            properties: HashMap::from([
                ("alias".to_string(), self.inner.alias.clone()),
                ("device".to_string(), self.inner.device.as_str().to_string()),
                ("abi".to_string(), turboembed::abi_version().to_string()),
                ("engine".to_string(), "turboembed".to_string()),
            ]),
        })
    }

    async fn infer(&self, request: ModelInferRequest) -> Result<ModelInferResponse, BackendError> {
        let texts = text_inputs(&request)?;
        let opts = options_from_request(&request);
        let inner = Arc::clone(&self.inner);
        let alias = inner.alias.clone();
        let embeddings = tokio::task::spawn_blocking(move || {
            let views: Vec<&str> = texts.iter().map(String::as_str).collect();
            let engine = inner
                .engine
                .lock()
                .map_err(|_| BackendError::Internal("TurboEmbed engine mutex poisoned".into()))?;
            engine.embed(&alias, &views, &opts).map_err(map_te)
        })
        .await
        .map_err(|e| BackendError::Internal(format!("TurboEmbed embed task panicked: {e}")))??;
        pack_response(request, embeddings)
    }

    async fn embed_packed_into(
        &self,
        model_name: &str,
        texts: &[String],
        pooling: &str,
        normalize: Option<bool>,
        truncate_to: u32,
        dest: &mut Vec<u8>,
    ) -> Result<PackedEmbed, BackendError> {
        if texts.is_empty() {
            return Err(BackendError::InvalidRequest(
                "texts must not be empty".into(),
            ));
        }
        let mut opts = EmbedOptions {
            pooling: Pooling::Default,
            normalize,
            truncate_to: (truncate_to > 0).then_some(truncate_to),
            output_format: OutputFormat::PackedBytes,
        };
        if !pooling.is_empty() {
            opts.pooling = match pooling.to_ascii_lowercase().as_str() {
                "mean" => Pooling::Mean,
                "cls" => Pooling::Cls,
                "last" => Pooling::Last,
                _ => Pooling::Default,
            };
        }
        let inner = Arc::clone(&self.inner);
        let alias = inner.alias.clone();
        let texts = texts.to_vec();
        let embeddings = tokio::task::spawn_blocking(move || {
            let views: Vec<&str> = texts.iter().map(String::as_str).collect();
            let engine = inner
                .engine
                .lock()
                .map_err(|_| BackendError::Internal("TurboEmbed engine mutex poisoned".into()))?;
            engine.embed(&alias, &views, &opts).map_err(map_te)
        })
        .await
        .map_err(|e| BackendError::Internal(format!("TurboEmbed embed task panicked: {e}")))??;
        let dim = embeddings.dim();
        let count = embeddings.count();
        if dim == 0 || count == 0 {
            return Err(BackendError::Internal(
                "TurboEmbed returned an empty embedding batch".into(),
            ));
        }
        if dim == 8 {
            return Err(BackendError::Internal(
                "FAKE: TurboEmbed returned dim=8 (FNV mock) for a catalog embed".into(),
            ));
        }
        let values = embeddings.values();
        if values.len() != count * dim {
            return Err(BackendError::Internal(format!(
                "TurboEmbed ragged blob (len={}, count={count}, dim={dim})",
                values.len()
            )));
        }
        output_scratch::pack_le_f32_into(values, dest);
        Ok(PackedEmbed {
            dim: dim as u32,
            count: count as u32,
            model_name: model_name.to_string(),
            model_version: String::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_map_host_gpu_defaults() {
        assert_eq!(device_from_config("ort", None).unwrap(), Device::Auto);
        assert_eq!(
            device_from_config("ort", Some("cuda")).unwrap(),
            Device::Cuda
        );
        assert_eq!(device_from_config("ort", Some("CPU")).unwrap(), Device::Cpu);
        assert_eq!(
            device_from_config("openvino", Some("GPU")).unwrap(),
            Device::OpenVinoGpu
        );
        assert_eq!(device_from_config("openvino", None).unwrap(), Device::Auto);
        assert_eq!(device_from_config("mlx", None).unwrap(), Device::Auto);
        assert_eq!(
            device_from_config("mlx", Some("metal")).unwrap(),
            Device::Auto
        );
    }

    #[test]
    fn device_map_rejects_mock_and_unknown() {
        assert!(matches!(
            device_from_config("ort", Some("mock")),
            Err(BackendError::Unavailable(_))
        ));
        assert!(matches!(
            device_from_config("ort", Some("npu")),
            Err(BackendError::InvalidRequest(_))
        ));
        assert!(matches!(
            device_from_config("mlx", Some("cpu")),
            Err(BackendError::Unavailable(_))
        ));
        assert!(matches!(
            device_from_config("llama-cpp", Some("cuda")),
            Err(BackendError::InvalidRequest(_))
        ));
    }

    fn open_err(alias: &str, device: Device) -> BackendError {
        match TurboEmbedBackend::open(alias, device) {
            Ok(_) => panic!("expected {alias} on {device:?} to fail"),
            Err(e) => e,
        }
    }

    #[test]
    fn catalog_alias_refuses_mock_device() {
        let err = open_err("minilm", Device::Mock);
        let msg = err.to_string();
        assert!(
            msg.contains("Mock") || msg.contains("mock"),
            "expected mock refusal, got {msg}"
        );
    }

    #[test]
    fn mock_alias_is_refused_on_auto() {
        let err = open_err("mock-embed", Device::Auto);
        assert!(
            err.to_string().contains("mock"),
            "expected mock-alias refusal, got {err}"
        );
    }

    #[cfg(not(any(feature = "ort-cuda", feature = "genai")))]
    #[test]
    fn catalog_minilm_fails_loud_without_provider() {
        for device in [
            Device::Auto,
            Device::Cuda,
            Device::OpenVinoGpu,
            Device::Metal,
        ] {
            let err = match TurboEmbedBackend::open("minilm", device) {
                Ok(backend) => panic!(
                    "{device:?} opened minilm dim={} without a provider feature",
                    backend.embedding_dim()
                ),
                Err(e) => e,
            };
            let msg = err.to_string();
            assert!(
                matches!(err, BackendError::Unavailable(_)),
                "{device:?}: expected Unavailable, got {err:?}"
            );
            assert!(
                !msg.to_ascii_lowercase().contains("serving mock"),
                "{device:?} must not mention serving mock: {msg}"
            );
        }
    }
}
