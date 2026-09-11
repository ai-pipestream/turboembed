//! Real ONNX Runtime engine (crate feature `runtime`).
//!
//! One `ort` session per [`OrtBackend`], loaded eagerly at construction so a
//! bad model path or a missing execution provider fails at startup, not at
//! request time. Tokenization runs server-side with the HuggingFace
//! `tokenizers` crate; pooling and normalization are the pure-Rust helpers in
//! [`crate::pool`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use inferstream_backend::{Backend, BackendError, ModelMetadata, TokenizeOptions};
use inferstream_protocol::extension::{Encoding, Offset};
use inferstream_protocol::inference::{
    model_infer_response::InferOutputTensor, model_metadata_response::TensorMetadata,
    ModelInferRequest, ModelInferResponse,
};
use inferstream_protocol::tensor::{pack_fp32, DataType};
use ort::session::{builder::GraphOptimizationLevel, Session, SessionInputValue};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::pool::{cls_pool, l2_normalize, mean_pool};
use crate::{text_inputs, OrtConfig, OrtDevice, Pooling};

const DEFAULT_MAX_SEQ_LEN: usize = 512;

/// ONNX Runtime embedding backend (real engine).
pub struct OrtBackend {
    inner: Arc<Inner>,
}

struct Inner {
    /// `Session::run` takes `&mut self`, so the session is serialized behind
    /// a mutex for now; ONNX Runtime batches within a run, and intra-op
    /// threading still parallelizes a single request. Session pooling is a
    /// later optimization, not a correctness issue.
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    model_path: String,
    device: OrtDevice,
    pooling: Pooling,
    normalize: bool,
    /// Model graph input names we will feed (subset of
    /// `input_ids` / `attention_mask` / `token_type_ids`).
    input_names: Vec<String>,
    /// Name of the hidden-state output tensor (first graph output).
    output_name: String,
}

impl std::fmt::Debug for OrtBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrtBackend")
            .field("model_path", &self.inner.model_path)
            .field("device", &self.inner.device)
            .field("pooling", &self.inner.pooling)
            .finish()
    }
}

fn load_error(what: &str, detail: impl std::fmt::Display) -> BackendError {
    BackendError::Unavailable(format!("{what}: {detail}"))
}

/// Locate `tokenizer.json`: explicit config path, next to the model, or one
/// directory up (HF snapshots keep the model in `onnx/` with the tokenizer at
/// the snapshot root).
fn find_tokenizer(config: &OrtConfig) -> Result<PathBuf, BackendError> {
    if let Some(path) = &config.tokenizer_path {
        let path = PathBuf::from(path);
        let file = if path.is_dir() {
            path.join("tokenizer.json")
        } else {
            path
        };
        if file.is_file() {
            return Ok(file);
        }
        return Err(load_error(
            "tokenizer not found",
            format!("{} does not exist", file.display()),
        ));
    }
    let model = Path::new(&config.model_path);
    let mut candidates = Vec::new();
    if let Some(dir) = model.parent() {
        candidates.push(dir.join("tokenizer.json"));
        if let Some(up) = dir.parent() {
            candidates.push(up.join("tokenizer.json"));
        }
    }
    candidates
        .iter()
        .find(|c| c.is_file())
        .cloned()
        .ok_or_else(|| {
            load_error(
                "tokenizer not found",
                format!(
                    "no tokenizer.json next to {} (searched {:?}); set tokenizer_dir in the model config",
                    model.display(),
                    candidates
                ),
            )
        })
}

impl OrtBackend {
    /// Load the tokenizer and ONNX session eagerly. Fails with
    /// [`BackendError::Unavailable`] when the model, tokenizer, or requested
    /// execution provider cannot be brought up.
    pub fn new(config: OrtConfig) -> Result<Self, BackendError> {
        if config.model_path.is_empty() {
            return Err(BackendError::InvalidRequest(
                "ort backend requires `path` pointing at a .onnx file".into(),
            ));
        }
        if !Path::new(&config.model_path).is_file() {
            return Err(load_error(
                "onnx model not found",
                format!("{} is not a file", config.model_path),
            ));
        }

        let tokenizer_file = find_tokenizer(&config)?;
        let mut tokenizer = Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| load_error("failed to load tokenizer", e))?;
        let max_len = config.max_seq_len.unwrap_or(DEFAULT_MAX_SEQ_LEN);
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: max_len,
                ..Default::default()
            }))
            .map_err(|e| load_error("failed to configure truncation", e))?;
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        let builder = Session::builder()
            .map_err(|e| load_error("failed to create ort session builder", e))?;
        #[allow(unused_mut)]
        let mut builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| load_error("failed to configure ort session", e))?;

        match config.device {
            OrtDevice::Cpu => {}
            OrtDevice::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    builder = builder
                        .with_execution_providers([ort::ep::CUDA::default()
                            .build()
                            .error_on_failure()])
                        .map_err(|e| load_error("CUDA execution provider unavailable", e))?;
                }
                #[cfg(not(feature = "cuda"))]
                return Err(BackendError::Unavailable(
                    "device = \"cuda\" but this binary was built without the CUDA EP; \
                     rebuild with --features ort-cuda"
                        .into(),
                ));
            }
            OrtDevice::TensorRt => {
                #[cfg(feature = "tensorrt")]
                {
                    builder = builder
                        .with_execution_providers([ort::ep::TensorRT::default()
                            .build()
                            .error_on_failure()])
                        .map_err(|e| {
                            load_error(
                                "TensorRT execution provider unavailable \
                                 (are TensorRT 10 libs installed?)",
                                e,
                            )
                        })?;
                }
                #[cfg(not(feature = "tensorrt"))]
                return Err(BackendError::Unavailable(
                    "device = \"tensorrt\" but this binary was built without the TensorRT EP; \
                     rebuild with --features ort-tensorrt (requires TensorRT libs on the host)"
                        .into(),
                ));
            }
        }

        let session = builder
            .commit_from_file(&config.model_path)
            .map_err(|e| load_error("failed to load onnx model", e))?;

        const KNOWN: [&str; 3] = ["input_ids", "attention_mask", "token_type_ids"];
        let mut input_names = Vec::new();
        for input in session.inputs() {
            if KNOWN.contains(&input.name()) {
                input_names.push(input.name().to_string());
            } else {
                return Err(load_error(
                    "unsupported model input",
                    format!(
                        "graph input {:?} is not one of {KNOWN:?}; \
                         only tokenizer-fed transformer embedding models are supported",
                        input.name()
                    ),
                ));
            }
        }
        if !input_names.iter().any(|n| n == "input_ids") {
            return Err(load_error(
                "unsupported model",
                "graph has no `input_ids` input",
            ));
        }
        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or_else(|| load_error("unsupported model", "graph has no outputs"))?;

        tracing::info!(
            model = %config.model_path,
            tokenizer = %tokenizer_file.display(),
            device = config.device.as_str(),
            pooling = ?config.pooling,
            output = %output_name,
            "ort session loaded"
        );

        Ok(Self {
            inner: Arc::new(Inner {
                session: Mutex::new(session),
                tokenizer,
                model_path: config.model_path,
                device: config.device,
                pooling: config.pooling,
                normalize: config.normalize.unwrap_or(true),
                input_names,
                output_name,
            }),
        })
    }
}

impl Inner {
    /// Tokenize + run the session + pool. Blocking; called via
    /// `spawn_blocking` from the async trait methods.
    fn embed(&self, texts: Vec<String>) -> Result<(usize, Vec<f32>), BackendError> {
        let batch = texts.len();
        let encodings = self
            .tokenizer
            .encode_batch(texts, true)
            .map_err(|e| BackendError::InvalidRequest(format!("tokenization failed: {e}")))?;
        let seq = encodings.first().map(|e| e.len()).unwrap_or(0);
        if seq == 0 {
            return Err(BackendError::InvalidRequest(
                "tokenization produced an empty sequence".into(),
            ));
        }

        let mut input_ids = Vec::with_capacity(batch * seq);
        let mut attention_mask = Vec::with_capacity(batch * seq);
        let mut token_type_ids = Vec::with_capacity(batch * seq);
        for encoding in &encodings {
            input_ids.extend(encoding.get_ids().iter().map(|&v| v as i64));
            attention_mask.extend(encoding.get_attention_mask().iter().map(|&v| v as i64));
            token_type_ids.extend(encoding.get_type_ids().iter().map(|&v| v as i64));
        }

        let shape = [batch as i64, seq as i64];
        let mut feed: Vec<(&str, SessionInputValue<'_>)> = Vec::new();
        for name in &self.input_names {
            let data = match name.as_str() {
                "input_ids" => input_ids.clone(),
                "attention_mask" => attention_mask.clone(),
                "token_type_ids" => token_type_ids.clone(),
                _ => unreachable!("input names validated at load"),
            };
            let tensor = Tensor::from_array((shape, data))
                .map_err(|e| BackendError::Internal(format!("tensor build failed: {e}")))?;
            feed.push((name.as_str(), tensor.into()));
        }

        let mut session = self
            .session
            .lock()
            .map_err(|_| BackendError::Internal("ort session mutex poisoned".into()))?;
        let outputs = session
            .run(feed)
            .map_err(|e| BackendError::Internal(format!("onnx runtime inference failed: {e}")))?;
        let (out_shape, hidden) = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| BackendError::Internal(format!("output extraction failed: {e}")))?;

        let dims: Vec<i64> = out_shape.iter().copied().collect();
        let mut pooled = match (self.pooling, dims.as_slice()) {
            // [batch, seq, dim] hidden states → pool over tokens.
            (Pooling::Mean, [b, s, d]) if *b as usize == batch && *s as usize == seq => {
                mean_pool(hidden, &attention_mask, batch, seq, *d as usize)
            }
            (Pooling::Cls, [b, s, d]) if *b as usize == batch && *s as usize == seq => {
                cls_pool(hidden, batch, seq, *d as usize)
            }
            // [batch, dim]: the graph already pools (e.g. exported
            // sentence-transformers with pooling baked in).
            (_, [b, _d]) if *b as usize == batch => hidden.to_vec(),
            _ => {
                return Err(BackendError::Internal(format!(
                    "unexpected output shape {dims:?} from {:?} (batch={batch}, seq={seq})",
                    self.output_name
                )))
            }
        };
        let dim = pooled.len() / batch;
        if self.normalize {
            l2_normalize(&mut pooled, dim);
        }
        Ok((dim, pooled))
    }

    /// Tokenize with request-scoped options. The session tokenizer is
    /// configured for embedding inference (fixed truncation, pad-to-longest),
    /// so the Tokenize RPC works on a clone with the caller's options applied.
    fn tokenize_with_options(
        &self,
        texts: &[String],
        options: &TokenizeOptions,
    ) -> Result<Vec<Encoding>, BackendError> {
        use tokenizers::{PaddingParams, PaddingStrategy, TruncationParams};

        let mut tokenizer = self.tokenizer.clone();
        let truncation = options.truncate_to.map(|len| TruncationParams {
            max_length: len,
            ..Default::default()
        });
        tokenizer
            .with_truncation(truncation)
            .map_err(|e| BackendError::InvalidRequest(format!("invalid truncation: {e}")))?;
        if options.pad_to_longest {
            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            }));
        } else {
            tokenizer.with_padding(None);
        }

        let encodings = tokenizer
            .encode_batch(texts.to_vec(), options.add_special_tokens)
            .map_err(|e| BackendError::InvalidRequest(format!("tokenization failed: {e}")))?;

        Ok(encodings
            .into_iter()
            .map(|encoding| {
                let offsets = if options.with_offsets {
                    encoding
                        .get_offsets()
                        .iter()
                        .map(|&(start, end)| Offset {
                            start: start as u32,
                            end: end as u32,
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                Encoding {
                    input_ids: encoding.get_ids().to_vec(),
                    attention_mask: encoding.get_attention_mask().to_vec(),
                    tokens: encoding.get_tokens().to_vec(),
                    offsets,
                }
            })
            .collect())
    }
}

#[async_trait]
impl Backend for OrtBackend {
    fn id(&self) -> &str {
        "onnxruntime"
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
            platform: "onnxruntime_onnx".to_string(),
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
            properties: HashMap::from([
                ("model_path".to_string(), self.inner.model_path.clone()),
                ("device".to_string(), self.inner.device.as_str().to_string()),
                ("pooling".to_string(), format!("{:?}", self.inner.pooling)),
                ("normalize".to_string(), self.inner.normalize.to_string()),
            ]),
        })
    }

    async fn infer(&self, request: ModelInferRequest) -> Result<ModelInferResponse, BackendError> {
        let texts = text_inputs(&request)?;
        let single = texts.len() == 1;
        let batch = texts.len();
        let inner = self.inner.clone();
        let (dim, embeddings) = tokio::task::spawn_blocking(move || inner.embed(texts))
            .await
            .map_err(|e| BackendError::Internal(format!("inference task panicked: {e}")))??;

        // Contract: `[d]` for one text (matches the mock), `[n, d]` for a batch.
        let shape = if single {
            vec![dim as i64]
        } else {
            vec![batch as i64, dim as i64]
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
            raw_output_contents: vec![pack_fp32(&embeddings)],
        })
    }

    /// The engine's own HF tokenizer answers Tokenize when the server has no
    /// `tokenizer_dir`-configured local tokenizer for the model.
    async fn tokenize(
        &self,
        _model_name: &str,
        texts: &[String],
        options: &TokenizeOptions,
    ) -> Result<Vec<Encoding>, BackendError> {
        let inner = self.inner.clone();
        let texts = texts.to_vec();
        let options = options.clone();
        tokio::task::spawn_blocking(move || inner.tokenize_with_options(&texts, &options))
            .await
            .map_err(|e| BackendError::Internal(format!("tokenize task panicked: {e}")))?
    }

    async fn detokenize(
        &self,
        _model_name: &str,
        sequences: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>, BackendError> {
        let inner = self.inner.clone();
        let sequences = sequences.to_vec();
        tokio::task::spawn_blocking(move || {
            sequences
                .iter()
                .map(|ids| {
                    inner
                        .tokenizer
                        .decode(ids, skip_special_tokens)
                        .map_err(|e| BackendError::InvalidRequest(format!("decode failed: {e}")))
                })
                .collect()
        })
        .await
        .map_err(|e| BackendError::Internal(format!("detokenize task panicked: {e}")))?
    }
}
