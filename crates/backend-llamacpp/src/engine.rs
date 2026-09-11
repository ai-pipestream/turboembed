//! In-process llama.cpp engine (crate feature `runtime`).
//!
//! One [`LlamaModel`] per configured model, loaded eagerly at construction so
//! a bad GGUF path or an unsupported device fails at startup. Generation runs
//! on a fresh per-request context inside `spawn_blocking`; decoded token
//! pieces are pushed through a channel onto the response stream as they are
//! produced, so clients see tokens live. Wire shapes are identical to
//! server-client mode (see the crate docs).

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use inferstream_backend::{BackendError, ResponseStream, TokenizeOptions};
use inferstream_protocol::extension::Encoding;
use inferstream_protocol::inference::{
    model_infer_response::InferOutputTensor, model_metadata_response::TensorMetadata,
    ModelInferRequest, ModelInferResponse,
};
use inferstream_protocol::tensor::{pack_bytes, DataType};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend as LlamaRuntime;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use tokio_stream::wrappers::ReceiverStream;

use inferstream_backend::ModelMetadata;

use crate::{bool_param, gen_params, int_param, prompt_from, GenParams, LlamaCppConfig, LlamaDevice};

/// Default context window when the config does not set one; capped to the
/// model's training context at load.
const DEFAULT_N_CTX: u32 = 4096;

/// Buffered token chunks per in-flight generation before backpressure
/// applies to the decode loop.
const TOKEN_CHANNEL_CAPACITY: usize = 32;

/// llama.cpp requires exactly one global backend initialization per process.
fn runtime() -> &'static LlamaRuntime {
    static RUNTIME: OnceLock<LlamaRuntime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
        LlamaRuntime::init().expect("llama backend initializes exactly once via OnceLock")
    })
}

fn load_error(what: &str, detail: impl std::fmt::Display) -> BackendError {
    BackendError::Unavailable(format!("{what}: {detail}"))
}

/// In-process llama.cpp engine for one GGUF model.
#[derive(Clone)]
pub(crate) struct LlamaEngine {
    inner: Arc<Inner>,
}

struct Inner {
    model: LlamaModel,
    model_path: String,
    device: LlamaDevice,
    /// Effective context window (config `n_ctx` capped to the model's
    /// training context).
    n_ctx: u32,
}

impl std::fmt::Debug for LlamaEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaEngine")
            .field("model_path", &self.inner.model_path)
            .field("device", &self.inner.device)
            .field("n_ctx", &self.inner.n_ctx)
            .finish()
    }
}

/// Decode one token to its byte piece, retrying with the exact buffer size
/// llama.cpp reports when the initial guess is too small.
fn token_piece(
    model: &LlamaModel,
    token: LlamaToken,
    special: bool,
) -> Result<Vec<u8>, BackendError> {
    use llama_cpp_2::TokenToStringError;
    match model.token_to_piece_bytes(token, 32, special, None) {
        Err(TokenToStringError::InsufficientBufferSpace(needed)) => model
            .token_to_piece_bytes(token, usize::try_from(-needed).unwrap_or(256), special, None)
            .map_err(|e| BackendError::Internal(format!("token decode failed: {e}"))),
        Ok(bytes) => Ok(bytes),
        Err(e) => Err(BackendError::Internal(format!("token decode failed: {e}"))),
    }
}

impl LlamaEngine {
    /// Load the GGUF model eagerly. Fails with [`BackendError::Unavailable`]
    /// when the file is missing or the requested device is not compiled in.
    pub(crate) fn new(config: &LlamaCppConfig) -> Result<Self, BackendError> {
        if !Path::new(&config.model_path).is_file() {
            return Err(load_error(
                "gguf model not found",
                format!("{} is not a file", config.model_path),
            ));
        }

        // Devices this build can actually serve. CPU always works; an
        // accelerator device needs the matching crate feature, otherwise
        // llama.cpp would silently fall back to CPU — fail loudly instead.
        let n_gpu_layers = match config.device {
            LlamaDevice::Cpu => 0,
            LlamaDevice::Cuda => {
                #[cfg(not(feature = "cuda"))]
                return Err(BackendError::Unavailable(
                    "device = \"cuda\" but this binary was built without the llama.cpp CUDA \
                     backend; rebuild with --features llamacpp-cuda (or full-cuda)"
                        .into(),
                ));
                #[cfg(feature = "cuda")]
                config.n_gpu_layers.unwrap_or(u32::MAX)
            }
            LlamaDevice::Metal => {
                #[cfg(not(feature = "metal"))]
                return Err(BackendError::Unavailable(
                    "device = \"metal\" but this binary was built without the llama.cpp Metal \
                     backend; rebuild with --features metal"
                        .into(),
                ));
                #[cfg(feature = "metal")]
                config.n_gpu_layers.unwrap_or(u32::MAX)
            }
            LlamaDevice::Sycl | LlamaDevice::Vulkan => {
                return Err(BackendError::Unavailable(format!(
                    "llama.cpp device {:?} is not compiled in-process by this crate \
                     (supported: cpu, cuda, metal); use server-client mode: set `endpoint` \
                     to a llama-server built for that device",
                    config.device
                )));
            }
        };

        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = LlamaModel::load_from_file(runtime(), &config.model_path, &model_params)
            .map_err(|e| load_error("failed to load gguf model", e))?;

        let n_ctx_train = model.n_ctx_train();
        let n_ctx = config
            .n_ctx
            .unwrap_or(DEFAULT_N_CTX)
            .min(n_ctx_train.max(1));

        tracing::info!(
            model = %config.model_path,
            device = ?config.device,
            n_gpu_layers,
            n_ctx,
            n_ctx_train,
            "llama.cpp model loaded (in-process)"
        );

        Ok(Self {
            inner: Arc::new(Inner {
                model,
                model_path: config.model_path.clone(),
                device: config.device,
                n_ctx,
            }),
        })
    }

    pub(crate) fn metadata(&self, model_name: &str) -> ModelMetadata {
        let bytes = DataType::Bytes.as_oip().to_string();
        ModelMetadata {
            name: model_name.to_string(),
            versions: vec!["1".to_string()],
            platform: "llama_cpp".to_string(),
            inputs: vec![TensorMetadata {
                name: "text".to_string(),
                datatype: bytes.clone(),
                shape: vec![1],
            }],
            outputs: vec![
                TensorMetadata {
                    name: "text".to_string(),
                    datatype: bytes.clone(),
                    shape: vec![1],
                },
                TensorMetadata {
                    name: "token".to_string(),
                    datatype: bytes,
                    shape: vec![1],
                },
            ],
            properties: HashMap::from([
                ("model_path".to_string(), self.inner.model_path.clone()),
                ("device".to_string(), format!("{:?}", self.inner.device)),
                ("mode".to_string(), "in-process".to_string()),
                ("n_ctx".to_string(), self.inner.n_ctx.to_string()),
            ]),
        }
    }

    /// Unary generation: the whole completion in one `text` BYTES output
    /// plus a `tokens_predicted` parameter (server-client parity).
    pub(crate) async fn infer(
        &self,
        request: ModelInferRequest,
    ) -> Result<ModelInferResponse, BackendError> {
        let prompt = prompt_from(&request)?;
        let params = gen_params(&request.parameters);
        let inner = self.inner.clone();
        let (completion, produced) = tokio::task::spawn_blocking(move || {
            let mut collected = Vec::new();
            let mut produced = 0i64;
            inner.generate(&prompt, &params, |piece, _| {
                if !piece.is_empty() {
                    produced += 1;
                }
                collected.extend_from_slice(&piece);
                true
            })?;
            Ok::<(Vec<u8>, i64), BackendError>((collected, produced))
        })
        .await
        .map_err(|e| BackendError::Internal(format!("generation task panicked: {e}")))??;

        Ok(ModelInferResponse {
            model_name: request.model_name,
            model_version: request.model_version,
            id: request.id,
            parameters: HashMap::from([("tokens_predicted".to_string(), int_param(produced))]),
            outputs: vec![InferOutputTensor {
                name: "text".to_string(),
                datatype: DataType::Bytes.as_oip().to_string(),
                shape: vec![1],
                parameters: HashMap::new(),
                contents: None,
            }],
            raw_output_contents: vec![pack_bytes(&[completion.as_slice()])],
        })
    }

    /// Streaming generation: one `token` BYTES chunk per decoded piece, the
    /// last chunk flagged `final = true`. Chunks flow as they are decoded.
    pub(crate) fn infer_stream(
        &self,
        request: ModelInferRequest,
    ) -> Result<ResponseStream, BackendError> {
        let prompt = prompt_from(&request)?;
        let params = gen_params(&request.parameters);
        let meta = (
            request.model_name.clone(),
            request.model_version.clone(),
            request.id.clone(),
        );
        let inner = self.inner.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<ModelInferResponse, BackendError>>(
            TOKEN_CHANNEL_CAPACITY,
        );
        tokio::task::spawn_blocking(move || {
            let result = inner.generate(&prompt, &params, |piece, is_final| {
                tx.blocking_send(Ok(Inner::token_chunk(&meta, piece, is_final)))
                    .is_ok()
            });
            if let Err(error) = result {
                let _ = tx.blocking_send(Err(error));
            }
        });
        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    /// Tokenize with the GGUF's own vocabulary. Byte offsets are not
    /// available from llama.cpp, so `with_offsets` returns empty offsets;
    /// `pad_to_longest` is unsupported (GGUF models have no universal pad
    /// token).
    pub(crate) fn tokenize(
        &self,
        texts: &[String],
        options: &TokenizeOptions,
    ) -> Result<Vec<Encoding>, BackendError> {
        if options.pad_to_longest {
            return Err(BackendError::InvalidRequest(
                "pad_to_longest is not supported by the in-process llama.cpp tokenizer".into(),
            ));
        }
        let add_bos = if options.add_special_tokens {
            AddBos::Always
        } else {
            AddBos::Never
        };
        let mut encodings = Vec::with_capacity(texts.len());
        for text in texts {
            let mut tokens = self
                .inner
                .model
                .str_to_token(text, add_bos)
                .map_err(|e| BackendError::InvalidRequest(format!("tokenization failed: {e}")))?;
            if let Some(limit) = options.truncate_to {
                tokens.truncate(limit);
            }
            let mut encoding = Encoding {
                input_ids: Vec::with_capacity(tokens.len()),
                attention_mask: vec![1; tokens.len()],
                tokens: Vec::with_capacity(tokens.len()),
                offsets: Vec::new(),
            };
            for token in tokens {
                encoding.input_ids.push(token.0 as u32);
                let piece = token_piece(&self.inner.model, token, true)?;
                encoding
                    .tokens
                    .push(String::from_utf8_lossy(&piece).into_owned());
            }
            encodings.push(encoding);
        }
        Ok(encodings)
    }

    pub(crate) fn detokenize(
        &self,
        sequences: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>, BackendError> {
        let n_vocab = self.inner.model.n_vocab();
        sequences
            .iter()
            .map(|ids| {
                let mut bytes = Vec::new();
                for &id in ids {
                    let signed = i32::try_from(id).map_err(|_| {
                        BackendError::InvalidRequest(format!("token id {id} out of range"))
                    })?;
                    if signed >= n_vocab {
                        return Err(BackendError::InvalidRequest(format!(
                            "token id {id} is outside the vocabulary (n_vocab = {n_vocab})"
                        )));
                    }
                    let token = LlamaToken(signed);
                    if skip_special_tokens && self.inner.model.is_eog_token(token) {
                        continue;
                    }
                    // `special = !skip` renders control tokens as text when
                    // the caller wants them kept.
                    bytes.extend(token_piece(&self.inner.model, token, !skip_special_tokens)?);
                }
                Ok(String::from_utf8_lossy(&bytes).into_owned())
            })
            .collect()
    }
}

impl Inner {
    /// Run one generation. `emit(piece, is_final)` is called for every
    /// decoded token piece (the final call carries `is_final = true`, with an
    /// empty piece when generation ended on EOG/budget without a buffered
    /// token); returning `false` aborts (client went away).
    ///
    /// Blocking — always called via `spawn_blocking`.
    fn generate(
        &self,
        prompt: &str,
        params: &GenParams,
        mut emit: impl FnMut(Vec<u8>, bool) -> bool,
    ) -> Result<(), BackendError> {
        if params.stop.is_some() {
            return Err(BackendError::InvalidRequest(
                "the `stop` parameter is only supported in server-client mode today".into(),
            ));
        }
        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| {
                BackendError::InvalidRequest(format!("prompt tokenization failed: {e}"))
            })?;
        if tokens.is_empty() {
            return Err(BackendError::InvalidRequest(
                "prompt tokenized to nothing".into(),
            ));
        }
        let n_ctx = self.n_ctx as usize;
        if tokens.len() + 1 > n_ctx {
            return Err(BackendError::InvalidRequest(format!(
                "prompt is {} tokens; the context window is {n_ctx}",
                tokens.len()
            )));
        }
        // `n_predict <= 0` mirrors llama-server: generate up to the context
        // budget.
        let budget = n_ctx - tokens.len();
        let max_tokens = if params.n_predict > 0 {
            (params.n_predict as usize).min(budget)
        } else {
            budget
        };

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx))
            .with_n_batch(self.n_ctx);
        let mut ctx = self
            .model
            .new_context(runtime(), ctx_params)
            .map_err(|e| load_error("failed to create llama context", e))?;

        let mut batch = LlamaBatch::new(n_ctx, 1);
        let last = tokens.len() as i32 - 1;
        for (i, token) in (0_i32..).zip(tokens.iter()) {
            batch
                .add(*token, i, &[0], i == last)
                .map_err(|e| BackendError::Internal(format!("batch build failed: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| BackendError::Internal(format!("prompt decode failed: {e}")))?;

        // Greedy unless sampling knobs are present (temperature 0 = greedy,
        // matching llama-server semantics closely enough for parity).
        let mut chain: Vec<LlamaSampler> = Vec::new();
        if let Some(p) = params.top_p {
            chain.push(LlamaSampler::top_p(p as f32, 1));
        }
        if let Some(t) = params.temperature {
            if t > 0.0 {
                chain.push(LlamaSampler::temp(t as f32));
            }
        }
        let mut sampler = if chain.is_empty() {
            LlamaSampler::greedy()
        } else {
            chain.push(LlamaSampler::dist(params.seed.unwrap_or(42) as u32));
            LlamaSampler::chain_simple(chain)
        };

        // One piece is always buffered so the last emitted chunk can carry
        // the final flag the moment generation stops.
        let mut pending: Option<Vec<u8>> = None;
        let mut n_cur = batch.n_tokens();
        let mut produced = 0usize;
        while produced < max_tokens {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                break;
            }
            let piece = token_piece(&self.model, token, false)?;
            if let Some(previous) = pending.replace(piece) {
                if !emit(previous, false) {
                    return Ok(());
                }
            }
            produced += 1;
            if produced >= max_tokens {
                break;
            }
            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| BackendError::Internal(format!("batch build failed: {e}")))?;
            n_cur += 1;
            ctx.decode(&mut batch)
                .map_err(|e| BackendError::Internal(format!("token decode step failed: {e}")))?;
        }
        emit(pending.unwrap_or_default(), true);
        Ok(())
    }

    fn token_chunk(
        request_meta: &(String, String, String),
        piece: Vec<u8>,
        is_final: bool,
    ) -> ModelInferResponse {
        let (model_name, model_version, id) = request_meta;
        ModelInferResponse {
            model_name: model_name.clone(),
            model_version: model_version.clone(),
            id: id.clone(),
            parameters: HashMap::from([("final".to_string(), bool_param(is_final))]),
            outputs: vec![InferOutputTensor {
                name: "token".to_string(),
                datatype: DataType::Bytes.as_oip().to_string(),
                shape: vec![1],
                parameters: HashMap::new(),
                contents: None,
            }],
            raw_output_contents: vec![pack_bytes(&[piece.as_slice()])],
        }
    }
}
