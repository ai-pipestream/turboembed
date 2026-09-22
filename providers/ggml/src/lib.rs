//! ggml / llama.cpp provider for Turbo: GGUF generation.
//!
//! One Turbo device per ggml backend device (GPU devices, plus the CPU,
//! which AUTO never selects). A model is a `llama_model` loaded with every
//! layer on the selected device; a generation owns its own `llama_context`
//! (the KV cache) and produces one token per `step`, so cancellation and
//! backpressure are the caller's. Sampling is llama.cpp's sampler chain
//! built from the request: logit bias, repetition penalties, top-k, top-p,
//! min-p, temperature, then a seeded distribution sample or greedy.
//!
//! Bundle contract: a `gguf` artifact; `contract.max_seq` is the context
//! length; the chat template comes from the bundle's `tokenizer.chat_template`
//! when present, else from the GGUF metadata. Prompt roles and the special
//! tokens are the model's own (the GGUF vocabulary).
//!
//! Embeddings: an embedding bundle with a `gguf` artifact (a BERT-family
//! encoder converted by llama.cpp) runs through the same llama context with
//! pooling in the graph (llama.cpp's pooling type from the bundle contract)
//! and L2 normalization on the host. llama.cpp hands pooled vectors back in
//! host memory whatever the device, so the result placement is HOST and the
//! device does not advertise `TURBO_CAP_DEVICE_RESULT`; the bytes llama.cpp
//! moved back are counted in `d2h_bytes`. The pooling type is fixed at
//! context creation, so a pooling override is not offered.
//!
//! Honesty: structured output is honored for GBNF grammars only (a JSON
//! schema is refused naming the field), `n_sequences > 1` and tool
//! definitions are not offered. Every cell is `EXPERIMENTAL` until receipts
//! land.

#![deny(missing_docs)]

use std::ffi::CStr;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex, OnceLock};

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::logit_bias::LlamaLogitBias;
use llama_cpp_2::token::LlamaToken;

use turbo_core::abi;
use turbo_core::buffer::{BufferDesc, HostBuffer, NativeHandle, ProviderBuffer};
use turbo_core::bundle::Bundle;
use turbo_core::error::{Error, Result};
use turbo_core::provider::{
    Capability, Chunk, ContextDesc, DeviceInfo, EmbedOptions, GenerateDesc, Message, ModelDesc, ModelInfo, Output,
    Provider, ProviderContext, ProviderGeneration, ProviderModel, ProviderResult, ProviderSession, RunOptions,
    SessionDesc, SessionStats, TokenBatch,
};
use turbo_core::types::{
    CapStatus, DType, DeviceKind, FinishReason, Modality, ModelKind, Normalize, OutputDType, Placement, Pooling,
    PromptRole, Stage, StagePlacement, StagePlacements, StructuredKind, Task, Truncate,
};

/// Provider id.
pub const GGML_PROVIDER_ID: &str = "ggml";
/// Artifact format the provider loads.
pub const GGUF_ARTIFACT: &str = "gguf";
/// Default `max_new_tokens` when the request leaves it at 0.
pub const DEFAULT_MAX_NEW_TOKENS: u32 = 512;
/// Tokens the repetition penalties look back over.
const PENALTY_LAST_N: i32 = 64;

/// Capability bits every ggml device reports.
pub const GGML_CAPS: u64 = abi::TURBO_CAP_DYNAMIC_SHAPE
    | abi::TURBO_CAP_WEIGHT_SHARING
    | abi::TURBO_CAP_OPT_TRUNCATE
    | abi::TURBO_CAP_OPT_MAX_TOKENS
    | abi::TURBO_CAP_OPT_PROMPT_ROLE
    | abi::TURBO_CAP_OPT_NORMALIZE
    | abi::TURBO_CAP_OPT_OUTPUT_DIM
    | abi::TURBO_CAP_OPT_GEN_SEED
    | abi::TURBO_CAP_OPT_GEN_SAMPLING
    | abi::TURBO_CAP_OPT_GEN_PENALTIES
    | abi::TURBO_CAP_OPT_GEN_LOGIT_BIAS
    | abi::TURBO_CAP_OPT_GEN_LOGPROBS
    | abi::TURBO_CAP_OPT_GEN_STOP_STRINGS
    | abi::TURBO_CAP_OPT_GEN_STOP_TOKENS
    | abi::TURBO_CAP_OPT_GEN_MIN_TOKENS
    | abi::TURBO_CAP_OPT_GEN_ECHO
    | abi::TURBO_CAP_OPT_GEN_STRUCTURED;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn backend() -> Result<&'static LlamaBackend> {
    static BACKEND: OnceLock<std::result::Result<LlamaBackend, String>> = OnceLock::new();
    match BACKEND.get_or_init(|| LlamaBackend::init().map_err(|e| e.to_string())) {
        Ok(b) => Ok(b),
        Err(e) => Err(Error::provider_load(format!("llama.cpp backend init failed: {e}"))),
    }
}

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

/// One ggml backend device.
#[derive(Clone, Debug)]
struct GgmlDevice {
    /// Index in ggml's device registry.
    index: usize,
    kind: DeviceKind,
    name: String,
    description: String,
    memory_total: u64,
    memory_free: u64,
}

fn enumerate() -> Result<Vec<GgmlDevice>> {
    backend()?;
    let mut out = Vec::new();
    // SAFETY: plain reads from ggml's static device registry after backend init.
    unsafe {
        use llama_cpp_sys_2 as sys;
        let n = sys::ggml_backend_dev_count();
        for i in 0..n {
            let dev = sys::ggml_backend_dev_get(i);
            if dev.is_null() {
                return Err(Error::internal(format!("ggml device {i} is NULL")));
            }
            let kind = match sys::ggml_backend_dev_type(dev) {
                sys::GGML_BACKEND_DEVICE_TYPE_CPU => DeviceKind::Cpu,
                sys::GGML_BACKEND_DEVICE_TYPE_GPU => DeviceKind::Gpu,
                sys::GGML_BACKEND_DEVICE_TYPE_IGPU => DeviceKind::IGpu,
                sys::GGML_BACKEND_DEVICE_TYPE_ACCEL => DeviceKind::Accel,
                other => return Err(Error::internal(format!("ggml device {i} has unknown type {other}"))),
            };
            let name = CStr::from_ptr(sys::ggml_backend_dev_name(dev)).to_string_lossy().into_owned();
            let description = CStr::from_ptr(sys::ggml_backend_dev_description(dev)).to_string_lossy().into_owned();
            let (mut free, mut total) = (0usize, 0usize);
            sys::ggml_backend_dev_memory(dev, &mut free, &mut total);
            out.push(GgmlDevice {
                index: i,
                kind,
                name,
                description,
                memory_total: total as u64,
                memory_free: free as u64,
            });
        }
    }
    if out.is_empty() {
        return Err(Error::device_not_found("ggml registered no backend devices"));
    }
    Ok(out)
}

/// The provider.
#[derive(Default)]
pub struct GgmlProvider {
    devices: OnceLock<std::result::Result<Vec<GgmlDevice>, Error>>,
}

impl GgmlProvider {
    /// Construct. llama.cpp is initialized on first use.
    pub fn new() -> Self {
        Self::default()
    }

    fn list(&self) -> Result<&[GgmlDevice]> {
        match self.devices.get_or_init(enumerate) {
            Ok(v) => Ok(v),
            Err(e) => Err(e.clone()),
        }
    }

    fn device(&self, ordinal: u32) -> Result<&GgmlDevice> {
        let all = self.list()?;
        all.get(ordinal as usize).ok_or_else(|| {
            Error::device_not_found(format!("ggml has {} device(s); ordinal {ordinal} does not exist", all.len()))
        })
    }
}

impl Provider for GgmlProvider {
    fn id(&self) -> &str {
        GGML_PROVIDER_ID
    }

    fn version(&self) -> &str {
        VERSION
    }

    fn devices(&self) -> Result<Vec<DeviceInfo>> {
        Ok(self
            .list()?
            .iter()
            .enumerate()
            .map(|(ordinal, d)| DeviceInfo {
                kind: d.kind,
                ordinal: ordinal as u32,
                vendor_id: 0,
                caps: GGML_CAPS,
                memory_total: d.memory_total,
                memory_free: d.memory_free,
                name: format!("{} ({})", d.description, d.name),
                vendor: String::new(),
                provider_id: GGML_PROVIDER_ID.into(),
                provider_version: VERSION.into(),
                runtime_version: format!("llama.cpp (llama-cpp-2 {})", llama_cpp_2_version()),
                driver_version: String::new(),
            })
            .collect())
    }

    fn capability(&self, ordinal: u32, task: Task, modality: Modality) -> Capability {
        let Ok(d) = self.device(ordinal) else { return Capability::unsupported() };
        if modality != Modality::Text || !matches!(task, Task::Generate | Task::Embed) {
            return Capability::unsupported();
        }
        Capability {
            status: CapStatus::Experimental,
            dtype: None,
            reference_dtype: None,
            cosine_floor: 0.0,
            max_abs_error: 0.0,
            // Embeddings are bit-reproducible on the CPU; sampled generation
            // draws a seed when none is given, so it never claims to be.
            deterministic: d.kind == DeviceKind::Cpu && task == Task::Embed,
            notes: match task {
                Task::Embed => format!(
                    "llama.cpp on {}; pooling in the graph, L2 on the host, result in host memory; receipts pending",
                    d.name
                ),
                _ => format!("llama.cpp on {}; GGUF quantization decides the compute dtype; receipts pending", d.name),
            },
        }
    }

    fn can_run(&self, ordinal: u32, bundle: &Bundle, task: Task, modality: Modality) -> Result<()> {
        self.device(ordinal)?;
        if modality != Modality::Text || !matches!(task, Task::Generate | Task::Embed) {
            return Err(Error::unsupported_task(format!(
                "ggml provider offers GENERATE and EMBED on TEXT, not {task:?} x {modality:?}"
            )));
        }
        if bundle.artifact(GGUF_ARTIFACT).is_none() {
            return Err(Error::bundle_no_artifact(format!(
                "bundle `{}` has no `gguf` artifact",
                bundle.manifest().model_id
            )));
        }
        if !matches!(bundle.kind(), ModelKind::Generative | ModelKind::Embedding) {
            return Err(Error::unsupported_task(format!(
                "ggml provider serves generative and embedding bundles, not {:?}",
                bundle.kind()
            )));
        }
        Ok(())
    }

    fn create_context(&self, ordinal: u32, desc: &ContextDesc) -> Result<Arc<dyn ProviderContext>> {
        let device = self.device(ordinal)?.clone();
        desc.options.reject_unknown(&[], "ggml context")?;
        Ok(Arc::new(GgmlContext { inner: Arc::new(ContextInner { ordinal, device }) }))
    }
}

fn llama_cpp_2_version() -> &'static str {
    "0.1.156"
}

// ---------------------------------------------------------------------------
// Context and model
// ---------------------------------------------------------------------------

struct ContextInner {
    ordinal: u32,
    device: GgmlDevice,
}

struct GgmlContext {
    inner: Arc<ContextInner>,
}

impl ProviderContext for GgmlContext {
    fn ordinal(&self) -> u32 {
        self.inner.ordinal
    }

    fn alloc(&self, desc: &BufferDesc) -> Result<Arc<dyn ProviderBuffer>> {
        if desc.placement != Placement::Host {
            return Err(Error::unsupported_placement(format!(
                "ggml provider allocates TURBO_PLACE_HOST only, not {:?}",
                desc.placement
            )));
        }
        Ok(HostBuffer::new(desc.clone())?)
    }

    fn import(&self, _desc: &BufferDesc, _handle: &NativeHandle) -> Result<Arc<dyn ProviderBuffer>> {
        Err(Error::unsupported("ggml provider does not import buffers"))
    }

    fn load_model(&self, bundle: Arc<Bundle>, desc: &ModelDesc) -> Result<Arc<dyn ProviderModel>> {
        desc.options.reject_unknown(&[], "ggml model")?;
        let kind = bundle.kind();
        if !matches!(kind, ModelKind::Generative | ModelKind::Embedding) {
            return Err(Error::unsupported_task(format!(
                "ggml provider serves generative and embedding bundles, not {kind:?}"
            )));
        }
        let path = bundle.artifact_path(GGUF_ARTIFACT)?;
        let device = &self.inner.device;
        let mut params = LlamaModelParams::default();
        params = match device.kind {
            DeviceKind::Cpu => params.with_n_gpu_layers(0),
            _ => params.with_n_gpu_layers(u32::MAX).with_devices(&[device.index]).map_err(|e| {
                Error::device_unavailable(format!("ggml device {} cannot be selected: {e}", device.name))
            })?,
        };
        let model = LlamaModel::load_from_file(backend()?, &path, &params)
            .map_err(|e| Error::runtime(format!("llama.cpp could not load `{}`: {e}", path.display())))?;
        let c = bundle.contract();
        let max_seq = if c.max_seq == 0 { model.n_ctx_train() } else { c.max_seq };
        if max_seq > model.n_ctx_train() {
            return Err(Error::bundle_invalid(format!(
                "contract.max_seq {max_seq} exceeds the model's training context {}",
                model.n_ctx_train()
            )));
        }
        let placement = if device.kind == DeviceKind::Cpu { StagePlacement::Host } else { StagePlacement::Device };
        // Generative bundles need a chat template: the bundle's wins over the
        // GGUF metadata's. Embedding bundles need the contract instead.
        let (template, embed) = match kind {
            ModelKind::Generative => {
                let t = match bundle.manifest().tokenizer.as_ref().and_then(|t| t.chat_template.clone()) {
                    Some(t) => LlamaChatTemplate::new(&t)
                        .map_err(|e| Error::bundle_invalid(format!("tokenizer.chat_template is not usable: {e}")))?,
                    None => model.chat_template(None).map_err(|e| {
                        Error::bundle_invalid(format!(
                            "bundle `{}` declares no chat template and the GGUF carries none: {e}",
                            bundle.manifest().model_id
                        ))
                    })?,
                };
                (Some(t), None)
            }
            _ => {
                let pool = bundle
                    .pooling()?
                    .ok_or_else(|| Error::bundle_invalid("embedding bundle must declare contract.pooling"))?;
                let normalize = bundle
                    .normalize()?
                    .ok_or_else(|| Error::bundle_invalid("embedding bundle must declare contract.normalize"))?;
                if c.dim == 0 {
                    return Err(Error::bundle_invalid("embedding bundle must declare contract.dim"));
                }
                if model.n_embd() != c.dim as i32 {
                    return Err(Error::bundle_invalid(format!(
                        "GGUF embedding width {} does not match contract.dim {}",
                        model.n_embd(),
                        c.dim
                    )));
                }
                (None, Some(EmbedContract { pool, normalize: normalize == Normalize::L2, dim: c.dim }))
            }
        };
        let max_batch = if bundle.manifest().limits.max_batch == 0 { 32 } else { bundle.manifest().limits.max_batch };
        let info = ModelInfo {
            task: if kind == ModelKind::Generative { Task::Generate } else { Task::Embed },
            kind,
            modality: Modality::Text,
            dim: embed.as_ref().map(|e| e.dim).unwrap_or(0),
            labels: Vec::new(),
            pooling: embed.as_ref().map(|e| e.pool),
            normalize: embed.as_ref().map(|e| if e.normalize { Normalize::L2 } else { Normalize::None }),
            aggregation: None,
            max_seq,
            max_batch: if kind == ModelKind::Generative { 1 } else { max_batch },
            // Pooled embeddings come back as f32 whatever the weight
            // quantization; a generative model's dtype is not one value.
            dtype_used: if kind == ModelKind::Embedding { Some(DType::F32) } else { None },
            stages: if kind == ModelKind::Generative {
                StagePlacements::NONE
                    .with(Stage::Tokenize, StagePlacement::Host)
                    .with(Stage::Encode, placement)
                    .with(Stage::Postprocess, StagePlacement::Host)
            } else {
                StagePlacements::NONE
                    .with(Stage::Tokenize, StagePlacement::Host)
                    .with(Stage::Encode, placement)
                    .with(Stage::Pool, placement)
                    .with(Stage::Normalize, StagePlacement::Host)
            },
            inputs: Vec::new(),
            outputs: Vec::new(),
            vocab_size: model.n_vocab().max(0) as u32,
            model_id: bundle.manifest().model_id.clone(),
            revision: bundle.manifest().revision.clone(),
            tokenizer_sha256: bundle.tokenizer_sha256().to_string(),
            provider_id: GGML_PROVIDER_ID.into(),
            prefix_query: c.prompts.query.clone(),
            prefix_document: c.prompts.document.clone(),
        };
        let inner =
            Arc::new(ModelInner { model, template, embed, info, device_kind: device.kind, _ctx: self.inner.clone() });
        Ok(Arc::new(GgmlModel { inner }))
    }
}

/// The embedding contract of an embedding bundle.
#[derive(Clone, Copy)]
struct EmbedContract {
    pool: Pooling,
    normalize: bool,
    dim: u32,
}

struct ModelInner {
    model: LlamaModel,
    /// Present for generative bundles.
    template: Option<LlamaChatTemplate>,
    /// Present for embedding bundles.
    embed: Option<EmbedContract>,
    info: ModelInfo,
    device_kind: DeviceKind,
    _ctx: Arc<ContextInner>,
}

struct GgmlModel {
    inner: Arc<ModelInner>,
}

impl ProviderModel for GgmlModel {
    fn info(&self) -> &ModelInfo {
        &self.inner.info
    }

    fn create_session(&self, desc: &SessionDesc) -> Result<Box<dyn ProviderSession>> {
        if self.inner.embed.is_none() {
            return Err(Error::unsupported_task(format!(
                "model `{}` is generative; create a generation, not a session",
                self.inner.info.model_id
            )));
        }
        desc.options.reject_unknown(&[], "ggml session")?;
        if desc.max_batch < 1 || desc.max_seq < 2 {
            return Err(Error::invalid_argument("session needs max_batch >= 1 and max_seq >= 2"));
        }
        EmbedSession::new(self.inner.clone(), desc.max_batch, desc.max_seq)
            .map(|s| Box::new(s) as Box<dyn ProviderSession>)
    }

    fn create_generation(&self, desc: &GenerateDesc) -> Result<Box<dyn ProviderGeneration>> {
        if self.inner.template.is_none() {
            return Err(Error::unsupported_task(format!(
                "model `{}` is an embedding model; create a session, not a generation",
                self.inner.info.model_id
            )));
        }
        Generation::new(self.inner.clone(), desc).map(|g| Box::new(g) as Box<dyn ProviderGeneration>)
    }
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// Sampler chains and request state for one generation.
struct Plan {
    max_new: u32,
    min_new: u32,
    stop: Vec<String>,
    stop_tokens: Vec<i32>,
    logprobs: u32,
    echo: bool,
    seed: u32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    repeat_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    logit_bias: Vec<(i32, f32)>,
    grammar: Option<String>,
}

impl Plan {
    fn from_desc(desc: &GenerateDesc, model: &ModelInner) -> Result<Self> {
        desc.options.reject_unknown(&[], "ggml generation")?;
        if desc.n_sequences > 1 {
            return Err(Error::unsupported_option(GenerateDesc::FIELD_N_SEQUENCES, "n_sequences", GGML_PROVIDER_ID));
        }
        if !desc.tools.is_empty() {
            return Err(Error::unsupported_option(GenerateDesc::FIELD_N_TOOLS, "tools", GGML_PROVIDER_ID));
        }
        let grammar = match desc.structured_kind {
            StructuredKind::None => None,
            StructuredKind::Grammar => Some(desc.structured.clone()),
            StructuredKind::JsonSchema => {
                return Err(Error::unsupported_option(
                    GenerateDesc::FIELD_STRUCTURED_KIND,
                    "structured_kind=JSON_SCHEMA (the ggml provider takes GBNF grammars only)",
                    GGML_PROVIDER_ID,
                ))
            }
        };
        let max_new = if desc.max_new_tokens == 0 { DEFAULT_MAX_NEW_TOKENS } else { desc.max_new_tokens };
        if desc.min_new_tokens > max_new {
            return Err(Error::invalid_argument(format!(
                "min_new_tokens {} exceeds max_new_tokens {max_new}",
                desc.min_new_tokens
            ))
            .with_field(3));
        }
        let vocab = model.model.n_vocab();
        for &(id, _) in &desc.logit_bias {
            if id < 0 || id >= vocab {
                return Err(Error::invalid_argument(format!(
                    "logit_bias token {id} is outside the vocabulary of {vocab}"
                ))
                .with_field(GenerateDesc::FIELD_N_LOGIT_BIAS));
            }
        }
        for &id in &desc.stop_tokens {
            if id < 0 || id >= vocab {
                return Err(Error::invalid_argument(format!("stop token {id} is outside the vocabulary of {vocab}"))
                    .with_field(15));
            }
        }
        if desc.top_k != 0 && u64::from(desc.top_k) > vocab as u64 {
            return Err(Error::invalid_argument(format!("top_k {} exceeds the vocabulary of {vocab}", desc.top_k))
                .with_field(6));
        }
        if u64::from(desc.logprobs) > vocab as u64 {
            return Err(Error::invalid_argument(format!("logprobs {} exceeds the vocabulary of {vocab}", desc.logprobs))
                .with_field(GenerateDesc::FIELD_LOGPROBS));
        }
        // The ABI seed is 64 bits; llama.cpp's is 32 and reserves
        // 0xFFFFFFFF as "draw one". Neither is silently mapped.
        let seed = match desc.seed {
            None => fresh_seed(),
            Some(s) if s > u64::from(u32::MAX) => {
                return Err(Error::invalid_argument(format!("seed {s} does not fit llama.cpp's 32-bit seed"))
                    .with_field(GenerateDesc::FIELD_SEED))
            }
            Some(s) if s == u64::from(u32::MAX) => {
                return Err(Error::invalid_argument("seed 0xFFFFFFFF is llama.cpp's random-seed sentinel and cannot be honored as a fixed seed")
                    .with_field(GenerateDesc::FIELD_SEED))
            }
            Some(s) => s as u32,
        };
        Ok(Self {
            max_new,
            min_new: desc.min_new_tokens,
            stop: desc.stop.clone(),
            stop_tokens: desc.stop_tokens.clone(),
            logprobs: desc.logprobs,
            echo: desc.echo,
            // An unseeded sampled generation is a fresh draw, as a caller who
            // omitted the seed expects; the chunk stream does not report the
            // seed, so a caller who wants reproduction sets one.
            seed,
            temperature: desc.temperature,
            top_k: desc.top_k,
            top_p: desc.top_p,
            min_p: desc.min_p,
            repeat_penalty: desc.repeat_penalty,
            presence_penalty: desc.presence_penalty,
            frequency_penalty: desc.frequency_penalty,
            logit_bias: desc.logit_bias.clone(),
            grammar,
        })
    }

    /// Build the sampler chain. With `suppress_eog`, end-of-generation tokens
    /// get a -inf bias so `min_new_tokens` is honored without resampling.
    fn sampler(&self, model: &LlamaModel, suppress_eog: bool) -> Result<LlamaSampler> {
        let vocab = model.n_vocab();
        let mut chain = Vec::new();
        if let Some(g) = &self.grammar {
            chain.push(
                LlamaSampler::grammar(model, g, "root")
                    .map_err(|e| Error::invalid_argument(format!("grammar is not valid GBNF: {e}")).with_field(23))?,
            );
        }
        let mut biases: Vec<LlamaLogitBias> =
            self.logit_bias.iter().map(|&(id, b)| LlamaLogitBias::new(LlamaToken(id), b)).collect();
        if suppress_eog {
            for id in 0..vocab {
                let t = LlamaToken(id);
                if model.is_eog_token(t) {
                    biases.push(LlamaLogitBias::new(t, f32::NEG_INFINITY));
                }
            }
        }
        if !biases.is_empty() {
            chain.push(LlamaSampler::logit_bias(vocab, &biases));
        }
        let repeat = if self.repeat_penalty == 0.0 { 1.0 } else { self.repeat_penalty };
        if repeat != 1.0 || self.presence_penalty != 0.0 || self.frequency_penalty != 0.0 {
            chain.push(LlamaSampler::penalties(
                vocab,
                PENALTY_LAST_N,
                repeat,
                self.frequency_penalty,
                self.presence_penalty,
            ));
        }
        if self.temperature > 0.0 {
            if self.top_k != 0 {
                chain.push(LlamaSampler::top_k(self.top_k as i32));
            }
            if self.top_p > 0.0 && self.top_p < 1.0 {
                chain.push(LlamaSampler::top_p(self.top_p, 1));
            }
            if self.min_p > 0.0 {
                chain.push(LlamaSampler::min_p(self.min_p, 1));
            }
            chain.push(LlamaSampler::temp(self.temperature));
            chain.push(LlamaSampler::dist(self.seed));
        } else {
            chain.push(LlamaSampler::greedy());
        }
        Ok(LlamaSampler::chain_simple(chain))
    }
}

/// A generation: its own llama context (KV cache), samplers, and decode state.
struct Generation {
    // Field order is load-bearing: the context borrows the model in
    // `inner` and must drop first.
    ctx: Option<LlamaContext<'static>>,
    batch: LlamaBatch<'static>,
    sampler: LlamaSampler,
    sampler_no_eog: Option<LlamaSampler>,
    plan: Plan,
    prompt_tokens: Vec<LlamaToken>,
    prompt_text: String,
    n_past: i32,
    generated: u32,
    /// Decoded text not yet handed out: the last `hold_max` bytes stay
    /// here so a stop string that spans token boundaries can be withheld
    /// whole instead of leaking its prefix in earlier chunks.
    hold: String,
    hold_max: usize,
    pending: Vec<u8>,
    prompted: bool,
    cancelled: bool,
    done: bool,
    inner: Arc<ModelInner>,
    _guard: Mutex<()>,
}

// SAFETY: a generation is single-owner by the core's contract; llama.cpp
// contexts are not shared across threads concurrently.
unsafe impl Send for Generation {}

impl Generation {
    fn new(inner: Arc<ModelInner>, desc: &GenerateDesc) -> Result<Self> {
        let plan = Plan::from_desc(desc, &inner)?;
        let hold_max = hold_max_of(&plan.stop);
        let sampler = plan.sampler(&inner.model, false)?;
        let sampler_no_eog = if plan.min_new > 0 { Some(plan.sampler(&inner.model, true)?) } else { None };
        let n_ctx = inner.info.max_seq;
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx.min(2048))
            .with_n_seq_max(1);
        let ctx = inner
            .model
            .new_context(backend()?, params)
            .map_err(|e| Error::runtime(format!("llama.cpp could not create a context of {n_ctx} tokens: {e}")))?;
        // SAFETY: `ctx` borrows `inner.model`; `inner` is an Arc held by this
        // struct for its whole life and `ctx` is declared first so it drops
        // before the Arc. The lifetime is extended only for storage.
        let ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
        Ok(Self {
            ctx: Some(ctx),
            batch: LlamaBatch::new(n_ctx.min(2048) as usize, 1),
            sampler,
            sampler_no_eog,
            plan,
            prompt_tokens: Vec::new(),
            prompt_text: String::new(),
            n_past: 0,
            generated: 0,
            hold: String::new(),
            hold_max,
            pending: Vec::new(),
            prompted: false,
            cancelled: false,
            done: false,
            inner,
            _guard: Mutex::new(()),
        })
    }

    fn ctx(&mut self) -> &mut LlamaContext<'static> {
        self.ctx.as_mut().expect("context lives as long as the generation")
    }

    /// Decode the prompt so the next sample sees the last prompt token's logits.
    fn prefill(&mut self) -> Result<()> {
        let n = self.prompt_tokens.len();
        let n_ctx = self.inner.info.max_seq as usize;
        if n == 0 {
            return Err(Error::invalid_argument("the prompt tokenizes to nothing"));
        }
        if n + 1 > n_ctx {
            return Err(Error::capacity(format!("prompt is {n} tokens but the context holds {n_ctx}")));
        }
        self.ctx().clear_kv_cache();
        self.sampler.reset();
        if let Some(s) = &mut self.sampler_no_eog {
            s.reset();
        }
        let n_batch = self.batch_capacity();
        let tokens = self.prompt_tokens.clone();
        let mut pos = 0i32;
        for chunk in tokens.chunks(n_batch) {
            self.batch.clear();
            for (i, &t) in chunk.iter().enumerate() {
                let last = pos as usize + i + 1 == n;
                self.batch
                    .add(t, pos + i as i32, &[0], last)
                    .map_err(|e| Error::internal(format!("batch add: {e}")))?;
            }
            self.ctx
                .as_mut()
                .expect("context")
                .decode(&mut self.batch)
                .map_err(|e| Error::runtime(format!("llama.cpp decode failed: {e}")))?;
            pos += chunk.len() as i32;
        }
        self.n_past = pos;
        self.generated = 0;
        self.hold.clear();
        self.pending.clear();
        self.prompted = true;
        self.done = false;
        Ok(())
    }

    fn batch_capacity(&self) -> usize {
        (self.inner.info.max_seq as usize).min(2048)
    }

    fn finish(&mut self, out: &mut Chunk, reason: FinishReason) {
        // Whatever was held back for stop-string matching is text the
        // model produced; the stream ends with it, and so does a trailing
        // partial character, as U+FFFD.
        let tail = self.drain_decodable(true);
        self.hold.push_str(&tail);
        out.text.push_str(&self.hold);
        self.hold.clear();
        out.done = true;
        out.finish_reason = reason;
        self.done = true;
    }

    /// Hand out the held text except the last `hold_max` bytes, cut on a
    /// character boundary.
    fn release_held(&mut self, out: &mut Chunk) {
        if self.hold.len() <= self.hold_max {
            return;
        }
        let mut cut = self.hold.len() - self.hold_max;
        while cut > 0 && !self.hold.is_char_boundary(cut) {
            cut -= 1;
        }
        out.text.push_str(&self.hold[..cut]);
        self.hold.drain(..cut);
    }

    /// Append a token's bytes and return the text that became decodable.
    /// A control token (a chat delimiter the model emits mid-stream) has
    /// no text and contributes nothing; a piece longer than the first
    /// buffer is fetched again at the size llama.cpp reports.
    fn piece(&mut self, token: LlamaToken) -> Result<String> {
        use llama_cpp_2::token_type::LlamaTokenAttr;
        use llama_cpp_2::TokenToStringError;
        let attrs = self.inner.model.token_attr(token);
        if attrs.0.contains(LlamaTokenAttr::Control) {
            return Ok(String::new());
        }
        let bytes = match self.inner.model.token_to_piece_bytes(token, 16, false, None) {
            Ok(b) => b,
            Err(TokenToStringError::InsufficientBufferSpace(need)) => self
                .inner
                .model
                .token_to_piece_bytes(token, need.unsigned_abs() as usize, false, None)
                .map_err(|e| Error::internal(format!("token {} has no piece at {} bytes: {e}", token.0, need.unsigned_abs())))?,
            Err(e) => return Err(Error::internal(format!("token {} has no piece: {e}", token.0))),
        };
        self.pending.extend_from_slice(&bytes);
        Ok(self.drain_decodable(false))
    }

    /// Text from `pending` that is complete UTF-8. An incomplete trailing
    /// sequence waits for the next token unless `flush` is set; bytes that
    /// can never be valid become U+FFFD, so one bad byte cannot wedge the
    /// stream.
    fn drain_decodable(&mut self, flush: bool) -> String {
        let mut text = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    text.push_str(s);
                    self.pending.clear();
                    return text;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    text.push_str(std::str::from_utf8(&self.pending[..valid]).expect("validated prefix"));
                    match e.error_len() {
                        Some(bad) => {
                            text.push('\u{FFFD}');
                            self.pending.drain(..valid + bad);
                        }
                        None => {
                            self.pending.drain(..valid);
                            if flush && !self.pending.is_empty() {
                                text.push('\u{FFFD}');
                                self.pending.clear();
                            }
                            return text;
                        }
                    }
                }
            }
        }
    }

    /// Top-`k` log probabilities of the model's own distribution for the
    /// last decode: the log-softmax of the raw logits, before the sampler
    /// chain (temperature, top-k/p, penalties, logit bias, grammar) and in
    /// descending order without token ids. That is what the ABI's
    /// `logprobs` field carries (see the provider README); it is not the
    /// probability the sampled token was drawn with.
    fn top_logprobs(&mut self, k: u32, out: &mut Vec<f32>) {
        let idx = self.batch.n_tokens() - 1;
        let logits = self.ctx().get_logits_ith(idx);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = max + logits.iter().map(|l| (l - max).exp()).sum::<f32>().ln();
        let k = (k as usize).min(logits.len());
        if k == 0 {
            return;
        }
        let mut all: Vec<f32> = logits.iter().map(|l| l - lse).collect();
        all.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
        let top = &mut all[..k];
        top.sort_by(|a, b| b.total_cmp(a));
        out.extend_from_slice(top);
    }
}

/// Most tokens one embedding decode carries. llama.cpp's encoder needs a
/// whole decode inside one micro-batch and its compute buffer grows with
/// the square of the micro-batch, so a run is split into groups of rows
/// that fit this many tokens (or one sequence, when that is larger).
const EMBED_UBATCH_CAP: u32 = 4096;

/// A seed from the OS for generations that name none; never llama.cpp's
/// 0xFFFFFFFF "draw one" sentinel, which would re-randomize every prefill.
fn fresh_seed() -> u32 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0),
    );
    match (h.finish() & 0xFFFF_FFFF) as u32 {
        u32::MAX => u32::MAX - 1,
        s => s,
    }
}

/// Bytes to hold back so no stop string can be partly delivered: the
/// longest stop string minus one byte (a match needs the new piece).
fn hold_max_of(stop: &[String]) -> usize {
    stop.iter().map(|s| s.len()).max().unwrap_or(0).saturating_sub(1)
}

impl ProviderGeneration for Generation {
    fn prompt(&mut self, messages: &[Message<'_>]) -> Result<()> {
        let chat: Vec<LlamaChatMessage> = messages
            .iter()
            .map(|m| LlamaChatMessage::new(m.role.to_string(), m.content.to_string()))
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| Error::invalid_argument(format!("message contains a NUL byte: {e}")))?;
        let template = self.inner.template.as_ref().expect("generative model");
        let text = self
            .inner
            .model
            .apply_chat_template(template, &chat, true)
            .map_err(|e| Error::invalid_argument(format!("chat template could not render the messages: {e}")))?;
        let tokens = self
            .inner
            .model
            .str_to_token(&text, AddBos::Never)
            .map_err(|e| Error::invalid_argument(format!("prompt tokenization failed: {e}")))?;
        self.prompt_text = text;
        self.prompt_tokens = tokens;
        self.prefill()
    }

    fn prompt_tokens(&mut self, ids: &[i32]) -> Result<()> {
        let vocab = self.inner.model.n_vocab();
        for &id in ids {
            if id < 0 || id >= vocab {
                return Err(Error::invalid_argument(format!("prompt token {id} is outside the vocabulary of {vocab}")));
            }
        }
        self.prompt_tokens = ids.iter().map(|&i| LlamaToken(i)).collect();
        // Text of the prompt (for `echo`), byte-accurate across token boundaries.
        let mut bytes = Vec::new();
        for &t in &self.prompt_tokens {
            if let Ok(piece) = self.inner.model.token_to_piece_bytes(t, 16, true, None) {
                bytes.extend_from_slice(&piece);
            }
        }
        self.prompt_text = String::from_utf8_lossy(&bytes).into_owned();
        self.prefill()
    }

    fn step(&mut self, out: &mut Chunk) -> Result<()> {
        out.prompt_tokens = self.prompt_tokens.len() as u32;
        out.generated_tokens = self.generated;
        if !self.prompted {
            return Err(Error::invalid_state("no prompt has been given"));
        }
        if self.done {
            return Err(Error::invalid_state("generation already finished"));
        }
        if self.cancelled {
            self.finish(out, FinishReason::Cancelled);
            return Ok(());
        }
        if self.generated >= self.plan.max_new {
            self.finish(out, FinishReason::Length);
            return Ok(());
        }
        if self.n_past as u32 >= self.inner.info.max_seq {
            self.finish(out, FinishReason::Length);
            return Ok(());
        }
        let idx = self.batch.n_tokens() - 1;
        if self.plan.logprobs > 0 {
            let k = self.plan.logprobs;
            let mut lp = Vec::new();
            self.top_logprobs(k, &mut lp);
            out.logprobs.extend(lp);
        }
        let token = {
            let ctx = self.ctx.as_ref().expect("context");
            let use_no_eog = self.generated < self.plan.min_new;
            match (&mut self.sampler_no_eog, use_no_eog) {
                (Some(s), true) => {
                    let t = s.sample(ctx, idx);
                    self.sampler.accept(t);
                    t
                }
                _ => self.sampler.sample(ctx, idx),
            }
        };
        self.generated += 1;
        out.generated_tokens = self.generated;
        out.tokens.push(token.0);
        if self.generated == 1 && self.plan.echo {
            out.text.push_str(&self.prompt_text);
        }
        if self.inner.model.is_eog_token(token) {
            self.finish(out, FinishReason::Eos);
            return Ok(());
        }
        if self.plan.stop_tokens.contains(&token.0) {
            // Like EOS: the token ends the stream and its text is withheld.
            self.finish(out, FinishReason::Stop);
            return Ok(());
        }
        let piece = self.piece(token)?;
        self.hold.push_str(&piece);
        // A stop string can end anywhere inside the new piece, not only at
        // its end, and the earliest match wins. Only the bytes this piece
        // could complete are searched: the piece plus `hold_max` before it
        // (everything earlier was searched when it arrived).
        let mut from = self.hold.len().saturating_sub(piece.len() + self.hold_max);
        while from > 0 && !self.hold.is_char_boundary(from) {
            from -= 1;
        }
        let mut earliest: Option<usize> = None;
        for s in self.plan.stop.iter().filter(|s| !s.is_empty()) {
            if let Some(i) = self.hold[from..].find(s.as_str()) {
                let at = from + i;
                if earliest.is_none_or(|e| at < e) {
                    earliest = Some(at);
                }
            }
        }
        if let Some(at) = earliest {
            // Everything before the match goes out; the match and whatever
            // followed it in the same piece are withheld.
            out.text.push_str(&self.hold[..at]);
            self.hold.clear();
            self.pending.clear();
            out.done = true;
            out.finish_reason = FinishReason::Stop;
            self.done = true;
            return Ok(());
        }
        self.release_held(out);
        // Feed the token back for the next step.
        self.batch.clear();
        self.batch.add(token, self.n_past, &[0], true).map_err(|e| Error::internal(format!("batch add: {e}")))?;
        self.n_past += 1;
        self.ctx
            .as_mut()
            .expect("context")
            .decode(&mut self.batch)
            .map_err(|e| Error::runtime(format!("llama.cpp decode failed: {e}")))?;
        if self.generated >= self.plan.max_new {
            self.finish(out, FinishReason::Length);
        }
        Ok(())
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }
}

// ---------------------------------------------------------------------------
// Embedding session
// ---------------------------------------------------------------------------

/// A fixed-shape embedding workspace: one llama context sized for
/// `max_batch * max_seq` tokens with pooling in the graph.
struct EmbedSession {
    // Field order is load-bearing: the context borrows the model in `inner`.
    ctx: Option<LlamaContext<'static>>,
    batch: LlamaBatch<'static>,
    rows: Vec<Vec<LlamaToken>>,
    /// Tokens one decode call may carry (the context's micro-batch).
    ubatch: usize,
    out: Arc<HostBuffer>,
    opts: EmbedOptions,
    n_rows: u32,
    runs: u64,
    d2h: u64,
    max_batch: u32,
    max_seq: u32,
    name: Arc<str>,
    inner: Arc<ModelInner>,
}

// SAFETY: sessions are single-owner by the core's contract.
unsafe impl Send for EmbedSession {}

impl EmbedSession {
    fn new(inner: Arc<ModelInner>, max_batch: u32, max_seq: u32) -> Result<Self> {
        let embed = inner.embed.expect("embedding model");
        let pooling = match embed.pool {
            Pooling::Mean => LlamaPoolingType::Mean,
            Pooling::Cls => LlamaPoolingType::Cls,
            Pooling::Last => LlamaPoolingType::Last,
            Pooling::Model => unreachable!("the bundle names a pooling"),
        };
        let n_tokens =
            max_batch.checked_mul(max_seq).ok_or_else(|| Error::invalid_shape("max_batch * max_seq overflows"))?;
        // llama.cpp's encoder path requires every token of a decode call to
        // fit one micro-batch (`n_ubatch >= n_tokens`, a hard assert), and
        // its compute buffer (the attention mask included) grows with the
        // square of the micro-batch. So the micro-batch is capped at
        // `EMBED_UBATCH_CAP` tokens (never below one sequence) and a run
        // decodes its rows in groups that fit; the context only ever holds
        // one group.
        let ubatch = n_tokens.min(EMBED_UBATCH_CAP).max(max_seq);
        // `n_ctx` is per context and llama.cpp divides it by `n_seq_max`
        // for each sequence's KV stream, so it must hold every row of the
        // batch at `max_seq`, not one micro-batch; BERT encoders keep no KV
        // cache and are indifferent, decoder-style embedders are not.
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_tokens))
            .with_n_batch(ubatch)
            .with_n_ubatch(ubatch)
            .with_n_seq_max(max_batch)
            .with_embeddings(true)
            .with_pooling_type(pooling);
        let ctx = inner.model.new_context(backend()?, params).map_err(|e| {
            Error::runtime(format!("llama.cpp could not create an embedding context of {ubatch} tokens: {e}"))
        })?;
        if ctx.n_ctx() < n_tokens {
            return Err(Error::runtime(format!(
                "llama.cpp created an embedding context of {} tokens; {n_tokens} ({max_batch} rows of {max_seq}) are needed",
                ctx.n_ctx()
            )));
        }
        // SAFETY: as for `Generation`: `inner` outlives the context, which is
        // declared first so it drops first.
        let ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
        Ok(Self {
            ctx: Some(ctx),
            batch: LlamaBatch::new(ubatch as usize, max_batch as i32),
            ubatch: ubatch as usize,
            rows: (0..max_batch).map(|_| Vec::with_capacity(max_seq as usize)).collect(),
            out: HostBuffer::packed(DType::F32, &[max_batch as u64, embed.dim as u64])?,
            opts: EmbedOptions::default(),
            n_rows: 0,
            runs: 0,
            d2h: 0,
            max_batch,
            max_seq,
            name: Arc::from("embeddings"),
            inner,
        })
    }

    fn budget(&self, max_tokens: u32) -> Result<u32> {
        let b = if max_tokens == 0 { self.max_seq } else { max_tokens };
        if b > self.max_seq {
            return Err(Error::capacity(format!("max_tokens {b} exceeds the session's max_seq {}", self.max_seq)));
        }
        if b < 2 {
            return Err(Error::capacity("token budget must be at least 2 for the special tokens"));
        }
        Ok(b)
    }
}

impl ProviderSession for EmbedSession {
    fn write_text(&mut self, texts: &[&str], opts: &EmbedOptions) -> Result<()> {
        if texts.is_empty() || texts.len() > self.max_batch as usize {
            return Err(Error::capacity(format!(
                "{} inputs written but the session holds 1..{} rows",
                texts.len(),
                self.max_batch
            )));
        }
        if !matches!(opts.output_dtype, OutputDType::Model | OutputDType::F32) {
            return Err(Error::unsupported_option(EmbedOptions::FIELD_OUTPUT_DTYPE, "output_dtype", GGML_PROVIDER_ID));
        }
        if opts.pooling != Pooling::Model && Some(opts.pooling) != self.inner.info.pooling {
            return Err(Error::unsupported_option(EmbedOptions::FIELD_POOLING, "pooling", GGML_PROVIDER_ID));
        }
        let dim = self.inner.info.dim;
        if opts.output_dim > dim {
            return Err(Error::invalid_argument(format!(
                "output_dim {} exceeds the model dimension {dim}",
                opts.output_dim
            ))
            .with_field(EmbedOptions::FIELD_OUTPUT_DIM));
        }
        let budget = self.budget(opts.max_tokens)? as usize;
        let prefix = match opts.prompt_role {
            PromptRole::None => "",
            PromptRole::Query => self.inner.info.prefix_query.as_str(),
            PromptRole::Document => self.inner.info.prefix_document.as_str(),
        };
        let model = self.inner.clone();
        for (r, text) in texts.iter().enumerate() {
            // llama.cpp adds the model's special tokens ([CLS] ... [SEP]).
            let full = if prefix.is_empty() { (*text).to_string() } else { format!("{prefix}{text}") };
            let tokens = model
                .model
                .str_to_token(&full, AddBos::Always)
                .map_err(|e| Error::invalid_argument(format!("row {r} could not be tokenized: {e}")))?;
            let row = &mut self.rows[r];
            row.clear();
            if tokens.len() <= budget {
                row.extend_from_slice(&tokens);
            } else {
                // Keep the leading special token, and the trailing one when
                // the vocabulary added one (BERT's [SEP]; a decoder-style
                // embedder adds BOS only and its last token is content), and
                // cut the content between them. The prompt prefix sits at
                // the front of the content, so left truncation would drop
                // it; that combination is refused rather than embedded in
                // the wrong subspace.
                if opts.truncate == Truncate::Left && !prefix.is_empty() {
                    return Err(Error::invalid_argument(format!(
                        "input row {r} needs truncation and truncate LEFT would remove the prompt prefix `{prefix}`; use RIGHT or a shorter input"
                    ))
                    .with_field(EmbedOptions::FIELD_TRUNCATE));
                }
                let first = tokens[0];
                let last_is_special = model
                    .model
                    .token_attr(tokens[tokens.len() - 1])
                    .0
                    .contains(llama_cpp_2::token_type::LlamaTokenAttr::Control);
                let last = if last_is_special { Some(tokens[tokens.len() - 1]) } else { None };
                let content = if last_is_special { &tokens[1..tokens.len() - 1] } else { &tokens[1..] };
                let keep = budget - 1 - usize::from(last_is_special);
                let kept = match opts.truncate {
                    Truncate::None => {
                        return Err(Error::capacity(format!(
                            "input row {r} tokenizes to {} tokens but the budget is {budget} and truncation is NONE",
                            tokens.len()
                        )))
                    }
                    Truncate::Left => &content[content.len() - keep..],
                    Truncate::Right | Truncate::Model => &content[..keep],
                };
                row.push(first);
                row.extend_from_slice(kept);
                if let Some(l) = last {
                    row.push(l);
                }
            }
        }
        self.opts = *opts;
        self.n_rows = texts.len() as u32;
        Ok(())
    }

    fn write_tokens(&mut self, batch: &TokenBatch<'_>) -> Result<()> {
        if batch.batch > self.max_batch || batch.seq > self.max_seq {
            return Err(Error::capacity(format!(
                "token batch [{}, {}] exceeds the session shape [{}, {}]",
                batch.batch, batch.seq, self.max_batch, self.max_seq
            )));
        }
        for r in 0..batch.batch as usize {
            let ids = batch.ids_row(r);
            let mask = batch.mask_row(r);
            // The mask is honored as trailing padding: live tokens, then
            // zeros. A zero inside the live run would have to be dropped or
            // attended to, and llama.cpp offers neither for an encoder, so
            // it is refused rather than reinterpreted.
            let live = mask.iter().position(|&m| m == 0).unwrap_or(mask.len());
            if let Some(bad) = mask[live..].iter().position(|&m| m != 0) {
                return Err(Error::invalid_argument(format!(
                    "token batch row {r} has a masked position at column {live} followed by a live token at column {}; \
                     the ggml provider takes the mask as trailing padding only",
                    live + bad
                )));
            }
            if live == 0 {
                return Err(Error::invalid_argument(format!("token batch row {r} has no live tokens")));
            }
            if let Some(types) = batch.types {
                let stride = batch.row_stride as usize;
                if types[r * stride..r * stride + live].iter().any(|&t| t != 0) {
                    return Err(Error::unsupported(format!(
                        "token batch row {r} carries token_type_ids other than 0; GGUF encoders take no token types"
                    )));
                }
            }
            let row = &mut self.rows[r];
            row.clear();
            row.extend(ids[..live].iter().map(|&id| LlamaToken(id)));
        }
        self.opts = EmbedOptions::default();
        self.n_rows = batch.batch;
        Ok(())
    }

    fn run(&mut self, opts: &RunOptions) -> Result<ProviderResult> {
        opts.params.reject_unknown(&[], "ggml run")?;
        if self.n_rows == 0 {
            return Err(Error::invalid_state("no inputs written"));
        }
        let n = self.n_rows as usize;
        let embed = self.inner.embed.expect("embedding model");
        let dim = embed.dim as usize;
        let dim_out = if self.opts.output_dim == 0 { dim } else { self.opts.output_dim as usize };
        let normalize = match self.opts.normalize {
            Normalize::Model => embed.normalize,
            Normalize::L2 => true,
            Normalize::None => false,
        };
        let ctx = self.ctx.as_mut().expect("context");
        // SAFETY: the core holds the session lock with no result lease outstanding.
        let out = unsafe { self.out.as_f32_mut()? };
        // Rows are decoded in groups whose token total fits the micro-batch;
        // a row is never longer than one sequence, so every group holds at
        // least one row.
        let mut start = 0usize;
        while start < n {
            let mut end = start;
            let mut total = 0usize;
            while end < n && (end == start || total + self.rows[end].len() <= self.ubatch) {
                total += self.rows[end].len();
                end += 1;
            }
            if total > self.ubatch {
                return Err(Error::internal(format!(
                    "row {start} holds {total} tokens but the micro-batch holds {}",
                    self.ubatch
                )));
            }
            self.batch.clear();
            for r in start..end {
                self.batch
                    .add_sequence(&self.rows[r], (r - start) as i32, true)
                    .map_err(|e| Error::internal(format!("batch add for row {r}: {e}")))?;
            }
            ctx.clear_kv_cache();
            ctx.decode(&mut self.batch).map_err(|e| Error::runtime(format!("llama.cpp decode failed: {e}")))?;
            for r in start..end {
                let v = ctx
                    .embeddings_seq_ith((r - start) as i32)
                    .map_err(|e| Error::runtime(format!("llama.cpp returned no embedding for row {r}: {e}")))?;
                if v.len() != dim {
                    return Err(Error::internal(format!(
                        "llama.cpp returned {} values for row {r}, expected {dim}",
                        v.len()
                    )));
                }
                let dst = &mut out[r * dim_out..(r + 1) * dim_out];
                dst.copy_from_slice(&v[..dim_out]);
                if normalize {
                    let norm = dst.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm <= 1e-12 {
                        return Err(Error::runtime(format!(
                            "row {r} pooled to a zero vector, which cannot be L2-normalized as the contract promises"
                        )));
                    }
                    for x in dst.iter_mut() {
                        *x /= norm;
                    }
                }
            }
            start = end;
        }
        if self.inner.device_kind != DeviceKind::Cpu {
            // llama.cpp copied the full pooled vectors (the model dimension)
            // from the device; `output_dim` is a host-side cut afterwards,
            // so the counter reports what crossed the bus.
            self.d2h += (n * dim * 4) as u64;
        }
        self.runs += 1;
        Ok(ProviderResult {
            outputs: vec![Output {
                name: self.name.clone(),
                buffer: self.out.clone(),
                shape: vec![n as u64, dim_out as u64],
            }],
            spans: Vec::new(),
        })
    }

    fn stats(&self) -> Result<SessionStats> {
        Ok(SessionStats {
            runs: self.runs,
            // Not counted: the result API and llama.cpp both allocate on
            // the run path (see the CUDA provider's note).
            host_allocs: None,
            h2d_bytes: 0,
            d2h_bytes: self.d2h,
            input_bytes: (self.max_batch as u64) * (self.max_seq as u64) * 4,
            output_bytes: self.out.desc().bytes,
            // llama.cpp's own allocations are not observable here.
            provider_allocs: None,
        })
    }
}

turbo_core::export_provider!(c"ggml", c"2.0.0-alpha.0", || Arc::new(GgmlProvider::new()));
