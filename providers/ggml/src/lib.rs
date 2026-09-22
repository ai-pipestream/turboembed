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
//! Honesty: structured output is honored for GBNF grammars only (a JSON
//! schema is refused naming the field), `n_sequences > 1` and tool
//! definitions are not offered, and embeddings through GGUF are not offered
//! yet. Every cell is `EXPERIMENTAL` until receipts land.

#![deny(missing_docs)]

use std::ffi::CStr;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex, OnceLock};

use llama_cpp_2::context::params::LlamaContextParams;
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
    Capability, Chunk, ContextDesc, DeviceInfo, GenerateDesc, Message, ModelDesc, ModelInfo, Provider, ProviderContext,
    ProviderGeneration, ProviderModel, ProviderSession, SessionDesc,
};
use turbo_core::types::{
    CapStatus, DeviceKind, FinishReason, Modality, ModelKind, Placement, Stage, StagePlacement, StagePlacements,
    StructuredKind, Task,
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
        if task != Task::Generate || modality != Modality::Text {
            return Capability::unsupported();
        }
        Capability {
            status: CapStatus::Experimental,
            dtype: None,
            reference_dtype: None,
            cosine_floor: 0.0,
            max_abs_error: 0.0,
            deterministic: d.kind == DeviceKind::Cpu,
            notes: format!("llama.cpp on {}; GGUF quantization decides the compute dtype; receipts pending", d.name),
        }
    }

    fn can_run(&self, ordinal: u32, bundle: &Bundle, task: Task, modality: Modality) -> Result<()> {
        self.device(ordinal)?;
        if task != Task::Generate || modality != Modality::Text {
            return Err(Error::unsupported_task(format!(
                "ggml provider offers GENERATE on TEXT, not {task:?} x {modality:?}"
            )));
        }
        if bundle.artifact(GGUF_ARTIFACT).is_none() {
            return Err(Error::bundle_no_artifact(format!(
                "bundle `{}` has no `gguf` artifact",
                bundle.manifest().model_id
            )));
        }
        if bundle.kind() != ModelKind::Generative {
            return Err(Error::unsupported_task(format!(
                "ggml provider serves generative bundles, not {:?}",
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
        if bundle.kind() != ModelKind::Generative {
            return Err(Error::unsupported_task(format!(
                "ggml provider serves generative bundles, not {:?}",
                bundle.kind()
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
        // The bundle's chat template wins over the GGUF metadata's.
        let template = match bundle.manifest().tokenizer.as_ref().and_then(|t| t.chat_template.clone()) {
            Some(t) => LlamaChatTemplate::new(&t)
                .map_err(|e| Error::bundle_invalid(format!("tokenizer.chat_template is not usable: {e}")))?,
            None => model.chat_template(None).map_err(|e| {
                Error::bundle_invalid(format!(
                    "bundle `{}` declares no chat template and the GGUF carries none: {e}",
                    bundle.manifest().model_id
                ))
            })?,
        };
        let placement = if device.kind == DeviceKind::Cpu { StagePlacement::Host } else { StagePlacement::Device };
        let info = ModelInfo {
            task: Task::Generate,
            kind: ModelKind::Generative,
            modality: Modality::Text,
            dim: 0,
            labels: Vec::new(),
            pooling: None,
            normalize: None,
            aggregation: None,
            max_seq,
            max_batch: 1,
            dtype_used: None,
            stages: StagePlacements::NONE
                .with(Stage::Tokenize, StagePlacement::Host)
                .with(Stage::Encode, placement)
                .with(Stage::Postprocess, StagePlacement::Host),
            inputs: Vec::new(),
            outputs: Vec::new(),
            vocab_size: model.n_vocab().max(0) as u32,
            model_id: bundle.manifest().model_id.clone(),
            revision: bundle.manifest().revision.clone(),
            tokenizer_sha256: bundle.tokenizer_sha256().to_string(),
            provider_id: GGML_PROVIDER_ID.into(),
            prefix_query: String::new(),
            prefix_document: String::new(),
        };
        Ok(Arc::new(GgmlModel { inner: Arc::new(ModelInner { model, template, info, _ctx: self.inner.clone() }) }))
    }
}

struct ModelInner {
    model: LlamaModel,
    template: LlamaChatTemplate,
    info: ModelInfo,
    _ctx: Arc<ContextInner>,
}

struct GgmlModel {
    inner: Arc<ModelInner>,
}

impl ProviderModel for GgmlModel {
    fn info(&self) -> &ModelInfo {
        &self.inner.info
    }

    fn create_session(&self, _desc: &SessionDesc) -> Result<Box<dyn ProviderSession>> {
        Err(Error::unsupported_task(format!(
            "model `{}` is generative; create a generation, not a session",
            self.inner.info.model_id
        )))
    }

    fn create_generation(&self, desc: &GenerateDesc) -> Result<Box<dyn ProviderGeneration>> {
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
        if desc.top_k != 0 && desc.top_k as i32 > vocab {
            return Err(Error::invalid_argument(format!("top_k {} exceeds the vocabulary of {vocab}", desc.top_k))
                .with_field(6));
        }
        Ok(Self {
            max_new,
            min_new: desc.min_new_tokens,
            stop: desc.stop.clone(),
            stop_tokens: desc.stop_tokens.clone(),
            logprobs: desc.logprobs,
            echo: desc.echo,
            seed: desc.seed.map(|s| (s & 0xFFFF_FFFF) as u32).unwrap_or(0xDEAD_BEEF),
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
    text_acc: String,
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
            text_acc: String::new(),
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
        self.text_acc.clear();
        self.pending.clear();
        self.prompted = true;
        self.done = false;
        Ok(())
    }

    fn batch_capacity(&self) -> usize {
        (self.inner.info.max_seq as usize).min(2048)
    }

    fn finish(&mut self, out: &mut Chunk, reason: FinishReason) {
        out.done = true;
        out.finish_reason = reason;
        self.done = true;
    }

    /// Append a token's bytes and return the text that became decodable.
    fn piece(&mut self, token: LlamaToken) -> Result<String> {
        let bytes = self
            .inner
            .model
            .token_to_piece_bytes(token, 16, false, None)
            .map_err(|e| Error::internal(format!("token {} has no piece: {e}", token.0)))?;
        self.pending.extend_from_slice(&bytes);
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(e) => e.valid_up_to(),
        };
        let text = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
        self.pending.drain(..valid);
        Ok(text)
    }

    /// Top-`k` log probabilities of the distribution the last decode produced.
    fn top_logprobs(&mut self, k: u32, out: &mut Vec<f32>) {
        let idx = self.batch.n_tokens() - 1;
        let logits = self.ctx().get_logits_ith(idx);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = max + logits.iter().map(|l| (l - max).exp()).sum::<f32>().ln();
        let mut top: Vec<f32> = logits.iter().map(|l| l - lse).collect();
        top.sort_by(|a, b| b.total_cmp(a));
        out.extend(top.into_iter().take(k as usize));
    }
}

impl ProviderGeneration for Generation {
    fn prompt(&mut self, messages: &[Message<'_>]) -> Result<()> {
        let chat: Vec<LlamaChatMessage> = messages
            .iter()
            .map(|m| LlamaChatMessage::new(m.role.to_string(), m.content.to_string()))
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| Error::invalid_argument(format!("message contains a NUL byte: {e}")))?;
        let text = self
            .inner
            .model
            .apply_chat_template(&self.inner.template, &chat, true)
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
        let piece = self.piece(token)?;
        out.text.push_str(&piece);
        self.text_acc.push_str(&piece);
        if self.plan.stop_tokens.contains(&token.0) {
            self.finish(out, FinishReason::Stop);
            return Ok(());
        }
        if let Some(s) = self.plan.stop.iter().find(|s| !s.is_empty() && self.text_acc.ends_with(s.as_str())) {
            // Hand back the text before the stop string only.
            let keep = out.text.len().saturating_sub(s.len());
            out.text.truncate(keep);
            self.finish(out, FinishReason::Stop);
            return Ok(());
        }
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

turbo_core::export_provider!(c"ggml", c"2.0.0-alpha.0", || Arc::new(GgmlProvider::new()));
