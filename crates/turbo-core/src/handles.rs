//! Owned handles with lifetime, lease, and option enforcement.
//!
//! Context → Model → Session → Result, and Model → Generation. Every child
//! holds an `Arc` to its parent, so releasing a parent never invalidates a
//! child. Sessions and generations are single-owner: a concurrent call
//! returns `TURBO_E_BUSY` without touching provider state. A result leases
//! the session's output storage; writes and runs on that session return
//! `TURBO_E_BUSY` until the result is released.
//!
//! Option validation against the capability matrix happens here, once, for
//! every provider: an option that differs from the model contract and is not
//! covered by a capability bit fails with `TURBO_E_UNSUPPORTED_OPTION`
//! naming the field.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

use turbo_abi as abi;

use crate::buffer::{BufferDesc, NativeHandle, ProviderBuffer};
use crate::bundle::Bundle;
use crate::error::{Error, Result};
use crate::provider::{
    Chunk, ClassifyOptions, ContextDesc, DeviceInfo, EmbedOptions, GenerateDesc, Message, ModelDesc, ModelInfo,
    Options, Provider, ProviderContext, ProviderGeneration, ProviderModel, ProviderResult, ProviderSession,
    RerankOptions, RunOptions, SessionDesc, SessionStats, Span, TokenBatch,
};
use crate::runtime::Runtime;
use crate::types::{
    Aggregation, HandleKind, ModelKind, Normalize, OutputDType, Pooling, PromptRole, StructuredKind, Task, Truncate,
};

/// Device plus memory domain.
pub struct Context {
    runtime: Arc<Runtime>,
    device_index: u32,
    provider: Arc<dyn Provider>,
    inner: Arc<dyn ProviderContext>,
}

impl Context {
    /// Create a context on a device of the runtime.
    pub fn create(runtime: Arc<Runtime>, device_index: u32, desc: &ContextDesc) -> Result<Arc<Self>> {
        let entry = runtime.device(device_index)?.clone();
        let provider = runtime.provider_for(device_index)?.clone();
        let inner = provider.create_context(entry.info.ordinal, desc)?;
        Ok(Arc::new(Self { runtime, device_index, provider, inner }))
    }

    /// Owning runtime.
    pub fn runtime(&self) -> &Arc<Runtime> {
        &self.runtime
    }

    /// Device index within the runtime.
    pub fn device_index(&self) -> u32 {
        self.device_index
    }

    /// Static device info.
    pub fn device_info(&self) -> &DeviceInfo {
        &self.runtime.devices()[self.device_index as usize].info
    }

    /// Provider.
    pub fn provider(&self) -> &Arc<dyn Provider> {
        &self.provider
    }

    /// Allocate a buffer.
    pub fn alloc(self: &Arc<Self>, desc: &BufferDesc) -> Result<Arc<Buffer>> {
        let inner = self.inner.alloc(desc)?;
        Ok(Arc::new(Buffer { context: self.clone(), inner, lease: None }))
    }

    /// Import caller memory.
    pub fn import(self: &Arc<Self>, desc: &BufferDesc, handle: &NativeHandle) -> Result<Arc<Buffer>> {
        if self.device_info().caps & abi::TURBO_CAP_HOST_PTR_IMPORT == 0 && handle.kind == HandleKind::HostPtr {
            return Err(Error::unsupported(format!(
                "device `{}` does not advertise TURBO_CAP_HOST_PTR_IMPORT",
                self.device_info().name
            )));
        }
        let inner = self.inner.import(desc, handle)?;
        Ok(Arc::new(Buffer { context: self.clone(), inner, lease: None }))
    }

    /// Load a bundle directory.
    pub fn load_model(self: &Arc<Self>, bundle_dir: &Path, desc: &ModelDesc) -> Result<Arc<Model>> {
        let bundle = Arc::new(Bundle::open(bundle_dir)?);
        self.load_bundle(bundle, desc)
    }

    /// Load an already-verified bundle.
    pub fn load_bundle(self: &Arc<Self>, bundle: Arc<Bundle>, desc: &ModelDesc) -> Result<Arc<Model>> {
        let ordinal = self.device_info().ordinal;
        let cap = self.provider.capability(ordinal, bundle.task(), bundle.modality());
        if !cap.is_offered() {
            return Err(Error::unsupported_task(format!(
                "device `{}` (provider `{}`) does not offer {:?} for {:?}; bundle `{}` cannot load here",
                self.device_info().name,
                self.provider.id(),
                bundle.task(),
                bundle.modality(),
                bundle.manifest().model_id
            )));
        }
        self.provider.can_run(ordinal, &bundle, bundle.task(), bundle.modality())?;
        let inner = self.inner.load_model(bundle.clone(), desc)?;
        let info = inner.info();
        if info.provider_id != self.provider.id() {
            return Err(Error::internal(format!(
                "provider `{}` returned a model claiming provider `{}`",
                self.provider.id(),
                info.provider_id
            )));
        }
        if info.max_seq == 0 || info.max_batch == 0 {
            return Err(Error::internal(format!(
                "provider `{}` returned max_seq {} and max_batch {}; both must be non-zero",
                self.provider.id(),
                info.max_seq,
                info.max_batch
            )));
        }
        Ok(Arc::new(Model { context: self.clone(), bundle, inner }))
    }
}

/// Typed memory. Retains its context, and its result lease when it came from a result.
pub struct Buffer {
    context: Arc<Context>,
    inner: Arc<dyn ProviderBuffer>,
    lease: Option<Arc<ResultHandle>>,
}

impl Buffer {
    /// Description.
    pub fn desc(&self) -> &BufferDesc {
        self.inner.desc()
    }

    /// Owning context.
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Provider buffer.
    pub fn provider_buffer(&self) -> &Arc<dyn ProviderBuffer> {
        &self.inner
    }

    /// Host pointer for host-visible placements.
    pub fn host_ptr(&self) -> Option<std::ptr::NonNull<u8>> {
        self.inner.host_ptr()
    }

    /// Whether this buffer is leased from a result.
    pub fn is_result_view(&self) -> bool {
        self.lease.is_some()
    }

    /// Export a native handle.
    pub fn export(&self, kind: HandleKind) -> Result<NativeHandle> {
        self.inner.export(kind)
    }

    /// Blocking copy to host.
    pub fn read_to_host(&self, dst: &mut [u8]) -> Result<()> {
        self.inner.read_to_host(dst)
    }
}

/// Loaded model. Immutable.
pub struct Model {
    context: Arc<Context>,
    bundle: Arc<Bundle>,
    inner: Arc<dyn ProviderModel>,
}

impl Model {
    /// Owning context.
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Bundle.
    pub fn bundle(&self) -> &Arc<Bundle> {
        &self.bundle
    }

    /// What loaded.
    pub fn info(&self) -> &ModelInfo {
        self.inner.info()
    }

    /// Device capability bits.
    fn caps(&self) -> u64 {
        self.context.device_info().caps
    }

    /// Provider id.
    fn provider_id(&self) -> &str {
        self.context.provider.id()
    }

    /// Label by index.
    pub fn label(&self, index: u32) -> Result<&str> {
        self.info().labels.get(index as usize).map(String::as_str).ok_or_else(|| {
            Error::invalid_argument(format!(
                "label index {index} is out of range; the model has {} labels",
                self.info().labels.len()
            ))
        })
    }

    /// Create a session. Requested maxima default to the model's and may not exceed them.
    pub fn create_session(self: &Arc<Self>, desc: &SessionDesc) -> Result<Arc<Session>> {
        let info = self.info();
        let mut resolved = desc.clone();
        if resolved.max_batch == 0 {
            resolved.max_batch = info.max_batch;
        }
        if resolved.max_seq == 0 {
            resolved.max_seq = info.max_seq;
        }
        if resolved.max_batch > info.max_batch {
            return Err(Error::capacity(format!(
                "session max_batch {} exceeds the model's {}",
                resolved.max_batch, info.max_batch
            ))
            .with_field(2));
        }
        if resolved.max_seq > info.max_seq {
            return Err(Error::capacity(format!(
                "session max_seq {} exceeds the model's {}",
                resolved.max_seq, info.max_seq
            ))
            .with_field(3));
        }
        let inner = self.inner.create_session(&resolved)?;
        Ok(Arc::new(Session {
            model: self.clone(),
            desc: resolved,
            state: Mutex::new(SessionState { inner, inputs_ready: false, written_task: None }),
            lease: AtomicBool::new(false),
        }))
    }

    /// Create a generation, validating gated fields against capabilities.
    pub fn create_generation(self: &Arc<Self>, desc: &GenerateDesc) -> Result<Arc<Generation>> {
        if self.info().kind != ModelKind::Generative {
            return Err(Error::unsupported_task(format!(
                "model `{}` is {:?}, not generative",
                self.info().model_id,
                self.info().kind
            )));
        }
        self.validate_generate(desc)?;
        let inner = self.inner.create_generation(desc)?;
        Ok(Arc::new(Generation {
            model: self.clone(),
            state: Mutex::new(GenState { inner, prompted: false, finished: false, cancelled: false }),
            chunk: Mutex::new(Chunk::default()),
        }))
    }

    fn gated(&self, condition: bool, bit: u64, field: u32, name: &str) -> Result<()> {
        if condition && self.caps() & bit == 0 {
            return Err(Error::unsupported_option(field, name, self.provider_id()));
        }
        Ok(())
    }

    fn validate_generate(&self, d: &GenerateDesc) -> Result<()> {
        self.gated(d.n_sequences > 1, abi::TURBO_CAP_OPT_GEN_N, GenerateDesc::FIELD_N_SEQUENCES, "n_sequences")?;
        self.gated(
            d.repeat_penalty != 0.0 && d.repeat_penalty != 1.0,
            abi::TURBO_CAP_OPT_GEN_PENALTIES,
            GenerateDesc::FIELD_REPEAT_PENALTY,
            "repeat_penalty",
        )?;
        self.gated(
            d.presence_penalty != 0.0,
            abi::TURBO_CAP_OPT_GEN_PENALTIES,
            GenerateDesc::FIELD_PRESENCE_PENALTY,
            "presence_penalty",
        )?;
        self.gated(
            d.frequency_penalty != 0.0,
            abi::TURBO_CAP_OPT_GEN_PENALTIES,
            GenerateDesc::FIELD_FREQUENCY_PENALTY,
            "frequency_penalty",
        )?;
        self.gated(d.seed.is_some(), abi::TURBO_CAP_OPT_GEN_SEED, GenerateDesc::FIELD_HAS_SEED, "seed")?;
        self.gated(!d.stop.is_empty(), abi::TURBO_CAP_OPT_GEN_STOP_STRINGS, GenerateDesc::FIELD_N_STOP, "stop")?;
        self.gated(
            !d.logit_bias.is_empty(),
            abi::TURBO_CAP_OPT_GEN_LOGIT_BIAS,
            GenerateDesc::FIELD_N_LOGIT_BIAS,
            "logit_bias",
        )?;
        self.gated(d.logprobs > 0, abi::TURBO_CAP_OPT_GEN_LOGPROBS, GenerateDesc::FIELD_LOGPROBS, "logprobs")?;
        self.gated(
            d.structured_kind != StructuredKind::None,
            abi::TURBO_CAP_OPT_GEN_STRUCTURED,
            GenerateDesc::FIELD_STRUCTURED_KIND,
            "structured_kind",
        )?;
        self.gated(!d.tools.is_empty(), abi::TURBO_CAP_OPT_GEN_TOOLS, GenerateDesc::FIELD_N_TOOLS, "tools")?;
        if d.structured_kind != StructuredKind::None && d.structured.trim().is_empty() {
            return Err(Error::invalid_argument("structured_kind is set but `structured` text is empty").with_field(23));
        }
        if !(0.0..=2.0).contains(&d.temperature) || d.temperature.is_nan() {
            return Err(
                Error::invalid_argument(format!("temperature {} is outside 0..=2", d.temperature)).with_field(5)
            );
        }
        if !(0.0..=1.0).contains(&d.top_p) || d.top_p.is_nan() {
            return Err(Error::invalid_argument(format!("top_p {} is outside 0..=1", d.top_p)).with_field(7));
        }
        if !(0.0..=1.0).contains(&d.min_p) || d.min_p.is_nan() {
            return Err(Error::invalid_argument(format!("min_p {} is outside 0..=1", d.min_p)).with_field(8));
        }
        Ok(())
    }

    /// Validate embed options against the contract and capabilities.
    pub fn validate_embed(&self, o: &EmbedOptions) -> Result<()> {
        let info = self.info();
        if info.kind != ModelKind::Embedding {
            return Err(Error::unsupported_task(format!(
                "model `{}` is {:?}, not an embedder",
                info.model_id, info.kind
            )));
        }
        self.gated(
            o.truncate != Truncate::Model,
            abi::TURBO_CAP_OPT_TRUNCATE,
            EmbedOptions::FIELD_TRUNCATE,
            "truncate",
        )?;
        self.gated(o.max_tokens != 0, abi::TURBO_CAP_OPT_MAX_TOKENS, EmbedOptions::FIELD_MAX_TOKENS, "max_tokens")?;
        if o.max_tokens > info.max_seq {
            return Err(Error::capacity(format!(
                "max_tokens {} exceeds the model's max_seq {}",
                o.max_tokens, info.max_seq
            ))
            .with_field(EmbedOptions::FIELD_MAX_TOKENS));
        }
        self.gated(
            o.prompt_role != PromptRole::None,
            abi::TURBO_CAP_OPT_PROMPT_ROLE,
            EmbedOptions::FIELD_PROMPT_ROLE,
            "prompt_role",
        )?;
        let differs_norm = o.normalize != Normalize::Model && Some(o.normalize) != info.normalize;
        self.gated(differs_norm, abi::TURBO_CAP_OPT_NORMALIZE, EmbedOptions::FIELD_NORMALIZE, "normalize")?;
        let differs_pool = o.pooling != Pooling::Model && Some(o.pooling) != info.pooling;
        self.gated(differs_pool, abi::TURBO_CAP_OPT_POOLING_OVERRIDE, EmbedOptions::FIELD_POOLING, "pooling")?;
        if o.output_dim != 0 && o.output_dim != info.dim {
            self.gated(true, abi::TURBO_CAP_OPT_OUTPUT_DIM, EmbedOptions::FIELD_OUTPUT_DIM, "output_dim")?;
            let allowed = &self.bundle.contract().truncate_dims;
            if !allowed.contains(&o.output_dim) {
                return Err(Error::invalid_argument(format!(
                    "output_dim {} is not one of the bundle's truncate_dims {:?}",
                    o.output_dim, allowed
                ))
                .with_field(EmbedOptions::FIELD_OUTPUT_DIM));
            }
        }
        let differs_dtype = o.output_dtype != OutputDType::Model && o.output_dtype != OutputDType::F32;
        self.gated(differs_dtype, abi::TURBO_CAP_OPT_OUTPUT_DTYPE, EmbedOptions::FIELD_OUTPUT_DTYPE, "output_dtype")?;
        Ok(())
    }

    /// Validate rerank options.
    pub fn validate_rerank(&self, o: &RerankOptions) -> Result<()> {
        let info = self.info();
        if info.kind != ModelKind::Reranker {
            return Err(Error::unsupported_task(format!(
                "model `{}` is {:?}, not a reranker",
                info.model_id, info.kind
            )));
        }
        self.gated(
            o.truncate != Truncate::Model,
            abi::TURBO_CAP_OPT_TRUNCATE,
            RerankOptions::FIELD_TRUNCATE,
            "truncate",
        )?;
        self.gated(o.max_tokens != 0, abi::TURBO_CAP_OPT_MAX_TOKENS, RerankOptions::FIELD_MAX_TOKENS, "max_tokens")?;
        if o.max_tokens > info.max_seq {
            return Err(Error::capacity(format!(
                "max_tokens {} exceeds the model's max_seq {}",
                o.max_tokens, info.max_seq
            ))
            .with_field(RerankOptions::FIELD_MAX_TOKENS));
        }
        self.gated(o.top_n != 0 || o.return_sorted, abi::TURBO_CAP_OPT_TOP_N, RerankOptions::FIELD_TOP_N, "top_n")?;
        Ok(())
    }

    /// Validate classification options.
    pub fn validate_classify(&self, o: &ClassifyOptions) -> Result<()> {
        let info = self.info();
        if !matches!(info.kind, ModelKind::Classifier | ModelKind::TokenClassifier) {
            return Err(Error::unsupported_task(format!(
                "model `{}` is {:?}, not a classifier",
                info.model_id, info.kind
            )));
        }
        self.gated(
            o.truncate != Truncate::Model,
            abi::TURBO_CAP_OPT_TRUNCATE,
            ClassifyOptions::FIELD_TRUNCATE,
            "truncate",
        )?;
        self.gated(o.max_tokens != 0, abi::TURBO_CAP_OPT_MAX_TOKENS, ClassifyOptions::FIELD_MAX_TOKENS, "max_tokens")?;
        if o.max_tokens > info.max_seq {
            return Err(Error::capacity(format!(
                "max_tokens {} exceeds the model's max_seq {}",
                o.max_tokens, info.max_seq
            ))
            .with_field(ClassifyOptions::FIELD_MAX_TOKENS));
        }
        if o.aggregation != Aggregation::Model {
            if info.kind != ModelKind::TokenClassifier {
                return Err(Error::invalid_argument("aggregation applies only to token classifiers")
                    .with_field(ClassifyOptions::FIELD_AGGREGATION));
            }
            let differs = Some(o.aggregation) != info.aggregation;
            self.gated(differs, abi::TURBO_CAP_OPT_AGGREGATION, ClassifyOptions::FIELD_AGGREGATION, "aggregation")?;
        }
        Ok(())
    }
}

struct SessionState {
    inner: Box<dyn ProviderSession>,
    inputs_ready: bool,
    written_task: Option<Task>,
}

/// Execution workspace. Single owner.
pub struct Session {
    model: Arc<Model>,
    desc: SessionDesc,
    state: Mutex<SessionState>,
    lease: AtomicBool,
}

impl Session {
    /// Owning model.
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    /// Resolved maxima.
    pub fn desc(&self) -> &SessionDesc {
        &self.desc
    }

    fn lock(&self) -> Result<MutexGuard<'_, SessionState>> {
        match self.state.try_lock() {
            Ok(g) => Ok(g),
            Err(TryLockError::WouldBlock) => {
                Err(Error::busy("session has an operation in flight on another thread; sessions are single-owner"))
            }
            Err(TryLockError::Poisoned(p)) => Ok(p.into_inner()),
        }
    }

    fn require_no_lease(&self) -> Result<()> {
        if self.lease.load(Ordering::Acquire) {
            return Err(Error::busy(
                "a result from this session is still held; release it before writing or running again",
            ));
        }
        Ok(())
    }

    fn check_batch(&self, n: usize) -> Result<()> {
        if n == 0 {
            return Err(Error::invalid_argument("at least one input is required"));
        }
        if n > self.desc.max_batch as usize {
            return Err(Error::capacity(format!("{n} inputs exceed the session's max_batch {}", self.desc.max_batch)));
        }
        Ok(())
    }

    /// Write texts for embedding.
    pub fn write_text(&self, texts: &[&str], opts: &EmbedOptions) -> Result<()> {
        self.model.validate_embed(opts)?;
        self.check_batch(texts.len())?;
        self.require_no_lease()?;
        let mut st = self.lock()?;
        st.inputs_ready = false;
        st.inner.write_text(texts, opts)?;
        st.inputs_ready = true;
        st.written_task = Some(Task::Embed);
        Ok(())
    }

    /// Write prepared tokens (embedding models).
    pub fn write_tokens(&self, batch: &TokenBatch<'_>) -> Result<()> {
        let info = self.model.info();
        if !matches!(
            info.kind,
            ModelKind::Embedding | ModelKind::Reranker | ModelKind::Classifier | ModelKind::TokenClassifier
        ) {
            return Err(Error::unsupported_task(format!(
                "model `{}` is {:?}; prepared tokens apply to encoder models",
                info.model_id, info.kind
            )));
        }
        batch.validate(info.vocab_size)?;
        self.check_batch(batch.batch as usize)?;
        if batch.seq > self.desc.max_seq {
            return Err(Error::capacity(format!(
                "seq {} exceeds the session's max_seq {}",
                batch.seq, self.desc.max_seq
            )));
        }
        self.require_no_lease()?;
        let mut st = self.lock()?;
        st.inputs_ready = false;
        st.inner.write_tokens(batch)?;
        st.inputs_ready = true;
        st.written_task = Some(info.task);
        Ok(())
    }

    /// Write a query and documents for reranking.
    pub fn write_pairs(&self, query: &str, docs: &[&str], opts: &RerankOptions) -> Result<()> {
        self.model.validate_rerank(opts)?;
        self.check_batch(docs.len())?;
        if opts.top_n as usize > docs.len() {
            return Err(Error::invalid_argument(format!(
                "top_n {} exceeds the {} documents supplied",
                opts.top_n,
                docs.len()
            ))
            .with_field(RerankOptions::FIELD_TOP_N));
        }
        self.require_no_lease()?;
        let mut st = self.lock()?;
        st.inputs_ready = false;
        st.inner.write_pairs(query, docs, opts)?;
        st.inputs_ready = true;
        st.written_task = Some(Task::Rerank);
        Ok(())
    }

    /// Write texts for classification.
    pub fn write_text_classify(&self, texts: &[&str], opts: &ClassifyOptions) -> Result<()> {
        self.model.validate_classify(opts)?;
        self.check_batch(texts.len())?;
        self.require_no_lease()?;
        let mut st = self.lock()?;
        st.inputs_ready = false;
        st.inner.write_text_classify(texts, opts)?;
        st.inputs_ready = true;
        st.written_task = Some(self.model.info().task);
        Ok(())
    }

    /// Bind a named tensor (RUN models).
    pub fn bind(&self, name: &str, buffer: &Arc<Buffer>) -> Result<()> {
        let info = self.model.info();
        if info.kind != ModelKind::Generic {
            return Err(Error::unsupported_task(format!(
                "model `{}` is {:?}; named bindings apply to generic RUN models",
                info.model_id, info.kind
            )));
        }
        if !Arc::ptr_eq(&buffer.context, &self.model.context) {
            return Err(Error::invalid_argument("buffer belongs to a different context than the session"));
        }
        if buffer.is_result_view() {
            return Err(Error::invalid_argument("a result view cannot be bound as an input; copy it first"));
        }
        let known = info.inputs.iter().chain(info.outputs.iter()).any(|t| t.name == name);
        if !known {
            return Err(Error::invalid_argument(format!(
                "`{name}` is not an input or output of model `{}`; inputs: {:?}, outputs: {:?}",
                info.model_id,
                info.inputs.iter().map(|t| &t.name).collect::<Vec<_>>(),
                info.outputs.iter().map(|t| &t.name).collect::<Vec<_>>()
            )));
        }
        self.require_no_lease()?;
        let mut st = self.lock()?;
        st.inner.bind(name, buffer.inner.clone())?;
        st.inputs_ready = true;
        st.written_task = Some(Task::Run);
        Ok(())
    }

    /// Execute and lease the result.
    pub fn run(self: &Arc<Self>, opts: &RunOptions) -> Result<Arc<ResultHandle>> {
        self.require_no_lease()?;
        let mut st = self.lock()?;
        if !st.inputs_ready {
            return Err(Error::invalid_state("no valid inputs are written; call a write function before run"));
        }
        let result = st.inner.run(opts)?;
        if result.outputs.is_empty() {
            return Err(Error::internal(format!(
                "provider `{}` returned a result with no outputs",
                self.model.provider_id()
            )));
        }
        for (i, out) in result.outputs.iter().enumerate() {
            let logical = out.logical_bytes()?;
            if logical > out.buffer.desc().bytes {
                return Err(Error::internal(format!(
                    "provider `{}` output {i} `{}` claims shape {:?} ({logical} bytes) but its buffer holds {} bytes",
                    self.model.provider_id(),
                    out.name,
                    out.shape,
                    out.buffer.desc().bytes
                )));
            }
        }
        self.lease.store(true, Ordering::Release);
        Ok(Arc::new(ResultHandle { session: self.clone(), result }))
    }

    /// Counters.
    pub fn stats(&self) -> Result<SessionStats> {
        let st = self.lock()?;
        Ok(st.inner.stats())
    }
}

/// Leased result. Dropping it (all clones) returns the lease.
pub struct ResultHandle {
    session: Arc<Session>,
    result: ProviderResult,
}

impl ResultHandle {
    /// Owning session.
    pub fn session(&self) -> &Arc<Session> {
        &self.session
    }

    /// Outputs.
    pub fn outputs(&self) -> &[crate::provider::Output] {
        &self.result.outputs
    }

    /// Spans (token classification).
    pub fn spans(&self) -> &[Span] {
        &self.result.spans
    }

    /// Output by index.
    pub fn output(&self, index: u32) -> Result<&crate::provider::Output> {
        self.result.outputs.get(index as usize).ok_or_else(|| {
            Error::invalid_argument(format!(
                "output index {index} is out of range; the result has {} outputs",
                self.result.outputs.len()
            ))
        })
    }

    /// A buffer view of an output that keeps the lease alive.
    pub fn buffer(self: &Arc<Self>, index: u32) -> Result<Arc<Buffer>> {
        let out = self.output(index)?;
        Ok(Arc::new(Buffer {
            context: self.session.model.context.clone(),
            inner: out.buffer.clone(),
            lease: Some(self.clone()),
        }))
    }

    /// Blocking copy of the logical bytes of an output into `dst`.
    pub fn read(&self, index: u32, dst: &mut [u8]) -> Result<usize> {
        let out = self.output(index)?;
        let logical = out.logical_bytes()? as usize;
        if dst.len() < logical {
            return Err(Error::capacity(format!(
                "destination holds {} bytes but output {index} `{}` is {logical} bytes",
                dst.len(),
                out.name
            )));
        }
        if let Some(ptr) = out.buffer.host_ptr() {
            // SAFETY: the lease guarantees the provider is not writing this
            // buffer, and `logical <= bytes` was checked at run.
            let src = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), logical) };
            dst[..logical].copy_from_slice(src);
        } else {
            let total = out.buffer.desc().bytes as usize;
            if total == logical {
                out.buffer.read_to_host(&mut dst[..logical])?;
            } else {
                let mut scratch = vec![0u8; total];
                out.buffer.read_to_host(&mut scratch)?;
                dst[..logical].copy_from_slice(&scratch[..logical]);
            }
        }
        Ok(logical)
    }
}

impl std::fmt::Debug for ResultHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResultHandle")
            .field("outputs", &self.result.outputs.iter().map(|o| (&*o.name, &o.shape)).collect::<Vec<_>>())
            .field("spans", &self.result.spans.len())
            .finish()
    }
}

impl Drop for ResultHandle {
    fn drop(&mut self) {
        self.session.lease.store(false, Ordering::Release);
    }
}

struct GenState {
    inner: Box<dyn ProviderGeneration>,
    prompted: bool,
    finished: bool,
    cancelled: bool,
}

/// Streaming generation. Single owner.
pub struct Generation {
    model: Arc<Model>,
    state: Mutex<GenState>,
    chunk: Mutex<Chunk>,
}

impl Generation {
    /// Owning model.
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    fn lock(&self) -> Result<MutexGuard<'_, GenState>> {
        match self.state.try_lock() {
            Ok(g) => Ok(g),
            Err(TryLockError::WouldBlock) => Err(Error::busy(
                "generation has an operation in flight on another thread; generations are single-owner",
            )),
            Err(TryLockError::Poisoned(p)) => Ok(p.into_inner()),
        }
    }

    /// Apply the chat template and tokenize.
    pub fn prompt(&self, messages: &[Message<'_>]) -> Result<()> {
        if messages.is_empty() {
            return Err(Error::invalid_argument("at least one message is required"));
        }
        for (i, m) in messages.iter().enumerate() {
            if m.role.is_empty() {
                return Err(Error::invalid_argument(format!("message {i} has an empty role")));
            }
        }
        let mut st = self.lock()?;
        if st.prompted {
            return Err(Error::invalid_state("generation was already prompted; create a new generation"));
        }
        st.inner.prompt(messages)?;
        st.prompted = true;
        Ok(())
    }

    /// Use prompt tokens directly.
    pub fn prompt_tokens(&self, ids: &[i32]) -> Result<()> {
        if ids.is_empty() {
            return Err(Error::invalid_argument("at least one prompt token is required"));
        }
        let vocab = self.model.info().vocab_size;
        if let Some(bad) = ids.iter().find(|&&t| t < 0 || (vocab != 0 && t as u32 >= vocab)) {
            return Err(Error::invalid_argument(format!("prompt token {bad} is outside 0..{vocab}")));
        }
        let mut st = self.lock()?;
        if st.prompted {
            return Err(Error::invalid_state("generation was already prompted; create a new generation"));
        }
        st.inner.prompt_tokens(ids)?;
        st.prompted = true;
        Ok(())
    }

    /// Next chunk. Returns a guard over the reused chunk storage.
    pub fn step(&self) -> Result<MutexGuard<'_, Chunk>> {
        let mut st = self.lock()?;
        if !st.prompted {
            return Err(Error::invalid_state("generation has no prompt; call prompt or prompt_tokens first"));
        }
        if st.finished {
            return Err(Error::invalid_state("generation already finished; create a new generation"));
        }
        let mut chunk = self.chunk.lock().unwrap_or_else(|p| p.into_inner());
        chunk.clear();
        st.inner.step(&mut chunk)?;
        if chunk.done {
            st.finished = true;
            if st.cancelled && chunk.finish_reason != crate::types::FinishReason::Cancelled {
                return Err(Error::internal("provider reported a cancelled generation as finished for another reason"));
            }
        }
        Ok(chunk)
    }

    /// Cancel.
    pub fn cancel(&self) -> Result<()> {
        let mut st = self.lock()?;
        if !st.finished {
            st.cancelled = true;
            st.inner.cancel();
        }
        Ok(())
    }
}

/// Convenience: build `Options` from pairs.
pub fn options_from_pairs<I, K, V>(pairs: I) -> Options
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<String>,
    V: Into<String>,
{
    Options(pairs.into_iter().map(|(k, v)| (k.into(), v.into())).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{self, MockProvider};
    use crate::runtime::{DeviceSelector, RuntimeDesc};

    fn setup() -> (tempfile::TempDir, Arc<Context>) {
        let tmp = tempfile::tempdir().unwrap();
        mock::write_mock_bundle(tmp.path(), mock::MockBundleKind::Embedding).unwrap();
        let rt = Runtime::new(RuntimeDesc::default(), vec![Arc::new(MockProvider::new())]).unwrap();
        let idx = rt.select(&DeviceSelector::default()).unwrap();
        let ctx = Context::create(rt, idx, &ContextDesc::default()).unwrap();
        (tmp, ctx)
    }

    #[test]
    fn result_lease_blocks_writes_and_runs() {
        let (tmp, ctx) = setup();
        let model = ctx.load_model(tmp.path(), &ModelDesc::default()).unwrap();
        let session = model.create_session(&SessionDesc::default()).unwrap();
        session.write_text(&["hello world"], &EmbedOptions::default()).unwrap();
        let result = session.run(&RunOptions::default()).unwrap();
        assert_eq!(session.write_text(&["x"], &EmbedOptions::default()).unwrap_err().code(), abi::TURBO_E_BUSY);
        assert_eq!(session.run(&RunOptions::default()).unwrap_err().code(), abi::TURBO_E_BUSY);
        let view = result.buffer(0).unwrap();
        drop(result);
        // The view still holds the lease.
        assert_eq!(session.run(&RunOptions::default()).unwrap_err().code(), abi::TURBO_E_BUSY);
        drop(view);
        session.run(&RunOptions::default()).unwrap();
    }

    #[test]
    fn run_without_inputs_is_invalid_state() {
        let (tmp, ctx) = setup();
        let model = ctx.load_model(tmp.path(), &ModelDesc::default()).unwrap();
        let session = model.create_session(&SessionDesc::default()).unwrap();
        assert_eq!(session.run(&RunOptions::default()).unwrap_err().code(), abi::TURBO_E_INVALID_STATE);
    }

    #[test]
    fn ungated_option_is_rejected_with_field() {
        let (tmp, ctx) = setup();
        let model = ctx.load_model(tmp.path(), &ModelDesc::default()).unwrap();
        let session = model.create_session(&SessionDesc::default()).unwrap();
        let opts = EmbedOptions { pooling: Pooling::Cls, ..Default::default() };
        let err = session.write_text(&["a"], &opts).unwrap_err();
        assert_eq!(err.code(), abi::TURBO_E_UNSUPPORTED_OPTION);
        assert_eq!(err.field(), EmbedOptions::FIELD_POOLING);
    }

    #[test]
    fn children_outlive_parents() {
        let (tmp, ctx) = setup();
        let model = ctx.load_model(tmp.path(), &ModelDesc::default()).unwrap();
        let session = model.create_session(&SessionDesc::default()).unwrap();
        drop(ctx);
        drop(model);
        session.write_text(&["still alive"], &EmbedOptions::default()).unwrap();
        let r = session.run(&RunOptions::default()).unwrap();
        drop(session);
        let mut dst = vec![0u8; r.output(0).unwrap().logical_bytes().unwrap() as usize];
        r.read(0, &mut dst).unwrap();
    }
}
