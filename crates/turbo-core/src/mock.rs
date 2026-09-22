//! The mock provider: deterministic, CPU-only, allocation-free after warmup.
//!
//! It exists so the contract can be tested without hardware. It serves only
//! bundles that carry a `mock` artifact and never impersonates a real model.
//! Every capability bit it advertises is honored exactly; every bit it does
//! not advertise is rejected by the core before the mock is reached. Its
//! outputs are pure functions of the inputs and the bundle's salt, so the
//! conformance suite can assert determinism and cross-binding parity.
//!
//! Tokenization is whitespace splitting with FNV-hashed ids, `[CLS]` = 1 and
//! `[SEP]` = 2. Embeddings are the mean of hash-derived unit vectors. Rerank
//! scores are token overlap. Classifier logits are hash-derived. Generation
//! emits a deterministic id sequence. The generic RUN model computes `y = 2x`.
//!
//! It also carries the suite's two honesty hooks, both of them part of the
//! mock's own contract rather than a debug switch: the `fault` option
//! ([`FAULT_OPTION`]) and the planned `CHUNK x TEXT` cell. See
//! [`FAULT_OPTION`] for the option's grammar and [`MockProvider::capability`]
//! for the cell.

use std::fmt::Write as _;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::Deserialize;
use turbo_abi as abi;

use crate::buffer::{BufferDesc, HostBuffer, NativeHandle, ProviderBuffer};
use crate::bundle::{sha256_bytes, Bundle, MANIFEST_NAME};
use crate::error::{Error, Result};
use crate::provider::{
    Capability, Chunk, ClassifyOptions, ContextDesc, DeviceInfo, EmbedOptions, GenerateDesc, Message, ModelDesc,
    ModelInfo, Options, Output, Provider, ProviderContext, ProviderGeneration, ProviderModel, ProviderResult,
    ProviderSession, RerankOptions, RunOptions, SessionDesc, SessionStats, Span, TensorInfo, TokenBatch,
};
use crate::types::{
    Aggregation, CapStatus, DType, DeviceKind, FinishReason, HandleKind, Modality, ModelKind, Normalize, Placement,
    PromptRole, Stage, StagePlacement, StagePlacements, Task, Truncate,
};

/// Provider id.
pub const MOCK_PROVIDER_ID: &str = "mock";
/// Artifact format name in bundles.
pub const MOCK_ARTIFACT: &str = "mock";
/// `[CLS]` id.
pub const MOCK_CLS: i32 = 1;
/// `[SEP]` id.
pub const MOCK_SEP: i32 = 2;
/// First ordinary token id.
pub const MOCK_FIRST_WORD: i32 = 3;
/// Default new-token budget for generation.
pub const MOCK_DEFAULT_MAX_NEW: u32 = 16;
/// Default vocabulary size written by [`write_mock_bundle`].
pub const MOCK_VOCAB: u32 = 1000;

/// Capability bits the mock honors. Everything else is rejected by the core.
pub const MOCK_CAPS: u64 = abi::TURBO_CAP_HOST_PTR_IMPORT
    | abi::TURBO_CAP_DYNAMIC_SHAPE
    | abi::TURBO_CAP_WEIGHT_SHARING
    | abi::TURBO_CAP_DETERMINISTIC
    | abi::TURBO_CAP_OPT_TRUNCATE
    | abi::TURBO_CAP_OPT_MAX_TOKENS
    | abi::TURBO_CAP_OPT_PROMPT_ROLE
    | abi::TURBO_CAP_OPT_NORMALIZE
    | abi::TURBO_CAP_OPT_OUTPUT_DIM
    | abi::TURBO_CAP_OPT_TOP_N
    | abi::TURBO_CAP_OPT_AGGREGATION
    | abi::TURBO_CAP_OPT_RAW_SCORES
    | abi::TURBO_CAP_OPT_GEN_STOP_STRINGS
    | abi::TURBO_CAP_OPT_GEN_STOP_TOKENS
    | abi::TURBO_CAP_OPT_GEN_SEED
    | abi::TURBO_CAP_OPT_GEN_LOGPROBS
    | abi::TURBO_CAP_OPT_GEN_SAMPLING
    | abi::TURBO_CAP_OPT_GEN_MIN_TOKENS
    | abi::TURBO_CAP_OPT_GEN_ECHO;

/// The one provider option the mock accepts, on a context, a model, or a
/// session descriptor.
///
/// Its value is `<stage>=<status>`, and it makes that stage fail with that
/// status. It exists so every documented status code the contract can produce
/// has a conformance case that reaches it without a real failure: a device
/// that vanishes, a runtime that errors, a provider that panics. Nothing
/// outside the conformance suite sets it, and no other provider reads it.
///
/// | descriptor | value | what fails, and how |
/// |---|---|---|
/// | [`ContextDesc`] | `context=device_unavailable` | context creation, `TURBO_E_DEVICE_UNAVAILABLE` |
/// | [`ModelDesc`] | `load=unsupported_dtype` | model load, `TURBO_E_UNSUPPORTED_DTYPE` |
/// | [`SessionDesc`] | `run=overloaded` | every run, `TURBO_E_OVERLOADED` |
/// | [`SessionDesc`] | `run=runtime` | every run, `TURBO_E_RUNTIME` |
/// | [`SessionDesc`] | `run=internal` | every run, `TURBO_E_INTERNAL` |
/// | [`SessionDesc`] | `run=panic` | every run panics inside the provider call |
///
/// A value that is not one of these, or one whose stage does not belong to
/// the descriptor it was set on, is `TURBO_E_INVALID_ARGUMENT` naming the
/// option's 1-based index, exactly as an unknown option key is.
pub const FAULT_OPTION: &str = "fault";

/// The stage a [`FAULT_OPTION`] value may name, per descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FaultStage {
    Context,
    Load,
    Run,
}

impl FaultStage {
    fn name(self) -> &'static str {
        match self {
            FaultStage::Context => "context",
            FaultStage::Load => "load",
            FaultStage::Run => "run",
        }
    }

    /// The statuses this stage may be told to fail with.
    fn codes(self) -> &'static [&'static str] {
        match self {
            FaultStage::Context => &["device_unavailable"],
            FaultStage::Load => &["unsupported_dtype"],
            FaultStage::Run => &["overloaded", "runtime", "internal", "panic"],
        }
    }
}

/// The parsed value of a [`FAULT_OPTION`] for one stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    DeviceUnavailable,
    UnsupportedDtype,
    Overloaded,
    Runtime,
    Internal,
    Panic,
}

impl Fault {
    fn of(code: &str) -> Option<Self> {
        Some(match code {
            "device_unavailable" => Fault::DeviceUnavailable,
            "unsupported_dtype" => Fault::UnsupportedDtype,
            "overloaded" => Fault::Overloaded,
            "runtime" => Fault::Runtime,
            "internal" => Fault::Internal,
            "panic" => Fault::Panic,
            _ => return None,
        })
    }

    /// Raise it: an error, or a panic for `run=panic`.
    fn raise(self, what: &str) -> Error {
        match self {
            Fault::DeviceUnavailable => {
                Error::device_unavailable(format!("mock {what}: the injected fault `device_unavailable`"))
            }
            Fault::UnsupportedDtype => {
                Error::unsupported_dtype(format!("mock {what}: the injected fault `unsupported_dtype`"))
            }
            Fault::Overloaded => Error::overloaded(format!("mock {what}: the injected fault `overloaded`")),
            Fault::Runtime => Error::runtime(format!("mock {what}: the injected fault `runtime`")),
            Fault::Internal => Error::internal(format!("mock {what}: the injected fault `internal`")),
            Fault::Panic => panic!("mock {what}: the injected fault `panic`"),
        }
    }
}

/// The [`FAULT_OPTION`] value set on one descriptor, if any.
///
/// `Ok(None)` means the option is absent. An unparseable value, or one for
/// another descriptor's stage, is an argument error naming the option's
/// 1-based index.
fn fault_of(options: &Options, stage: FaultStage, what: &str) -> Result<Option<Fault>> {
    for (i, (k, v)) in options.0.iter().enumerate() {
        if k != FAULT_OPTION {
            continue;
        }
        let field = i as u32 + 1;
        let (named, code) = v.split_once('=').ok_or_else(|| {
            Error::invalid_argument(format!(
                "mock {what} option `{FAULT_OPTION}` is `{v}`; the value is `<stage>=<status>`, here `{}=<{}>`",
                stage.name(),
                stage.codes().join("|")
            ))
            .with_field(field)
        })?;
        if named != stage.name() {
            return Err(Error::invalid_argument(format!(
                "mock {what} option `{FAULT_OPTION}` names stage `{named}`, which is not this descriptor's stage `{}`",
                stage.name()
            ))
            .with_field(field));
        }
        let fault = Fault::of(code).filter(|_| stage.codes().contains(&code)).ok_or_else(|| {
            Error::invalid_argument(format!(
                "mock {what} option `{FAULT_OPTION}` names status `{code}`; stage `{}` accepts {}",
                stage.name(),
                stage.codes().join(", ")
            ))
            .with_field(field)
        })?;
        return Ok(Some(fault));
    }
    Ok(None)
}

/// Contents of the `mock.json` artifact.
#[derive(Clone, Debug, Deserialize)]
struct MockArtifact {
    vocab_size: u32,
    salt: u64,
}

/// The provider.
#[derive(Debug, Default)]
pub struct MockProvider;

impl MockProvider {
    /// Construct.
    pub fn new() -> Self {
        Self
    }

    fn device(ordinal: u32) -> DeviceInfo {
        let (kind, name) = match ordinal {
            0 => (DeviceKind::Cpu, "Mock CPU"),
            _ => (DeviceKind::Accel, "Mock accelerator"),
        };
        DeviceInfo {
            kind,
            ordinal,
            vendor_id: 0,
            caps: MOCK_CAPS,
            memory_total: 0,
            memory_free: 0,
            name: name.to_string(),
            vendor: "Pipestream".to_string(),
            provider_id: MOCK_PROVIDER_ID.to_string(),
            provider_version: env!("CARGO_PKG_VERSION").to_string(),
            runtime_version: "mock".to_string(),
            driver_version: String::new(),
        }
    }
}

impl Provider for MockProvider {
    fn id(&self) -> &str {
        MOCK_PROVIDER_ID
    }

    fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }

    fn devices(&self) -> Result<Vec<DeviceInfo>> {
        Ok(vec![Self::device(0), Self::device(1)])
    }

    fn capability(&self, ordinal: u32, task: Task, modality: Modality) -> Capability {
        if ordinal > 1 || modality != Modality::Text {
            return Capability::unsupported();
        }
        match task {
            Task::Embed
            | Task::Rerank
            | Task::Classify
            | Task::TokenClassify
            | Task::Generate
            | Task::Tokenize
            | Task::Run => Capability {
                status: CapStatus::Supported,
                dtype: Some(DType::F32),
                reference_dtype: Some(DType::F32),
                cosine_floor: 1.0,
                max_abs_error: 0.0,
                deterministic: true,
                notes: "mock: deterministic hash-derived outputs for contract testing; never a real model".into(),
            },
            // The one planned cell in the tree. Chunking is a host utility
            // (`turbo_chunk_plan_*`, `crates/turbo-core/src/chunker.rs`), so
            // the code path exists but nothing runs it on a device and it is
            // not qualified: PLANNED says exactly that, and the core refuses
            // a call that lands on it the same way it refuses an unsupported
            // one. It is what the suite gates the PLANNED rule on.
            Task::Chunk => Capability::planned(
                "mock: chunking is a host utility, not a device path; planned so the PLANNED gate has a cell",
            ),
        }
    }

    fn can_run(&self, ordinal: u32, bundle: &Bundle, task: Task, modality: Modality) -> Result<()> {
        if !self.capability(ordinal, task, modality).is_offered() {
            return Err(Error::unsupported_task(format!(
                "mock device {ordinal} does not offer {task:?} for {modality:?}"
            )));
        }
        if bundle.artifact(MOCK_ARTIFACT).is_none() {
            return Err(Error::bundle_no_artifact(format!(
                "bundle `{}` has no `mock` artifact; the mock provider never serves real models",
                bundle.manifest().model_id
            )));
        }
        Ok(())
    }

    fn create_context(&self, ordinal: u32, desc: &ContextDesc) -> Result<Arc<dyn ProviderContext>> {
        if ordinal > 1 {
            return Err(Error::device_not_found(format!("mock has no device ordinal {ordinal}")));
        }
        desc.options.reject_unknown(&[FAULT_OPTION], "mock context")?;
        if let Some(f) = fault_of(&desc.options, FaultStage::Context, "context")? {
            return Err(f.raise("context create"));
        }
        Ok(Arc::new(MockContext { ordinal }))
    }
}

struct MockContext {
    ordinal: u32,
}

impl ProviderContext for MockContext {
    fn ordinal(&self) -> u32 {
        self.ordinal
    }

    fn alloc(&self, desc: &BufferDesc) -> Result<Arc<dyn ProviderBuffer>> {
        if desc.placement != Placement::Host {
            return Err(Error::unsupported_placement(format!(
                "mock allocates TURBO_PLACE_HOST only, not {:?}",
                desc.placement
            )));
        }
        Ok(HostBuffer::new(desc.clone())?)
    }

    fn import(&self, desc: &BufferDesc, handle: &NativeHandle) -> Result<Arc<dyn ProviderBuffer>> {
        if handle.kind != HandleKind::HostPtr {
            return Err(Error::unsupported(format!("mock imports TURBO_HANDLE_HOST_PTR only, not {:?}", handle.kind)));
        }
        if desc.placement != Placement::Host {
            return Err(Error::unsupported_placement("imported host memory must be TURBO_PLACE_HOST"));
        }
        let addr = handle
            .handle
            .checked_add(handle.offset)
            .ok_or_else(|| Error::invalid_argument("host pointer plus offset overflows"))?;
        let ptr =
            NonNull::new(addr as *mut u8).ok_or_else(|| Error::invalid_argument("host pointer handle is null"))?;
        Ok(Arc::new(ImportedHost { desc: desc.clone(), ptr }))
    }

    fn load_model(&self, bundle: Arc<Bundle>, desc: &ModelDesc) -> Result<Arc<dyn ProviderModel>> {
        desc.options.reject_unknown(&[FAULT_OPTION], "mock model")?;
        if let Some(f) = fault_of(&desc.options, FaultStage::Load, "model")? {
            return Err(f.raise("model load"));
        }
        let path = bundle.artifact_path(MOCK_ARTIFACT)?;
        let text = std::fs::read_to_string(&path)?;
        let artifact: MockArtifact = serde_json::from_str(&text)
            .map_err(|e| Error::bundle_invalid(format!("mock artifact `{}`: {e}", path.display())))?;
        if artifact.vocab_size <= MOCK_FIRST_WORD as u32 + 1 {
            return Err(Error::bundle_invalid(format!(
                "mock vocab_size {} must exceed {}",
                artifact.vocab_size,
                MOCK_FIRST_WORD + 1
            )));
        }
        let c = bundle.contract();
        let kind = bundle.kind();
        let max_batch = if bundle.manifest().limits.max_batch == 0 { 32 } else { bundle.manifest().limits.max_batch };
        let host = StagePlacement::Host;
        let stages = match kind {
            ModelKind::Embedding => StagePlacements::NONE
                .with(Stage::Tokenize, host)
                .with(Stage::Encode, host)
                .with(Stage::Pool, host)
                .with(Stage::Normalize, host),
            ModelKind::Generic => StagePlacements::NONE.with(Stage::Encode, host),
            _ => StagePlacements::NONE
                .with(Stage::Tokenize, host)
                .with(Stage::Encode, host)
                .with(Stage::Postprocess, host),
        };
        let (inputs, outputs) = if kind == ModelKind::Generic {
            (
                vec![TensorInfo { name: "x".into(), dtype: DType::F32, shape: vec![-1, -1] }],
                vec![TensorInfo { name: "y".into(), dtype: DType::F32, shape: vec![-1, -1] }],
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let info = ModelInfo {
            task: bundle.task(),
            kind,
            modality: bundle.modality(),
            dim: c.dim,
            labels: c.labels.clone(),
            pooling: bundle.pooling()?,
            normalize: bundle.normalize()?,
            aggregation: bundle.aggregation()?,
            max_seq: if c.max_seq == 0 { 64 } else { c.max_seq },
            max_batch,
            dtype_used: Some(DType::F32),
            stages,
            inputs,
            outputs,
            vocab_size: artifact.vocab_size,
            model_id: bundle.manifest().model_id.clone(),
            revision: bundle.manifest().revision.clone(),
            tokenizer_sha256: bundle.tokenizer_sha256().to_string(),
            provider_id: MOCK_PROVIDER_ID.into(),
            prefix_query: c.prompts.query.clone(),
            prefix_document: c.prompts.document.clone(),
        };
        let activation = match kind {
            ModelKind::Reranker | ModelKind::Classifier | ModelKind::TokenClassifier => {
                Some(Activation::from_contract(c.activation.as_deref())?)
            }
            _ => None,
        };
        Ok(Arc::new(MockModel { info, vocab: artifact.vocab_size, salt: artifact.salt, activation }))
    }
}

struct ImportedHost {
    desc: BufferDesc,
    ptr: NonNull<u8>,
}
// SAFETY: the importer promised `desc.bytes` readable/writable bytes for the buffer's lifetime.
unsafe impl Send for ImportedHost {}
unsafe impl Sync for ImportedHost {}

impl ProviderBuffer for ImportedHost {
    fn desc(&self) -> &BufferDesc {
        &self.desc
    }
    fn host_ptr(&self) -> Option<NonNull<u8>> {
        Some(self.ptr)
    }
    fn read_to_host(&self, dst: &mut [u8]) -> Result<()> {
        if dst.len() as u64 != self.desc.bytes {
            return Err(Error::capacity("destination size does not match the buffer"));
        }
        // SAFETY: see the Send/Sync note.
        let src = unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), dst.len()) };
        dst.copy_from_slice(src);
        Ok(())
    }
    fn export(&self, kind: HandleKind) -> Result<NativeHandle> {
        if kind != HandleKind::HostPtr {
            return Err(Error::unsupported("imported host memory exports TURBO_HANDLE_HOST_PTR only"));
        }
        Ok(NativeHandle { kind, handle: self.ptr.as_ptr() as u64, aux: 0, offset: 0 })
    }
}

/// Score activation declared by the bundle contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Activation {
    Softmax,
    Sigmoid,
    None,
}

impl Activation {
    fn from_contract(name: Option<&str>) -> Result<Self> {
        match name {
            Some("softmax") => Ok(Self::Softmax),
            Some("sigmoid") => Ok(Self::Sigmoid),
            Some("none") => Ok(Self::None),
            None => Err(Error::bundle_invalid("contract.activation is required for scored models")),
            Some(other) => Err(Error::bundle_invalid(format!("contract.activation `{other}` is not known"))),
        }
    }

    fn apply(self, row: &mut [f32]) {
        match self {
            Self::Softmax => softmax(row),
            Self::Sigmoid => {
                for v in row.iter_mut() {
                    *v = sigmoid(*v);
                }
            }
            Self::None => {}
        }
    }
}

struct MockModel {
    info: ModelInfo,
    vocab: u32,
    salt: u64,
    /// Activation for rerankers and classifiers; `None` for other kinds.
    activation: Option<Activation>,
}

impl ProviderModel for MockModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn create_session(&self, desc: &SessionDesc) -> Result<Box<dyn ProviderSession>> {
        desc.options.reject_unknown(&[FAULT_OPTION], "mock session")?;
        let run_fault = fault_of(&desc.options, FaultStage::Run, "session")?;
        let n_labels = self.info.labels.len().max(1) as u64;
        let width = match self.info.kind {
            ModelKind::Embedding => self.info.dim as u64,
            ModelKind::Reranker => 1,
            ModelKind::Classifier => n_labels,
            ModelKind::TokenClassifier => desc.max_seq as u64 * n_labels,
            ModelKind::Generic | ModelKind::Generative => 1,
        };
        let out = HostBuffer::packed(DType::F32, &[desc.max_batch as u64, width])?;
        let sorted = HostBuffer::packed(DType::I32, &[desc.max_batch as u64])?;
        let cap = desc.max_batch as usize * desc.max_seq as usize;
        Ok(Box::new(MockSession {
            run_fault,
            vocab: self.vocab,
            salt: self.salt,
            activation: self.activation.unwrap_or(Activation::None),
            info: self.info.clone(),
            max_batch: desc.max_batch,
            max_seq: desc.max_seq,
            ids: vec![0; cap],
            mask: vec![0; cap],
            n_rows: 0,
            out,
            sorted,
            full: vec![0.0; self.info.dim as usize],
            word_spans: Vec::with_capacity(cap),
            row_ranges: Vec::with_capacity(desc.max_batch as usize),
            spans: Vec::with_capacity(cap),
            embed_opts: EmbedOptions::default(),
            rerank_opts: RerankOptions::default(),
            classify_opts: ClassifyOptions::default(),
            query_tokens: Vec::with_capacity(desc.max_seq as usize),
            bound_x: None,
            bound_y: None,
            y: None,
            runs: 0,
            allocs: AtomicU64::new(0),
            names: Names {
                embeddings: Arc::from("embeddings"),
                scores: Arc::from("scores"),
                sorted: Arc::from("sorted"),
                y: Arc::from("y"),
            },
            no_prefix: Arc::from(""),
            prefix_query: Arc::from(self.info.prefix_query.as_str()),
            prefix_document: Arc::from(self.info.prefix_document.as_str()),
        }))
    }

    fn create_generation(&self, desc: &GenerateDesc) -> Result<Box<dyn ProviderGeneration>> {
        desc.options.reject_unknown(&[], "mock generation")?;
        if desc.top_k > self.vocab {
            return Err(
                Error::invalid_argument(format!("top_k {} exceeds vocab {}", desc.top_k, self.vocab)).with_field(6)
            );
        }
        let max_new = if desc.max_new_tokens == 0 { MOCK_DEFAULT_MAX_NEW } else { desc.max_new_tokens };
        if desc.min_new_tokens > max_new {
            return Err(Error::invalid_argument(format!(
                "min_new_tokens {} exceeds max_new_tokens {max_new}",
                desc.min_new_tokens
            ))
            .with_field(3));
        }
        Ok(Box::new(MockGeneration {
            vocab: self.vocab,
            salt: self.salt,
            max_seq: self.info.max_seq,
            max_new,
            min_new: desc.min_new_tokens,
            seed: desc.seed,
            sampling: if desc.temperature > 0.0 {
                let pool = if desc.top_k == 0 { self.vocab - MOCK_FIRST_WORD as u32 } else { desc.top_k };
                let keep = if desc.top_p > 0.0 && desc.top_p < 1.0 { desc.top_p } else { 1.0 };
                let keep = if desc.min_p > 0.0 { keep * (1.0 - desc.min_p) } else { keep };
                Some(((pool as f32 * keep).ceil() as u32).max(1))
            } else {
                None
            },
            stop: desc.stop.clone(),
            stop_tokens: desc.stop_tokens.clone(),
            logprobs: desc.logprobs,
            echo: desc.echo,
            prompt_ids: Vec::new(),
            prompt_text: String::new(),
            state: 0,
            generated: 0,
            hold: String::new(),
            hold_max: desc.stop.iter().map(String::len).max().unwrap_or(0).saturating_sub(1),
            cancelled: false,
            done: false,
        }))
    }
}

struct Names {
    embeddings: Arc<str>,
    scores: Arc<str>,
    sorted: Arc<str>,
    y: Arc<str>,
}

/// Session state. Scratch is sized at creation; the run path does not grow it.
struct MockSession {
    vocab: u32,
    salt: u64,
    activation: Activation,
    info: ModelInfo,
    max_batch: u32,
    max_seq: u32,
    ids: Vec<i32>,
    mask: Vec<i32>,
    n_rows: u32,
    out: Arc<HostBuffer>,
    sorted: Arc<HostBuffer>,
    full: Vec<f32>,
    /// `(byte_start, byte_end, label)` per word, all rows, filled at write time.
    word_spans: Vec<(u64, u64, u32)>,
    /// `word_spans` range per row.
    row_ranges: Vec<(usize, usize)>,
    spans: Vec<Span>,
    embed_opts: EmbedOptions,
    rerank_opts: RerankOptions,
    classify_opts: ClassifyOptions,
    query_tokens: Vec<i32>,
    bound_x: Option<Arc<dyn ProviderBuffer>>,
    bound_y: Option<Arc<dyn ProviderBuffer>>,
    y: Option<Arc<HostBuffer>>,
    runs: u64,
    allocs: AtomicU64,
    /// The `fault` option's `run=...` value, raised by every run.
    run_fault: Option<Fault>,
    names: Names,
    no_prefix: Arc<str>,
    prefix_query: Arc<str>,
    prefix_document: Arc<str>,
}

/// FNV-1a 64 with a seed.
pub fn fnv(bytes: &[u8], seed: u64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ seed;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Map a hash to [-1, 1).
fn unit(h: u64) -> f32 {
    ((h >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
}

/// Mock token id for a word.
pub fn word_id(word: &str, vocab: u32, salt: u64) -> i32 {
    MOCK_FIRST_WORD + (fnv(word.as_bytes(), salt) % (vocab - MOCK_FIRST_WORD as u32) as u64) as i32
}

/// Words with byte offsets, split on Unicode whitespace. Does not allocate.
pub fn words(text: &str) -> Words<'_> {
    Words { text, pos: 0 }
}

/// Iterator returned by [`words`].
pub struct Words<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> Iterator for Words<'a> {
    type Item = (usize, &'a str);
    fn next(&mut self) -> Option<Self::Item> {
        let rest = &self.text[self.pos..];
        let start_rel = rest.find(|c: char| !c.is_whitespace())?;
        let start = self.pos + start_rel;
        let after = &self.text[start..];
        let end = after.find(char::is_whitespace).map(|e| start + e).unwrap_or(self.text.len());
        self.pos = end;
        Some((start, &self.text[start..end]))
    }
}

impl MockSession {
    fn note_alloc(&self) {
        if self.runs > 0 {
            self.allocs.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The token budget for a write: `max_tokens`, or the session width. A
    /// budget the session cannot hold is an error, never clamped, so the
    /// caller's option is honored exactly or refused.
    fn budget(&self, max_tokens: u32) -> Result<u32> {
        if max_tokens == 0 {
            return Ok(self.max_seq);
        }
        if max_tokens > self.max_seq {
            return Err(Error::capacity(format!(
                "max_tokens {max_tokens} exceeds the session's max_seq {}",
                self.max_seq
            ))
            .with_field(EmbedOptions::FIELD_MAX_TOKENS));
        }
        Ok(max_tokens)
    }

    /// Tokenize one text into row `r`, recording word spans and labels.
    fn tokenize_row(&mut self, r: usize, prefix: &str, text: &str, truncate: Truncate, budget: u32) -> Result<()> {
        let seq = self.max_seq as usize;
        let base = r * seq;
        let budget = budget as usize;
        if budget < 2 {
            return Err(Error::capacity("token budget must be at least 2 for [CLS] and [SEP]"));
        }
        let content_budget = budget - 2;
        let n_prefix = words(prefix).count();
        if n_prefix > content_budget {
            return Err(Error::capacity(format!(
                "prompt prefix alone is {n_prefix} tokens but the content budget is {content_budget}"
            )));
        }
        // The prefix is the model's prompt and is never truncated away;
        // truncation applies to the text words after it.
        let n_words = words(text).count();
        let avail = content_budget - n_prefix;
        let (skip, take) = if n_words <= avail {
            (0, n_words)
        } else {
            match truncate {
                Truncate::None => {
                    return Err(Error::capacity(format!(
                        "input row {r} has {} tokens plus 2 specials but the budget is {budget} and truncation is NONE",
                        n_prefix + n_words
                    )));
                }
                Truncate::Model | Truncate::Right => (0, avail),
                Truncate::Left => (n_words - avail, avail),
            }
        };
        let (vocab, salt) = (self.vocab, self.salt);
        let n_labels = self.info.labels.len().max(1) as u64;
        let mut col = 0usize;
        self.ids[base] = MOCK_CLS;
        self.mask[base] = 1;
        col += 1;
        let span_start = self.word_spans.len();
        let prefix_words = words(prefix).map(|(_, w)| (usize::MAX, w));
        let text_words = words(text).enumerate().filter(|(i, _)| *i >= skip && *i < skip + take).map(|(_, w)| w);
        for (offset, w) in prefix_words.chain(text_words) {
            let id = word_id(w, vocab, salt);
            self.ids[base + col] = id;
            self.mask[base + col] = 1;
            col += 1;
            if offset != usize::MAX {
                let label = (fnv(&id.to_le_bytes(), salt) % n_labels) as u32;
                self.word_spans.push((offset as u64, (offset + w.len()) as u64, label));
            }
        }
        self.row_ranges.push((span_start, self.word_spans.len()));
        self.ids[base + col] = MOCK_SEP;
        self.mask[base + col] = 1;
        col += 1;
        for c in col..seq {
            self.ids[base + c] = 0;
            self.mask[base + c] = 0;
        }
        Ok(())
    }

    fn begin_write(&mut self, n: usize) {
        self.n_rows = n as u32;
        self.word_spans.clear();
        self.row_ranges.clear();
    }

    fn run_embed(&mut self) -> Result<ProviderResult> {
        let dim_full = self.info.dim as usize;
        let dim_out = if self.embed_opts.output_dim == 0 { dim_full } else { self.embed_opts.output_dim as usize };
        let normalize = match self.embed_opts.normalize {
            Normalize::Model => self.info.normalize == Some(Normalize::L2),
            Normalize::L2 => true,
            Normalize::None => false,
        };
        let n = self.n_rows as usize;
        // SAFETY: the core holds the session lock and no result lease is outstanding.
        let out = unsafe { self.out.as_f32_mut()? };
        for r in 0..n {
            let row = &mut out[r * dim_out..(r + 1) * dim_out];
            if dim_out == dim_full {
                embed_into(&self.ids, &self.mask, self.max_seq as usize, self.salt, r, row, normalize);
            } else {
                embed_into(&self.ids, &self.mask, self.max_seq as usize, self.salt, r, &mut self.full, false);
                row.copy_from_slice(&self.full[..dim_out]);
                if normalize {
                    l2(row);
                }
            }
        }
        Ok(ProviderResult {
            outputs: vec![Output {
                name: self.names.embeddings.clone(),
                buffer: self.out.clone(),
                shape: vec![n as u64, dim_out as u64],
            }],
            spans: Vec::new(),
        })
    }

    fn run_rerank(&mut self) -> Result<ProviderResult> {
        let n = self.n_rows as usize;
        let out = unsafe { self.out.as_f32_mut()? };
        for (r, slot) in out.iter_mut().enumerate().take(n) {
            let base = r * self.max_seq as usize;
            let mut hit = 0usize;
            let mut total = 0usize;
            for c in 0..self.max_seq as usize {
                if self.mask[base + c] == 0 {
                    continue;
                }
                let id = self.ids[base + c];
                if id == MOCK_CLS || id == MOCK_SEP {
                    continue;
                }
                total += 1;
                if self.query_tokens.contains(&id) {
                    hit += 1;
                }
            }
            let overlap = if total == 0 { 0.0 } else { hit as f32 / total as f32 };
            let logit = 4.0 * (overlap - 0.5);
            *slot = logit;
        }
        if !self.rerank_opts.raw_scores {
            self.activation.apply(&mut out[..n]);
        }
        let mut outputs =
            vec![Output { name: self.names.scores.clone(), buffer: self.out.clone(), shape: vec![n as u64] }];
        if self.rerank_opts.return_sorted || self.rerank_opts.top_n != 0 {
            let k = if self.rerank_opts.top_n == 0 { n } else { self.rerank_opts.top_n as usize };
            // SAFETY: as above; the sorted buffer is i32 [max_batch].
            let sorted = unsafe { self.sorted.bytes_mut() };
            let sorted: &mut [i32] =
                unsafe { std::slice::from_raw_parts_mut(sorted.as_mut_ptr().cast(), self.max_batch as usize) };
            for (i, s) in sorted.iter_mut().enumerate().take(n) {
                *s = i as i32;
            }
            let scores = &out[..n];
            sorted[..n].sort_by(|&a, &b| {
                scores[b as usize].partial_cmp(&scores[a as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
            });
            outputs.push(Output {
                name: self.names.sorted.clone(),
                buffer: self.sorted.clone(),
                shape: vec![k as u64],
            });
        }
        Ok(ProviderResult { outputs, spans: Vec::new() })
    }

    fn run_classify(&mut self) -> Result<ProviderResult> {
        let n = self.n_rows as usize;
        let l = self.info.labels.len();
        let out = unsafe { self.out.as_f32_mut()? };
        for r in 0..n {
            let base = r * self.max_seq as usize;
            let mut h = self.salt;
            for c in 0..self.max_seq as usize {
                if self.mask[base + c] == 1 {
                    h = fnv(&self.ids[base + c].to_le_bytes(), h);
                }
            }
            let row = &mut out[r * l..(r + 1) * l];
            for (i, v) in row.iter_mut().enumerate() {
                *v = 2.0 * unit(fnv(&(i as u64).to_le_bytes(), h));
            }
            if !self.classify_opts.raw_scores {
                self.activation.apply(row);
            }
        }
        Ok(ProviderResult {
            outputs: vec![Output {
                name: self.names.scores.clone(),
                buffer: self.out.clone(),
                shape: vec![n as u64, l as u64],
            }],
            spans: Vec::new(),
        })
    }

    fn run_token_classify(&mut self) -> Result<ProviderResult> {
        let n = self.n_rows as usize;
        let l = self.info.labels.len();
        let seq = self.max_seq as usize;
        let aggregation = match self.classify_opts.aggregation {
            Aggregation::Model => self.info.aggregation.unwrap_or(Aggregation::Simple),
            a => a,
        };
        let (major, minor) =
            if self.classify_opts.raw_scores { (3.0, -1.0) } else { (0.9, 0.1 / (l.max(2) - 1) as f32) };
        let out = unsafe { self.out.as_f32_mut()? };
        for r in 0..n {
            let base = r * seq;
            let row = &mut out[r * seq * l..(r + 1) * seq * l];
            for v in row.iter_mut() {
                *v = 0.0;
            }
            for c in 0..seq {
                if self.mask[base + c] == 0 {
                    continue;
                }
                let id = self.ids[base + c];
                let label = if id == MOCK_CLS || id == MOCK_SEP {
                    0
                } else {
                    (fnv(&id.to_le_bytes(), self.salt) % l as u64) as u32
                };
                let cell = &mut row[c * l..(c + 1) * l];
                for (i, v) in cell.iter_mut().enumerate() {
                    *v = if i == label as usize { major } else { minor };
                }
            }
        }
        let cap_before = self.spans.capacity();
        self.spans.clear();
        // Label 0 is the outside tag: an outside word yields no span and ends
        // the group before it, as in the real providers.
        for r in 0..n {
            let (s, e) = self.row_ranges[r];
            let mut prev: Option<(u64, u64, u32)> = None;
            for i in s..e {
                let (ws, we, label) = self.word_spans[i];
                if label == 0 {
                    if let Some((ps, pe, pl)) = prev.take() {
                        self.spans.push(Span { row: r as u32, byte_start: ps, byte_end: pe, label: pl, score: major });
                    }
                    continue;
                }
                match aggregation {
                    Aggregation::None => {
                        self.spans.push(Span { row: r as u32, byte_start: ws, byte_end: we, label, score: major })
                    }
                    // The mock has no sub-word tokens, so First and Max reduce to Simple.
                    Aggregation::Model | Aggregation::Simple | Aggregation::First | Aggregation::Max => {
                        if let Some((ps, pe, pl)) = prev {
                            if pl == label {
                                prev = Some((ps, we, pl));
                                continue;
                            }
                            self.spans.push(Span {
                                row: r as u32,
                                byte_start: ps,
                                byte_end: pe,
                                label: pl,
                                score: major,
                            });
                        }
                        prev = Some((ws, we, label));
                    }
                }
            }
            if let Some((ps, pe, pl)) = prev {
                self.spans.push(Span { row: r as u32, byte_start: ps, byte_end: pe, label: pl, score: major });
            }
        }
        if self.spans.capacity() != cap_before {
            self.note_alloc();
        }
        Ok(ProviderResult {
            outputs: vec![Output {
                name: self.names.scores.clone(),
                buffer: self.out.clone(),
                shape: vec![n as u64, seq as u64, l as u64],
            }],
            spans: self.spans.clone(),
        })
    }

    fn run_generic(&mut self) -> Result<ProviderResult> {
        let x = self.bound_x.clone().ok_or_else(|| Error::invalid_state("input `x` is not bound"))?;
        let xd = x.desc().clone();
        if xd.dtype != DType::F32 {
            return Err(Error::unsupported_dtype(format!("mock RUN input `x` must be f32, got {}", xd.dtype.name())));
        }
        if !xd.is_packed() {
            return Err(Error::unsupported("mock RUN input `x` must be packed row-major"));
        }
        let xp = x.host_ptr().ok_or_else(|| Error::unsupported_placement("mock RUN input `x` must be host-visible"))?;
        let n = xd.element_count() as usize;
        let y: Arc<dyn ProviderBuffer> = match &self.bound_y {
            Some(y) => {
                if y.desc().bytes < xd.bytes {
                    return Err(Error::capacity(format!(
                        "bound output `y` holds {} bytes but `x` is {} bytes",
                        y.desc().bytes,
                        xd.bytes
                    )));
                }
                if y.desc().dtype != DType::F32 {
                    return Err(Error::unsupported_dtype("mock RUN output `y` must be f32"));
                }
                y.clone()
            }
            None => match self.y.as_ref().filter(|b| b.desc().bytes == xd.bytes).cloned() {
                Some(b) => b,
                None => {
                    self.note_alloc();
                    let b = HostBuffer::packed(DType::F32, &xd.shape)?;
                    self.y = Some(b.clone());
                    b
                }
            },
        };
        let yp =
            y.host_ptr().ok_or_else(|| Error::unsupported_placement("mock RUN output `y` must be host-visible"))?;
        if xp == yp {
            return Err(Error::invalid_argument("mock RUN input `x` and output `y` alias the same memory"));
        }
        // SAFETY: sizes checked above; the core guarantees exclusive access during run.
        let xs = unsafe { std::slice::from_raw_parts(xp.as_ptr().cast::<f32>(), n) };
        let ys = unsafe { std::slice::from_raw_parts_mut(yp.as_ptr().cast::<f32>(), n) };
        for (o, i) in ys.iter_mut().zip(xs) {
            *o = i * 2.0;
        }
        Ok(ProviderResult {
            outputs: vec![Output { name: self.names.y.clone(), buffer: y, shape: xd.shape.clone() }],
            spans: Vec::new(),
        })
    }
}

impl ProviderSession for MockSession {
    fn write_text(&mut self, texts: &[&str], opts: &EmbedOptions) -> Result<()> {
        self.begin_write(texts.len());
        self.embed_opts = *opts;
        let budget = self.budget(opts.max_tokens)?;
        // Shared prefix strings: cloning the Arc does not allocate.
        let prefix = match opts.prompt_role {
            PromptRole::None => Arc::clone(&self.no_prefix),
            PromptRole::Query => Arc::clone(&self.prefix_query),
            PromptRole::Document => Arc::clone(&self.prefix_document),
        };
        for (r, t) in texts.iter().enumerate() {
            self.tokenize_row(r, &prefix, t, opts.truncate, budget)?;
        }
        Ok(())
    }

    fn write_tokens(&mut self, batch: &TokenBatch<'_>) -> Result<()> {
        self.begin_write(batch.batch as usize);
        let seq = self.max_seq as usize;
        for r in 0..batch.batch as usize {
            let base = r * seq;
            let ids = batch.ids_row(r);
            let mask = batch.mask_row(r);
            self.ids[base..base + ids.len()].copy_from_slice(ids);
            self.mask[base..base + mask.len()].copy_from_slice(mask);
            for c in ids.len()..seq {
                self.ids[base + c] = 0;
                self.mask[base + c] = 0;
            }
            self.row_ranges.push((self.word_spans.len(), self.word_spans.len()));
        }
        Ok(())
    }

    fn write_pairs(&mut self, query: &str, docs: &[&str], opts: &RerankOptions) -> Result<()> {
        self.begin_write(docs.len());
        self.rerank_opts = *opts;
        let budget = self.budget(opts.max_tokens)?;
        self.query_tokens.clear();
        for (_, w) in words(query) {
            if self.query_tokens.len() == self.query_tokens.capacity() {
                self.note_alloc();
            }
            self.query_tokens.push(word_id(w, self.vocab, self.salt));
        }
        for (r, d) in docs.iter().enumerate() {
            self.tokenize_row(r, "", d, opts.truncate, budget)?;
        }
        Ok(())
    }

    fn write_text_classify(&mut self, texts: &[&str], opts: &ClassifyOptions) -> Result<()> {
        self.begin_write(texts.len());
        self.classify_opts = *opts;
        let budget = self.budget(opts.max_tokens)?;
        for (r, t) in texts.iter().enumerate() {
            self.tokenize_row(r, "", t, opts.truncate, budget)?;
        }
        Ok(())
    }

    fn bind(&mut self, name: &str, buffer: Arc<dyn ProviderBuffer>) -> Result<()> {
        match name {
            "x" => self.bound_x = Some(buffer),
            "y" => self.bound_y = Some(buffer),
            other => return Err(Error::invalid_argument(format!("mock RUN has no tensor `{other}`"))),
        }
        Ok(())
    }

    fn run(&mut self, opts: &RunOptions) -> Result<ProviderResult> {
        opts.params.reject_unknown(&[], "mock run")?;
        // The session's injected fault, if it has one. It is raised before
        // any state moves, so a refused run leaves the session exactly as it
        // was; `run=panic` is the deliberate exception, and unwinds.
        if let Some(f) = self.run_fault {
            return Err(f.raise("session run"));
        }
        let result = match self.info.kind {
            ModelKind::Embedding => self.run_embed(),
            ModelKind::Reranker => self.run_rerank(),
            ModelKind::Classifier => self.run_classify(),
            ModelKind::TokenClassifier => self.run_token_classify(),
            ModelKind::Generic => self.run_generic(),
            ModelKind::Generative => {
                Err(Error::unsupported_task("generative models use turbo_generation_*, not sessions"))
            }
        }?;
        self.runs += 1;
        Ok(result)
    }

    fn stats(&self) -> Result<SessionStats> {
        let in_bytes = (self.ids.len() * 4 * 2) as u64;
        Ok(SessionStats {
            runs: self.runs,
            host_allocs: None,
            h2d_bytes: 0,
            d2h_bytes: 0,
            input_bytes: in_bytes,
            output_bytes: self.out.desc().bytes + self.sorted.desc().bytes,
            provider_allocs: Some(self.allocs.load(Ordering::Relaxed)),
        })
    }
}

fn embed_into(ids: &[i32], mask: &[i32], max_seq: usize, salt: u64, r: usize, out: &mut [f32], normalize: bool) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    let base = r * max_seq;
    let mut n = 0usize;
    for c in 0..max_seq {
        if mask[base + c] == 0 {
            continue;
        }
        let id = ids[base + c] as u64;
        for (k, v) in out.iter_mut().enumerate() {
            *v += unit(fnv(&id.to_le_bytes(), salt ^ (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
        }
        n += 1;
    }
    if n > 0 {
        for v in out.iter_mut() {
            *v /= n as f32;
        }
    }
    if normalize {
        l2(out);
    }
}

fn l2(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in v.iter_mut() {
        *x /= sum;
    }
}

struct MockGeneration {
    vocab: u32,
    salt: u64,
    max_seq: u32,
    max_new: u32,
    min_new: u32,
    seed: Option<u64>,
    /// `Some(pool)` when sampling: each step draws from `pool` candidate ids
    /// with a seed-derived generator, so temperature > 0 makes the output
    /// depend on the seed and top_k / top_p / min_p visibly shrink the pool.
    /// `None` is greedy: the same prompt always yields the same tokens.
    sampling: Option<u32>,
    stop: Vec<String>,
    stop_tokens: Vec<i32>,
    logprobs: u32,
    echo: bool,
    prompt_ids: Vec<i32>,
    prompt_text: String,
    state: u64,
    generated: u32,
    /// Text not yet delivered: with stop strings set, the last `hold_max`
    /// bytes wait until no stop string can still complete through them.
    hold: String,
    hold_max: usize,
    cancelled: bool,
    done: bool,
}

impl MockGeneration {
    fn start(&mut self) {
        // Greedy decoding depends on the prompt only; the seed enters through
        // the sampling draw, so temperature 0 ignores it like a real decoder.
        let mut h = self.salt;
        for id in &self.prompt_ids {
            h = fnv(&id.to_le_bytes(), h);
        }
        self.state = h;
        self.generated = 0;
        self.hold.clear();
    }

    /// End the stream; text held back for stop matching is the model's
    /// and goes out with the last chunk.
    fn finish(&mut self, out: &mut Chunk, reason: FinishReason) {
        out.text.push_str(&self.hold);
        self.hold.clear();
        out.done = true;
        out.finish_reason = reason;
        self.done = true;
    }

    /// Append one piece of text and deliver what no stop string can still
    /// reach: the piece plus the `hold_max` bytes before it are searched,
    /// the earliest match ends the stream before itself, and otherwise all
    /// but the last `hold_max` bytes go out.
    fn deliver(&mut self, piece: &str, out: &mut Chunk) -> bool {
        self.hold.push_str(piece);
        let mut from = self.hold.len().saturating_sub(piece.len() + self.hold_max);
        while from > 0 && !self.hold.is_char_boundary(from) {
            from -= 1;
        }
        let earliest = self
            .stop
            .iter()
            .filter(|s| !s.is_empty())
            .filter_map(|s| self.hold[from..].find(s.as_str()).map(|i| from + i))
            .min();
        if let Some(at) = earliest {
            out.text.push_str(&self.hold[..at]);
            self.hold.clear();
            return true;
        }
        if self.hold.len() > self.hold_max {
            let mut cut = self.hold.len() - self.hold_max;
            while cut > 0 && !self.hold.is_char_boundary(cut) {
                cut -= 1;
            }
            out.text.push_str(&self.hold[..cut]);
            self.hold.drain(..cut);
        }
        false
    }
}

impl ProviderGeneration for MockGeneration {
    fn prompt(&mut self, messages: &[Message<'_>]) -> Result<()> {
        self.prompt_ids.clear();
        self.prompt_text.clear();
        self.prompt_ids.push(MOCK_CLS);
        for m in messages {
            let _ = writeln!(self.prompt_text, "<{}>{}", m.role, m.content);
            for (_, w) in words(m.role).chain(words(m.content)) {
                self.prompt_ids.push(word_id(w, self.vocab, self.salt));
            }
        }
        self.prompt_ids.push(MOCK_SEP);
        if self.prompt_ids.len() > self.max_seq as usize {
            return Err(Error::capacity(format!(
                "prompt is {} tokens but the model's max_seq is {}",
                self.prompt_ids.len(),
                self.max_seq
            )));
        }
        self.start();
        Ok(())
    }

    fn prompt_tokens(&mut self, ids: &[i32]) -> Result<()> {
        if ids.len() > self.max_seq as usize {
            return Err(Error::capacity(format!(
                "prompt is {} tokens but the model's max_seq is {}",
                ids.len(),
                self.max_seq
            )));
        }
        self.prompt_ids.clear();
        self.prompt_ids.extend_from_slice(ids);
        self.prompt_text.clear();
        for id in ids {
            let _ = write!(self.prompt_text, "tok{id} ");
        }
        self.start();
        Ok(())
    }

    fn step(&mut self, out: &mut Chunk) -> Result<()> {
        out.prompt_tokens = self.prompt_ids.len() as u32;
        if self.done {
            return Err(Error::invalid_state("generation already finished"));
        }
        if self.cancelled {
            out.generated_tokens = self.generated;
            self.finish(out, FinishReason::Cancelled);
            return Ok(());
        }
        if self.generated >= self.max_new {
            out.generated_tokens = self.generated;
            self.finish(out, FinishReason::Length);
            return Ok(());
        }
        self.state = fnv(&self.generated.to_le_bytes(), self.state);
        // The greedy token is the rank-0 candidate; sampling draws a rank
        // below the pool size with the seeded generator and takes the
        // candidate at that rank, so a pool of one is greedy decoding and
        // a wider pool is a superset of it, as top_k is for a real model.
        let greedy = (self.state % self.vocab as u64) as i32;
        let mut token = match self.sampling {
            None => greedy,
            Some(pool) => {
                let draw = fnv(&self.seed.unwrap_or(self.salt).to_le_bytes(), self.state);
                let rank = (draw % pool as u64) as i32;
                if rank == 0 {
                    greedy
                } else {
                    let words = self.vocab as i32 - MOCK_FIRST_WORD;
                    MOCK_FIRST_WORD + (greedy.max(MOCK_FIRST_WORD) - MOCK_FIRST_WORD + rank).rem_euclid(words)
                }
            }
        };
        // EOS only once the minimum is met.
        if token == MOCK_SEP && self.generated < self.min_new {
            token = MOCK_FIRST_WORD;
        }
        if token == MOCK_CLS {
            token = MOCK_FIRST_WORD + 1;
        }
        self.generated += 1;
        out.generated_tokens = self.generated;
        out.tokens.push(token);
        if self.generated == 1 && self.echo {
            out.text.push_str(&self.prompt_text);
        }
        if self.logprobs > 0 {
            for k in 0..self.logprobs {
                out.logprobs.push(-0.05 * (k as f32 + 1.0));
            }
        }
        if token == MOCK_SEP {
            self.finish(out, FinishReason::Eos);
            return Ok(());
        }
        if self.stop_tokens.contains(&token) {
            // Like EOS: the token ends the stream and its text is withheld.
            self.finish(out, FinishReason::Stop);
            return Ok(());
        }
        let piece = format!("tok{token} ");
        if self.deliver(&piece, out) {
            out.done = true;
            out.finish_reason = FinishReason::Stop;
            self.done = true;
            return Ok(());
        }
        if self.generated >= self.max_new {
            self.finish(out, FinishReason::Length);
        }
        Ok(())
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }
}

/// Kinds of mock bundle [`write_mock_bundle`] can produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MockBundleKind {
    /// 8-dimensional embedder, mean pooling, L2, truncate_dims [4], prompts.
    Embedding,
    /// Cross-encoder reranker.
    Reranker,
    /// Three-label classifier.
    Classifier,
    /// Three-label token classifier (`O`, `PER`, `LOC`), simple aggregation.
    TokenClassifier,
    /// Generative model.
    Generative,
    /// Generic RUN model with `x` and `y`.
    Generic,
}

/// Write a verified mock bundle into `dir` (which must exist). Returns the manifest text.
pub fn write_mock_bundle(dir: &Path, kind: MockBundleKind) -> Result<String> {
    let artifact = serde_json::json!({ "vocab_size": MOCK_VOCAB, "salt": 7 }).to_string();
    std::fs::write(dir.join("mock.json"), &artifact)?;
    let sha = sha256_bytes(artifact.as_bytes());
    let (task, kind_name, contract) = match kind {
        MockBundleKind::Embedding => (
            "embed",
            "embedding",
            serde_json::json!({
                "pooling": "mean", "normalize": "l2", "max_seq": 16, "dim": 8,
                "truncate_dims": [4],
                "prompts": {"query": "query:", "document": "passage:"},
                "similarity_fn": "cosine", "dtype": "f32"
            }),
        ),
        MockBundleKind::Reranker => {
            ("rerank", "reranker", serde_json::json!({ "max_seq": 16, "activation": "sigmoid" }))
        }
        MockBundleKind::Classifier => (
            "classify",
            "classifier",
            serde_json::json!({ "max_seq": 16, "labels": ["negative", "neutral", "positive"], "activation": "softmax" }),
        ),
        MockBundleKind::TokenClassifier => (
            "token_classify",
            "token_classifier",
            serde_json::json!({ "max_seq": 16, "labels": ["O", "PER", "LOC"], "aggregation": "simple", "activation": "softmax", "tagging": "BIO" }),
        ),
        MockBundleKind::Generative => ("generate", "generative", serde_json::json!({ "max_seq": 512 })),
        MockBundleKind::Generic => ("run", "generic", serde_json::json!({})),
    };
    let manifest = serde_json::json!({
        "bundle_version": 2,
        "model_id": format!("turbo/mock-{kind_name}"),
        "revision": "mock",
        "license": "Apache-2.0",
        "task": task,
        "kind": kind_name,
        "modality": "text",
        "family": "mock",
        "tokenizer": { "kind": "mock", "files": {} },
        "contract": contract,
        "artifacts": { "mock": { "path": "mock.json", "sha256": sha } },
        "limits": { "max_batch": 8 }
    });
    let text = serde_json::to_string_pretty(&manifest)
        .map_err(|e| Error::internal(format!("serialize mock manifest: {e}")))?;
    std::fs::write(dir.join(MANIFEST_NAME), &text)?;
    Bundle::open(dir)?;
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_report_byte_offsets() {
        let w: Vec<_> = words("  héllo  world\tx").collect();
        assert_eq!(w, vec![(2, "héllo"), (10, "world"), (16, "x")]);
        assert_eq!(words("").count(), 0);
        assert_eq!(words("   ").count(), 0);
    }

    #[test]
    fn word_ids_stay_in_vocab() {
        for w in ["a", "b", "zzz", "héllo"] {
            let id = word_id(w, MOCK_VOCAB, 7);
            assert!(id >= MOCK_FIRST_WORD && (id as u32) < MOCK_VOCAB);
        }
    }
}
