//! NVIDIA CUDA provider for Turbo.
//!
//! The encoder runs through ONNX Runtime's CUDA execution provider with
//! IoBinding: token rows are tokenized natively on the host into pinned
//! memory, copied once to device input buffers that stay bound for the
//! session's lifetime, and the model's output stays on the device. Pooling,
//! L2 normalization, sigmoid, and softmax are the provider's own kernels
//! (`src/kernels.cu`), launched on the session's stream straight from the
//! model output. Results are device buffers exported as
//! `TURBO_HANDLE_CUDA_PTR`; nothing is pulled to the host unless the caller
//! reads it, or the task needs host post-processing (rerank ordering, span
//! aggregation), in which case the bytes moved are counted in the session's
//! `d2h_bytes`.
//!
//! Bundle contract: an `onnx` artifact with inputs `input_ids`,
//! `attention_mask`, and optionally `token_type_ids` (all rank 2, int64 or
//! int32) and exactly one float32 output; a BERT WordPiece `tokenizer.json`;
//! `contract.pooling`, `contract.normalize`, and `contract.dim` for embedding
//! models; `contract.labels` for classifiers.
//!
//! Device policy: one device per CUDA ordinal, kind GPU. Every cell is
//! `EXPERIMENTAL` until receipts land under `testdata/receipts/`.
//!
//! Runtime dependencies: the ONNX Runtime CUDA execution provider library
//! (`libonnxruntime_providers_cuda.so`, next to this provider's library) and
//! the CUDA 13 user-space libraries it links (cuBLAS, cuDNN 9, NVRTC). Those
//! are found through the loader's normal search, or preloaded from the
//! directory named by the context option `cuda_lib_dir` or the environment
//! variable `TURBO_CUDA_LIB_DIR`. A missing library is an error at context
//! creation, never a fallback to the CPU execution provider.

#![deny(missing_docs)]

pub mod cuda;
pub mod wordpiece;

use std::ffi::{c_void, CString};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

use ort::ep::ExecutionProvider;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{IoBinding, Session};
use ort::value::{DynTensorValueType, Shape, TensorElementType, TensorRefMut, ValueType};

use turbo_core::abi;
use turbo_core::buffer::{BufferDesc, HostBuffer, NativeHandle, ProviderBuffer};
use turbo_core::bundle::Bundle;
use turbo_core::error::{Error, Result};
use turbo_core::provider::{
    Capability, ClassifyOptions, ContextDesc, DeviceInfo, EmbedOptions, ModelDesc, ModelInfo, Output, Provider,
    ProviderContext, ProviderModel, ProviderResult, ProviderSession, RerankOptions, RunOptions, SessionDesc,
    SessionStats, Span, TokenBatch,
};
use turbo_core::types::{
    Aggregation, CapStatus, DType, DeviceKind, HandleKind, Modality, ModelKind, Normalize, OutputDType, Placement,
    Pooling, PromptRole, Stage, StagePlacement, StagePlacements, Task, Truncate,
};

use cuda::{DeviceMem, DeviceProps, PinnedMem, Stream};
use wordpiece::{RowScratch, Vocab, WordSpan};

/// Provider id.
pub const CUDA_PROVIDER_ID: &str = "cuda";
/// Artifact format the provider loads.
pub const ONNX_ARTIFACT: &str = "onnx";
/// Environment variable naming a directory of CUDA user-space libraries to preload.
pub const LIB_DIR_ENV: &str = "TURBO_CUDA_LIB_DIR";
/// Context option naming that directory.
pub const LIB_DIR_OPTION: &str = "cuda_lib_dir";

/// Capability bits every CUDA device reports.
pub const CUDA_CAPS: u64 = abi::TURBO_CAP_HOST_PTR_IMPORT
    | abi::TURBO_CAP_DEVICE_RESULT
    | abi::TURBO_CAP_DYNAMIC_SHAPE
    | abi::TURBO_CAP_WEIGHT_SHARING
    | abi::TURBO_CAP_DEVICE_POSTPROCESS
    | abi::TURBO_CAP_OPT_TRUNCATE
    | abi::TURBO_CAP_OPT_MAX_TOKENS
    | abi::TURBO_CAP_OPT_PROMPT_ROLE
    | abi::TURBO_CAP_OPT_NORMALIZE
    | abi::TURBO_CAP_OPT_POOLING_OVERRIDE
    | abi::TURBO_CAP_OPT_OUTPUT_DIM
    | abi::TURBO_CAP_OPT_TOP_N
    | abi::TURBO_CAP_OPT_AGGREGATION
    | abi::TURBO_CAP_OPT_RAW_SCORES;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const NVIDIA_VENDOR_ID: u32 = 0x10DE;

fn ort_err(what: &str, e: ort::Error) -> Error {
    Error::runtime(format!("onnxruntime {what}: {e}"))
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// The provider. Devices are probed once; free memory is refreshed per query.
#[derive(Default)]
pub struct CudaProvider {
    probe: OnceLock<std::result::Result<Vec<DeviceProps>, Error>>,
}

impl CudaProvider {
    /// Construct. No CUDA call happens until the runtime asks for devices.
    pub fn new() -> Self {
        Self::default()
    }

    fn props(&self) -> Result<&[DeviceProps]> {
        match self.probe.get_or_init(cuda::devices) {
            Ok(v) => Ok(v),
            Err(e) => Err(e.clone()),
        }
    }

    fn device(&self, ordinal: u32) -> Result<&DeviceProps> {
        let all = self.props()?;
        all.get(ordinal as usize).ok_or_else(|| {
            Error::device_not_found(format!(
                "cuda provider has {} device(s); ordinal {ordinal} does not exist",
                all.len()
            ))
        })
    }

    fn offers(task: Task, modality: Modality) -> bool {
        modality == Modality::Text && matches!(task, Task::Embed | Task::Rerank | Task::Classify | Task::TokenClassify)
    }
}

impl Provider for CudaProvider {
    fn id(&self) -> &str {
        CUDA_PROVIDER_ID
    }

    fn version(&self) -> &str {
        VERSION
    }

    fn devices(&self) -> Result<Vec<DeviceInfo>> {
        let props = self.props()?;
        let mut out = Vec::with_capacity(props.len());
        for p in props {
            let free = cuda::free_memory(p.index)?;
            out.push(DeviceInfo {
                kind: DeviceKind::Gpu,
                ordinal: p.index as u32,
                vendor_id: NVIDIA_VENDOR_ID,
                caps: CUDA_CAPS,
                memory_total: p.total_mem,
                memory_free: free,
                name: format!("{} (sm_{}{})", p.name, p.major, p.minor),
                vendor: "NVIDIA".into(),
                provider_id: CUDA_PROVIDER_ID.into(),
                provider_version: VERSION.into(),
                runtime_version: format!(
                    "onnxruntime {} / cudart {}",
                    ort_version(),
                    cuda::version_string(p.runtime_version)
                ),
                driver_version: cuda::version_string(p.driver_version),
            });
        }
        Ok(out)
    }

    fn capability(&self, ordinal: u32, task: Task, modality: Modality) -> Capability {
        let Ok(p) = self.device(ordinal) else { return Capability::unsupported() };
        if !Self::offers(task, modality) {
            return Capability::unsupported();
        }
        Capability {
            status: CapStatus::Experimental,
            dtype: Some(DType::F32),
            reference_dtype: Some(DType::F32),
            cosine_floor: 0.0,
            max_abs_error: 0.0,
            deterministic: false,
            notes: format!(
                "onnxruntime CUDA EP on {} with device-side pooling and activation kernels; precision and benchmark receipts pending",
                p.name
            ),
        }
    }

    fn can_run(&self, ordinal: u32, bundle: &Bundle, task: Task, modality: Modality) -> Result<()> {
        self.device(ordinal)?;
        if !Self::offers(task, modality) {
            return Err(Error::unsupported_task(format!(
                "cuda provider offers EMBED, RERANK, CLASSIFY, and TOKEN_CLASSIFY on TEXT, not {task:?} x {modality:?}"
            )));
        }
        if bundle.artifact(ONNX_ARTIFACT).is_none() {
            return Err(Error::bundle_no_artifact(format!(
                "bundle `{}` has no `onnx` artifact",
                bundle.manifest().model_id
            )));
        }
        let tokenizer_kind = bundle.manifest().tokenizer.as_ref().map(|t| t.kind.as_str()).unwrap_or("");
        if tokenizer_kind != "wordpiece" {
            return Err(Error::unsupported(format!(
                "cuda provider tokenizes BERT WordPiece natively; tokenizer kind `{tokenizer_kind}` is not supported here"
            )));
        }
        Kind::of(bundle.kind())?;
        Ok(())
    }

    fn create_context(&self, ordinal: u32, desc: &ContextDesc) -> Result<Arc<dyn ProviderContext>> {
        let props = self.device(ordinal)?.clone();
        desc.options.reject_unknown(&[LIB_DIR_OPTION], "cuda context")?;
        let lib_dir = match desc.options.get(LIB_DIR_OPTION) {
            Some(v) => Some(PathBuf::from(v)),
            None => std::env::var_os(LIB_DIR_ENV).map(PathBuf::from),
        };
        if let Some(dir) = &lib_dir {
            preload_libraries(dir)?;
        }
        let stream = Stream::new(props.index)?;
        Ok(Arc::new(CudaContext { inner: Arc::new(ContextInner { ordinal, props, stream }) }))
    }
}

fn ort_version() -> &'static str {
    // The version string of the ONNX Runtime library actually linked or
    // loaded (a Jetson build links a local one).
    ort::info()
}

/// Preload every `lib*.so*` in `dir` with `RTLD_GLOBAL` so the ONNX Runtime
/// CUDA execution provider resolves its CUDA and cuDNN dependencies from
/// there. Libraries are loaded in passes until every one is in, so ordering
/// within the directory does not matter. Any library that never loads is an
/// error naming it and the loader's reason.
fn preload_libraries(dir: &Path) -> Result<()> {
    static DONE: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
    let mut done = DONE.lock().map_err(|_| Error::internal("preload lock poisoned"))?;
    if done.iter().any(|d| d == dir) {
        return Ok(());
    }
    let mut pending: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| Error::device_unavailable(format!("cuda_lib_dir `{}`: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.starts_with("lib") && name.contains(".so") && p.is_file()
        })
        .collect();
    pending.sort();
    if pending.is_empty() {
        return Err(Error::device_unavailable(format!("cuda_lib_dir `{}` holds no shared libraries", dir.display())));
    }
    let mut last_errors = Vec::new();
    loop {
        let before = pending.len();
        let mut next = Vec::new();
        last_errors.clear();
        for p in pending {
            match dlopen_global(&p) {
                Ok(()) => {}
                Err(msg) => {
                    last_errors.push(format!("{}: {msg}", p.display()));
                    next.push(p);
                }
            }
        }
        pending = next;
        if pending.is_empty() {
            break;
        }
        if pending.len() == before {
            return Err(Error::device_unavailable(format!(
                "could not preload CUDA libraries from `{}`: {}",
                dir.display(),
                last_errors.join("; ")
            )));
        }
    }
    done.push(dir.to_path_buf());
    Ok(())
}

fn dlopen_global(path: &Path) -> std::result::Result<(), String> {
    let c = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| "path contains NUL".to_string())?;
    // SAFETY: plain dlopen of a file path; the handle is intentionally leaked
    // so the library stays resident for the process lifetime.
    let h = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
    if h.is_null() {
        // SAFETY: dlerror returns a static or thread-local string.
        let msg = unsafe { libc::dlerror() };
        let msg = if msg.is_null() {
            "unknown dlopen failure".to_string()
        } else {
            unsafe { std::ffi::CStr::from_ptr(msg) }.to_string_lossy().into_owned()
        };
        return Err(msg);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Context and buffers
// ---------------------------------------------------------------------------

/// Device state shared by a context and everything created from it.
struct ContextInner {
    ordinal: u32,
    props: DeviceProps,
    /// Transfer stream for buffer reads; sessions own their own streams.
    stream: Stream,
}

impl ContextInner {
    fn device(&self) -> i32 {
        self.props.index
    }
}

struct CudaContext {
    inner: Arc<ContextInner>,
}

enum Storage {
    Device(DeviceMem),
    Pinned(PinnedMem),
    ImportedDevice(NonNull<c_void>),
    ImportedHost(NonNull<u8>),
}

/// A buffer owned or wrapped by the CUDA provider.
pub struct CudaBuffer {
    desc: BufferDesc,
    storage: Storage,
    ctx: Arc<ContextInner>,
}

// SAFETY: imported pointers are used only through explicit copies the ABI
// makes the caller synchronize; owned memory is exclusively ours.
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    fn device_ptr(&self) -> Option<*mut c_void> {
        match &self.storage {
            Storage::Device(m) => Some(m.ptr()),
            Storage::ImportedDevice(p) => Some(p.as_ptr()),
            _ => None,
        }
    }
}

impl ProviderBuffer for CudaBuffer {
    fn desc(&self) -> &BufferDesc {
        &self.desc
    }

    fn host_ptr(&self) -> Option<NonNull<u8>> {
        match &self.storage {
            Storage::Pinned(m) => NonNull::new(m.ptr()),
            Storage::ImportedHost(p) => Some(*p),
            _ => None,
        }
    }

    fn read_to_host(&self, dst: &mut [u8]) -> Result<()> {
        if dst.len() as u64 != self.desc.bytes {
            return Err(Error::capacity(format!(
                "destination is {} bytes but the buffer holds {}",
                dst.len(),
                self.desc.bytes
            )));
        }
        match &self.storage {
            Storage::Device(_) | Storage::ImportedDevice(_) => {
                let src = self.device_ptr().expect("device storage");
                cuda::set_device(self.ctx.device())?;
                // SAFETY: dst is a live slice; src is our device allocation of desc.bytes.
                unsafe { cuda::copy_d2h(dst.as_mut_ptr(), src, dst.len(), &self.ctx.stream) }?;
                self.ctx.stream.synchronize()
            }
            Storage::Pinned(_) | Storage::ImportedHost(_) => {
                let src = self.host_ptr().expect("host storage");
                // SAFETY: src covers `bytes` bytes; the caller synchronizes access.
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), dst.len()) };
                Ok(())
            }
        }
    }

    fn export(&self, kind: HandleKind) -> Result<NativeHandle> {
        match (&self.storage, kind) {
            (Storage::Device(_) | Storage::ImportedDevice(_), HandleKind::CudaPtr) => Ok(NativeHandle {
                kind: HandleKind::CudaPtr,
                handle: self.device_ptr().expect("device storage") as u64,
                aux: self.ctx.device() as u64,
                offset: 0,
            }),
            (Storage::Pinned(_) | Storage::ImportedHost(_), HandleKind::HostPtr) => Ok(NativeHandle {
                kind: HandleKind::HostPtr,
                handle: self.host_ptr().expect("host storage").as_ptr() as u64,
                aux: 0,
                offset: 0,
            }),
            (Storage::Device(_) | Storage::ImportedDevice(_), other) => {
                Err(Error::unsupported(format!("cuda device buffers export TURBO_HANDLE_CUDA_PTR only, not {other:?}")))
            }
            (_, other) => {
                Err(Error::unsupported(format!("cuda host buffers export TURBO_HANDLE_HOST_PTR only, not {other:?}")))
            }
        }
    }
}

impl ProviderContext for CudaContext {
    fn ordinal(&self) -> u32 {
        self.inner.ordinal
    }

    fn alloc(&self, desc: &BufferDesc) -> Result<Arc<dyn ProviderBuffer>> {
        self.inner.alloc(desc)
    }

    fn import(&self, desc: &BufferDesc, handle: &NativeHandle) -> Result<Arc<dyn ProviderBuffer>> {
        self.inner.import(desc, handle)
    }

    fn load_model(&self, bundle: Arc<Bundle>, desc: &ModelDesc) -> Result<Arc<dyn ProviderModel>> {
        self.inner.load_model(bundle, desc)
    }
}

impl ContextInner {
    fn alloc(self: &Arc<Self>, desc: &BufferDesc) -> Result<Arc<dyn ProviderBuffer>> {
        let bytes = usize::try_from(desc.bytes).map_err(|_| Error::invalid_shape("byte count exceeds usize"))?;
        let storage = match desc.placement {
            Placement::Host => return Ok(HostBuffer::new(desc.clone())?),
            Placement::Pinned => Storage::Pinned(PinnedMem::new(bytes.max(1))?),
            Placement::Device => Storage::Device(DeviceMem::new(self.device(), bytes.max(1))?),
            Placement::Shared => {
                return Err(Error::unsupported_placement(
                    "cuda provider allocates HOST, PINNED, and DEVICE; SHARED (managed memory) is not offered",
                ))
            }
        };
        Ok(Arc::new(CudaBuffer { desc: desc.clone(), storage, ctx: self.clone() }))
    }

    fn import(self: &Arc<Self>, desc: &BufferDesc, handle: &NativeHandle) -> Result<Arc<dyn ProviderBuffer>> {
        if handle.offset != 0 {
            return Err(Error::unsupported("cuda provider imports handles with offset 0 only"));
        }
        let storage = match (handle.kind, desc.placement) {
            (HandleKind::HostPtr, Placement::Host | Placement::Pinned) => Storage::ImportedHost(
                NonNull::new(handle.handle as *mut u8)
                    .ok_or_else(|| Error::invalid_argument("imported host pointer is NULL"))?,
            ),
            (HandleKind::CudaPtr, Placement::Device) => {
                if handle.aux != 0 && handle.aux != self.device() as u64 {
                    return Err(Error::invalid_argument(format!(
                        "imported CUDA pointer belongs to device {} but this context is device {}",
                        handle.aux,
                        self.device()
                    )));
                }
                Storage::ImportedDevice(
                    NonNull::new(handle.handle as *mut c_void)
                        .ok_or_else(|| Error::invalid_argument("imported CUDA pointer is NULL"))?,
                )
            }
            (k, p) => {
                return Err(Error::unsupported(format!(
                    "cuda provider imports HOST_PTR as HOST/PINNED and CUDA_PTR as DEVICE, not {k:?} as {p:?}"
                )))
            }
        };
        Ok(Arc::new(CudaBuffer { desc: desc.clone(), storage, ctx: self.clone() }))
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Embedding,
    Reranker,
    Classifier,
    TokenClassifier,
}

impl Kind {
    fn of(kind: ModelKind) -> Result<Self> {
        match kind {
            ModelKind::Embedding => Ok(Self::Embedding),
            ModelKind::Reranker => Ok(Self::Reranker),
            ModelKind::Classifier => Ok(Self::Classifier),
            ModelKind::TokenClassifier => Ok(Self::TokenClassifier),
            other => Err(Error::unsupported_task(format!(
                "cuda provider does not serve {other:?} bundles (embedding, reranker, classifier, token_classifier)"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Activation {
    Softmax,
    Sigmoid,
    None,
}

impl Activation {
    fn parse(name: Option<&str>, default: Activation) -> Result<Self> {
        match name {
            None => Ok(default),
            Some("softmax") => Ok(Self::Softmax),
            Some("sigmoid") => Ok(Self::Sigmoid),
            Some("none") => Ok(Self::None),
            Some(other) => {
                Err(Error::bundle_invalid(format!("contract.activation `{other}` must be softmax, sigmoid, or none")))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Elem {
    I32,
    I64,
}

impl Elem {
    fn width(self) -> usize {
        match self {
            Elem::I32 => 4,
            Elem::I64 => 8,
        }
    }
}

struct ModelInner {
    ctx: Arc<ContextInner>,
    info: ModelInfo,
    kind: Kind,
    pool: Pooling,
    normalize: bool,
    activation: Activation,
    labels: Vec<String>,
    /// Output width: embedding dim or label count.
    width: u32,
    vocab: Vocab,
    session: Mutex<Session>,
    in_ids: String,
    in_mask: String,
    in_types: Option<String>,
    out_name: String,
    elem: Elem,
    aggregation: Aggregation,
}

struct CudaModel {
    inner: Arc<ModelInner>,
}

impl ContextInner {
    fn load_model(self: &Arc<Self>, bundle: Arc<Bundle>, desc: &ModelDesc) -> Result<Arc<dyn ProviderModel>> {
        desc.options.reject_unknown(&[], "cuda model")?;
        let m = bundle.manifest();
        if bundle.modality() != Modality::Text {
            return Err(Error::unsupported_modality("cuda provider serves text bundles only"));
        }
        let kind = Kind::of(bundle.kind())?;
        let c = bundle.contract();
        let (pool, normalize, activation, width) = match kind {
            Kind::Embedding => {
                if c.dim == 0 {
                    return Err(Error::bundle_invalid("embedding bundle must declare contract.dim"));
                }
                let pool = bundle
                    .pooling()?
                    .ok_or_else(|| Error::bundle_invalid("embedding bundle must declare contract.pooling"))?;
                let normalize = bundle.normalize()?.ok_or_else(|| {
                    Error::bundle_invalid("embedding bundle must declare contract.normalize (l2 or none)")
                })?;
                (pool, normalize == Normalize::L2, Activation::None, c.dim)
            }
            Kind::Reranker => {
                (Pooling::Cls, false, Activation::parse(c.activation.as_deref(), Activation::Sigmoid)?, 1)
            }
            Kind::Classifier => {
                if c.labels.is_empty() {
                    return Err(Error::bundle_invalid("classifier bundle must declare contract.labels"));
                }
                (
                    Pooling::Cls,
                    false,
                    Activation::parse(c.activation.as_deref(), Activation::Softmax)?,
                    c.labels.len() as u32,
                )
            }
            Kind::TokenClassifier => {
                if c.labels.is_empty() {
                    return Err(Error::bundle_invalid("token classifier bundle must declare contract.labels"));
                }
                (
                    Pooling::Cls,
                    false,
                    Activation::parse(c.activation.as_deref(), Activation::Softmax)?,
                    c.labels.len() as u32,
                )
            }
        };
        let aggregation = bundle.aggregation()?.unwrap_or(Aggregation::Simple);
        if aggregation == Aggregation::Model {
            return Err(Error::bundle_invalid("contract.aggregation must name a strategy, not `model`"));
        }
        let path = bundle.artifact_path(ONNX_ARTIFACT)?;

        // Tokenizer.
        let spec = m.tokenizer.as_ref().ok_or_else(|| {
            Error::bundle_invalid("bundle has no tokenizer section; the cuda provider tokenizes natively")
        })?;
        let tok = spec.files.get("tokenizer.json").ok_or_else(|| {
            Error::bundle_invalid("bundle has no tokenizer.json; the cuda provider tokenizes natively")
        })?;
        if spec.kind != "wordpiece" {
            return Err(Error::unsupported(format!(
                "cuda provider tokenizes BERT WordPiece natively; tokenizer kind `{}` is not supported here",
                spec.kind
            )));
        }
        let vocab = Vocab::load(&bundle.resolve(&tok.path)?)?;

        // ONNX Runtime session on this device. Registration failure is an
        // error: the CPU execution provider is never a substitute.
        cuda::set_device(self.device())?;
        let ep = ort::ep::CUDA::default().with_device_id(self.device());
        if !ep.is_available().map_err(|e| ort_err("probe the CUDA execution provider", e))? {
            return Err(Error::device_unavailable(
                "the ONNX Runtime CUDA execution provider is not available in this build",
            ));
        }
        let session = Session::builder()
            .map_err(|e| ort_err("session builder", e))?
            .with_execution_providers([ep.build().error_on_failure()])
            .map_err(|e| ort_err("register the CUDA execution provider", e.into()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| ort_err("optimization level", e.into()))?
            .with_intra_threads(1)
            .map_err(|e| ort_err("thread count", e.into()))?
            .commit_from_file(&path)
            .map_err(|e| ort_err(&format!("load `{}` on CUDA device {}", path.display(), self.device()), e))?;

        // Inputs: input_ids, attention_mask, optional token_type_ids; rank 2; one integer width.
        let (mut in_ids, mut in_mask, mut in_types, mut elem) = (None, None, None, None);
        for input in session.inputs() {
            let name = input.name();
            let ValueType::Tensor { ty, shape, .. } = input.dtype() else {
                return Err(Error::unsupported(format!("model input `{name}` is not a tensor")));
            };
            if shape.len() != 2 {
                return Err(Error::unsupported(format!(
                    "model input `{name}` must be rank 2 [batch, seq]; it is rank {}",
                    shape.len()
                )));
            }
            let e = match ty {
                TensorElementType::Int64 => Elem::I64,
                TensorElementType::Int32 => Elem::I32,
                other => {
                    return Err(Error::unsupported_dtype(format!(
                        "model input `{name}` is {other:?}; int64 or int32 expected"
                    )))
                }
            };
            if let Some(prev) = elem {
                if prev != e {
                    return Err(Error::unsupported("model inputs mix int32 and int64 element types"));
                }
            }
            elem = Some(e);
            match name {
                "input_ids" => in_ids = Some(name.to_string()),
                "attention_mask" => in_mask = Some(name.to_string()),
                "token_type_ids" => in_types = Some(name.to_string()),
                other => {
                    return Err(Error::unsupported(format!(
                        "model input `{other}` is not one of input_ids, attention_mask, token_type_ids"
                    )))
                }
            }
        }
        let (Some(in_ids), Some(in_mask), Some(elem)) = (in_ids, in_mask, elem) else {
            return Err(Error::unsupported("model must have input_ids and attention_mask inputs"));
        };
        // Output: exactly one float32 tensor of the rank the task needs.
        if session.outputs().len() != 1 {
            return Err(Error::unsupported(format!(
                "model must have exactly one output; `{}` has {}",
                m.model_id,
                session.outputs().len()
            )));
        }
        let output = &session.outputs()[0];
        let out_name = output.name().to_string();
        let ValueType::Tensor { ty, shape, .. } = output.dtype() else {
            return Err(Error::unsupported("model output is not a tensor"));
        };
        if *ty != TensorElementType::Float32 {
            return Err(Error::unsupported_dtype(format!("model output is {ty:?}; float32 expected")));
        }
        let want_rank = match kind {
            Kind::Embedding | Kind::TokenClassifier => 3,
            Kind::Reranker | Kind::Classifier => 2,
        };
        if shape.len() != want_rank {
            return Err(Error::unsupported(format!(
                "{kind:?} model output must be rank {want_rank}; `{out_name}` is rank {}",
                shape.len()
            )));
        }
        let last = *shape.last().expect("rank >= 2");
        if last >= 0 && last as u32 != width {
            return Err(Error::bundle_invalid(format!(
                "model output width {last} does not match the contract ({width})"
            )));
        }

        let max_seq = if c.max_seq == 0 { 512 } else { c.max_seq };
        let max_batch = if m.limits.max_batch == 0 { 32 } else { m.limits.max_batch };
        let dev = StagePlacement::Device;
        let stages = StagePlacements::NONE
            .with(Stage::Tokenize, StagePlacement::Host)
            .with(Stage::Encode, dev)
            .with(Stage::Pool, if kind == Kind::Embedding { dev } else { StagePlacement::Unused })
            .with(Stage::Normalize, if kind == Kind::Embedding && normalize { dev } else { StagePlacement::Unused })
            .with(
                Stage::Postprocess,
                match kind {
                    Kind::Embedding => StagePlacement::Unused,
                    Kind::Reranker | Kind::Classifier => dev,
                    // Softmax runs on the device; span aggregation is host work.
                    Kind::TokenClassifier => StagePlacement::Host,
                },
            );
        let info = ModelInfo {
            task: match kind {
                Kind::Embedding => Task::Embed,
                Kind::Reranker => Task::Rerank,
                Kind::Classifier => Task::Classify,
                Kind::TokenClassifier => Task::TokenClassify,
            },
            kind: bundle.kind(),
            modality: Modality::Text,
            dim: if kind == Kind::Embedding { width } else { 0 },
            labels: c.labels.clone(),
            pooling: (kind == Kind::Embedding).then_some(pool),
            normalize: (kind == Kind::Embedding).then_some(if normalize { Normalize::L2 } else { Normalize::None }),
            aggregation: (kind == Kind::TokenClassifier).then_some(aggregation),
            max_seq,
            max_batch,
            dtype_used: Some(DType::F32),
            stages,
            inputs: Vec::new(),
            outputs: Vec::new(),
            vocab_size: c.vocab_size,
            model_id: m.model_id.clone(),
            revision: m.revision.clone(),
            tokenizer_sha256: bundle.tokenizer_sha256().to_string(),
            provider_id: CUDA_PROVIDER_ID.into(),
            prefix_query: c.prompts.query.clone(),
            prefix_document: c.prompts.document.clone(),
        };
        let inner = Arc::new(ModelInner {
            ctx: self.clone(),
            info,
            kind,
            pool,
            normalize,
            activation,
            labels: c.labels.clone(),
            width,
            vocab,
            session: Mutex::new(session),
            in_ids,
            in_mask,
            in_types,
            out_name,
            elem,
            aggregation,
        });
        Ok(Arc::new(CudaModel { inner }))
    }
}

impl ProviderModel for CudaModel {
    fn info(&self) -> &ModelInfo {
        &self.inner.info
    }

    fn create_session(&self, desc: &SessionDesc) -> Result<Box<dyn ProviderSession>> {
        desc.options.reject_unknown(&[], "cuda session")?;
        if desc.max_batch < 1 || desc.max_seq < 2 {
            return Err(Error::invalid_argument("session needs max_batch >= 1 and max_seq >= 2"));
        }
        Ok(Box::new(CudaSession::new(self.inner.clone(), desc.max_batch, desc.max_seq)?))
    }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Written {
    Nothing,
    Embed,
    Tokens,
    Pairs,
    Classify,
}

struct CudaSession {
    model: Arc<ModelInner>,
    batch: u32,
    seq: u32,
    stream: Stream,
    // Host staging (i32), row stride `seq`.
    ids: Vec<i32>,
    mask: Vec<i32>,
    types: Vec<i32>,
    lengths: Vec<u32>,
    scratch: RowScratch,
    pos_scratch: Vec<i32>,
    // Pinned staging in the model's input width, compacted to `used_seq`.
    staging: PinnedMem,
    d_ids: DeviceMem,
    d_mask: DeviceMem,
    d_types: Option<DeviceMem>,
    // Result storage.
    out: Arc<CudaBuffer>,
    sorted: Arc<HostBuffer>,
    readback: PinnedMem,
    readback_valid: bool,
    binding: IoBinding,
    words: Vec<Vec<WordSpan>>,
    spans: Vec<Span>,
    eopts: EmbedOptions,
    ropts: RerankOptions,
    copts: ClassifyOptions,
    written: Written,
    n_rows: u32,
    used_seq: u32,
    runs: u64,
    h2d: u64,
    d2h: u64,
    name0: Arc<str>,
    name1: Arc<str>,
}

// SAFETY: sessions are single-owner by the core's contract; the raw device
// pointers and ORT binding are touched only under that ownership.
unsafe impl Send for CudaSession {}

impl CudaSession {
    fn new(model: Arc<ModelInner>, batch: u32, seq: u32) -> Result<Self> {
        let device = model.ctx.device();
        cuda::set_device(device)?;
        let n = batch as usize * seq as usize;
        let width = model.elem.width();
        let stream = Stream::new(device)?;
        let staging = PinnedMem::new(3 * n * width)?;
        let d_ids = DeviceMem::new(device, n * width)?;
        let d_mask = DeviceMem::new(device, n * width)?;
        let d_types = if model.in_types.is_some() { Some(DeviceMem::new(device, n * width)?) } else { None };
        let out_shape: Vec<u64> = match model.kind {
            Kind::Embedding | Kind::Classifier => vec![batch as u64, model.width as u64],
            Kind::Reranker => vec![batch as u64],
            Kind::TokenClassifier => vec![batch as u64, seq as u64, model.width as u64],
        };
        let out_desc = BufferDesc::packed(Placement::Device, DType::F32, &out_shape)?;
        let out = Arc::new(CudaBuffer {
            storage: Storage::Device(DeviceMem::new(device, out_desc.bytes as usize)?),
            desc: out_desc.clone(),
            ctx: model.ctx.clone(),
        });
        let sorted = HostBuffer::packed(DType::I32, &[batch as u64])?;
        let readback = PinnedMem::new(out_desc.bytes as usize)?;
        let binding = {
            let session = model.session.lock().map_err(|_| Error::internal("model session lock poisoned"))?;
            let mut b = session.create_binding().map_err(|e| ort_err("create IoBinding", e))?;
            let mem = MemoryInfo::new(AllocationDevice::CUDA, device, AllocatorType::Device, MemoryType::Default)
                .map_err(|e| ort_err("CUDA memory info", e))?;
            b.bind_output_to_device(model.out_name.as_str(), &mem).map_err(|e| ort_err("bind output", e))?;
            b
        };
        let name0: Arc<str> = Arc::from(if model.kind == Kind::Embedding { "embeddings" } else { "scores" });
        Ok(Self {
            batch,
            seq,
            stream,
            ids: vec![0; n],
            mask: vec![0; n],
            types: vec![0; n],
            lengths: vec![0; batch as usize],
            scratch: RowScratch::new(seq),
            pos_scratch: vec![0; seq as usize],
            staging,
            d_ids,
            d_mask,
            d_types,
            out,
            sorted,
            readback,
            readback_valid: false,
            binding,
            words: (0..batch).map(|_| Vec::with_capacity(seq as usize)).collect(),
            spans: Vec::with_capacity(n),
            eopts: EmbedOptions::default(),
            ropts: RerankOptions::default(),
            copts: ClassifyOptions::default(),
            written: Written::Nothing,
            n_rows: 0,
            used_seq: 0,
            runs: 0,
            h2d: 0,
            d2h: 0,
            name0,
            name1: Arc::from("sorted"),
            model,
        })
    }

    fn budget(&self, max_tokens: u32) -> Result<u32> {
        let b = if max_tokens == 0 { self.seq } else { max_tokens };
        if b > self.seq {
            return Err(Error::capacity(format!("max_tokens {b} exceeds the session's max_seq {}", self.seq)));
        }
        if b < 2 {
            return Err(Error::capacity("token budget must be at least 2 for [CLS] and [SEP]"));
        }
        Ok(b)
    }

    fn check_count(&self, count: usize) -> Result<u32> {
        if count == 0 || count > self.batch as usize {
            return Err(Error::capacity(format!(
                "{count} inputs written but the session holds 1..{} rows",
                self.batch
            )));
        }
        Ok(count as u32)
    }

    fn row(&mut self, r: usize) -> (&mut [i32], &mut [i32], &mut [i32]) {
        let s = self.seq as usize;
        let base = r * s;
        (&mut self.ids[base..base + s], &mut self.mask[base..base + s], &mut self.types[base..base + s])
    }

    fn encode_rows(
        &mut self,
        texts: &[&str],
        prefix: &str,
        truncate: Truncate,
        budget: u32,
        words: bool,
    ) -> Result<()> {
        let seq = self.seq;
        let vocab_model = self.model.clone();
        for (r, text) in texts.iter().enumerate() {
            let s = seq as usize;
            let base = r * s;
            let mut word_vec = if words { Some(std::mem::take(&mut self.words[r])) } else { None };
            let n = wordpiece::encode_row(
                &vocab_model.vocab,
                r as u32,
                prefix,
                text,
                truncate,
                budget,
                seq,
                &mut self.scratch,
                &mut self.ids[base..base + s],
                &mut self.mask[base..base + s],
                word_vec.as_mut(),
            )?;
            if let Some(w) = word_vec {
                self.words[r] = w;
            } else {
                self.words[r].clear();
            }
            for t in &mut self.types[base..base + s] {
                *t = 0;
            }
            self.lengths[r] = n;
        }
        Ok(())
    }

    /// Compact rows to `used_seq` columns in the model's element width,
    /// upload, and bind the input tensors at `[n_rows, used_seq]`.
    fn upload_and_bind(&mut self) -> Result<()> {
        let n_rows = self.n_rows as usize;
        let used = self.used_seq as usize;
        let seq = self.seq as usize;
        let elems = n_rows * used;
        let width = self.model.elem.width();
        let section = self.batch as usize * seq * width;
        // Sections: ids, mask, types.
        let base = self.staging.ptr();
        for (k, src) in [&self.ids, &self.mask, &self.types].into_iter().enumerate() {
            if k == 2 && self.d_types.is_none() {
                break;
            }
            let dst = base.wrapping_add(k * section);
            for r in 0..n_rows {
                let row = &src[r * seq..r * seq + used];
                match self.model.elem {
                    Elem::I32 => {
                        // SAFETY: dst section holds batch*seq i32; r*used+used <= batch*seq.
                        let d = unsafe { std::slice::from_raw_parts_mut(dst.cast::<i32>().add(r * used), used) };
                        d.copy_from_slice(row);
                    }
                    Elem::I64 => {
                        let d = unsafe { std::slice::from_raw_parts_mut(dst.cast::<i64>().add(r * used), used) };
                        for (o, &v) in d.iter_mut().zip(row) {
                            *o = v as i64;
                        }
                    }
                }
            }
        }
        cuda::set_device(self.model.ctx.device())?;
        let bytes = elems * width;
        // SAFETY: the pinned staging sections are `section` bytes each and outlive the copies (synchronized below).
        unsafe {
            cuda::copy_h2d(&self.d_ids, base, bytes, &self.stream)?;
            cuda::copy_h2d(&self.d_mask, base.wrapping_add(section), bytes, &self.stream)?;
        }
        self.h2d += 2 * bytes as u64;
        if let Some(t) = &self.d_types {
            unsafe { cuda::copy_h2d(t, base.wrapping_add(2 * section), bytes, &self.stream) }?;
            self.h2d += bytes as u64;
        }
        // The execution provider runs on its own stream; make the uploads
        // visible before it starts.
        self.stream.synchronize()?;
        let device = self.model.ctx.device();
        let shape = [n_rows as i64, used as i64];
        let bind = |b: &mut IoBinding, name: &str, mem: &DeviceMem| -> Result<()> {
            let info = MemoryInfo::new(AllocationDevice::CUDA, device, AllocatorType::Device, MemoryType::Default)
                .map_err(|e| ort_err("CUDA memory info", e))?;
            match self.model.elem {
                Elem::I64 => {
                    // SAFETY: the device buffer holds at least n_rows*used i64 and outlives the binding's use.
                    let t = unsafe { TensorRefMut::<i64>::from_raw(info, mem.ptr(), Shape::new(shape)) }
                        .map_err(|e| ort_err("wrap device input", e))?;
                    b.bind_input(name, &*t).map_err(|e| ort_err("bind input", e))
                }
                Elem::I32 => {
                    let t = unsafe { TensorRefMut::<i32>::from_raw(info, mem.ptr(), Shape::new(shape)) }
                        .map_err(|e| ort_err("wrap device input", e))?;
                    b.bind_input(name, &*t).map_err(|e| ort_err("bind input", e))
                }
            }
        };
        bind(&mut self.binding, self.model.in_ids.as_str(), &self.d_ids)?;
        bind(&mut self.binding, self.model.in_mask.as_str(), &self.d_mask)?;
        if let (Some(name), Some(mem)) = (&self.model.in_types, &self.d_types) {
            bind(&mut self.binding, name.as_str(), mem)?;
        }
        // ONNX Runtime keeps the OrtValue it allocated for a bound output
        // and reuses it on the next run of the same binding. A run whose
        // shape is smaller than the previous one then fails inside the
        // graph with `INVALID_ARGUMENT: The output OrtValue provided for
        // output ...`, so the output is bound afresh for every run and the
        // execution provider allocates for this shape.
        self.binding.clear_outputs();
        let out_info = MemoryInfo::new(AllocationDevice::CUDA, device, AllocatorType::Device, MemoryType::Default)
            .map_err(|e| ort_err("CUDA memory info", e))?;
        self.binding
            .bind_output_to_device(self.model.out_name.as_str(), &out_info)
            .map_err(|e| ort_err("bind output", e))?;
        Ok(())
    }

    fn out_ptr(&self) -> *mut f32 {
        self.out.device_ptr().expect("device result").cast()
    }

    /// Run the graph and the post-processing kernels; leaves the result in `out`.
    fn execute(&mut self) -> Result<()> {
        let n_rows = self.n_rows as i32;
        let used = self.used_seq as i32;
        let width = self.model.width as i32;
        let raw = match self.written {
            Written::Pairs => self.ropts.raw_scores,
            Written::Classify => self.copts.raw_scores,
            _ => false,
        };
        let model = self.model.clone();
        let mut session = model.session.lock().map_err(|_| Error::internal("model session lock poisoned"))?;
        let outputs = session.run_binding(&self.binding).map_err(|e| ort_err("run", e))?;
        let value = outputs
            .get(model.out_name.as_str())
            .ok_or_else(|| Error::internal(format!("output `{}` missing from the run", model.out_name)))?;
        let tensor = value.downcast_ref::<DynTensorValueType>().map_err(|e| ort_err("output is not a tensor", e))?;
        let shape: Vec<i64> = tensor.shape().to_vec();
        let expect: Vec<i64> = match model.kind {
            Kind::Embedding | Kind::TokenClassifier => vec![n_rows as i64, used as i64, width as i64],
            Kind::Reranker | Kind::Classifier => vec![n_rows as i64, width as i64],
        };
        if shape != expect {
            return Err(Error::runtime(format!("model output shape {shape:?} does not match the expected {expect:?}")));
        }
        let src = tensor.data_ptr().cast::<f32>();
        if src.is_null() {
            return Err(Error::runtime("model output has no data pointer"));
        }
        let mask_width = model.elem.width() as i32;
        let stream = self.stream.raw();
        let out = self.out_ptr();
        // The EP synchronized its stream when `run` returned; our kernels
        // consume the output on the session stream.
        let rc = match model.kind {
            Kind::Embedding => {
                let pool = match self.eopts.pooling {
                    Pooling::Model => model.pool,
                    other => other,
                };
                let mode = match pool {
                    Pooling::Mean => 0,
                    Pooling::Cls => 1,
                    Pooling::Last => 2,
                    Pooling::Model => unreachable!("resolved above"),
                };
                let normalize = match self.eopts.normalize {
                    Normalize::Model => model.normalize,
                    Normalize::L2 => true,
                    Normalize::None => false,
                };
                let out_dim = if self.eopts.output_dim == 0 { width } else { self.eopts.output_dim as i32 };
                // SAFETY: all pointers are live device allocations of the stated extents.
                unsafe {
                    cuda::turbo_cuda_pool(
                        src,
                        self.d_mask.ptr(),
                        mask_width,
                        out,
                        n_rows,
                        used,
                        width,
                        out_dim,
                        mode,
                        normalize as i32,
                        stream,
                    )
                }
            }
            Kind::Reranker | Kind::Classifier => {
                let n = n_rows * width;
                let act = if raw { Activation::None } else { model.activation };
                match act {
                    Activation::None => {
                        // SAFETY: both are device allocations of at least n f32.
                        unsafe { cuda::copy_d2d(out.cast(), src.cast(), n as usize * 4, &self.stream) }?;
                        0
                    }
                    Activation::Sigmoid => unsafe { cuda::turbo_cuda_sigmoid(src, out, n, stream) },
                    Activation::Softmax => unsafe { cuda::turbo_cuda_softmax_rows(src, out, n_rows, width, stream) },
                }
            }
            Kind::TokenClassifier => {
                let act = if raw { Activation::None } else { model.activation };
                match act {
                    Activation::None => {
                        // SAFETY: both are device allocations of at least n_rows*used*width f32.
                        unsafe {
                            cuda::copy_d2d(out.cast(), src.cast(), (n_rows * used * width) as usize * 4, &self.stream)
                        }?;
                        0
                    }
                    Activation::Sigmoid => unsafe { cuda::turbo_cuda_sigmoid(src, out, n_rows * used * width, stream) },
                    Activation::Softmax => unsafe {
                        cuda::turbo_cuda_softmax_rows(src, out, n_rows * used, width, stream)
                    },
                }
            }
        };
        cuda::check(rc, "post-processing kernel launch")?;
        self.stream.synchronize()?;
        drop(outputs);
        Ok(())
    }

    /// Copy the logical result to pinned host memory once per run.
    fn read_back(&mut self, bytes: usize) -> Result<()> {
        if self.readback_valid {
            return Ok(());
        }
        // SAFETY: readback and out are both at least `bytes` long (the result's logical size).
        unsafe { cuda::copy_d2h(self.readback.ptr(), self.out_ptr().cast(), bytes, &self.stream) }?;
        self.stream.synchronize()?;
        self.d2h += bytes as u64;
        self.readback_valid = true;
        Ok(())
    }

    fn token_label(probs: &[f32], width: usize, row: usize, seq: usize, col: usize) -> (u32, f32) {
        let p = &probs[(row * seq + col) * width..(row * seq + col + 1) * width];
        let mut best = 0usize;
        for l in 1..width {
            if p[l] > p[best] {
                best = l;
            }
        }
        (best as u32, p[best])
    }

    fn entity_of(label: &str) -> &str {
        let b = label.as_bytes();
        if b.len() > 2 && b[1] == b'-' && matches!(b[0], b'B' | b'I' | b'L' | b'U' | b'E' | b'S') {
            &label[2..]
        } else {
            label
        }
    }

    /// Word-aligned span aggregation over the device softmax.
    ///
    /// Word label: first sub-token (`SIMPLE`, `FIRST`) or the highest-scoring
    /// sub-token (`MAX`). Consecutive words of one entity merge unless a
    /// `B-`/`U-`/`S-` tag starts a new one; a group's score is the mean of
    /// its word scores. Spans are word-aligned: a sub-word entity change
    /// inside one word is not represented, which is where `SIMPLE` differs
    /// from Hugging Face's token-level grouping.
    fn aggregate_spans(&mut self) -> Result<()> {
        let agg = match self.copts.aggregation {
            Aggregation::Model => self.model.aggregation,
            other => other,
        };
        let n_rows = self.n_rows as usize;
        let seq = self.used_seq as usize;
        let width = self.model.width as usize;
        let bytes = n_rows * seq * width * 4;
        self.read_back(bytes)?;
        // SAFETY: readback holds `bytes` initialized f32 written by the D2H copy above.
        let probs: &[f32] =
            unsafe { std::slice::from_raw_parts(self.readback.ptr().cast::<f32>(), n_rows * seq * width) };
        self.spans.clear();
        let labels = &self.model.labels;
        for r in 0..n_rows {
            let mut open: Option<(Span, usize, f32, &str)> = None; // span, word count, score sum, entity
            for w in &self.words[r] {
                if (w.first_token + w.n_tokens) as usize > seq {
                    break;
                }
                let (mut label, mut score) = Self::token_label(probs, width, r, seq, w.first_token as usize);
                if agg == Aggregation::Max {
                    for t in 1..w.n_tokens as usize {
                        let (l2, s2) = Self::token_label(probs, width, r, seq, w.first_token as usize + t);
                        if s2 > score {
                            score = s2;
                            label = l2;
                        }
                    }
                }
                let name = labels[label as usize].as_str();
                let outside = name == "O";
                if agg == Aggregation::None {
                    if !outside {
                        self.spans.push(Span { row: r as u32, byte_start: w.start, byte_end: w.end, label, score });
                    }
                    continue;
                }
                let entity = Self::entity_of(name);
                let begins =
                    name.len() > 1 && matches!(name.as_bytes()[0], b'B' | b'U' | b'S') && name.as_bytes()[1] == b'-';
                if let Some((span, count, sum, ent)) = open.take() {
                    if outside || ent != entity || begins {
                        self.spans.push(Span { score: sum / count as f32, ..span });
                    } else {
                        open = Some((Span { byte_end: w.end, ..span }, count + 1, sum + score, ent));
                        continue;
                    }
                }
                if !outside {
                    open = Some((
                        Span { row: r as u32, byte_start: w.start, byte_end: w.end, label, score },
                        1,
                        score,
                        entity,
                    ));
                }
            }
            if let Some((span, count, sum, _)) = open {
                self.spans.push(Span { score: sum / count as f32, ..span });
            }
        }
        Ok(())
    }
}

impl ProviderSession for CudaSession {
    fn write_text(&mut self, texts: &[&str], opts: &EmbedOptions) -> Result<()> {
        if self.model.kind != Kind::Embedding {
            return Err(Error::unsupported_task("write_text needs an embedding model"));
        }
        let count = self.check_count(texts.len())?;
        if !matches!(opts.output_dtype, OutputDType::Model | OutputDType::F32) {
            return Err(Error::unsupported_option(EmbedOptions::FIELD_OUTPUT_DTYPE, "output_dtype", CUDA_PROVIDER_ID));
        }
        if opts.output_dim > self.model.width {
            return Err(Error::invalid_argument(format!(
                "output_dim {} exceeds the model dimension {}",
                opts.output_dim, self.model.width
            ))
            .with_field(EmbedOptions::FIELD_OUTPUT_DIM));
        }
        let budget = self.budget(opts.max_tokens)?;
        let prefix = match opts.prompt_role {
            PromptRole::None => String::new(),
            PromptRole::Query => self.model.info.prefix_query.clone(),
            PromptRole::Document => self.model.info.prefix_document.clone(),
        };
        self.encode_rows(texts, &prefix, opts.truncate, budget, false)?;
        self.eopts = *opts;
        self.finish_write(count, Written::Embed, false)
    }

    fn write_tokens(&mut self, batch: &TokenBatch<'_>) -> Result<()> {
        if batch.batch > self.batch || batch.seq > self.seq {
            return Err(Error::capacity(format!(
                "token batch [{}, {}] exceeds the session shape [{}, {}]",
                batch.batch, batch.seq, self.batch, self.seq
            )));
        }
        let pad = self.model.vocab.pad_id();
        let seq = self.seq as usize;
        for r in 0..batch.batch as usize {
            let ids = batch.ids_row(r);
            let mask = batch.mask_row(r);
            let (row_ids, row_mask, row_types) = self.row(r);
            row_ids[..ids.len()].copy_from_slice(ids);
            row_mask[..mask.len()].copy_from_slice(mask);
            for c in ids.len()..seq {
                row_ids[c] = pad;
                row_mask[c] = 0;
            }
            match batch.types {
                Some(t) => {
                    let s = r * batch.row_stride as usize;
                    row_types[..ids.len()].copy_from_slice(&t[s..s + ids.len()]);
                    for v in &mut row_types[ids.len()..] {
                        *v = 0;
                    }
                }
                None => {
                    for v in row_types.iter_mut() {
                        *v = 0;
                    }
                }
            }
            self.lengths[r] = batch.seq;
            self.words[r].clear();
        }
        self.eopts = EmbedOptions::default();
        self.copts = ClassifyOptions::default();
        self.ropts = RerankOptions::default();
        self.finish_write(batch.batch, Written::Tokens, true)
    }

    fn write_pairs(&mut self, query: &str, docs: &[&str], opts: &RerankOptions) -> Result<()> {
        if self.model.kind != Kind::Reranker {
            return Err(Error::unsupported_task("write_pairs needs a reranker model"));
        }
        let count = self.check_count(docs.len())?;
        let budget = self.budget(opts.max_tokens)?;
        let trunc = match opts.truncate {
            Truncate::Model => wordpiece::TRUNC_LONGEST_FIRST,
            Truncate::Right => wordpiece::TRUNC_QUERY_PRIORITY,
            Truncate::None => wordpiece::TRUNC_ERROR,
            Truncate::Left => {
                return Err(Error::unsupported_option(RerankOptions::FIELD_TRUNCATE, "truncate=LEFT", CUDA_PROVIDER_ID))
            }
        };
        let seq = self.seq;
        let model = self.model.clone();
        for (r, doc) in docs.iter().enumerate() {
            let s = seq as usize;
            let base = r * s;
            model.vocab.pack_pair(
                query.as_bytes(),
                doc.as_bytes(),
                &mut self.ids[base..base + s],
                &mut self.mask[base..base + s],
                &mut self.types[base..base + s],
                &mut self.pos_scratch,
                seq,
                trunc,
                budget,
            )?;
            self.lengths[r] = self.mask[base..base + s].iter().filter(|&&m| m != 0).count() as u32;
            self.words[r].clear();
        }
        self.ropts = *opts;
        self.finish_write(count, Written::Pairs, false)
    }

    fn write_text_classify(&mut self, texts: &[&str], opts: &ClassifyOptions) -> Result<()> {
        if !matches!(self.model.kind, Kind::Classifier | Kind::TokenClassifier) {
            return Err(Error::unsupported_task("write_text_classify needs a classifier model"));
        }
        let count = self.check_count(texts.len())?;
        let budget = self.budget(opts.max_tokens)?;
        let want_words = self.model.kind == Kind::TokenClassifier;
        self.encode_rows(texts, "", opts.truncate, budget, want_words)?;
        self.copts = *opts;
        // Token classifiers keep the session's full width so the output
        // shape is [rows, max_seq, labels] regardless of input length.
        self.finish_write(count, Written::Classify, want_words)
    }

    fn run(&mut self, opts: &RunOptions) -> Result<ProviderResult> {
        opts.params.reject_unknown(&[], "cuda run")?;
        if self.written == Written::Nothing || self.n_rows == 0 {
            return Err(Error::invalid_state("no inputs written"));
        }
        self.readback_valid = false;
        self.spans.clear();
        self.upload_and_bind()?;
        self.execute()?;
        self.runs += 1;
        let n = self.n_rows as u64;
        let w = self.model.width as u64;
        let shape = match self.model.kind {
            Kind::Embedding => {
                let d = if self.eopts.output_dim == 0 { w } else { self.eopts.output_dim as u64 };
                vec![n, d]
            }
            Kind::Classifier => vec![n, w],
            Kind::Reranker => vec![n],
            Kind::TokenClassifier => vec![n, self.used_seq as u64, w],
        };
        let mut outputs = vec![Output { name: self.name0.clone(), buffer: self.out.clone(), shape }];
        if self.model.kind == Kind::Reranker && (self.ropts.return_sorted || self.ropts.top_n != 0) {
            let rows = self.n_rows as usize;
            self.read_back(rows * 4)?;
            // SAFETY: readback holds `rows` f32 written by the D2H copy.
            let scores: &[f32] = unsafe { std::slice::from_raw_parts(self.readback.ptr().cast::<f32>(), rows) };
            // SAFETY: the core holds the session lock with no result lease outstanding.
            let sorted = unsafe { self.sorted.as_i32_mut()? };
            for (i, s) in sorted[..rows].iter_mut().enumerate() {
                *s = i as i32;
            }
            sorted[..rows].sort_by(|&a, &b| scores[b as usize].total_cmp(&scores[a as usize]));
            let k = if self.ropts.top_n == 0 { rows } else { (self.ropts.top_n as usize).min(rows) };
            outputs.push(Output { name: self.name1.clone(), buffer: self.sorted.clone(), shape: vec![k as u64] });
        }
        if self.model.kind == Kind::TokenClassifier {
            self.aggregate_spans()?;
        }
        Ok(ProviderResult { outputs, spans: self.spans.clone() })
    }

    fn stats(&self) -> SessionStats {
        let width = self.model.elem.width() as u64;
        let n = self.batch as u64 * self.seq as u64;
        let inputs = if self.d_types.is_some() { 3 } else { 2 };
        SessionStats {
            runs: self.runs,
            host_allocs: 0,
            h2d_bytes: self.h2d,
            d2h_bytes: self.d2h,
            input_bytes: inputs * n * width,
            output_bytes: self.out.desc.bytes,
            // ONNX Runtime allocates small host objects per bound run
            // (tensor wrappers, output maps); they are not observable here.
            provider_allocs: None,
        }
    }
}

impl CudaSession {
    fn finish_write(&mut self, count: u32, written: Written, full_width: bool) -> Result<()> {
        self.n_rows = count;
        self.written = written;
        self.readback_valid = false;
        // Compact to the longest live row, rounded up to 8 columns, unless the
        // task's output shape depends on the sequence axis.
        let longest = self.lengths[..count as usize].iter().copied().max().unwrap_or(1).max(1);
        self.used_seq = if full_width { self.seq } else { longest.div_ceil(8).saturating_mul(8).min(self.seq) };
        Ok(())
    }
}

turbo_core::export_provider!(c"cuda", c"2.0.0-alpha.0", || Arc::new(CudaProvider::new()));
