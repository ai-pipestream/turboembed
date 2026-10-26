//! The core's side of include/turbo/turbo_backend.h: the backends linked
//! into this build, and the devices they list.

use std::ffi::{CStr, c_char, c_void};

use crate::status::{Error, INTERNAL, Result, UNSUPPORTED};
use crate::{
    TURBO_STAGE_MAX, turbo_buffer_desc, turbo_device_info, turbo_error, turbo_log_fn, turbo_native_handle,
    turbo_token_range, write_str,
};

pub const TURBO_CAP_UNSUPPORTED: u32 = 0;
pub const TURBO_CAP_EXPERIMENTAL: u32 = 1;
pub const TURBO_CAP_SUPPORTED: u32 = 2;

pub const TURBO_BERT_EMBEDDING_TENSORS: u32 = 5;
pub const TURBO_BERT_LAYER_TENSORS: u32 = 16;

pub const TURBO_FAMILY_BERT: u32 = 1;
pub const TURBO_FAMILY_ROBERTA: u32 = 2;
pub const TURBO_FAMILY_STATIC: u32 = 3;
pub const TURBO_FAMILY_DEBERTA_V2: u32 = 4;

pub const TURBO_DEBERTA_WORD_EMBEDDINGS: u32 = 0;
pub const TURBO_DEBERTA_EMBEDDINGS_LN_WEIGHT: u32 = 1;
pub const TURBO_DEBERTA_EMBEDDINGS_LN_BIAS: u32 = 2;
pub const TURBO_DEBERTA_RELATIVE_EMBEDDINGS: u32 = 3;
pub const TURBO_DEBERTA_RELATIVE_LN_WEIGHT: u32 = 4;
pub const TURBO_DEBERTA_RELATIVE_LN_BIAS: u32 = 5;
pub const TURBO_DEBERTA_EMBEDDING_TENSORS: u32 = 6;

pub const TURBO_STATIC_EMBEDDINGS: u32 = 0;
pub const TURBO_STATIC_WEIGHTS: u32 = 1;
pub const TURBO_STATIC_MAPPING: u32 = 2;
pub const TURBO_STATIC_TENSORS: u32 = 3;

pub const TURBO_FORMAT_SAFETENSORS: u32 = 1;
pub const TURBO_FORMAT_OPENVINO_IR: u32 = 2;
pub const TURBO_FORMAT_HEF: u32 = 3;
pub const TURBO_FORMAT_GGUF: u32 = 4;
pub const TURBO_FORMAT_ONNX: u32 = 5;

/// TURBO_FORMAT_BIT: format `f`'s bit in turbo_backend.formats.
pub const fn format_bit(f: u32) -> u32 {
    1 << (f - 1)
}

pub const TURBO_NUMERIC_F32_FMA: u32 = 1;
pub const TURBO_NUMERIC_TF32: u32 = 2;
pub const TURBO_NUMERIC_F16_F32ACC: u32 = 4;
pub const TURBO_NUMERIC_F16_CHUNKACC: u32 = 8;

/// The core's tuning policy for a session, and what the backend chose.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turbo_backend_tuning {
    pub struct_size: u32,
    pub mode: u32,
    pub budget_ms: u32,
    pub numerics_allowed: u32,
    pub cached: *const c_char,
    pub tuned: u32,
    pub tune_ms: u32,
    pub choices: [c_char; crate::TURBO_CHOICES_LEN],
    pub timings: *mut c_char,
    pub timings_len: u32,
    pub numerics_used: u32,
}

pub const TURBO_INPUT_TOKEN_IDS: u32 = 1;
pub const TURBO_INPUT_EMBEDDINGS: u32 = 2;
pub const TURBO_OUTPUT_HIDDEN_STATES: u32 = 1;
pub const TURBO_OUTPUT_EMBEDDINGS: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turbo_backend_tensor {
    pub name: *const c_char,
    pub data: *const c_void,
    pub shape: [u64; 2],
    pub ndim: u32,
    pub dtype: u32,
    pub bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turbo_backend_model {
    pub struct_size: u32,
    pub family: u32,
    pub dtype: u32,
    pub layers: u32,
    pub hidden: u32,
    pub heads: u32,
    pub intermediate: u32,
    pub vocab_size: u32,
    pub max_positions: u32,
    pub token_types: u32,
    pub layer_norm_eps: f64,
    pub tensor_count: u32,
    pub position_offset: u32,
    pub tensors: *const turbo_backend_tensor,
    pub format: u32,
    pub graph_input: u32,
    pub graph_output: u32,
    pub compute_dtype: u32,
    pub fixed_seq: u32,
    pub fixed_batch: u32,
    pub artifact: *const c_void,
    pub artifact_bytes: u64,
    pub artifact2: *const c_void,
    pub artifact2_bytes: u64,
    pub head: *const turbo_backend_tensor,
    pub head_out: u32,
    /// Limited lookahead: keys a query attends to past itself, 0 for the
    /// whole row (architecture.lookahead). Only the CPU backend runs a
    /// band; the core refuses a banded model on any other.
    pub lookahead: u32,
    /// TURBO_FAMILY_DEBERTA_V2: half the relative table's rows, and the
    /// distance its log buckets reach; 0 for every other family.
    pub position_buckets: u32,
    pub max_relative_positions: u32,
}

/// A session as session_open receives it: turbo_session_desc's fields with
/// every 0 resolved, and the task.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turbo_backend_session_desc {
    pub struct_size: u32,
    pub task: u32,
    pub max_batch: u32,
    pub max_seq: u32,
    pub precision: u32,
    pub cpu_threads: u32,
    pub max_ranges: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turbo_backend_embed_rows {
    pub struct_size: u32,
    pub batch: u32,
    pub seq: u32,
    pub row_stride: u32,
    pub ids: *const i32,
    pub mask: *const i32,
    pub types: *const i32,
    pub pooling: u32,
    pub normalize: u32,
    pub output_dim: u32,
    pub head: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turbo_backend_run {
    pub struct_size: u32,
    pub placement: u32,
    pub output: *mut c_void,
    pub host: *mut c_void,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub host_allocs: u64,
    pub device_allocs: u64,
    pub stage: [u32; TURBO_STAGE_MAX],
}

#[repr(C)]
pub struct turbo_backend {
    pub struct_size: u32,
    pub reserved: u32,
    pub name: *const c_char,
    pub runtime_version: *const c_char,
    pub device_count: unsafe extern "C" fn(out: *mut u32, err: *mut turbo_error) -> i32,
    pub device_info: unsafe extern "C" fn(ordinal: u32, out: *mut turbo_device_info, err: *mut turbo_error) -> i32,
    #[allow(clippy::type_complexity)]
    pub capability: unsafe extern "C" fn(
        ordinal: u32,
        task: u32,
        precision: u32,
        status: *mut u32,
        dtype: *mut u32,
        options_honored: *mut u32,
        reason: *mut c_char,
        reason_len: u32,
        err: *mut turbo_error,
    ) -> i32,
    pub context_create: Option<
        unsafe extern "C" fn(
            ordinal: u32,
            log: turbo_log_fn,
            log_user_data: *mut c_void,
            out: *mut *mut c_void,
            err: *mut turbo_error,
        ) -> i32,
    >,
    pub context_release: Option<unsafe extern "C" fn(ctx: *mut c_void)>,
    pub buffer_alloc: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            desc: *const turbo_buffer_desc,
            out: *mut *mut c_void,
            host: *mut *mut c_void,
            err: *mut turbo_error,
        ) -> i32,
    >,
    #[allow(clippy::type_complexity)]
    pub buffer_import: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            desc: *const turbo_buffer_desc,
            handle: *const turbo_native_handle,
            out: *mut *mut c_void,
            host: *mut *mut c_void,
            err: *mut turbo_error,
        ) -> i32,
    >,
    pub buffer_release: Option<unsafe extern "C" fn(buf: *mut c_void)>,
    pub buffer_export: Option<
        unsafe extern "C" fn(buf: *mut c_void, kind: u32, out: *mut turbo_native_handle, err: *mut turbo_error) -> i32,
    >,
    pub model_load: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            desc: *const turbo_backend_model,
            out: *mut *mut c_void,
            err: *mut turbo_error,
        ) -> i32,
    >,
    pub model_release: Option<unsafe extern "C" fn(model: *mut c_void)>,
    #[allow(clippy::type_complexity)]
    pub session_create: Option<
        unsafe extern "C" fn(
            model: *mut c_void,
            task: u32,
            max_batch: u32,
            max_seq: u32,
            precision: u32,
            compute_dtype: *mut u32,
            out: *mut *mut c_void,
            err: *mut turbo_error,
        ) -> i32,
    >,
    pub session_release: Option<unsafe extern "C" fn(session: *mut c_void)>,
    pub embed_write: Option<
        unsafe extern "C" fn(session: *mut c_void, rows: *const turbo_backend_embed_rows, err: *mut turbo_error) -> i32,
    >,
    pub session_run:
        Option<unsafe extern "C" fn(session: *mut c_void, out: *mut turbo_backend_run, err: *mut turbo_error) -> i32>,
    pub buffer_read:
        Option<unsafe extern "C" fn(buf: *mut c_void, dst: *mut c_void, bytes: u64, err: *mut turbo_error) -> i32>,
    pub formats: u32,
    pub reserved2: u32,
    #[allow(clippy::type_complexity)]
    pub session_create_tuned: Option<
        unsafe extern "C" fn(
            model: *mut c_void,
            task: u32,
            max_batch: u32,
            max_seq: u32,
            precision: u32,
            tuning: *mut turbo_backend_tuning,
            compute_dtype: *mut u32,
            out: *mut *mut c_void,
            err: *mut turbo_error,
        ) -> i32,
    >,
    pub session_open: Option<
        unsafe extern "C" fn(
            model: *mut c_void,
            desc: *const turbo_backend_session_desc,
            tuning: *mut turbo_backend_tuning,
            compute_dtype: *mut u32,
            out: *mut *mut c_void,
            err: *mut turbo_error,
        ) -> i32,
    >,
    pub embed_write_ranges: Option<
        unsafe extern "C" fn(
            session: *mut c_void,
            rows: *const turbo_backend_embed_rows,
            ranges: *const turbo_token_range,
            count: u32,
            err: *mut turbo_error,
        ) -> i32,
    >,
    pub token_head: u32,
    pub reserved3: u32,
}

// The table is immutable static data, read from any thread.
unsafe impl Sync for turbo_backend {}

impl turbo_backend {
    pub fn name(&self) -> &str {
        unsafe { CStr::from_ptr(self.name) }.to_str().unwrap_or("")
    }

    /// The TURBO_FORMAT_* bits model_load takes: FORMAT_SAFETENSORS alone
    /// for a table that ends before formats, or that leaves it 0.
    pub fn formats(&self) -> u32 {
        let covers = self.struct_size as usize >= std::mem::offset_of!(turbo_backend, formats) + size_of::<u32>();
        match covers.then_some(self.formats) {
            None | Some(0) => format_bit(TURBO_FORMAT_SAFETENSORS),
            Some(f) => f,
        }
    }

    /// Whether this backend runs a rerank bundle: the CPU, and the device
    /// backends whose rerank scores are recorded (docs/reranking.md), CUDA
    /// and Level Zero. Metal and the rest take the same code path but stay
    /// refused until they have been run.
    pub fn runs_rerank(&self) -> bool {
        matches!(self.name(), "cpu" | "cuda" | "levelzero")
    }

    /// Whether model_load takes a token head and embed_write_ranges
    /// applies it: 0 for a table that ends before token_head.
    pub fn token_head(&self) -> bool {
        let covers = self.struct_size as usize >= std::mem::offset_of!(turbo_backend, token_head) + size_of::<u32>();
        covers && self.token_head == 1
    }
}

/// The backends linked into this build, in the order their devices are
/// numbered: a GPU backend's before the CPU's, so its devices come first
/// wherever order decides.
pub fn linked() -> &'static [&'static turbo_backend] {
    LINKED
}

static LINKED: &[&turbo_backend] = &[
    #[cfg(feature = "cuda")]
    // A table the C++ side fills at compile time and never writes.
    unsafe {
        &crate::cuda::turbo_cuda_backend
    },
    #[cfg(feature = "levelzero")]
    &crate::levelzero::BACKEND,
    #[cfg(feature = "metal")]
    // A table the Objective-C++ side fills at compile time and never writes.
    unsafe {
        &crate::metal::turbo_metal_backend
    },
    #[cfg(feature = "hailo")]
    // A table the C++ side fills at compile time and never writes.
    unsafe {
        &crate::hailo::turbo_hailo_backend
    },
    #[cfg(feature = "npu")]
    &crate::npu::BACKEND,
    #[cfg(feature = "cpu")]
    &crate::cpu::BACKEND,
];

/// The sizes the table has had, one per group of functions appended to it.
const TABLE_SIZES: [usize; 9] = [
    std::mem::offset_of!(turbo_backend, context_create),
    std::mem::offset_of!(turbo_backend, model_load),
    std::mem::offset_of!(turbo_backend, session_create),
    std::mem::offset_of!(turbo_backend, buffer_read),
    std::mem::offset_of!(turbo_backend, formats),
    std::mem::offset_of!(turbo_backend, session_create_tuned),
    std::mem::offset_of!(turbo_backend, session_open),
    std::mem::offset_of!(turbo_backend, token_head),
    size_of::<turbo_backend>(),
];

/// The table's size is one this core knows, and what it offers comes with
/// its release. A table built against another header is a build fault,
/// not a missing driver.
pub fn check_table(backend: &turbo_backend) -> Result<()> {
    if !TABLE_SIZES.contains(&(backend.struct_size as usize)) {
        return Err(Error::new(
            INTERNAL,
            format!(
                "{} backend: its table is {} bytes, this core knows {TABLE_SIZES:?}",
                backend.name(),
                backend.struct_size
            ),
        ));
    }
    let b = backend;
    // A function past struct_size is not read.
    macro_rules! has {
        ($f:ident) => {
            b.struct_size as usize > std::mem::offset_of!(turbo_backend, $f) && b.$f.is_some()
        };
    }
    let pairs = [
        ("context_create", has!(context_create), "context_release", has!(context_release)),
        ("buffer_alloc", has!(buffer_alloc), "buffer_release", has!(buffer_release)),
        ("buffer_import", has!(buffer_import), "buffer_release", has!(buffer_release)),
        ("model_load", has!(model_load), "model_release", has!(model_release)),
        ("session_create", has!(session_create), "session_release", has!(session_release)),
        ("session_create", has!(session_create), "session_run", has!(session_run)),
        ("session_create_tuned", has!(session_create_tuned), "session_release", has!(session_release)),
        ("session_create_tuned", has!(session_create_tuned), "session_run", has!(session_run)),
        ("session_open", has!(session_open), "session_release", has!(session_release)),
        ("session_open", has!(session_open), "session_run", has!(session_run)),
        ("embed_write_ranges", has!(embed_write_ranges), "session_open", has!(session_open)),
    ];
    for (f, has, release, has_release) in pairs {
        if has && !has_release {
            return Err(Error::new(INTERNAL, format!("{} backend: its table has {f} and no {release}", b.name())));
        }
    }
    Ok(())
}

/// A function of the table that may be NULL, when struct_size covers it
/// and it is there; else TURBO_E_UNSUPPORTED naming the backend and the
/// function.
macro_rules! offered {
    ($b:expr, $f:ident) => {
        $crate::backend::covered(
            $b,
            ::std::mem::offset_of!($crate::backend::turbo_backend, $f),
            || $b.$f,
            stringify!($f),
        )
    };
}
pub(crate) use offered;

/// `f` reads the function, and is called only when struct_size covers it.
pub(crate) fn covered<F>(
    backend: &turbo_backend,
    offset: usize,
    f: impl FnOnce() -> Option<F>,
    what: &str,
) -> Result<F> {
    let covers = backend.struct_size as usize >= offset + size_of::<Option<F>>();
    covers
        .then(f)
        .flatten()
        .ok_or_else(|| Error::new(UNSUPPORTED, format!("the {} backend does not offer {what}", backend.name())))
}

/// A backend's status as an Error, with its message.
fn failed(backend: &turbo_backend, what: &str, code: i32, err: &turbo_error) -> Error {
    let mut e = Error::new(code, format!("{} backend, {what}: {}", backend.name(), cstr(&err.message)));
    e.field = err.field;
    e
}

/// A backend refusing a call: fill `err` when the caller passed one.
///
/// # Safety
/// `err` is NULL or valid for the call.
pub unsafe fn refuse(err: *mut turbo_error, code: i32, message: &str) -> i32 {
    if let Some(e) = unsafe { err.as_mut() } {
        e.code = code;
        e.field = 0;
        write_str(&mut e.message, message);
    }
    code
}

/// As `refuse`, naming the 1-based option field the two field codes carry.
///
/// # Safety
/// `err` is NULL or valid for the call.
pub unsafe fn refuse_field(err: *mut turbo_error, code: i32, field: u32, message: &str) -> i32 {
    unsafe { refuse(err, code, message) };
    if let Some(e) = unsafe { err.as_mut() } {
        e.field = field;
    }
    code
}

/// A C string in a fixed buffer, read only up to the buffer's end whether
/// or not it holds a NUL.
pub(crate) fn cstr(b: &[c_char]) -> String {
    let bytes: Vec<u8> = b.iter().take_while(|&&c| c != 0).map(|&c| u8::from_ne_bytes(c.to_ne_bytes())).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Call a backend function with a fresh turbo_error and turn a failure into
/// an Error.
pub fn check(backend: &turbo_backend, what: &str, f: impl FnOnce(*mut turbo_error) -> i32) -> Result<()> {
    let mut err = crate::new_error();
    let code = f(&mut err);
    if code != 0 {
        return Err(failed(backend, what, code, &err));
    }
    Ok(())
}

#[cfg(all(test, feature = "cpu"))]
mod tests;
