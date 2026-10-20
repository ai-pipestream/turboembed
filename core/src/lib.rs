//! libturbo: the C interface in include/turbo/turbo.h.
//!
//! This cut has the runtime handle, status names, contexts, buffers,
//! models, the tokenizer, and embed sessions and their results. The types
//! below mirror the header's; tests/abi.rs checks their layout against a
//! C compiler's.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Arc;

pub mod backend;
pub mod bpe;
pub mod bundle;
mod classes;
#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "hailo")]
pub mod hailo;
pub mod head;
#[cfg(feature = "levelzero")]
pub mod levelzero;
pub mod manifest;
#[cfg(feature = "metal")]
pub mod metal;
pub mod model;
#[cfg(feature = "npu")]
pub mod npu;
pub mod projection;
pub mod record;
pub mod safetensors;
pub mod sentencepiece;
mod session;
pub mod status;
pub mod tokenizer;
pub mod tuning;
pub mod unigram;
#[cfg(any(feature = "levelzero", feature = "npu"))]
pub mod ze;

pub use session::*;

use backend::cstr;
use manifest::{PromptRole, Truncation};
use status::{
    Error, INVALID_ARGUMENT, INVALID_ENUM, INVALID_HANDLE, INVALID_SHAPE, INVALID_STRUCT_SIZE, INVALID_UTF8, PANIC,
    Result,
};
use tokenizer::{Encode, Tokenizer};

pub const TURBO_ERROR_MESSAGE_LEN: usize = 496;

pub const TURBO_TASK_EMBED: u32 = 1;

pub const TURBO_DEVICE_CPU: u32 = 1;
pub const TURBO_DEVICE_GPU: u32 = 2;
pub const TURBO_DEVICE_IGPU: u32 = 3;
pub const TURBO_DEVICE_NPU: u32 = 4;

pub const TURBO_DTYPE_I8: u32 = 6;
pub const TURBO_DTYPE_I32: u32 = 8;
pub const TURBO_DTYPE_I64: u32 = 9;
pub const TURBO_DTYPE_F16: u32 = 10;
pub const TURBO_DTYPE_BF16: u32 = 11;
pub const TURBO_DTYPE_F32: u32 = 12;
pub const TURBO_DTYPE_F64: u32 = 13;

pub const TURBO_PLACE_HOST: u32 = 1;
pub const TURBO_PLACE_PINNED: u32 = 2;
pub const TURBO_PLACE_DEVICE: u32 = 3;
pub const TURBO_PLACE_SHARED: u32 = 4;

pub const TURBO_HANDLE_HOST_PTR: u32 = 1;
pub const TURBO_HANDLE_CUDA_PTR: u32 = 2;
pub const TURBO_HANDLE_CL_MEM: u32 = 3;
pub const TURBO_HANDLE_ZE_USM: u32 = 4;
pub const TURBO_HANDLE_MTL_BUFFER: u32 = 5;
pub const TURBO_HANDLE_DMABUF_FD: u32 = 6;

pub const TURBO_PRECISION_MODEL: u32 = 0;
pub const TURBO_PRECISION_FASTEST: u32 = 1;
pub const TURBO_PRECISION_EXACT: u32 = 2;

pub const TURBO_AUTOTUNE_RUNTIME: u32 = 0;
pub const TURBO_AUTOTUNE_OFF: u32 = 1;
pub const TURBO_AUTOTUNE_ON: u32 = 2;
pub const TURBO_AUTOTUNE_RETUNE: u32 = 3;

pub const TURBO_TUNED_DEFAULT: u32 = 0;
pub const TURBO_TUNED_FORCED: u32 = 1;
pub const TURBO_TUNED_MEASURED: u32 = 2;
pub const TURBO_TUNED_CACHE: u32 = 3;

pub const TURBO_CHOICES_LEN: usize = 1024;

pub const TURBO_TRUNCATE_MODEL: u32 = 0;
pub const TURBO_TRUNCATE_NONE: u32 = 1;
pub const TURBO_TRUNCATE_RIGHT: u32 = 2;
pub const TURBO_TRUNCATE_LEFT: u32 = 3;

pub const TURBO_PROMPT_NONE: u32 = 0;
pub const TURBO_PROMPT_QUERY: u32 = 1;
pub const TURBO_PROMPT_DOCUMENT: u32 = 2;

pub const TURBO_NORMALIZE_MODEL: u32 = 0;
pub const TURBO_NORMALIZE_NONE: u32 = 1;
pub const TURBO_NORMALIZE_L2: u32 = 2;

pub const TURBO_POOLING_MODEL: u32 = 0;
pub const TURBO_POOLING_MEAN: u32 = 1;
pub const TURBO_POOLING_CLS: u32 = 2;
pub const TURBO_POOLING_LAST: u32 = 3;

pub const TURBO_STAGE_MAX: usize = 16;

pub const TURBO_EMBED_STAGE_TOKENIZE: usize = 0;
pub const TURBO_EMBED_STAGE_UPLOAD: usize = 1;
pub const TURBO_EMBED_STAGE_LOOKUP: usize = 2;
pub const TURBO_EMBED_STAGE_ENCODE: usize = 3;
pub const TURBO_EMBED_STAGE_POOL: usize = 4;
pub const TURBO_EMBED_STAGE_NORMALIZE: usize = 5;
pub const TURBO_EMBED_STAGE_DOWNLOAD: usize = 6;
pub const TURBO_EMBED_STAGE_COUNT: usize = 7;

pub const TURBO_OUTPUT_DIMS_MAX: usize = 16;

pub const TURBO_STAGE_UNUSED: u32 = 0;
pub const TURBO_STAGE_HOST: u32 = 1;
pub const TURBO_STAGE_DEVICE: u32 = 2;
pub const TURBO_STAGE_FUSED: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct turbo_text {
    pub ptr: *const c_char,
    pub len: u64,
}

#[repr(C)]
pub struct turbo_error {
    pub struct_size: u32,
    pub code: i32,
    pub field: u32,
    pub message: [c_char; TURBO_ERROR_MESSAGE_LEN],
}

pub type turbo_log_fn = Option<unsafe extern "C" fn(user_data: *mut c_void, level: u32, message: turbo_text)>;

#[repr(C)]
pub struct turbo_runtime_desc {
    pub struct_size: u32,
    pub reserved: u32,
    pub log: turbo_log_fn,
    pub log_user_data: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct turbo_device_info {
    pub struct_size: u32,
    pub kind: u32,
    pub ordinal: u32,
    pub unified_memory: u32,
    pub memory_total: u64,
    pub memory_free: u64,
    pub arch: [c_char; 32],
    pub name: [c_char; 128],
    pub vendor: [c_char; 64],
    pub backend: [c_char; 32],
    pub runtime_version: [c_char; 64],
    pub driver_version: [c_char; 64],
}

#[repr(C)]
#[derive(Debug)]
pub struct turbo_capability {
    pub struct_size: u32,
    pub status: u32,
    pub dtype: u32,
    pub options_honored: u32,
    pub cosine_floor: f32,
    pub speed_ratio: f32,
    pub benchmark: [c_char; 96],
    pub reason: [c_char; 160],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct turbo_native_handle {
    pub struct_size: u32,
    pub kind: u32,
    pub handle: u64,
    pub aux: u64,
    pub offset: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct turbo_buffer_desc {
    pub struct_size: u32,
    pub placement: u32,
    pub dtype: u32,
    pub ndim: u32,
    pub shape: [u64; 2],
    pub bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct turbo_model_info {
    pub struct_size: u32,
    pub task: u32,
    pub dim: u32,
    pub pooling: u32,
    pub normalize: u32,
    pub max_seq: u32,
    pub max_batch: u32,
    pub dtype: u32,
    pub model_id: [c_char; 128],
    pub revision: [c_char; 64],
    pub manifest_sha256: [c_char; 72],
    pub artifact_sha256: [c_char; 72],
    pub tokenizer_sha256: [c_char; 72],
    pub prefix_query: [c_char; 128],
    pub prefix_document: [c_char; 128],
    pub output_dims_count: u32,
    pub output_dims: [u32; TURBO_OUTPUT_DIMS_MAX],
}

/// The size of turbo_model_info before output_dims_count was appended,
/// which turbo_model_get_info still accepts.
pub const TURBO_MODEL_INFO_SIZE_V1: usize = std::mem::offset_of!(turbo_model_info, output_dims_count);

#[repr(C)]
pub struct turbo_tokenizer_info {
    pub struct_size: u32,
    pub vocab_size: u32,
    pub max_seq: u32,
    pub specials_per_sequence: u32,
    pub pad_id: i32,
    pub bos_id: i32,
    pub eos_id: i32,
    pub unk_id: i32,
    pub kind: [c_char; 32],
    pub sha256: [c_char; 72],
    pub manifest_sha256: [c_char; 72],
}

#[repr(C)]
pub struct turbo_encode_options {
    pub struct_size: u32,
    pub omit_special_tokens: u32,
    pub truncate: u32,
    pub max_tokens: u32,
    pub prompt_role: u32,
}

// ---- Handles -------------------------------------------------------------

const RUNTIME_MAGIC: u64 = 0x7475_7262_6f72_7431; // "turbort1"
const TOKENIZER_MAGIC: u64 = 0x7475_7262_6f74_6b31; // "turbotk1"
const CONTEXT_MAGIC: u64 = 0x7475_7262_6f63_7431; // "turboct1"
const BUFFER_MAGIC: u64 = 0x7475_7262_6f62_6631; // "turbobf1"
const MODEL_MAGIC: u64 = 0x7475_7262_6f6d_6431; // "turbomd1"

struct Runtime {
    log: turbo_log_fn,
    log_user_data: usize,
    devices: Vec<Device>,
    /// The kernel choices its tuned sessions measured (docs/autotune.md).
    tune_cache: tuning::Cache,
}

/// A device a linked backend listed when the runtime was made.
struct Device {
    backend: &'static backend::turbo_backend,
    ordinal: u32,
    info: turbo_device_info,
}

const LOG_WARNING: u32 = 1;
const LOG_INFO: u32 = 2;

pub struct turbo_runtime {
    magic: u64,
    inner: Arc<Runtime>,
}

pub struct turbo_tokenizer {
    magic: u64,
    _runtime: Arc<Runtime>,
    tok: Tokenizer,
}

unsafe fn runtime<'a>(rt: *mut turbo_runtime) -> Result<&'a turbo_runtime> {
    match unsafe { rt.as_ref() } {
        Some(r) if r.magic == RUNTIME_MAGIC => Ok(r),
        _ => Err(Error::new(INVALID_HANDLE, "not a turbo_runtime")),
    }
}

/// A device's context: the backend's own, released when the last handle,
/// buffer or model holding it goes.
struct Context {
    runtime: Arc<Runtime>,
    device: u32,
    backend: &'static backend::turbo_backend,
    raw: *mut c_void,
    release: unsafe extern "C" fn(*mut c_void),
}

// turbo_backend.h: every function in the table may be called from any
// thread, and what the context points to is the backend's.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { (self.release)(self.raw) };
    }
}

pub struct turbo_context {
    magic: u64,
    inner: Arc<Context>,
}

/// A buffer the backend made or wrapped, or a result's output. It holds
/// its context, so the backend's buffer is released before the backend's
/// context.
struct Buffer {
    context: Arc<Context>,
    desc: turbo_buffer_desc,
    raw: *mut c_void,
    host: *mut c_void,
    release: Release,
}

/// What releasing a buffer does.
enum Release {
    /// The backend's buffer, released through its table.
    Backend(unsafe extern "C" fn(*mut c_void)),
    /// A session's output, which the session owns: the buffer holds the
    /// result, and releasing it lets the result go.
    Result(Arc<session::SessionInner>),
}

impl Drop for Buffer {
    fn drop(&mut self) {
        match &self.release {
            Release::Backend(release) => unsafe { release(self.raw) },
            Release::Result(s) => s.let_go(),
        }
    }
}

pub struct turbo_buffer {
    magic: u64,
    inner: Buffer,
}

unsafe fn context<'a>(c: *mut turbo_context) -> Result<&'a turbo_context> {
    match unsafe { c.as_ref() } {
        Some(r) if r.magic == CONTEXT_MAGIC => Ok(r),
        _ => Err(Error::new(INVALID_HANDLE, "not a turbo_context")),
    }
}

unsafe fn buffer<'a>(b: *mut turbo_buffer) -> Result<&'a turbo_buffer> {
    match unsafe { b.as_ref() } {
        Some(r) if r.magic == BUFFER_MAGIC => Ok(r),
        _ => Err(Error::new(INVALID_HANDLE, "not a turbo_buffer")),
    }
}

unsafe fn tokenizer<'a>(t: *mut turbo_tokenizer) -> Result<&'a turbo_tokenizer> {
    match unsafe { t.as_ref() } {
        Some(r) if r.magic == TOKENIZER_MAGIC => Ok(r),
        _ => Err(Error::new(INVALID_HANDLE, "not a turbo_tokenizer")),
    }
}

// ---- Model -----------------------------------------------------------------------

/// A bundle loaded on a context's device. It holds its context, so the
/// backend's model is released before the backend's context.
struct Model {
    context: Arc<Context>,
    raw: *mut c_void,
    release: unsafe extern "C" fn(*mut c_void),
    /// The core's one verified host copy of the weights, and of a compiled
    /// artifact's bytes. A backend with its own memory copied them there at
    /// load; the CPU backend reads them here, in place, so they live exactly
    /// as long as the backend's model.
    weights: model::Weights,
    /// The bundle's tokenizer, checked against its reference at load, for
    /// turbo_embed_write_text.
    tokenizer: Tokenizer,
    info: turbo_model_info,
    /// The widths embed.output_dims allows besides dim.
    output_dims: Vec<u32>,
}

// As for Context: the backend's model may be used from any thread.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Drop for Model {
    fn drop(&mut self) {
        // The weights and the context are dropped after this, with the
        // struct's fields.
        unsafe { (self.release)(self.raw) };
    }
}

pub struct turbo_model {
    magic: u64,
    inner: Arc<Model>,
}

unsafe fn model_handle<'a>(m: *mut turbo_model) -> Result<&'a turbo_model> {
    match unsafe { m.as_ref() } {
        Some(r) if r.magic == MODEL_MAGIC => Ok(r),
        _ => Err(Error::new(INVALID_HANDLE, "not a turbo_model")),
    }
}

mod runtime;
use runtime::*;

mod ffi;
pub use ffi::*;

mod runtime_api;
pub use runtime_api::*;

mod buffer_api;
pub use buffer_api::*;

mod model_api;
pub use model_api::*;

mod tokenizer_api;
pub use tokenizer_api::*;
