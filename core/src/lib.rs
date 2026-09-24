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
pub mod bundle;
#[cfg(feature = "cpu")]
pub mod cpu;
pub mod manifest;
pub mod model;
pub mod safetensors;
mod session;
pub mod status;
pub mod tokenizer;

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

pub const TURBO_DTYPE_I32: u32 = 8;
pub const TURBO_DTYPE_F16: u32 = 10;
pub const TURBO_DTYPE_BF16: u32 = 11;
pub const TURBO_DTYPE_F32: u32 = 12;

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
}

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
}

/// A device a linked backend listed when the runtime was made.
struct Device {
    backend: &'static backend::turbo_backend,
    ordinal: u32,
    info: turbo_device_info,
}

const LOG_WARNING: u32 = 1;

impl Runtime {
    fn log(&self, level: u32, message: &str) {
        if let Some(f) = self.log {
            let t = turbo_text { ptr: message.as_ptr() as *const c_char, len: message.len() as u64 };
            unsafe { f(self.log_user_data as *mut c_void, level, t) };
        }
    }

    fn device(&self, index: u32) -> Result<&Device> {
        self.devices.get(index as usize).ok_or_else(|| {
            Error::new(INVALID_ARGUMENT, format!("device {index}: the runtime has {} devices", self.devices.len()))
        })
    }

    /// Every device each linked backend lists. A backend whose probe fails
    /// lists nothing, and the log says why; a table of the wrong size fails
    /// the runtime.
    fn enumerate(&mut self) -> Result<()> {
        for b in backend::linked() {
            backend::check_table(b)?;
            let mut n = 0u32;
            if let Err(e) = backend::check(b, "device_count", |err| unsafe { (b.device_count)(&mut n, err) }) {
                self.log(LOG_WARNING, &e.message);
                continue;
            }
            for ordinal in 0..n {
                match probe(b, ordinal) {
                    Ok(info) => self.devices.push(Device { backend: b, ordinal, info }),
                    Err(e) => self.log(LOG_WARNING, &e.message),
                }
            }
        }
        Ok(())
    }

    /// The (device, task, precision) cell. The backend says what it has
    /// built; SUPPORTED needs a benchmark record, and this build reads none.
    fn capability(&self, index: u32, task: u32, precision: u32) -> Result<turbo_capability> {
        let d = self.device(index)?;
        if task != TURBO_TASK_EMBED {
            return Err(Error::new(INVALID_ENUM, format!("task: {task} is not a TURBO_TASK_* value")));
        }
        if precision > TURBO_PRECISION_EXACT {
            return Err(Error::new(INVALID_ENUM, format!("precision: {precision} is not a TURBO_PRECISION_* value")));
        }
        let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
        cap.struct_size = size_of::<turbo_capability>() as u32;
        let b = d.backend;
        backend::check(b, "capability", |err| unsafe {
            (b.capability)(
                d.ordinal,
                task,
                precision,
                &mut cap.status,
                &mut cap.dtype,
                &mut cap.options_honored,
                cap.reason.as_mut_ptr(),
                cap.reason.len() as u32,
                err,
            )
        })?;
        match cap.status {
            backend::TURBO_CAP_UNSUPPORTED => {
                // A cell that does not run computes in nothing.
                cap.dtype = 0;
                cap.options_honored = 0;
            }
            backend::TURBO_CAP_EXPERIMENTAL => {
                write_str(&mut cap.reason, "no benchmark record for this cell");
                // The options the core applies before a backend sees the rows.
                cap.options_honored |= session::CORE_HONORED;
            }
            s => {
                return Err(Error::new(
                    status::INTERNAL,
                    format!(
                        "{} backend reported status {s}; only EXPERIMENTAL or UNSUPPORTED are its to report",
                        b.name()
                    ),
                ));
            }
        }
        Ok(cap)
    }
}

/// Ask a backend about one of its devices, now.
fn probe(b: &'static backend::turbo_backend, ordinal: u32) -> Result<turbo_device_info> {
    let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_device_info>() as u32;
    backend::check(b, "device_info", |err| unsafe { (b.device_info)(ordinal, &mut info, err) })?;
    info.struct_size = size_of::<turbo_device_info>() as u32;
    write_str(&mut info.backend, b.name());
    Ok(info)
}

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

// ---- Calling convention ----------------------------------------------------

/// Run `f` behind the header's rules: the error struct's size is checked
/// before anything else, a panic stops at this boundary, and the status is
/// both returned and written to `err`.
unsafe fn call(err: *mut turbo_error, f: impl FnOnce() -> Result<()>) -> i32 {
    if let Some(e) = unsafe { err.as_ref() }
        && e.struct_size as usize != size_of::<turbo_error>()
    {
        return INVALID_STRUCT_SIZE;
    }
    let outcome = catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|p| {
        let why = p
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| p.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        Err(Error::new(PANIC, format!("panic in libturbo: {why}")))
    });
    let e = match outcome {
        Ok(()) => Error::new(status::OK, ""),
        Err(e) => e,
    };
    if let Some(out) = unsafe { err.as_mut() } {
        out.code = e.code;
        out.field = e.field;
        write_str(&mut out.message, &e.message);
    }
    e.code
}

/// Copy `s` into a NUL-terminated buffer, cut at a character boundary.
pub(crate) fn write_str(dst: &mut [c_char], s: &str) {
    let mut n = s.len().min(dst.len() - 1);
    while !s.is_char_boundary(n) {
        n -= 1;
    }
    for (d, b) in dst.iter_mut().zip(&s.as_bytes()[..n]) {
        *d = *b as c_char;
    }
    dst[n] = 0;
}

unsafe fn text<'a>(t: turbo_text, what: &str) -> Result<&'a str> {
    if t.len == 0 {
        return Ok("");
    }
    if t.ptr.is_null() {
        return Err(Error::new(INVALID_ARGUMENT, format!("{what}: NULL with length {}", t.len)));
    }
    let bytes = unsafe { std::slice::from_raw_parts(t.ptr as *const u8, t.len as usize) };
    std::str::from_utf8(bytes).map_err(|e| Error::new(INVALID_UTF8, format!("{what}: {e}")))
}

unsafe fn out_ptr<'a, T>(p: *mut T, what: &str) -> Result<&'a mut T> {
    unsafe { p.as_mut() }.ok_or_else(|| Error::new(INVALID_ARGUMENT, format!("{what} is NULL")))
}

fn sized(struct_size: u32, want: usize, what: &str) -> Result<()> {
    if struct_size as usize != want {
        return Err(Error::new(
            INVALID_STRUCT_SIZE,
            format!("{what}.struct_size is {struct_size}, this library knows {want}"),
        ));
    }
    Ok(())
}

// ---- Library -----------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn turbo_version() -> *const c_char {
    VERSION.as_ptr()
}

/// The version and the backends linked into this build, in device order.
#[cfg(feature = "cpu")]
const VERSION: &std::ffi::CStr = c"0.1.0 cpu";
#[cfg(not(feature = "cpu"))]
const VERSION: &std::ffi::CStr = c"0.1.0";

pub(crate) fn new_error() -> turbo_error {
    turbo_error {
        struct_size: size_of::<turbo_error>() as u32,
        code: 0,
        field: 0,
        message: [0; TURBO_ERROR_MESSAGE_LEN],
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn turbo_status_name(code: i32) -> *const c_char {
    status::name(code).as_ptr()
}

// ---- Runtime -------------------------------------------------------------------

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_runtime_create(
    desc: *const turbo_runtime_desc,
    out: *mut *mut turbo_runtime,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let out = out_ptr(out, "out")?;
            let (log, user) = match desc.as_ref() {
                Some(d) => {
                    sized(d.struct_size, size_of::<turbo_runtime_desc>(), "turbo_runtime_desc")?;
                    (d.log, d.log_user_data as usize)
                }
                None => (None, 0),
            };
            let mut inner = Runtime { log, log_user_data: user, devices: Vec::new() };
            inner.enumerate()?;
            let rt = turbo_runtime { magic: RUNTIME_MAGIC, inner: Arc::new(inner) };
            *out = Box::into_raw(Box::new(rt));
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_runtime_device_count(
    rt: *mut turbo_runtime,
    out: *mut u32,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let rt = runtime(rt)?;
            *out_ptr(out, "out")? = rt.inner.devices.len() as u32;
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_runtime_device_info(
    rt: *mut turbo_runtime,
    index: u32,
    out: *mut turbo_device_info,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let rt = runtime(rt)?;
            let out = out_ptr(out, "out")?;
            sized(out.struct_size, size_of::<turbo_device_info>(), "turbo_device_info")?;
            // Probed again, so memory_free is the device's now.
            let d = rt.inner.device(index)?;
            *out = probe(d.backend, d.ordinal)?;
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_runtime_capability(
    rt: *mut turbo_runtime,
    index: u32,
    task: u32,
    precision: u32,
    out: *mut turbo_capability,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let rt = runtime(rt)?;
            let out = out_ptr(out, "out")?;
            sized(out.struct_size, size_of::<turbo_capability>(), "turbo_capability")?;
            *out = rt.inner.capability(index, task, precision)?;
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says; `reason` is
/// NULL or holds `reason_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_runtime_select(
    rt: *mut turbo_runtime,
    task: u32,
    out: *mut u32,
    reason: *mut c_char,
    reason_len: u32,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let rt = &runtime(rt)?.inner;
            let out = out_ptr(out, "out")?;
            let (pick, why) = select(rt, task)?;
            if !reason.is_null() && reason_len > 0 {
                write_str(std::slice::from_raw_parts_mut(reason, reason_len as usize), &why);
            }
            match pick {
                Some(i) => {
                    *out = i;
                    Ok(())
                }
                None => Err(Error::new(status::DEVICE_NOT_FOUND, why)),
            }
        })
    }
}

/// turbo.h's rule: the highest status at TURBO_PRECISION_MODEL, then the
/// lowest speed_ratio among equals (0 is no record, so it loses), never a
/// CPU. Returns the device, or none, and one line saying why.
fn select(rt: &Runtime, task: u32) -> Result<(Option<u32>, String)> {
    if task != TURBO_TASK_EMBED {
        return Err(Error::new(INVALID_ENUM, format!("task: {task} is not a TURBO_TASK_* value")));
    }
    let mut best: Option<(u32, turbo_capability)> = None;
    let mut ties = 0u32;
    let mut skipped = Vec::new();
    for (i, d) in rt.devices.iter().enumerate() {
        let i = i as u32;
        let label = format!("device {i} ({} {})", d.backend.name(), cstr(&d.info.name));
        if d.info.kind == TURBO_DEVICE_CPU {
            skipped.push(format!("{label}: a CPU is never selected"));
            continue;
        }
        let cap = rt.capability(i, task, TURBO_PRECISION_MODEL)?;
        if cap.status == backend::TURBO_CAP_UNSUPPORTED {
            skipped.push(format!("{label}: {}", cstr(&cap.reason)));
            continue;
        }
        match &best {
            Some((_, b)) if !ranks_above(&cap, b) => {
                if !ranks_above(b, &cap) {
                    ties += 1;
                }
            }
            _ => {
                best = Some((i, cap));
                ties = 1;
            }
        }
    }
    Ok(match best {
        Some((i, cap)) => {
            let d = &rt.devices[i as usize];
            let status = if cap.status == backend::TURBO_CAP_SUPPORTED { "supported" } else { "experimental" };
            let tie = if ties > 1 { format!(", first in device order of {ties} that tie") } else { String::new() };
            (Some(i), format!("device {i} ({} {}): {status}{tie}", d.backend.name(), cstr(&d.info.name)))
        }
        None if skipped.is_empty() => (None, "no device is listed".to_owned()),
        None => (None, format!("no device offers the task: {}", skipped.join("; "))),
    })
}

/// Whether cell `a` ranks above cell `b`: a higher status, or the same
/// status and a lower speed_ratio, where 0 is no record and ranks below any.
/// Neither ranking above the other is a tie.
pub fn ranks_above(a: &turbo_capability, b: &turbo_capability) -> bool {
    a.status > b.status
        || (a.status == b.status && a.speed_ratio > 0.0 && (b.speed_ratio == 0.0 || a.speed_ratio < b.speed_ratio))
}

/// # Safety
/// `rt` is NULL or a handle from turbo_runtime_create, released once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_runtime_release(rt: *mut turbo_runtime) {
    if unsafe { runtime(rt) }.is_ok() {
        let mut b = unsafe { Box::from_raw(rt) };
        b.magic = 0;
    }
}

// ---- Context and buffers ---------------------------------------------------------

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_context_create(
    rt: *mut turbo_runtime,
    device: u32,
    out: *mut *mut turbo_context,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let rt = runtime(rt)?;
            let out = out_ptr(out, "out")?;
            let d = rt.inner.device(device)?;
            let b = d.backend;
            let create = backend::offered!(b, context_create)?;
            let release = backend::offered!(b, context_release)?;
            let mut raw = std::ptr::null_mut();
            let user = rt.inner.log_user_data as *mut c_void;
            backend::check(b, "context_create", |err| create(d.ordinal, rt.inner.log, user, &mut raw, err))?;
            let inner = Context { runtime: rt.inner.clone(), device, backend: b, raw, release };
            *out = Box::into_raw(Box::new(turbo_context { magic: CONTEXT_MAGIC, inner: Arc::new(inner) }));
            Ok(())
        })
    }
}

/// # Safety
/// `ctx` is NULL or a handle from turbo_context_create, released once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_context_release(ctx: *mut turbo_context) {
    if unsafe { context(ctx) }.is_ok() {
        let mut b = unsafe { Box::from_raw(ctx) };
        b.magic = 0;
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_context_device(ctx: *mut turbo_context, out: *mut u32, err: *mut turbo_error) -> i32 {
    unsafe {
        call(err, || {
            let ctx = context(ctx)?;
            *out_ptr(out, "out")? = ctx.inner.device;
            Ok(())
        })
    }
}

fn dtype_size(dtype: u32) -> Option<u64> {
    match dtype {
        TURBO_DTYPE_I32 | TURBO_DTYPE_F32 => Some(4),
        TURBO_DTYPE_F16 | TURBO_DTYPE_BF16 => Some(2),
        _ => None,
    }
}

/// The caller's desc as a backend gets it: every field checked, and bytes
/// filled in from shape and dtype.
unsafe fn buffer_desc(desc: *const turbo_buffer_desc) -> Result<turbo_buffer_desc> {
    let d = *unsafe { desc.as_ref() }.ok_or_else(|| Error::new(INVALID_ARGUMENT, "desc is NULL"))?;
    sized(d.struct_size, size_of::<turbo_buffer_desc>(), "turbo_buffer_desc")?;
    if !(TURBO_PLACE_HOST..=TURBO_PLACE_SHARED).contains(&d.placement) {
        return Err(Error::new(INVALID_ENUM, format!("placement: {} is not a TURBO_PLACE_* value", d.placement)));
    }
    let size = dtype_size(d.dtype)
        .ok_or_else(|| Error::new(INVALID_ENUM, format!("dtype: {} is not a TURBO_DTYPE_* value", d.dtype)))?;
    // Entries past ndim are not read, and the backend sees them as 0.
    let mut d = d;
    if d.ndim == 1 {
        d.shape[1] = 0;
    }
    let dims = match d.ndim {
        1 => &d.shape[..1],
        2 => &d.shape[..],
        n => return Err(Error::new(INVALID_SHAPE, format!("ndim is {n}, not 1 or 2"))),
    };
    if let Some(i) = dims.iter().position(|&n| n == 0) {
        return Err(Error::new(INVALID_SHAPE, format!("shape[{i}] is 0")));
    }
    let bytes = dims.iter().try_fold(size, |a, &n| a.checked_mul(n)).ok_or_else(|| {
        Error::new(INVALID_SHAPE, format!("shape {dims:?} of {size}-byte elements is past 2^64 bytes"))
    })?;
    if d.bytes != 0 && d.bytes != bytes {
        return Err(Error::new(INVALID_ARGUMENT, format!("bytes is {}, shape and dtype make {bytes}", d.bytes)));
    }
    Ok(turbo_buffer_desc { bytes, ..d })
}

/// Wrap what the backend returned. Host memory has a host address; device
/// memory has none.
fn new_buffer(
    ctx: &turbo_context,
    desc: turbo_buffer_desc,
    raw: *mut c_void,
    host: *mut c_void,
    release: unsafe extern "C" fn(*mut c_void),
    out: &mut *mut turbo_buffer,
) -> Result<()> {
    let inner = Buffer { context: ctx.inner.clone(), desc, raw, host, release: Release::Backend(release) };
    let device = desc.placement == TURBO_PLACE_DEVICE;
    if host.is_null() != device {
        let b = ctx.inner.backend.name();
        return Err(Error::new(
            status::INTERNAL,
            format!("{b} backend: placement {} came back with host address {host:?}", desc.placement),
        ));
    }
    *out = Box::into_raw(Box::new(turbo_buffer { magic: BUFFER_MAGIC, inner }));
    Ok(())
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_buffer_alloc(
    ctx: *mut turbo_context,
    desc: *const turbo_buffer_desc,
    out: *mut *mut turbo_buffer,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let ctx = context(ctx)?;
            let out = out_ptr(out, "out")?;
            let desc = buffer_desc(desc)?;
            let b = ctx.inner.backend;
            let alloc = backend::offered!(b, buffer_alloc)?;
            let release = backend::offered!(b, buffer_release)?;
            let (mut raw, mut host) = (std::ptr::null_mut(), std::ptr::null_mut());
            backend::check(b, "buffer_alloc", |err| alloc(ctx.inner.raw, &desc, &mut raw, &mut host, err))?;
            new_buffer(ctx, desc, raw, host, release, out)
        })
    }
}

fn handle_kind(kind: u32) -> Result<u32> {
    if !(TURBO_HANDLE_HOST_PTR..=TURBO_HANDLE_DMABUF_FD).contains(&kind) {
        return Err(Error::new(INVALID_ENUM, format!("kind: {kind} is not a TURBO_HANDLE_* value")));
    }
    Ok(kind)
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says; the memory
/// `handle` names holds the buffer's bytes while the buffer is in use.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_buffer_import(
    ctx: *mut turbo_context,
    desc: *const turbo_buffer_desc,
    handle: *const turbo_native_handle,
    out: *mut *mut turbo_buffer,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let ctx = context(ctx)?;
            let out = out_ptr(out, "out")?;
            let desc = buffer_desc(desc)?;
            let handle = *handle.as_ref().ok_or_else(|| Error::new(INVALID_ARGUMENT, "handle is NULL"))?;
            sized(handle.struct_size, size_of::<turbo_native_handle>(), "turbo_native_handle")?;
            handle_kind(handle.kind)?;
            let b = ctx.inner.backend;
            let import = backend::offered!(b, buffer_import)?;
            let release = backend::offered!(b, buffer_release)?;
            let (mut raw, mut host) = (std::ptr::null_mut(), std::ptr::null_mut());
            backend::check(b, "buffer_import", |err| import(ctx.inner.raw, &desc, &handle, &mut raw, &mut host, err))?;
            new_buffer(ctx, desc, raw, host, release, out)
        })
    }
}

/// # Safety
/// `buf` is NULL or a handle from turbo_buffer_alloc or turbo_buffer_import,
/// released once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_buffer_release(buf: *mut turbo_buffer) {
    if unsafe { buffer(buf) }.is_ok() {
        let mut b = unsafe { Box::from_raw(buf) };
        b.magic = 0;
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_buffer_get_desc(
    buf: *mut turbo_buffer,
    out: *mut turbo_buffer_desc,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let buf = buffer(buf)?;
            let out = out_ptr(out, "out")?;
            sized(out.struct_size, size_of::<turbo_buffer_desc>(), "turbo_buffer_desc")?;
            *out = buf.inner.desc;
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_buffer_host_ptr(
    buf: *mut turbo_buffer,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let buf = &buffer(buf)?.inner;
            let out = out_ptr(out, "out")?;
            if buf.desc.placement == TURBO_PLACE_DEVICE {
                return Err(Error::new(status::UNSUPPORTED, "a DEVICE buffer has no host pointer"));
            }
            *out = buf.host;
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_buffer_export(
    buf: *mut turbo_buffer,
    kind: u32,
    out: *mut turbo_native_handle,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let buf = &buffer(buf)?.inner;
            let out = out_ptr(out, "out")?;
            sized(out.struct_size, size_of::<turbo_native_handle>(), "turbo_native_handle")?;
            handle_kind(kind)?;
            let b = buf.context.backend;
            let export = backend::offered!(b, buffer_export)?;
            let mut h: turbo_native_handle = std::mem::zeroed();
            backend::check(b, "buffer_export", |err| export(buf.raw, kind, &mut h, err))?;
            h.struct_size = size_of::<turbo_native_handle>() as u32;
            *out = h;
            Ok(())
        })
    }
}

// ---- Model -----------------------------------------------------------------------

/// A bundle loaded on a context's device. It holds its context, so the
/// backend's model is released before the backend's context.
struct Model {
    context: Arc<Context>,
    raw: *mut c_void,
    release: unsafe extern "C" fn(*mut c_void),
    /// The core's one verified host copy of the weights. A backend with its
    /// own memory copied them there at load; the CPU backend reads them
    /// here, in place, so they live exactly as long as the backend's model.
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

/// docs/bundle.md's loader rules, in order, for the context's device.
fn load_model(ctx: &turbo_context, path: &str) -> Result<Model> {
    let c = &ctx.inner;
    let b = c.backend;
    let load = backend::offered!(b, model_load)?;
    let release = backend::offered!(b, model_release)?;
    let bundle = bundle::Bundle::open(Path::new(path))?; // rules 1 to 4
    let tokenizer = Tokenizer::load(&bundle)?; // rule 5
    let m = &bundle.manifest;
    let arch = cstr(&c.runtime.devices[c.device as usize].info.arch);
    let index = model::choose(m, b.name(), &arch)?; // rule 6
    let weights = model::Weights::load(&bundle, index)?; // rules 7 and 8
    let art = &m.artifacts[index];

    let e = m.embed();
    let mut info: turbo_model_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_model_info>() as u32;
    info.task = TURBO_TASK_EMBED;
    info.dim = e.dim;
    info.pooling = match e.pooling {
        manifest::Pooling::Mean => TURBO_POOLING_MEAN,
        manifest::Pooling::Cls => TURBO_POOLING_CLS,
        manifest::Pooling::Last => TURBO_POOLING_LAST,
    };
    info.normalize = match e.normalize {
        manifest::Normalize::None => TURBO_NORMALIZE_NONE,
        manifest::Normalize::L2 => TURBO_NORMALIZE_L2,
    };
    // A fixed-shape artifact reports the smaller of its shape and the model's.
    let fixed = |model: u32, artifact: u32| if artifact == 0 { model } else { model.min(artifact) };
    info.max_seq = fixed(e.max_seq, art.fixed_seq);
    info.max_batch = fixed(e.max_batch, art.fixed_batch);
    // Raw weights fix no compute_dtype (the manifest refuses one), so the
    // artifact's dtype is the one its weights are stored in.
    info.dtype = weights.dtype;
    // The manifest's strings were checked against these buffers when it
    // was parsed (rule 2), and a hash is 64 hex digits: nothing is cut.
    write_str(&mut info.model_id, &m.model.id);
    write_str(&mut info.revision, &m.model.revision);
    write_str(&mut info.manifest_sha256, &bundle.manifest_sha256);
    write_str(&mut info.artifact_sha256, &model::artifact_sha256(m, art));
    write_str(&mut info.tokenizer_sha256, &tokenizer.sha256);
    write_str(&mut info.prefix_query, &e.prefix_query);
    write_str(&mut info.prefix_document, &e.prefix_document);

    let tensors = weights.tensors();
    let desc = weights.desc(&tensors);
    let mut raw = std::ptr::null_mut();
    backend::check(b, "model_load", |err| unsafe { load(c.raw, &desc, &mut raw, err) })?; // rule 9
    let output_dims = e.output_dims.clone();
    Ok(Model { context: c.clone(), raw, release, weights, tokenizer, info, output_dims })
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_model_load(
    ctx: *mut turbo_context,
    bundle_path: turbo_text,
    out: *mut *mut turbo_model,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let ctx = context(ctx)?;
            let out = out_ptr(out, "out")?;
            let path = text(bundle_path, "bundle_path")?;
            let inner = load_model(ctx, path)?;
            *out = Box::into_raw(Box::new(turbo_model { magic: MODEL_MAGIC, inner: Arc::new(inner) }));
            Ok(())
        })
    }
}

/// # Safety
/// `m` is NULL or a handle from turbo_model_load, released once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_model_release(m: *mut turbo_model) {
    if unsafe { model_handle(m) }.is_ok() {
        let mut b = unsafe { Box::from_raw(m) };
        b.magic = 0;
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_model_get_info(
    m: *mut turbo_model,
    out: *mut turbo_model_info,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let m = model_handle(m)?;
            let out = out_ptr(out, "out")?;
            sized(out.struct_size, size_of::<turbo_model_info>(), "turbo_model_info")?;
            *out = m.inner.info;
            Ok(())
        })
    }
}

/// Where a model's weights are, for the tests that check the CPU backend
/// holds no copy of its own. Not part of the C interface, and built only
/// with the `internals` feature.
#[cfg(feature = "internals")]
pub struct ModelWeights<'a> {
    /// The core's verified bytes of each weights file.
    pub files: Vec<&'a [u8]>,
    /// On the CPU backend, the address of each tensor the backend keeps.
    pub held: Option<Vec<*const c_void>>,
}

/// # Safety
/// `m` is a live handle from turbo_model_load, and outlives what is returned.
#[cfg(feature = "internals")]
pub unsafe fn model_weights<'a>(m: *mut turbo_model) -> Option<ModelWeights<'a>> {
    let m = &unsafe { model_handle(m) }.ok()?.inner;
    #[cfg(feature = "cpu")]
    let held = std::ptr::eq(m.context.backend, &cpu::BACKEND).then(|| unsafe { cpu::tensor_data(m.raw) });
    #[cfg(not(feature = "cpu"))]
    let held = None;
    Some(ModelWeights { files: m.weights.files(), held })
}

/// Where the CPU backend's F32 copy of an F16 or BF16 model's weights is,
/// once a session has made it. Built only with the `internals` feature.
///
/// # Safety
/// As for model_weights.
#[cfg(feature = "internals")]
pub unsafe fn model_converted_weights(m: *mut turbo_model) -> Option<Vec<*const c_void>> {
    let m = &unsafe { model_handle(m) }.ok()?.inner;
    #[cfg(feature = "cpu")]
    if std::ptr::eq(m.context.backend, &cpu::BACKEND) {
        return unsafe { cpu::converted_data(m.raw) };
    }
    None
}

// ---- Tokenizer -----------------------------------------------------------------

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_tokenizer_create(
    rt: *mut turbo_runtime,
    bundle_path: turbo_text,
    out: *mut *mut turbo_tokenizer,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let rt = runtime(rt)?;
            let out = out_ptr(out, "out")?;
            let path = text(bundle_path, "bundle_path")?;
            let bundle = bundle::Bundle::open(Path::new(path))?;
            let tok = Tokenizer::load(&bundle)?;
            let t = turbo_tokenizer { magic: TOKENIZER_MAGIC, _runtime: rt.inner.clone(), tok };
            *out = Box::into_raw(Box::new(t));
            Ok(())
        })
    }
}

/// # Safety
/// `t` is NULL or a handle from turbo_tokenizer_create, released once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_tokenizer_release(t: *mut turbo_tokenizer) {
    if unsafe { tokenizer(t) }.is_ok() {
        let mut b = unsafe { Box::from_raw(t) };
        b.magic = 0;
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_tokenizer_get_info(
    t: *mut turbo_tokenizer,
    out: *mut turbo_tokenizer_info,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let tok = &tokenizer(t)?.tok;
            let out = out_ptr(out, "out")?;
            sized(out.struct_size, size_of::<turbo_tokenizer_info>(), "turbo_tokenizer_info")?;
            out.vocab_size = tok.vocab_size();
            out.max_seq = tok.max_seq;
            out.specials_per_sequence = tok.specials_per_sequence();
            out.pad_id = tok.pad_id;
            out.bos_id = tok.bos_id;
            out.eos_id = tok.eos_id;
            out.unk_id = tok.unk_id;
            write_str(&mut out.kind, "wordpiece");
            write_str(&mut out.sha256, &tok.sha256);
            write_str(&mut out.manifest_sha256, &tok.manifest_sha256);
            Ok(())
        })
    }
}

/// The options as the tokenizer takes them; NULL, or 0 in every field, is
/// what the bundle says.
unsafe fn encode_options(tok: &Tokenizer, opts: *const turbo_encode_options) -> Result<Encode> {
    let mut e = Encode {
        add_special_tokens: true,
        truncation: tok.truncation(),
        max_tokens: tok.max_seq,
        prompt: PromptRole::None,
    };
    let Some(o) = (unsafe { opts.as_ref() }) else {
        return Ok(e);
    };
    sized(o.struct_size, size_of::<turbo_encode_options>(), "turbo_encode_options")?;
    e.add_special_tokens = match o.omit_special_tokens {
        0 => true,
        1 => false,
        v => {
            return Err(Error::field(INVALID_ARGUMENT, 1, format!("omit_special_tokens is {v}, not 0 or 1")));
        }
    };
    e.truncation = match o.truncate {
        TURBO_TRUNCATE_MODEL => tok.truncation(),
        TURBO_TRUNCATE_NONE => Truncation::None,
        TURBO_TRUNCATE_RIGHT => Truncation::Right,
        TURBO_TRUNCATE_LEFT => Truncation::Left,
        v => {
            return Err(Error::new(INVALID_ENUM, format!("truncate: {v} is not a TURBO_TRUNCATE_* value")));
        }
    };
    if o.max_tokens != 0 {
        e.max_tokens = o.max_tokens;
    }
    e.prompt = prompt_role(o.prompt_role)?;
    Ok(e)
}

fn prompt_role(v: u32) -> Result<PromptRole> {
    match v {
        TURBO_PROMPT_NONE => Ok(PromptRole::None),
        TURBO_PROMPT_QUERY => Ok(PromptRole::Query),
        TURBO_PROMPT_DOCUMENT => Ok(PromptRole::Document),
        v => Err(Error::new(INVALID_ENUM, format!("prompt_role: {v} is not a TURBO_PROMPT_* value"))),
    }
}

/// # Safety
/// `texts` holds `count` views; `ids`, `mask` and, when not NULL, `types`
/// hold `count * row_stride` elements and `lengths` holds `count`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_tokenizer_encode(
    t: *mut turbo_tokenizer,
    texts: *const turbo_text,
    count: u32,
    opts: *const turbo_encode_options,
    ids: *mut i32,
    mask: *mut i32,
    types: *mut i32,
    row_stride: u32,
    lengths: *mut u32,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let tok = &tokenizer(t)?.tok;
            let e = encode_options(tok, opts)?;
            if count == 0 {
                return Ok(());
            }
            if texts.is_null() || ids.is_null() || mask.is_null() {
                return Err(Error::new(INVALID_ARGUMENT, "texts, ids and mask may not be NULL"));
            }
            if row_stride == 0 {
                return Err(Error::new(INVALID_ARGUMENT, "row_stride is 0"));
            }
            let texts = std::slice::from_raw_parts(texts, count as usize);
            // Every row is encoded before any is written, so a failed call
            // leaves the caller's arrays as they were.
            let mut rows = Vec::with_capacity(texts.len());
            for (i, &tx) in texts.iter().enumerate() {
                let row = tok.encode(text(tx, &format!("texts[{i}]"))?, e).map_err(|mut err| {
                    err.message = format!("texts[{i}]: {}", err.message);
                    err
                })?;
                if row.len() > row_stride as usize {
                    return Err(Error::new(
                        status::CAPACITY,
                        format!("texts[{i}]: {} tokens, row_stride is {row_stride}", row.len()),
                    ));
                }
                rows.push(row);
            }
            let n = count as usize * row_stride as usize;
            let ids = std::slice::from_raw_parts_mut(ids, n);
            let mask = std::slice::from_raw_parts_mut(mask, n);
            let mut types = (!types.is_null()).then(|| std::slice::from_raw_parts_mut(types, n));
            for (i, row) in rows.iter().enumerate() {
                let at = i * row_stride as usize;
                let end = at + row_stride as usize;
                ids[at..at + row.len()].copy_from_slice(row);
                ids[at + row.len()..end].fill(tok.pad_id);
                mask[at..at + row.len()].fill(1);
                mask[at + row.len()..end].fill(0);
                if let Some(ty) = types.as_deref_mut() {
                    ty[at..end].fill(0);
                }
                if !lengths.is_null() {
                    *lengths.add(i) = row.len() as u32;
                }
            }
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_tokenizer_count(
    t: *mut turbo_tokenizer,
    txt: turbo_text,
    prompt: u32,
    out: *mut u32,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let tok = &tokenizer(t)?.tok;
            let prompt = prompt_role(prompt)?;
            let out = out_ptr(out, "out")?;
            *out = tok.count(text(txt, "text")?, prompt) as u32;
            Ok(())
        })
    }
}
