//! The library through its C interface, as any caller uses it: each handle
//! owned by a Rust value that releases it once, each struct passed with its
//! struct_size set, and each failure kept as the turbo_error the call filled.

use std::ffi::{CStr, c_char};
use std::ptr;

use turbo::*;

/// A failed call: the status code, turbo_error.field and turbo_error.message.
/// A refusal the server makes itself is one of these too, with its own
/// message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    pub code: i32,
    pub field: u32,
    pub message: String,
}

impl Failure {
    pub fn new(code: i32, field: u32, message: impl Into<String>) -> Failure {
        Failure { code, field, message: message.into() }
    }
}

pub type Result<T> = std::result::Result<T, Failure>;

fn new_error() -> turbo_error {
    turbo_error {
        struct_size: size_of::<turbo_error>() as u32,
        code: 0,
        field: 0,
        message: [0; TURBO_ERROR_MESSAGE_LEN],
    }
}

/// A fixed C string buffer as text, up to its NUL.
pub fn field(b: &[c_char]) -> String {
    let bytes: Vec<u8> = b.iter().take_while(|&&c| c != 0).map(|&c| u8::from_ne_bytes(c.to_ne_bytes())).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn text(s: &str) -> turbo_text {
    turbo_text { ptr: s.as_ptr() as *const c_char, len: s.len() as u64 }
}

/// Call `f` with a fresh turbo_error; a status other than 0 is a Failure.
fn check(f: impl FnOnce(*mut turbo_error) -> i32) -> Result<()> {
    let mut err = new_error();
    let rc = f(&mut err);
    if rc != 0 {
        return Err(Failure { code: rc, field: err.field, message: field(&err.message) });
    }
    Ok(())
}

/// turbo_version().
pub fn version() -> String {
    unsafe { CStr::from_ptr(turbo_version()) }.to_string_lossy().into_owned()
}

/// turbo_status_name(code).
pub fn status_name(code: i32) -> String {
    unsafe { CStr::from_ptr(turbo_status_name(code)) }.to_string_lossy().into_owned()
}

fn sized<T>() -> T {
    // Every struct the library fills starts with struct_size; the rest is
    // zero until the call writes it.
    let mut v: T = unsafe { std::mem::zeroed() };
    unsafe { *(&mut v as *mut T as *mut u32) = size_of::<T>() as u32 };
    v
}

// Runtime, context and model may be used from any thread (turbo.h,
// Conventions).
pub struct Runtime(*mut turbo_runtime);
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

impl Runtime {
    pub fn create() -> Result<Runtime> {
        let mut rt = ptr::null_mut();
        check(|e| unsafe { turbo_runtime_create(ptr::null(), &mut rt, e) })?;
        Ok(Runtime(rt))
    }

    pub fn device_info(&self, index: u32) -> Result<turbo_device_info> {
        let mut info: turbo_device_info = sized();
        check(|e| unsafe { turbo_runtime_device_info(self.0, index, &mut info, e) })?;
        Ok(info)
    }

    pub fn capability(&self, index: u32, task: u32, precision: u32) -> Result<turbo_capability> {
        let mut cap: turbo_capability = sized();
        check(|e| unsafe { turbo_runtime_capability(self.0, index, task, precision, &mut cap, e) })?;
        Ok(cap)
    }

    pub fn select(&self, task: u32) -> Result<u32> {
        let mut out = 0;
        check(|e| unsafe { turbo_runtime_select(self.0, task, &mut out, ptr::null_mut(), 0, e) })?;
        Ok(out)
    }

    pub fn context(&self, device: u32) -> Result<Context> {
        let mut c = ptr::null_mut();
        check(|e| unsafe { turbo_context_create(self.0, device, &mut c, e) })?;
        Ok(Context(c))
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        unsafe { turbo_runtime_release(self.0) };
    }
}

pub struct Context(*mut turbo_context);
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    pub fn load(&self, bundle_path: &str) -> Result<Model> {
        let mut m = ptr::null_mut();
        check(|e| unsafe { turbo_model_load(self.0, text(bundle_path), &mut m, e) })?;
        Ok(Model(m))
    }

    /// A packed [rows, cols] buffer of `dtype` at `placement`.
    pub fn alloc(&self, placement: u32, dtype: u32, rows: u32, cols: u32) -> Result<Buffer> {
        let desc = turbo_buffer_desc {
            struct_size: size_of::<turbo_buffer_desc>() as u32,
            placement,
            dtype,
            ndim: 2,
            shape: [rows as u64, cols as u64],
            bytes: 0,
        };
        let mut b = ptr::null_mut();
        check(|e| unsafe { turbo_buffer_alloc(self.0, &desc, &mut b, e) })?;
        Ok(Buffer(b))
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { turbo_context_release(self.0) };
    }
}

pub struct Model(*mut turbo_model);
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    pub fn info(&self) -> Result<turbo_model_info> {
        let mut info: turbo_model_info = sized();
        check(|e| unsafe { turbo_model_get_info(self.0, &mut info, e) })?;
        Ok(info)
    }

    pub fn session(&self, desc: &turbo_session_desc) -> Result<Session> {
        let mut s = ptr::null_mut();
        check(|e| unsafe { turbo_session_create(self.0, desc, &mut s, e) })?;
        Ok(Session(s))
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { turbo_model_release(self.0) };
    }
}

pub struct Buffer(*mut turbo_buffer);
// A buffer is released from whichever thread drops the response that holds
// it; it has one owner at a time.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn host_ptr(&self) -> Result<*mut u8> {
        let mut p = ptr::null_mut();
        check(|e| unsafe { turbo_buffer_host_ptr(self.0, &mut p, e) })?;
        Ok(p.cast())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { turbo_buffer_release(self.0) };
    }
}

/// A session. Writes and runs go only through the one request that holds it
/// in the model's pool; turbo_session_get_info reads what was fixed at
/// turbo_session_create.
pub struct Session(*mut turbo_session);
unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Session {
    pub fn info(&self) -> Result<turbo_session_info> {
        let mut info: turbo_session_info = sized();
        check(|e| unsafe { turbo_session_get_info(self.0, &mut info, e) })?;
        Ok(info)
    }

    pub fn write_text(&self, texts: &[turbo_text], opts: &turbo_embed_options) -> Result<()> {
        let count = u32::try_from(texts.len())
            .map_err(|_| Failure::new(TURBO_E_INVALID_SHAPE, 0, "more texts than a uint32_t holds"))?;
        check(|e| unsafe { turbo_embed_write_text(self.0, texts.as_ptr(), count, opts, e) })
    }

    pub fn write_tokens(&self, batch: &turbo_token_batch, opts: &turbo_embed_options) -> Result<()> {
        check(|e| unsafe { turbo_embed_write_tokens(self.0, batch, opts, e) })
    }

    pub fn run(&self) -> Result<Output> {
        let mut r = ptr::null_mut();
        check(|e| unsafe { turbo_session_run(self.0, &mut r, e) })?;
        Ok(Output(r))
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { turbo_session_release(self.0) };
    }
}

/// A run's result.
pub struct Output(*mut turbo_result);
unsafe impl Send for Output {}
unsafe impl Sync for Output {}

impl Output {
    pub fn info(&self) -> Result<turbo_result_info> {
        let mut info: turbo_result_info = sized();
        check(|e| unsafe { turbo_result_get_info(self.0, &mut info, e) })?;
        Ok(info)
    }

    /// turbo_result_read into a new vector of `bytes` bytes.
    pub fn read(&self, bytes: u64) -> Result<Vec<u8>> {
        let n = usize::try_from(bytes).map_err(|_| Failure::new(TURBO_E_OUT_OF_MEMORY, 0, "result too large"))?;
        let mut v: Vec<u8> = Vec::with_capacity(n);
        let mut written = 0;
        check(|e| unsafe { turbo_result_read(self.0, v.as_mut_ptr().cast(), bytes, &mut written, e) })?;
        // The library wrote `written` bytes, at most the capacity it was given.
        unsafe { v.set_len(written.min(bytes) as usize) };
        Ok(v)
    }

    pub fn buffer(&self) -> Result<Buffer> {
        let mut b = ptr::null_mut();
        check(|e| unsafe { turbo_result_buffer(self.0, &mut b, e) })?;
        Ok(Buffer(b))
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        unsafe { turbo_result_release(self.0) };
    }
}

// Status codes of turbo.h.
pub const TURBO_OK: i32 = 0;
pub const TURBO_E_INVALID_ARGUMENT: i32 = 256;
pub const TURBO_E_INVALID_STRUCT_SIZE: i32 = 257;
pub const TURBO_E_INVALID_UTF8: i32 = 258;
pub const TURBO_E_INVALID_HANDLE: i32 = 259;
pub const TURBO_E_INVALID_SHAPE: i32 = 260;
pub const TURBO_E_INVALID_STATE: i32 = 261;
pub const TURBO_E_INVALID_ENUM: i32 = 262;
pub const TURBO_E_UNSUPPORTED: i32 = 512;
pub const TURBO_E_UNSUPPORTED_OPTION: i32 = 513;
pub const TURBO_E_UNSUPPORTED_TASK: i32 = 514;
pub const TURBO_E_OUT_OF_MEMORY: i32 = 768;
pub const TURBO_E_BUSY: i32 = 769;
pub const TURBO_E_CAPACITY: i32 = 771;
pub const TURBO_E_DEVICE_NOT_FOUND: i32 = 1024;
pub const TURBO_E_DEVICE_UNAVAILABLE: i32 = 1025;
pub const TURBO_E_RUNTIME: i32 = 1026;
pub const TURBO_E_BUNDLE_NOT_FOUND: i32 = 1280;
pub const TURBO_E_BUNDLE_INVALID: i32 = 1281;
pub const TURBO_E_BUNDLE_INTEGRITY: i32 = 1282;
pub const TURBO_E_BUNDLE_NO_ARTIFACT: i32 = 1283;
pub const TURBO_E_INTERNAL: i32 = 1536;
pub const TURBO_E_PANIC: i32 = 1537;

/// Enum values as text: the header constant without `TURBO_`. A value the
/// header gives no constant for is its decimal.
pub mod names {
    use turbo::*;

    fn pick(table: &[(u32, &str)], v: u32) -> String {
        table.iter().find(|(k, _)| *k == v).map(|(_, n)| n.to_string()).unwrap_or_else(|| v.to_string())
    }

    pub const TRUNCATE: &[(u32, &str)] = &[
        (TURBO_TRUNCATE_MODEL, "TRUNCATE_MODEL"),
        (TURBO_TRUNCATE_NONE, "TRUNCATE_NONE"),
        (TURBO_TRUNCATE_RIGHT, "TRUNCATE_RIGHT"),
        (TURBO_TRUNCATE_LEFT, "TRUNCATE_LEFT"),
    ];
    pub const PROMPT: &[(u32, &str)] = &[
        (TURBO_PROMPT_NONE, "PROMPT_NONE"),
        (TURBO_PROMPT_QUERY, "PROMPT_QUERY"),
        (TURBO_PROMPT_DOCUMENT, "PROMPT_DOCUMENT"),
    ];
    pub const NORMALIZE: &[(u32, &str)] = &[
        (TURBO_NORMALIZE_MODEL, "NORMALIZE_MODEL"),
        (TURBO_NORMALIZE_NONE, "NORMALIZE_NONE"),
        (TURBO_NORMALIZE_L2, "NORMALIZE_L2"),
    ];
    pub const POOLING: &[(u32, &str)] = &[
        (TURBO_POOLING_MODEL, "POOLING_MODEL"),
        (TURBO_POOLING_MEAN, "POOLING_MEAN"),
        (TURBO_POOLING_CLS, "POOLING_CLS"),
        (TURBO_POOLING_LAST, "POOLING_LAST"),
    ];
    pub const PRECISION: &[(u32, &str)] = &[
        (TURBO_PRECISION_MODEL, "PRECISION_MODEL"),
        (TURBO_PRECISION_FASTEST, "PRECISION_FASTEST"),
        (TURBO_PRECISION_EXACT, "PRECISION_EXACT"),
    ];
    const TASK: &[(u32, &str)] = &[(TURBO_TASK_EMBED, "TASK_EMBED")];
    const DTYPE: &[(u32, &str)] = &[
        (TURBO_DTYPE_I32, "DTYPE_I32"),
        (TURBO_DTYPE_F16, "DTYPE_F16"),
        (TURBO_DTYPE_BF16, "DTYPE_BF16"),
        (TURBO_DTYPE_F32, "DTYPE_F32"),
    ];
    const DEVICE: &[(u32, &str)] = &[
        (TURBO_DEVICE_CPU, "DEVICE_CPU"),
        (TURBO_DEVICE_GPU, "DEVICE_GPU"),
        (TURBO_DEVICE_IGPU, "DEVICE_IGPU"),
        (TURBO_DEVICE_NPU, "DEVICE_NPU"),
    ];
    const CAP: &[(u32, &str)] = &[(0, "CAP_UNSUPPORTED"), (1, "CAP_EXPERIMENTAL"), (2, "CAP_SUPPORTED")];
    const PLACE: &[(u32, &str)] = &[
        (TURBO_PLACE_HOST, "PLACE_HOST"),
        (TURBO_PLACE_PINNED, "PLACE_PINNED"),
        (TURBO_PLACE_DEVICE, "PLACE_DEVICE"),
        (TURBO_PLACE_SHARED, "PLACE_SHARED"),
    ];
    const STAGE: &[(u32, &str)] = &[
        (TURBO_STAGE_UNUSED, "STAGE_UNUSED"),
        (TURBO_STAGE_HOST, "STAGE_HOST"),
        (TURBO_STAGE_DEVICE, "STAGE_DEVICE"),
        (TURBO_STAGE_FUSED, "STAGE_FUSED"),
    ];
    /// The embed task's stage constants, by index.
    pub const EMBED_STAGES: [&str; TURBO_EMBED_STAGE_COUNT] = [
        "EMBED_STAGE_TOKENIZE",
        "EMBED_STAGE_UPLOAD",
        "EMBED_STAGE_LOOKUP",
        "EMBED_STAGE_ENCODE",
        "EMBED_STAGE_POOL",
        "EMBED_STAGE_NORMALIZE",
        "EMBED_STAGE_DOWNLOAD",
    ];

    pub fn task(v: u32) -> String {
        pick(TASK, v)
    }
    pub fn dtype(v: u32) -> String {
        pick(DTYPE, v)
    }
    pub fn device_kind(v: u32) -> String {
        pick(DEVICE, v)
    }
    pub fn cap(v: u32) -> String {
        pick(CAP, v)
    }
    pub fn place(v: u32) -> String {
        pick(PLACE, v)
    }
    pub fn stage(v: u32) -> String {
        pick(STAGE, v)
    }
    pub fn pooling(v: u32) -> String {
        pick(POOLING, v)
    }
    pub fn normalize(v: u32) -> String {
        pick(NORMALIZE, v)
    }
    pub fn precision(v: u32) -> String {
        pick(PRECISION, v)
    }
}
