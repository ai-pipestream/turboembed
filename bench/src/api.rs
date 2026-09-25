//! The library through its C interface, as any caller uses it: each
//! handle owned by a Rust value that releases it once, and each failure
//! as its status name and message.

use std::ffi::{CStr, c_char};
use std::path::Path;
use std::ptr;

use turbo::*;

use crate::Result;

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

/// Call `f` with a fresh turbo_error; a status other than 0 is an error
/// naming `what`, the status and the library's message.
fn check(what: &str, f: impl FnOnce(*mut turbo_error) -> i32) -> Result<()> {
    let mut err = new_error();
    let rc = f(&mut err);
    if rc != 0 {
        let name = unsafe { CStr::from_ptr(turbo_status_name(rc)) }.to_string_lossy().into_owned();
        return Err(format!("{what}: {name}: {}", field(&err.message)));
    }
    Ok(())
}

/// turbo_version().
pub fn version() -> String {
    unsafe { CStr::from_ptr(turbo_version()) }.to_string_lossy().into_owned()
}

pub struct Runtime(*mut turbo_runtime);

impl Runtime {
    pub fn create() -> Result<Runtime> {
        let mut rt = ptr::null_mut();
        check("turbo_runtime_create", |e| unsafe { turbo_runtime_create(ptr::null(), &mut rt, e) })?;
        Ok(Runtime(rt))
    }

    pub fn device_count(&self) -> Result<u32> {
        let mut n = 0;
        check("turbo_runtime_device_count", |e| unsafe { turbo_runtime_device_count(self.0, &mut n, e) })?;
        Ok(n)
    }

    pub fn device_info(&self, index: u32) -> Result<turbo_device_info> {
        let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_device_info>() as u32;
        check("turbo_runtime_device_info", |e| unsafe { turbo_runtime_device_info(self.0, index, &mut info, e) })?;
        Ok(info)
    }

    pub fn capability(&self, index: u32, task: u32, precision: u32) -> Result<turbo_capability> {
        let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
        cap.struct_size = size_of::<turbo_capability>() as u32;
        check("turbo_runtime_capability", |e| unsafe {
            turbo_runtime_capability(self.0, index, task, precision, &mut cap, e)
        })?;
        Ok(cap)
    }

    /// The device `want` names: a runtime device index, or a backend name
    /// for the first device that backend lists.
    pub fn find(&self, want: &str) -> Result<u32> {
        let n = self.device_count()?;
        if let Ok(i) = want.parse::<u32>() {
            return if i < n { Ok(i) } else { Err(format!("device {i}: the runtime lists {n}")) };
        }
        for i in 0..n {
            if field(&self.device_info(i)?.backend) == want {
                return Ok(i);
            }
        }
        Err(format!("device {want}: no device of that backend is listed (turbo_version: {})", version()))
    }

    /// The first CPU device, if the build lists one.
    pub fn cpu(&self) -> Result<Option<turbo_device_info>> {
        for i in 0..self.device_count()? {
            let info = self.device_info(i)?;
            if info.kind == TURBO_DEVICE_CPU {
                return Ok(Some(info));
            }
        }
        Ok(None)
    }

    pub fn context(&self, device: u32) -> Result<Context> {
        let mut c = ptr::null_mut();
        check("turbo_context_create", |e| unsafe { turbo_context_create(self.0, device, &mut c, e) })?;
        Ok(Context(c))
    }

    pub fn tokenizer(&self, bundle: &Path) -> Result<Tokenizer> {
        let path = bundle.to_str().ok_or_else(|| format!("{}: not UTF-8", bundle.display()))?;
        let mut t = ptr::null_mut();
        check("turbo_tokenizer_create", |e| unsafe { turbo_tokenizer_create(self.0, text(path), &mut t, e) })?;
        Ok(Tokenizer(t))
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        unsafe { turbo_runtime_release(self.0) };
    }
}

pub struct Tokenizer(*mut turbo_tokenizer);

impl Tokenizer {
    pub fn info(&self) -> Result<turbo_tokenizer_info> {
        let mut info: turbo_tokenizer_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_tokenizer_info>() as u32;
        check("turbo_tokenizer_get_info", |e| unsafe { turbo_tokenizer_get_info(self.0, &mut info, e) })?;
        Ok(info)
    }
}

impl Drop for Tokenizer {
    fn drop(&mut self) {
        unsafe { turbo_tokenizer_release(self.0) };
    }
}

pub struct Context(*mut turbo_context);

impl Context {
    pub fn load(&self, bundle: &Path) -> Result<Model> {
        let path = bundle.to_str().ok_or_else(|| format!("{}: not UTF-8", bundle.display()))?;
        let mut m = ptr::null_mut();
        check("turbo_model_load", |e| unsafe { turbo_model_load(self.0, text(path), &mut m, e) })?;
        Ok(Model(m))
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { turbo_context_release(self.0) };
    }
}

pub struct Model(*mut turbo_model);

impl Model {
    pub fn info(&self) -> Result<turbo_model_info> {
        let mut info: turbo_model_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_model_info>() as u32;
        check("turbo_model_get_info", |e| unsafe { turbo_model_get_info(self.0, &mut info, e) })?;
        Ok(info)
    }

    pub fn session(&self, max_batch: u32, max_seq: u32, precision: u32) -> Result<Session> {
        let desc =
            turbo_session_desc { struct_size: size_of::<turbo_session_desc>() as u32, max_batch, max_seq, precision };
        let mut s = ptr::null_mut();
        check("turbo_session_create", |e| unsafe { turbo_session_create(self.0, &desc, &mut s, e) })?;
        Ok(Session(s))
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { turbo_model_release(self.0) };
    }
}

/// Token rows as turbo_embed_write_tokens takes them: [batch, seq] int32,
/// packed.
pub struct Batch<'a> {
    pub batch: u32,
    pub seq: u32,
    pub ids: &'a [i32],
    pub mask: &'a [i32],
    pub types: &'a [i32],
}

pub struct Session(*mut turbo_session);

impl Session {
    pub fn info(&self) -> Result<turbo_session_info> {
        let mut info: turbo_session_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_session_info>() as u32;
        check("turbo_session_get_info", |e| unsafe { turbo_session_get_info(self.0, &mut info, e) })?;
        Ok(info)
    }

    /// Write `rows` with the bundle's options, run, and copy the vectors
    /// into `out`, which holds batch x dim values. Returns the result's
    /// info.
    pub fn embed_into(&self, rows: &Batch, out: &mut [f32]) -> Result<turbo_result_info> {
        let n = rows.batch as usize * rows.seq as usize;
        assert!(rows.ids.len() == n && rows.mask.len() == n && rows.types.len() == n, "rows are [batch, seq]");
        let b = turbo_token_batch {
            struct_size: size_of::<turbo_token_batch>() as u32,
            batch: rows.batch,
            seq: rows.seq,
            row_stride: rows.seq,
            ids: rows.ids.as_ptr(),
            mask: rows.mask.as_ptr(),
            types: rows.types.as_ptr(),
        };
        check("turbo_embed_write_tokens", |e| unsafe { turbo_embed_write_tokens(self.0, &b, ptr::null(), e) })?;
        let mut r = ptr::null_mut();
        check("turbo_session_run", |e| unsafe { turbo_session_run(self.0, &mut r, e) })?;
        let r = Output(r);
        let mut info: turbo_result_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_result_info>() as u32;
        check("turbo_result_get_info", |e| unsafe { turbo_result_get_info(r.0, &mut info, e) })?;
        let mut written = 0;
        let capacity = std::mem::size_of_val(out) as u64;
        check("turbo_result_read", |e| unsafe {
            turbo_result_read(r.0, out.as_mut_ptr().cast(), capacity, &mut written, e)
        })?;
        if written != capacity {
            return Err(format!("turbo_result_read wrote {written} bytes, the rows make {capacity}"));
        }
        Ok(info)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { turbo_session_release(self.0) };
    }
}

struct Output(*mut turbo_result);

impl Drop for Output {
    fn drop(&mut self) {
        unsafe { turbo_result_release(self.0) };
    }
}
