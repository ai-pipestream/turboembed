//! The library through its C interface, for the steps that run a model:
//! each handle owned by a Rust value that releases it once, and each
//! failure as its status name and message.

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
fn field(b: &[c_char]) -> String {
    let bytes: Vec<u8> = b.iter().take_while(|&&c| c != 0).map(|&c| u8::from_ne_bytes(c.to_ne_bytes())).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn text(s: &str) -> turbo_text {
    turbo_text { ptr: s.as_ptr() as *const c_char, len: s.len() as u64 }
}

fn check(what: &str, f: impl FnOnce(*mut turbo_error) -> i32) -> Result<()> {
    let mut err = new_error();
    let rc = f(&mut err);
    if rc != 0 {
        let name = unsafe { CStr::from_ptr(turbo_status_name(rc)) }.to_string_lossy().into_owned();
        return Err(format!("{what}: {name}: {}", field(&err.message)));
    }
    Ok(())
}

pub struct Runtime(*mut turbo_runtime);

impl Runtime {
    pub fn create() -> Result<Runtime> {
        let mut rt = ptr::null_mut();
        check("turbo_runtime_create", |e| unsafe { turbo_runtime_create(ptr::null(), &mut rt, e) })?;
        Ok(Runtime(rt))
    }

    /// A context on the host processor.
    pub fn cpu(&self) -> Result<Context> {
        let mut n = 0;
        check("turbo_runtime_device_count", |e| unsafe { turbo_runtime_device_count(self.0, &mut n, e) })?;
        for i in 0..n {
            let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
            info.struct_size = size_of::<turbo_device_info>() as u32;
            check("turbo_runtime_device_info", |e| unsafe { turbo_runtime_device_info(self.0, i, &mut info, e) })?;
            if info.kind == TURBO_DEVICE_CPU {
                let mut c = ptr::null_mut();
                check("turbo_context_create", |e| unsafe { turbo_context_create(self.0, i, &mut c, e) })?;
                return Ok(Context(c));
            }
        }
        Err("this build of the library lists no cpu device".into())
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        unsafe { turbo_runtime_release(self.0) };
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

    /// A session at TURBO_PRECISION_EXACT, F32 throughout.
    pub fn session(&self, max_batch: u32, max_seq: u32) -> Result<Session> {
        let desc = turbo_session_desc {
            struct_size: size_of::<turbo_session_desc>() as u32,
            max_batch,
            max_seq,
            precision: TURBO_PRECISION_EXACT,
            tuning: TURBO_AUTOTUNE_OFF,
            tuning_budget_ms: 0,
        };
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

/// turbo_embed_options with every field the bundle's but pooling and
/// normalize, which 0 leaves to the bundle too.
pub fn options(pooling: u32, normalize: u32) -> turbo_embed_options {
    turbo_embed_options {
        struct_size: size_of::<turbo_embed_options>() as u32,
        truncate: 0,
        max_tokens: 0,
        prompt_role: 0,
        normalize,
        pooling,
        output_dim: 0,
    }
}

pub struct Session(*mut turbo_session);

impl Session {
    /// Rows [batch, seq], packed, every type 0: run them and copy the
    /// vectors into `out`, batch x dim values.
    pub fn tokens(
        &self,
        batch: u32,
        seq: u32,
        ids: &[i32],
        mask: &[i32],
        o: &turbo_embed_options,
        out: &mut [f32],
    ) -> Result<()> {
        let b = turbo_token_batch {
            struct_size: size_of::<turbo_token_batch>() as u32,
            batch,
            seq,
            row_stride: seq,
            ids: ids.as_ptr(),
            mask: mask.as_ptr(),
            types: ptr::null(),
        };
        check("turbo_embed_write_tokens", |e| unsafe { turbo_embed_write_tokens(self.0, &b, o, e) })?;
        self.run(out)
    }

    /// Texts, tokenized by the bundle: run them and copy the vectors into
    /// `out`, texts.len() x dim values.
    pub fn texts(&self, texts: &[&str], o: &turbo_embed_options, out: &mut [f32]) -> Result<()> {
        let t: Vec<turbo_text> = texts.iter().map(|s| text(s)).collect();
        check("turbo_embed_write_text", |e| unsafe {
            turbo_embed_write_text(self.0, t.as_ptr(), t.len() as u32, o, e)
        })?;
        self.run(out)
    }

    fn run(&self, out: &mut [f32]) -> Result<()> {
        let mut r = ptr::null_mut();
        check("turbo_session_run", |e| unsafe { turbo_session_run(self.0, &mut r, e) })?;
        let r = Output(r);
        let mut written = 0;
        let capacity = std::mem::size_of_val(out) as u64;
        check("turbo_result_read", |e| unsafe {
            turbo_result_read(r.0, out.as_mut_ptr().cast(), capacity, &mut written, e)
        })?;
        if written != capacity {
            return Err(format!("turbo_result_read wrote {written} bytes, the rows make {capacity}"));
        }
        Ok(())
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
