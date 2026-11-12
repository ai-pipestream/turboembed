//! The library through its C interface, as any caller uses it: each
//! handle owned by a Rust value that releases it once, and each failure
//! as its status name and message.

use std::ffi::{CStr, c_char, c_void};
use std::path::Path;
use std::ptr;
use std::sync::{Arc, Mutex};

use turbo::*;

use crate::Result;

const LOG_WARNING: u32 = 1;

type Warnings = Arc<Mutex<Vec<String>>>;

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

pub struct Runtime {
    raw: *mut turbo_runtime,
    warnings: Warnings,
}

pub(crate) unsafe extern "C" fn collect_warning(user_data: *mut c_void, level: u32, message: turbo_text) {
    if level > LOG_WARNING || user_data.is_null() {
        return;
    }
    let bytes = if message.len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(message.ptr.cast::<u8>(), message.len as usize) }
    };
    let lock = unsafe { &*(user_data as *const Mutex<Vec<String>>) };
    lock.lock().unwrap_or_else(|p| p.into_inner()).push(String::from_utf8_lossy(bytes).into_owned());
}

impl Runtime {
    pub fn create() -> Result<Runtime> {
        let warnings: Warnings = Arc::new(Mutex::new(Vec::new()));
        let desc = turbo_runtime_desc {
            struct_size: size_of::<turbo_runtime_desc>() as u32,
            reserved: 0,
            log: Some(collect_warning),
            log_user_data: Arc::as_ptr(&warnings).cast_mut().cast(),
        };
        let mut raw = ptr::null_mut();
        check("turbo_runtime_create", |e| unsafe { turbo_runtime_create(&desc, &mut raw, e) })?;
        Ok(Runtime { raw, warnings })
    }

    /// Warning and error messages logged by the library on this runtime.
    pub fn warnings(&self) -> Vec<String> {
        self.warnings.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    pub fn device_count(&self) -> Result<u32> {
        let mut n = 0;
        check("turbo_runtime_device_count", |e| unsafe { turbo_runtime_device_count(self.raw, &mut n, e) })?;
        Ok(n)
    }

    pub fn device_info(&self, index: u32) -> Result<turbo_device_info> {
        let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_device_info>() as u32;
        check("turbo_runtime_device_info", |e| unsafe { turbo_runtime_device_info(self.raw, index, &mut info, e) })?;
        Ok(info)
    }

    pub fn capability(&self, index: u32, task: u32, precision: u32) -> Result<turbo_capability> {
        let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
        cap.struct_size = size_of::<turbo_capability>() as u32;
        check("turbo_runtime_capability", |e| unsafe {
            turbo_runtime_capability(self.raw, index, task, precision, &mut cap, e)
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
        check("turbo_context_create", |e| unsafe { turbo_context_create(self.raw, device, &mut c, e) })?;
        Ok(Context { raw: c, warnings: Arc::clone(&self.warnings) })
    }

    pub fn tokenizer(&self, bundle: &Path) -> Result<Tokenizer> {
        let path = bundle.to_str().ok_or_else(|| format!("{}: not UTF-8", bundle.display()))?;
        let mut t = ptr::null_mut();
        check("turbo_tokenizer_create", |e| unsafe { turbo_tokenizer_create(self.raw, text(path), &mut t, e) })?;
        Ok(Tokenizer { raw: t, _warnings: Arc::clone(&self.warnings) })
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        unsafe { turbo_runtime_release(self.raw) };
    }
}

pub struct Tokenizer {
    raw: *mut turbo_tokenizer,
    _warnings: Warnings,
}

impl Tokenizer {
    pub fn info(&self) -> Result<turbo_tokenizer_info> {
        let mut info: turbo_tokenizer_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_tokenizer_info>() as u32;
        check("turbo_tokenizer_get_info", |e| unsafe { turbo_tokenizer_get_info(self.raw, &mut info, e) })?;
        Ok(info)
    }

    /// One text's ids with the bundle's template and prompt role
    /// (TURBO_PROMPT_*), cut to `max_tokens` the way the bundle truncates.
    pub fn encode(&self, s: &str, prompt_role: u32, max_tokens: u32) -> Result<Vec<i32>> {
        let opts = turbo_encode_options {
            struct_size: size_of::<turbo_encode_options>() as u32,
            omit_special_tokens: 0,
            truncate: TURBO_TRUNCATE_MODEL,
            max_tokens,
            prompt_role,
        };
        let stride = max_tokens.max(1);
        let (mut ids, mut mask) = (vec![0i32; stride as usize], vec![0i32; stride as usize]);
        let mut len = 0u32;
        let t = text(s);
        check("turbo_tokenizer_encode", |e| {
            // SAFETY: the handle is live for &self; `t` views `s`, which
            // outlives the call; ids and mask hold one row of `stride`
            // values, types may be null, and `len` takes the one length.
            unsafe {
                turbo_tokenizer_encode(
                    self.raw,
                    &t,
                    1,
                    &opts,
                    ids.as_mut_ptr(),
                    mask.as_mut_ptr(),
                    ptr::null_mut(),
                    stride,
                    &mut len,
                    e,
                )
            }
        })?;
        ids.truncate(len as usize);
        Ok(ids)
    }
}

impl Drop for Tokenizer {
    fn drop(&mut self) {
        unsafe { turbo_tokenizer_release(self.raw) };
    }
}

pub struct Context {
    raw: *mut turbo_context,
    warnings: Warnings,
}

impl Context {
    pub fn load(&self, bundle: &Path) -> Result<Model> {
        let path = bundle.to_str().ok_or_else(|| format!("{}: not UTF-8", bundle.display()))?;
        let mut m = ptr::null_mut();
        check("turbo_model_load", |e| unsafe { turbo_model_load(self.raw, text(path), &mut m, e) })?;
        Ok(Model { raw: m, warnings: Arc::clone(&self.warnings) })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { turbo_context_release(self.raw) };
    }
}

pub struct Model {
    raw: *mut turbo_model,
    warnings: Warnings,
}

impl Model {
    pub fn info(&self) -> Result<turbo_model_info> {
        let mut info: turbo_model_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_model_info>() as u32;
        check("turbo_model_get_info", |e| unsafe { turbo_model_get_info(self.raw, &mut info, e) })?;
        Ok(info)
    }

    pub fn session(&self, max_batch: u32, max_seq: u32, precision: u32) -> Result<Session> {
        self.session_with_ranges(max_batch, max_seq, precision, 0)
    }

    /// A session that also writes up to `max_ranges` ranges per run; 0 is
    /// `max_batch`.
    pub fn session_with_ranges(
        &self,
        max_batch: u32,
        max_seq: u32,
        precision: u32,
        max_ranges: u32,
    ) -> Result<Session> {
        // Tuning as TURBO_AUTOTUNE says, which the record's settings name.
        let desc = turbo_session_desc {
            struct_size: size_of::<turbo_session_desc>() as u32,
            max_batch,
            max_seq,
            precision,
            tuning: 0,
            tuning_budget_ms: 0,
            cpu_threads: 0,
            max_ranges,
        };
        let mut s = ptr::null_mut();
        check("turbo_session_create", |e| unsafe { turbo_session_create(self.raw, &desc, &mut s, e) })?;
        Ok(Session { raw: s, _warnings: Arc::clone(&self.warnings) })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { turbo_model_release(self.raw) };
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

/// `rows` as the C interface takes them.
fn token_batch(rows: &Batch) -> turbo_token_batch {
    let n = rows.batch as usize * rows.seq as usize;
    assert!(rows.ids.len() == n && rows.mask.len() == n && rows.types.len() == n, "rows are [batch, seq]");
    turbo_token_batch {
        struct_size: size_of::<turbo_token_batch>() as u32,
        batch: rows.batch,
        seq: rows.seq,
        row_stride: rows.seq,
        ids: rows.ids.as_ptr(),
        mask: rows.mask.as_ptr(),
        types: rows.types.as_ptr(),
    }
}

pub struct Session {
    raw: *mut turbo_session,
    _warnings: Warnings,
}

impl Session {
    pub fn info(&self) -> Result<turbo_session_info> {
        let mut info: turbo_session_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_session_info>() as u32;
        check("turbo_session_get_info", |e| unsafe { turbo_session_get_info(self.raw, &mut info, e) })?;
        Ok(info)
    }

    /// Write `rows` with the bundle's options, run, and copy the vectors
    /// into `out`, which holds batch x dim values. Returns the result's
    /// info.
    pub fn embed_into(&self, rows: &Batch, out: &mut [f32]) -> Result<turbo_result_info> {
        let b = token_batch(rows);
        check("turbo_embed_write_tokens", |e| unsafe { turbo_embed_write_tokens(self.raw, &b, ptr::null(), e) })?;
        self.read_into(out)
    }

    /// Write `texts` with `opts`, run, and copy the vectors into `out`,
    /// which holds texts x dim values. Returns the result's info.
    pub fn embed_texts_into(
        &self,
        texts: &[&str],
        opts: &turbo_embed_options,
        out: &mut [f32],
    ) -> Result<turbo_result_info> {
        let views: Vec<turbo_text> = texts.iter().map(|s| text(s)).collect();
        let n = u32::try_from(views.len()).map_err(|_| "more texts than a u32 counts".to_owned())?;
        check("turbo_embed_write_text", |e| unsafe { turbo_embed_write_text(self.raw, views.as_ptr(), n, opts, e) })?;
        self.read_into(out)
    }

    /// Write `rows` and one vector per range of `ranges` (each mean
    /// pooled over its tokens), run, and copy the vectors into `out`,
    /// which holds ranges x dim values.
    pub fn embed_ranges_into(
        &self,
        rows: &Batch,
        ranges: &[turbo_token_range],
        out: &mut [f32],
    ) -> Result<turbo_result_info> {
        let b = token_batch(rows);
        let n = u32::try_from(ranges.len()).map_err(|_| "more ranges than a u32 counts".to_owned())?;
        check("turbo_embed_write_ranges", |e| unsafe {
            turbo_embed_write_ranges(self.raw, &b, ranges.as_ptr(), n, ptr::null(), e)
        })?;
        self.read_into(out)
    }

    fn read_into(&self, out: &mut [f32]) -> Result<turbo_result_info> {
        let mut r = ptr::null_mut();
        check("turbo_session_run", |e| unsafe { turbo_session_run(self.raw, &mut r, e) })?;
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
        unsafe { turbo_session_release(self.raw) };
    }
}

struct Output(*mut turbo_result);

impl Drop for Output {
    fn drop(&mut self) {
        unsafe { turbo_result_release(self.0) };
    }
}
