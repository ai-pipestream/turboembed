//! libturbo: the C interface in include/turbo/turbo.h.
//!
//! This cut has the runtime handle, status names and the tokenizer. The
//! types below mirror the header's; tests/abi.rs checks their layout
//! against a C compiler's.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Arc;

pub mod bundle;
pub mod manifest;
pub mod safetensors;
pub mod status;
pub mod tokenizer;

use manifest::{PromptRole, Truncation};
use status::{Error, INVALID_ARGUMENT, INVALID_ENUM, INVALID_HANDLE, INVALID_STRUCT_SIZE, INVALID_UTF8, PANIC, Result};
use tokenizer::{Encode, Tokenizer};

pub const TURBO_ERROR_MESSAGE_LEN: usize = 496;

pub const TURBO_TRUNCATE_MODEL: u32 = 0;
pub const TURBO_TRUNCATE_NONE: u32 = 1;
pub const TURBO_TRUNCATE_RIGHT: u32 = 2;
pub const TURBO_TRUNCATE_LEFT: u32 = 3;

pub const TURBO_PROMPT_NONE: u32 = 0;
pub const TURBO_PROMPT_QUERY: u32 = 1;
pub const TURBO_PROMPT_DOCUMENT: u32 = 2;

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
}

#[repr(C)]
pub struct turbo_encode_options {
    pub struct_size: u32,
    pub add_special_tokens: u32,
    pub truncate: u32,
    pub max_tokens: u32,
    pub prompt_role: u32,
}

// ---- Handles -------------------------------------------------------------

const RUNTIME_MAGIC: u64 = 0x7475_7262_6f72_7431; // "turbort1"
const TOKENIZER_MAGIC: u64 = 0x7475_7262_6f74_6b31; // "turbotk1"

struct Runtime {
    // Kept for the stages that log; nothing in this cut does.
    _log: turbo_log_fn,
    _log_user_data: usize,
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
fn write_str(dst: &mut [c_char], s: &str) {
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
    c"0.1.0".as_ptr()
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
            let rt =
                turbo_runtime { magic: RUNTIME_MAGIC, inner: Arc::new(Runtime { _log: log, _log_user_data: user }) };
            *out = Box::into_raw(Box::new(rt));
            Ok(())
        })
    }
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
            Ok(())
        })
    }
}

/// The options as the tokenizer takes them; NULL is the defaults.
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
    e.add_special_tokens = match o.add_special_tokens {
        0 => false,
        1 => true,
        v => {
            return Err(Error::field(INVALID_ARGUMENT, 1, format!("add_special_tokens is {v}, not 0 or 1")));
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
    e.prompt = match o.prompt_role {
        TURBO_PROMPT_NONE => PromptRole::None,
        TURBO_PROMPT_QUERY => PromptRole::Query,
        TURBO_PROMPT_DOCUMENT => PromptRole::Document,
        v => {
            return Err(Error::new(INVALID_ENUM, format!("prompt_role: {v} is not a TURBO_PROMPT_* value")));
        }
    };
    Ok(e)
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
    out: *mut u32,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let tok = &tokenizer(t)?.tok;
            let out = out_ptr(out, "out")?;
            *out = tok.count(text(txt, "text")?, PromptRole::None) as u32;
            Ok(())
        })
    }
}
