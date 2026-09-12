//! C ABI matching `include/turboembed.h`.
//!
//! Vectors are allocated as `[dim_tag, f32...]` so [`turboembed_free`] can
//! recover the length without a side table.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;

use crate::catalog::Arch;
use crate::engine::Engine;
use crate::error::Error;

fn set_error(err: *mut *mut c_char, message: impl AsRef<str>) {
    if err.is_null() {
        return;
    }
    let cstr = CString::new(message.as_ref()).unwrap_or_else(|_| {
        CString::new("turboembed error (message contained NUL)").expect("static")
    });
    unsafe {
        *err = cstr.into_raw();
    }
}

fn cstr<'a>(ptr: *const c_char, name: &str) -> Result<&'a str, Error> {
    if ptr.is_null() {
        return Err(Error::invalid(format!("{name} is NULL")));
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|e| Error::invalid(format!("{name} is not UTF-8: {e}")))
}

fn alloc_vec(vector: Vec<f32>) -> *mut f32 {
    let dim = vector.len();
    let mut packed = Vec::with_capacity(1 + dim);
    packed.push(f32::from_bits(dim as u32));
    packed.extend(vector);
    let mut boxed = packed.into_boxed_slice();
    let base = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    unsafe { base.add(1) }
}

/// Create a session for `arch` ("nvidia" / "intel" / "apple").
///
/// # Safety
/// `arch` must be a valid C string. `catalog_path` may be null. `err` may be null.
#[no_mangle]
pub unsafe extern "C" fn turboembed_create(
    arch: *const c_char,
    catalog_path: *const c_char,
    err: *mut *mut c_char,
) -> *mut Engine {
    let result = (|| -> Result<Engine, Error> {
        let arch = Arch::parse(cstr(arch, "arch")?)?;
        if catalog_path.is_null() {
            Engine::open(arch)
        } else {
            Engine::open_file(arch, cstr(catalog_path, "catalog_path")?)
        }
    })();
    match result {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(e) => {
            set_error(err, e.to_string());
            ptr::null_mut()
        }
    }
}

/// # Safety
/// `engine` must be a pointer from [`turboembed_create`] or null.
#[no_mangle]
pub unsafe extern "C" fn turboembed_destroy(engine: *mut Engine) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}

/// `embed(alias, text)` — catalog alias + UTF-8 text → one FP32 vector.
///
/// # Safety
/// `engine` must be a live [`turboembed_create`] pointer. `alias` and `text`
/// must be valid C strings. `out` / `out_dim` must be writable. Caller frees
/// `*out` with [`turboembed_free`].
#[no_mangle]
pub unsafe extern "C" fn turboembed_embed(
    engine: *mut Engine,
    alias: *const c_char,
    text: *const c_char,
    out: *mut *mut f32,
    out_dim: *mut usize,
    err: *mut *mut c_char,
) -> c_int {
    if engine.is_null() {
        set_error(err, "engine is NULL");
        return 1;
    }
    let result = (|| -> Result<Vec<f32>, Error> {
        let alias = cstr(alias, "alias")?;
        let text = cstr(text, "text")?;
        (*engine).embed(alias, text)
    })();
    match result {
        Ok(vector) => {
            if !out_dim.is_null() {
                *out_dim = vector.len();
            }
            if !out.is_null() {
                *out = alloc_vec(vector);
            }
            0
        }
        Err(e) => {
            set_error(err, e.to_string());
            1
        }
    }
}

/// # Safety
/// `engine` must be a live pointer or null.
#[no_mangle]
pub unsafe extern "C" fn turboembed_device(engine: *const Engine) -> *const c_char {
    if engine.is_null() {
        return ptr::null();
    }
    b"CUDA\0".as_ptr().cast()
}

/// # Safety
/// `ptr` must be null or `*out` from a successful [`turboembed_embed`].
#[no_mangle]
pub unsafe extern "C" fn turboembed_free(ptr: *mut std::ffi::c_void) {
    if ptr.is_null() {
        return;
    }
    let payload = ptr.cast::<f32>();
    let base = payload.sub(1);
    let dim = (*base).to_bits() as usize;
    let len = 1 + dim;
    drop(Vec::from_raw_parts(base, len, len));
}

/// # Safety
/// `ptr` must be null or a string written to `*err`.
#[no_mangle]
pub unsafe extern "C" fn turboembed_free_str(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}
