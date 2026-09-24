//! The core's side of include/turbo/turbo_backend.h: the backends linked
//! into this build, and the devices they list.

use std::ffi::{CStr, c_char, c_void};

use crate::status::{Error, INTERNAL, Result, UNSUPPORTED};
use crate::{turbo_buffer_desc, turbo_device_info, turbo_error, turbo_log_fn, turbo_native_handle, write_str};

pub const TURBO_CAP_UNSUPPORTED: u32 = 0;
pub const TURBO_CAP_EXPERIMENTAL: u32 = 1;
pub const TURBO_CAP_SUPPORTED: u32 = 2;

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
}

// The table is immutable static data, read from any thread.
unsafe impl Sync for turbo_backend {}

impl turbo_backend {
    pub fn name(&self) -> &str {
        unsafe { CStr::from_ptr(self.name) }.to_str().unwrap_or("")
    }
}

/// The backends linked into this build, in the order their devices are
/// numbered.
pub fn linked() -> &'static [&'static turbo_backend] {
    LINKED
}

static LINKED: &[&turbo_backend] = &[
    #[cfg(feature = "cpu")]
    &crate::cpu::BACKEND,
];

/// The table's size is one this core knows, and what it offers comes with
/// its release. A table built against another header is a build fault,
/// not a missing driver.
pub fn check_table(backend: &turbo_backend) -> Result<()> {
    let want = size_of::<turbo_backend>();
    if backend.struct_size as usize != want {
        return Err(Error::new(
            INTERNAL,
            format!("{} backend: its table is {} bytes, this core knows {want}", backend.name(), backend.struct_size),
        ));
    }
    let b = backend;
    let pairs = [
        ("context_create", b.context_create.is_some(), "context_release", b.context_release.is_some()),
        ("buffer_alloc", b.buffer_alloc.is_some(), "buffer_release", b.buffer_release.is_some()),
        ("buffer_import", b.buffer_import.is_some(), "buffer_release", b.buffer_release.is_some()),
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
        $crate::backend::covered($b, ::std::mem::offset_of!($crate::backend::turbo_backend, $f), $b.$f, stringify!($f))
    };
}
pub(crate) use offered;

pub(crate) fn covered<F>(backend: &turbo_backend, offset: usize, f: Option<F>, what: &str) -> Result<F> {
    let covers = backend.struct_size as usize >= offset + size_of::<Option<F>>();
    covers
        .then_some(f)
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

/// A C string in a fixed buffer, read only up to the buffer's end whether
/// or not it holds a NUL.
pub(crate) fn cstr(b: &[c_char]) -> String {
    let bytes: Vec<u8> = b.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
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
