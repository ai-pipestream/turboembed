//! The core's side of include/turbo/turbo_backend.h: the backends linked
//! into this build, and the devices they list.

use std::ffi::{CStr, c_char, c_void};

use crate::status::{Error, INTERNAL, Result, UNSUPPORTED};
use crate::{
    TURBO_STAGE_MAX, turbo_buffer_desc, turbo_device_info, turbo_error, turbo_log_fn, turbo_native_handle, write_str,
};

pub const TURBO_CAP_UNSUPPORTED: u32 = 0;
pub const TURBO_CAP_EXPERIMENTAL: u32 = 1;
pub const TURBO_CAP_SUPPORTED: u32 = 2;

pub const TURBO_BERT_EMBEDDING_TENSORS: u32 = 5;
pub const TURBO_BERT_LAYER_TENSORS: u32 = 16;

pub const TURBO_FAMILY_BERT: u32 = 1;

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
    pub reserved: u32,
    pub tensors: *const turbo_backend_tensor,
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
    pub reserved: u32,
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
}

// The table is immutable static data, read from any thread.
unsafe impl Sync for turbo_backend {}

impl turbo_backend {
    pub fn name(&self) -> &str {
        unsafe { CStr::from_ptr(self.name) }.to_str().unwrap_or("")
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
    #[cfg(feature = "cpu")]
    &crate::cpu::BACKEND,
];

/// The sizes the table has had, one per group of functions appended to it.
const TABLE_SIZES: [usize; 5] = [
    std::mem::offset_of!(turbo_backend, context_create),
    std::mem::offset_of!(turbo_backend, model_load),
    std::mem::offset_of!(turbo_backend, session_create),
    std::mem::offset_of!(turbo_backend, buffer_read),
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

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;

    /// The CPU backend's table as a backend built before `field` was
    /// appended would have it.
    fn cpu_table(struct_size: usize) -> turbo_backend {
        let mut t = unsafe { std::ptr::read(&crate::cpu::BACKEND) };
        t.struct_size = struct_size as u32;
        t
    }

    #[test]
    fn a_table_from_before_the_model_functions_is_known_and_offers_none() {
        let t = cpu_table(std::mem::offset_of!(turbo_backend, model_load));
        check_table(&t).unwrap();
        assert!(offered!(&t, buffer_alloc).is_ok());
        let e = offered!(&t, model_load).err().unwrap();
        assert_eq!(e, Error::new(UNSUPPORTED, "the cpu backend does not offer model_load"));
        let t = cpu_table(std::mem::offset_of!(turbo_backend, context_create));
        check_table(&t).unwrap();
        assert_eq!(offered!(&t, context_create).err().unwrap().code, UNSUPPORTED);
    }

    #[test]
    fn a_table_from_before_the_session_functions_is_known_and_offers_none() {
        let t = cpu_table(std::mem::offset_of!(turbo_backend, session_create));
        check_table(&t).unwrap();
        assert!(offered!(&t, model_load).is_ok());
        let e = offered!(&t, session_create).err().unwrap();
        assert_eq!(e, Error::new(UNSUPPORTED, "the cpu backend does not offer session_create"));
        assert_eq!(offered!(&t, session_run).err().unwrap().code, UNSUPPORTED);
    }

    #[test]
    fn a_table_from_before_buffer_read_is_known_and_offers_none() {
        let t = cpu_table(std::mem::offset_of!(turbo_backend, buffer_read));
        check_table(&t).unwrap();
        assert!(offered!(&t, session_run).is_ok());
        let e = offered!(&t, buffer_read).err().unwrap();
        assert_eq!(e, Error::new(UNSUPPORTED, "the cpu backend does not offer buffer_read"));
    }

    #[test]
    fn a_session_without_its_release_or_its_run_is_refused() {
        let mut t = cpu_table(size_of::<turbo_backend>());
        t.session_release = None;
        let e = check_table(&t).unwrap_err();
        assert_eq!(e, Error::new(INTERNAL, "cpu backend: its table has session_create and no session_release"));
        let mut t = cpu_table(size_of::<turbo_backend>());
        t.session_run = None;
        let e = check_table(&t).unwrap_err();
        assert_eq!(e, Error::new(INTERNAL, "cpu backend: its table has session_create and no session_run"));
        let t = cpu_table(std::mem::offset_of!(turbo_backend, session_release));
        assert_eq!(check_table(&t).unwrap_err().code, INTERNAL, "a size inside the session group is unknown");
    }

    #[test]
    fn a_table_of_an_unknown_size_or_a_load_without_its_release_is_refused() {
        let t = cpu_table(std::mem::offset_of!(turbo_backend, model_release));
        assert_eq!(check_table(&t).unwrap_err().code, INTERNAL);
        let mut t = cpu_table(size_of::<turbo_backend>());
        t.model_release = None;
        let e = check_table(&t).unwrap_err();
        assert_eq!(e, Error::new(INTERNAL, "cpu backend: its table has model_load and no model_release"));
    }
}
