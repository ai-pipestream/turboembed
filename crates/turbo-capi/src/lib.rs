//! C ABI exports for Turbo (`libturbo`).
//!
//! Every function here is a thin, panic-safe adapter over `turbo-core`:
//! validate the handle and descriptor sizes, convert views, call the core,
//! and copy any error into the caller-owned `turbo_error`. No function
//! allocates on the error path beyond the core's own message string. Handles
//! are `Arc`s leaked to the caller and reclaimed by the matching release.
//!
//! The header `include/turbo/turbo.h` is generated from this crate and
//! `turbo-abi` by cbindgen; see `cbindgen.toml` and `scripts/gen-header.sh`.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]

use std::ffi::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::Arc;

use turbo_abi::*;
use turbo_core::abi_convert::{read_sized, write_sized};
use turbo_core::buffer::{BufferDesc, NativeHandle};
use turbo_core::chunker::{chunk_source, ChunkError, ChunkPlan, ChunkerConfig};
use turbo_core::handles::{Buffer, Context, Generation, Model, ResultHandle, Session};
use turbo_core::provider::{
    ClassifyOptions, ContextDesc, EmbedOptions, GenerateDesc, Message, ModelDesc, Options, RerankOptions, RunOptions,
    SessionDesc, TokenBatch,
};
use turbo_core::runtime::{DeviceSelector, LogLevel, Runtime, RuntimeDesc};
use turbo_core::tokenizer::{EncodeOptions, EncodeTarget, Tokenizer};
use turbo_core::types::{
    Aggregation, DeviceKind, HandleKind, Modality, Normalize, OutputDType, Pooling, PromptRole, SelectPolicy,
    StructuredKind, Task, Truncate,
};
use turbo_core::{Error, Result};

// ---------------------------------------------------------------------------
// Error and boundary helpers
// ---------------------------------------------------------------------------

/// Copy an error into the caller's record (if any) and return its code.
fn fail(err: *mut turbo_error, e: &Error) -> i32 {
    if !err.is_null() {
        // SAFETY: the caller passed a pointer to a writable turbo_error, or NULL.
        let out = unsafe { &mut *err };
        // Only fields below the caller's struct_size are written.
        let size = out.struct_size as usize;
        if size >= 8 {
            out.code = e.code();
        }
        if size >= 12 {
            out.field = e.field();
        }
        if size > 12 {
            let cap = (size - 12).min(TURBO_ERROR_MESSAGE_LEN);
            let bytes = e.message().as_bytes();
            let mut n = bytes.len().min(cap - 1);
            // Do not cut a UTF-8 sequence in half.
            while n > 0 && n < bytes.len() && (bytes[n] & 0xC0) == 0x80 {
                n -= 1;
            }
            for (i, b) in bytes[..n].iter().enumerate() {
                out.message[i] = *b as c_char;
            }
            out.message[n] = 0;
        }
    }
    e.code()
}

/// Reset an error record to success.
fn clear(err: *mut turbo_error) {
    if !err.is_null() {
        // SAFETY: as in `fail`.
        let out = unsafe { &mut *err };
        let size = out.struct_size as usize;
        if size >= 8 {
            out.code = TURBO_OK;
        }
        if size >= 12 {
            out.field = 0;
        }
        if size > 12 {
            out.message[0] = 0;
        }
    }
}

/// Run `f`, converting panics and errors into status codes.
fn boundary(err: *mut turbo_error, f: impl FnOnce() -> Result<()>) -> i32 {
    if !err.is_null() {
        // SAFETY: as in `fail`.
        let size = unsafe { (*err).struct_size };
        if size < 8 {
            // Cannot even write a code; report through the return value only.
            return TURBO_E_INVALID_STRUCT_SIZE;
        }
    }
    clear(err);
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => TURBO_OK,
        Ok(Err(e)) => fail(err, &e),
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic with a non-string payload".to_string());
            fail(err, &Error::panic(format!("caught at the C boundary: {msg}")))
        }
    }
}

/// Check that a descriptor's `struct_size` is a layout this library
/// understands: the current size, or the end of an earlier field (an older
/// caller's layout; see [`turbo_abi::Versioned`]). Any other value is
/// `TURBO_E_INVALID_STRUCT_SIZE`, since a prefix ending inside a field could
/// hand the library half a pointer or half a count.
fn check_size<T: turbo_abi::Versioned>(what: &str, got: u32) -> Result<()> {
    if T::size_is_known(got) {
        return Ok(());
    }
    Err(Error::invalid_struct_size(what, got, core::mem::size_of::<T>()))
}

/// Borrow a `turbo_text` as `&str`, validating UTF-8.
unsafe fn text<'a>(t: &turbo_text, what: &str) -> Result<&'a str> {
    if t.len == 0 {
        return Ok("");
    }
    if t.ptr.is_null() {
        return Err(Error::invalid_argument(format!("{what}.ptr is NULL with len {}", t.len)));
    }
    let len = usize::try_from(t.len).map_err(|_| Error::invalid_argument(format!("{what}.len exceeds usize")))?;
    // SAFETY: caller promises ptr..ptr+len is readable for the call.
    let bytes = unsafe { std::slice::from_raw_parts(t.ptr.cast::<u8>(), len) };
    std::str::from_utf8(bytes).map_err(|e| Error::invalid_utf8(&format!("{what} (byte {})", e.valid_up_to())))
}

/// Borrow an array of `turbo_text`.
unsafe fn texts<'a>(ptr: *const turbo_text, n: u32, what: &str) -> Result<Vec<&'a str>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(Error::invalid_argument(format!("{what} is NULL with count {n}")));
    }
    // SAFETY: caller promises `n` readable entries.
    let raw = unsafe { std::slice::from_raw_parts(ptr, n as usize) };
    raw.iter().enumerate().map(|(i, t)| unsafe { text(t, &format!("{what}[{i}]")) }).collect()
}

/// Borrow key/value options.
unsafe fn kvs(ptr: *const turbo_kv, n: u32, what: &str) -> Result<Options> {
    if n == 0 {
        return Ok(Options::default());
    }
    if ptr.is_null() {
        return Err(Error::invalid_argument(format!("{what} is NULL with count {n}")));
    }
    let raw = unsafe { std::slice::from_raw_parts(ptr, n as usize) };
    let mut out = Vec::with_capacity(raw.len());
    for (i, kv) in raw.iter().enumerate() {
        let k = unsafe { text(&kv.key, &format!("{what}[{i}].key")) }?;
        let v = unsafe { text(&kv.value, &format!("{what}[{i}].value")) }?;
        if k.is_empty() {
            return Err(Error::invalid_argument(format!("{what}[{i}].key is empty")));
        }
        out.push((k.to_string(), v.to_string()));
    }
    Ok(Options(out))
}

/// Copy a string into a fixed `c_char` array, NUL-terminated, cut on a UTF-8 boundary.
fn put_str(dst: &mut [c_char], s: &str) {
    if dst.is_empty() {
        return;
    }
    let bytes = s.as_bytes();
    let mut n = bytes.len().min(dst.len() - 1);
    while n > 0 && n < bytes.len() && (bytes[n] & 0xC0) == 0x80 {
        n -= 1;
    }
    for (i, b) in bytes[..n].iter().enumerate() {
        dst[i] = *b as c_char;
    }
    dst[n] = 0;
}

/// Borrow a handle.
unsafe fn handle<'a, T>(ptr: *const T, what: &str) -> Result<&'a T> {
    if ptr.is_null() {
        return Err(Error::invalid_handle(what));
    }
    // SAFETY: non-null handles were produced by this library and not released.
    Ok(unsafe { &*ptr })
}

/// Clone the `Arc` behind a handle.
unsafe fn arc<T>(ptr: *const T, what: &str) -> Result<Arc<T>> {
    if ptr.is_null() {
        return Err(Error::invalid_handle(what));
    }
    // SAFETY: the pointer came from Arc::into_raw in this library.
    unsafe { Arc::increment_strong_count(ptr) };
    Ok(unsafe { Arc::from_raw(ptr) })
}

/// Hand an `Arc` to the caller.
fn leak<T>(a: Arc<T>) -> *mut T {
    Arc::into_raw(a) as *mut T
}

/// Reclaim an `Arc` from the caller. NULL is a no-op.
unsafe fn reclaim<T>(ptr: *mut T) {
    if !ptr.is_null() {
        // SAFETY: the pointer came from `leak` and is released exactly once.
        drop(unsafe { Arc::from_raw(ptr as *const T) });
    }
}

unsafe fn out_ptr<'a, T>(out: *mut *mut T, what: &str) -> Result<&'a mut *mut T> {
    if out.is_null() {
        return Err(Error::invalid_argument(format!("{what} out-pointer is NULL")));
    }
    let r = unsafe { &mut *out };
    *r = std::ptr::null_mut();
    Ok(r)
}

// ---------------------------------------------------------------------------
// Version and names
// ---------------------------------------------------------------------------

/// ABI version compiled into this library.
#[no_mangle]
pub extern "C" fn turbo_abi_version() -> u32 {
    TURBO_ABI_VERSION
}

/// Library version as a static C string (`major.minor.patch[-pre]`).
#[no_mangle]
pub extern "C" fn turbo_version() -> *const c_char {
    static V: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    V.as_ptr().cast()
}

/// Symbolic name of a status code. Static, never NULL.
#[no_mangle]
pub extern "C" fn turbo_status_name(code: i32) -> *const c_char {
    macro_rules! names {
        ($($k:ident),*) => { match code { $( $k => concat!(stringify!($k), "\0"), )* _ => "TURBO_E_UNKNOWN\0" } };
    }
    let s: &'static str = names!(
        TURBO_OK,
        TURBO_E_INVALID_ARGUMENT,
        TURBO_E_INVALID_STRUCT_SIZE,
        TURBO_E_INVALID_UTF8,
        TURBO_E_INVALID_HANDLE,
        TURBO_E_INVALID_SHAPE,
        TURBO_E_INVALID_STATE,
        TURBO_E_INVALID_ENUM,
        TURBO_E_UNSUPPORTED,
        TURBO_E_UNSUPPORTED_OPTION,
        TURBO_E_UNSUPPORTED_TASK,
        TURBO_E_UNSUPPORTED_DTYPE,
        TURBO_E_UNSUPPORTED_PLACEMENT,
        TURBO_E_NOT_IMPLEMENTED,
        TURBO_E_UNSUPPORTED_MODALITY,
        TURBO_E_OUT_OF_MEMORY,
        TURBO_E_BUSY,
        TURBO_E_OVERLOADED,
        TURBO_E_CAPACITY,
        TURBO_E_DEVICE_NOT_FOUND,
        TURBO_E_DEVICE_UNAVAILABLE,
        TURBO_E_RUNTIME,
        TURBO_E_PROVIDER_LOAD,
        TURBO_E_ABI_MISMATCH,
        TURBO_E_CANCELLED,
        TURBO_E_BUNDLE_NOT_FOUND,
        TURBO_E_BUNDLE_INVALID,
        TURBO_E_BUNDLE_INTEGRITY,
        TURBO_E_BUNDLE_NO_ARTIFACT,
        TURBO_E_INTERNAL,
        TURBO_E_PANIC
    );
    s.as_ptr().cast()
}

// ---------------------------------------------------------------------------
// Runtime and devices
// ---------------------------------------------------------------------------

struct CLogSink {
    f: unsafe extern "C" fn(*mut c_void, u32, turbo_text),
    user: usize,
}
// SAFETY: the caller's log callback must be callable from any thread; documented in the header.
unsafe impl Send for CLogSink {}
unsafe impl Sync for CLogSink {}

/// Create a runtime. `desc` may be NULL for defaults.
#[no_mangle]
pub unsafe extern "C" fn turbo_runtime_create(
    desc: *const turbo_runtime_desc,
    out: *mut *mut turbo_runtime,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let out = unsafe { out_ptr(out, "turbo_runtime_create") }?;
        let mut rd = RuntimeDesc::default();
        if !desc.is_null() {
            let d = unsafe { read_sized::<turbo_runtime_desc>(desc, "turbo_runtime_desc") }?;
            rd.no_default_providers = d.flags & TURBO_RUNTIME_NO_DEFAULT_PROVIDERS != 0;
            if d.flags & !TURBO_RUNTIME_NO_DEFAULT_PROVIDERS != 0 {
                return Err(Error::invalid_argument(format!(
                    "turbo_runtime_desc.flags has unknown bits {:#x}",
                    d.flags
                ))
                .with_field(2));
            }
            if d.reserved != 0 {
                return Err(Error::invalid_argument("turbo_runtime_desc.reserved must be 0").with_field(4));
            }
            rd.provider_paths =
                unsafe { texts(d.provider_paths, d.n_provider_paths, "turbo_runtime_desc.provider_paths") }?
                    .into_iter()
                    .map(str::to_string)
                    .collect();
            if let Some(f) = d.log {
                let sink = CLogSink { f, user: d.log_user_data as usize };
                rd.log = Some(Arc::new(move |level: LogLevel, msg: &str| {
                    let t = turbo_text { ptr: msg.as_ptr().cast(), len: msg.len() as u64 };
                    // SAFETY: the callback was supplied by the caller for this purpose.
                    unsafe { (sink.f)(sink.user as *mut c_void, level as u32, t) };
                }));
            }
        }
        let rt = turbo::create_runtime(rd)?;
        *out = leak(rt) as *mut turbo_runtime;
        Ok(())
    })
}

/// Release a runtime. NULL is a no-op. Contexts keep it alive.
#[no_mangle]
pub unsafe extern "C" fn turbo_runtime_release(rt: *mut turbo_runtime) {
    unsafe { reclaim(rt as *mut Runtime) };
}

/// Load a provider library (`turbo_provider.h`) and register its devices.
/// The library stays loaded for the runtime's lifetime. A provider whose id
/// is already registered is rejected with `TURBO_E_PROVIDER_LOAD`.
#[no_mangle]
pub unsafe extern "C" fn turbo_runtime_load_provider(
    rt: *mut turbo_runtime,
    path: turbo_text,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let rt = unsafe { handle(rt as *const Runtime, "turbo_runtime") }?;
        let p = unsafe { text(&path, "path") }?;
        if p.is_empty() {
            return Err(Error::invalid_argument("path is empty"));
        }
        rt.load_provider(Path::new(p))
    })
}

/// Number of enumerated devices.
#[no_mangle]
pub unsafe extern "C" fn turbo_runtime_device_count(
    rt: *mut turbo_runtime,
    out: *mut u32,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let rt = unsafe { handle(rt as *const Runtime, "turbo_runtime") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        unsafe { *out = rt.device_count() };
        Ok(())
    })
}

/// Static info for device `index`.
#[no_mangle]
pub unsafe extern "C" fn turbo_runtime_device_info(
    rt: *mut turbo_runtime,
    index: u32,
    out: *mut turbo_device_info,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let rt = unsafe { handle(rt as *const Runtime, "turbo_runtime") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        let o = unsafe { &mut *out };
        check_size::<turbo_device_info>("turbo_device_info", o.struct_size)?;
        let d = &rt.device(index)?.info;
        let mut full = turbo_device_info {
            struct_size: o.struct_size,
            kind: d.kind.as_abi(),
            ordinal: d.ordinal,
            vendor_id: d.vendor_id,
            caps: d.caps,
            memory_total: d.memory_total,
            memory_free: d.memory_free,
            name: [0; 128],
            vendor: [0; 64],
            provider_id: [0; 32],
            provider_version: [0; 32],
            runtime_version: [0; 64],
            driver_version: [0; 64],
        };
        put_str(&mut full.name, &d.name);
        put_str(&mut full.vendor, &d.vendor);
        put_str(&mut full.provider_id, &d.provider_id);
        put_str(&mut full.provider_version, &d.provider_version);
        put_str(&mut full.runtime_version, &d.runtime_version);
        put_str(&mut full.driver_version, &d.driver_version);
        // SAFETY: copy only the caller's declared size.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&full as *const turbo_device_info).cast::<u8>(),
                out.cast::<u8>(),
                o.struct_size as usize,
            )
        };
        Ok(())
    })
}

/// Select a device by policy. `sel` may be NULL for AUTO.
#[no_mangle]
pub unsafe extern "C" fn turbo_runtime_select_device(
    rt: *mut turbo_runtime,
    sel: *const turbo_device_selector,
    out: *mut u32,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let rt = unsafe { handle(rt as *const Runtime, "turbo_runtime") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        let mut ds = DeviceSelector::default();
        if !sel.is_null() {
            let s = unsafe { read_sized::<turbo_device_selector>(sel, "turbo_device_selector") }?;
            ds.policy = SelectPolicy::from_abi(s.policy).map_err(|e| e.with_field(2))?;
            if s.kind_mask != 0 {
                for kind in DeviceKind::ALL {
                    if s.kind_mask & (1u32 << kind.as_abi()) != 0 {
                        ds.kinds.push(*kind);
                    }
                }
                let known: u32 = DeviceKind::ALL.iter().map(|k| 1u32 << k.as_abi()).sum();
                if s.kind_mask & !known != 0 {
                    return Err(Error::invalid_argument(format!(
                        "turbo_device_selector.kind_mask has unknown bits {:#x}",
                        s.kind_mask & !known
                    ))
                    .with_field(3));
                }
            }
            ds.ordinal = s.ordinal;
            ds.provider_id = unsafe { text(&s.provider_id, "turbo_device_selector.provider_id") }?.to_string();
            ds.vendor = unsafe { text(&s.vendor, "turbo_device_selector.vendor") }?.to_string();
        }
        unsafe { *out = rt.select(&ds)? };
        Ok(())
    })
}

/// Capability cell for (device, task, modality).
#[no_mangle]
pub unsafe extern "C" fn turbo_runtime_capability(
    rt: *mut turbo_runtime,
    index: u32,
    task: u32,
    modality: u32,
    out: *mut turbo_capability,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let rt = unsafe { handle(rt as *const Runtime, "turbo_runtime") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        let o = unsafe { &mut *out };
        check_size::<turbo_capability>("turbo_capability", o.struct_size)?;
        let task = Task::from_abi(task)?;
        let modality = Modality::from_abi(modality)?;
        let c = rt.capability(index, task, modality)?;
        let mut full = turbo_capability {
            struct_size: o.struct_size,
            status: c.status.as_abi(),
            dtype: c.dtype.map(|d| d.as_abi()).unwrap_or(0),
            reference_dtype: c.reference_dtype.map(|d| d.as_abi()).unwrap_or(0),
            cosine_floor: c.cosine_floor,
            max_abs_error: c.max_abs_error,
            deterministic: c.deterministic as u32,
            reserved: 0,
            notes: [0; 128],
        };
        put_str(&mut full.notes, &c.notes);
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&full as *const turbo_capability).cast::<u8>(),
                out.cast::<u8>(),
                o.struct_size as usize,
            )
        };
        Ok(())
    })
}

/// Per-request feasibility: can `bundle_path` run `task`/`modality` on device `index`?
#[no_mangle]
pub unsafe extern "C" fn turbo_can_run(
    rt: *mut turbo_runtime,
    index: u32,
    bundle_path: turbo_text,
    task: u32,
    modality: u32,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let rt = unsafe { handle(rt as *const Runtime, "turbo_runtime") }?;
        let path = unsafe { text(&bundle_path, "bundle_path") }?;
        if path.is_empty() {
            return Err(Error::invalid_argument("bundle_path is empty"));
        }
        let task = Task::from_abi(task)?;
        let modality = Modality::from_abi(modality)?;
        rt.can_run(index, Path::new(path), task, modality)
    })
}

// ---------------------------------------------------------------------------
// Contexts and buffers
// ---------------------------------------------------------------------------

/// Create a context on device `index`. `desc` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn turbo_context_create(
    rt: *mut turbo_runtime,
    index: u32,
    desc: *const turbo_context_desc,
    out: *mut *mut turbo_context,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let out = unsafe { out_ptr(out, "turbo_context_create") }?;
        let rt = unsafe { arc(rt as *const Runtime, "turbo_runtime") }?;
        let mut cd = ContextDesc::default();
        if !desc.is_null() {
            let d = unsafe { read_sized::<turbo_context_desc>(desc, "turbo_context_desc") }?;
            if d.flags != 0 {
                return Err(Error::invalid_argument("turbo_context_desc.flags must be 0").with_field(2));
            }
            if !d.next.is_null() {
                return Err(Error::not_implemented("turbo_context_desc.next (external queue import)"));
            }
            cd.options = unsafe { kvs(d.options, d.n_options, "turbo_context_desc.options") }?;
        }
        let ctx = Context::create(rt, index, &cd)?;
        *out = leak(ctx) as *mut turbo_context;
        Ok(())
    })
}

/// Release a context. Models and buffers keep it alive.
#[no_mangle]
pub unsafe extern "C" fn turbo_context_release(ctx: *mut turbo_context) {
    unsafe { reclaim(ctx as *mut Context) };
}

/// Device index of a context.
#[no_mangle]
pub unsafe extern "C" fn turbo_context_device(ctx: *mut turbo_context, out: *mut u32, err: *mut turbo_error) -> i32 {
    boundary(err, || {
        let c = unsafe { handle(ctx as *const Context, "turbo_context") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        unsafe { *out = c.device_index() };
        Ok(())
    })
}

unsafe fn buffer_desc(desc: *const turbo_buffer_desc) -> Result<(BufferDesc, Option<NativeHandle>)> {
    if desc.is_null() {
        return Err(Error::invalid_argument("turbo_buffer_desc is NULL"));
    }
    let d = unsafe { read_sized::<turbo_buffer_desc>(desc, "turbo_buffer_desc") }?;
    let bd = BufferDesc::from_abi(&d)?;
    let native = if d.next.is_null() {
        None
    } else {
        let h =
            unsafe { read_sized::<turbo_native_handle>(d.next as *const turbo_native_handle, "turbo_native_handle") }?;
        Some(NativeHandle { kind: HandleKind::from_abi(h.kind)?, handle: h.handle, aux: h.aux, offset: h.offset })
    };
    Ok((bd, native))
}

/// Allocate a buffer in the context's memory.
#[no_mangle]
pub unsafe extern "C" fn turbo_buffer_alloc(
    ctx: *mut turbo_context,
    desc: *const turbo_buffer_desc,
    out: *mut *mut turbo_buffer,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let out = unsafe { out_ptr(out, "turbo_buffer_alloc") }?;
        let c = unsafe { arc(ctx as *const Context, "turbo_context") }?;
        let (bd, native) = unsafe { buffer_desc(desc) }?;
        if native.is_some() {
            return Err(Error::invalid_argument(
                "turbo_buffer_alloc does not take a native handle; use turbo_buffer_import",
            ));
        }
        let b = c.alloc(&bd)?;
        *out = leak(b) as *mut turbo_buffer;
        Ok(())
    })
}

/// Wrap caller memory described by a `turbo_native_handle` in `desc.next`.
#[no_mangle]
pub unsafe extern "C" fn turbo_buffer_import(
    ctx: *mut turbo_context,
    desc: *const turbo_buffer_desc,
    out: *mut *mut turbo_buffer,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let out = unsafe { out_ptr(out, "turbo_buffer_import") }?;
        let c = unsafe { arc(ctx as *const Context, "turbo_context") }?;
        let (bd, native) = unsafe { buffer_desc(desc) }?;
        let native = native.ok_or_else(|| {
            Error::invalid_argument("turbo_buffer_import requires a turbo_native_handle in turbo_buffer_desc.next")
        })?;
        let b = c.import(&bd, &native)?;
        *out = leak(b) as *mut turbo_buffer;
        Ok(())
    })
}

/// Release a buffer.
#[no_mangle]
pub unsafe extern "C" fn turbo_buffer_release(buf: *mut turbo_buffer) {
    unsafe { reclaim(buf as *mut Buffer) };
}

/// Description of a buffer.
#[no_mangle]
pub unsafe extern "C" fn turbo_buffer_get_desc(
    buf: *mut turbo_buffer,
    out: *mut turbo_buffer_desc,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let b = unsafe { handle(buf as *const Buffer, "turbo_buffer") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        let o = unsafe { &mut *out };
        check_size::<turbo_buffer_desc>("turbo_buffer_desc", o.struct_size)?;
        let d = b.desc();
        let mut full = turbo_buffer_desc {
            struct_size: o.struct_size,
            placement: d.placement.as_abi(),
            dtype: d.dtype.as_abi(),
            ndim: d.shape.len() as u32,
            shape: [0; TURBO_MAX_RANK],
            strides: [0; TURBO_MAX_RANK],
            bytes: d.bytes,
            next: std::ptr::null(),
        };
        full.shape[..d.shape.len()].copy_from_slice(&d.shape);
        full.strides[..d.strides.len()].copy_from_slice(&d.strides);
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&full as *const turbo_buffer_desc).cast::<u8>(),
                out.cast::<u8>(),
                o.struct_size as usize,
            )
        };
        Ok(())
    })
}

/// Host pointer for host-visible placements, else `TURBO_E_UNSUPPORTED_PLACEMENT`.
#[no_mangle]
pub unsafe extern "C" fn turbo_buffer_host_ptr(
    buf: *mut turbo_buffer,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let b = unsafe { handle(buf as *const Buffer, "turbo_buffer") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        match b.host_ptr() {
            Some(p) => {
                unsafe { *out = p.as_ptr().cast() };
                Ok(())
            }
            None => Err(Error::unsupported_placement(format!(
                "buffer placement {:?} is not host-visible; use turbo_buffer_export or a result read",
                b.desc().placement
            ))),
        }
    })
}

/// Export a native handle of `kind`.
#[no_mangle]
pub unsafe extern "C" fn turbo_buffer_export(
    buf: *mut turbo_buffer,
    kind: u32,
    out: *mut turbo_native_handle,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let b = unsafe { handle(buf as *const Buffer, "turbo_buffer") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        // SAFETY: `out` is non-null; only its leading `struct_size` field is read here.
        let size = unsafe { out.cast::<u32>().read_unaligned() };
        check_size::<turbo_native_handle>("turbo_native_handle", size)?;
        let h = b.export(HandleKind::from_abi(kind)?)?;
        let full = turbo_native_handle {
            struct_size: size,
            kind: h.kind.as_abi(),
            handle: h.handle,
            aux: h.aux,
            offset: h.offset,
        };
        // SAFETY: `size` was validated and `out` is writable for that many bytes.
        unsafe { write_sized(&full, out, size) };
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/// Load a bundle directory on a context. `desc` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn turbo_model_load(
    ctx: *mut turbo_context,
    bundle_path: turbo_text,
    desc: *const turbo_model_desc,
    out: *mut *mut turbo_model,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let out = unsafe { out_ptr(out, "turbo_model_load") }?;
        let c = unsafe { arc(ctx as *const Context, "turbo_context") }?;
        let path = unsafe { text(&bundle_path, "bundle_path") }?;
        if path.is_empty() {
            return Err(Error::invalid_argument("bundle_path is empty"));
        }
        let mut md = ModelDesc::default();
        if !desc.is_null() {
            let d = unsafe { read_sized::<turbo_model_desc>(desc, "turbo_model_desc") }?;
            if !d.next.is_null() {
                return Err(Error::invalid_argument("turbo_model_desc.next must be NULL"));
            }
            md.options = unsafe { kvs(d.options, d.n_options, "turbo_model_desc.options") }?;
        }
        let m = c.load_model(Path::new(path), &md)?;
        *out = leak(m) as *mut turbo_model;
        Ok(())
    })
}

/// Release a model. Sessions and generations keep it alive.
#[no_mangle]
pub unsafe extern "C" fn turbo_model_release(m: *mut turbo_model) {
    unsafe { reclaim(m as *mut Model) };
}

/// What loaded.
#[no_mangle]
pub unsafe extern "C" fn turbo_model_get_info(
    m: *mut turbo_model,
    out: *mut turbo_model_info,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let m = unsafe { handle(m as *const Model, "turbo_model") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        let o = unsafe { &mut *out };
        check_size::<turbo_model_info>("turbo_model_info", o.struct_size)?;
        let i = m.info();
        let mut full = turbo_model_info {
            struct_size: o.struct_size,
            task: i.task.as_abi(),
            kind: i.kind.as_abi(),
            modality: i.modality.as_abi(),
            dim: i.dim,
            n_labels: i.labels.len() as u32,
            pooling: i.pooling.map(|p| p.as_abi()).unwrap_or(0),
            normalize: i.normalize.map(|n| n.as_abi()).unwrap_or(0),
            max_seq: i.max_seq,
            max_batch: i.max_batch,
            dtype_used: i.dtype_used.map(|d| d.as_abi()).unwrap_or(0),
            fully_accelerated: i.stages.fully_accelerated() as u32,
            stage_placement: i.stages.as_abi(),
            n_inputs: i.inputs.len() as u32,
            n_outputs: i.outputs.len() as u32,
            vocab_size: i.vocab_size,
            model_id: [0; 128],
            revision: [0; 64],
            tokenizer_sha256: [0; 72],
            provider_id: [0; 32],
            prefix_query: [0; 128],
            prefix_document: [0; 128],
        };
        put_str(&mut full.model_id, &i.model_id);
        put_str(&mut full.revision, &i.revision);
        put_str(&mut full.tokenizer_sha256, &i.tokenizer_sha256);
        put_str(&mut full.provider_id, &i.provider_id);
        put_str(&mut full.prefix_query, &i.prefix_query);
        put_str(&mut full.prefix_document, &i.prefix_document);
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&full as *const turbo_model_info).cast::<u8>(),
                out.cast::<u8>(),
                o.struct_size as usize,
            )
        };
        Ok(())
    })
}

/// Named tensor `index` in `direction` (RUN models).
#[no_mangle]
pub unsafe extern "C" fn turbo_model_io_info(
    m: *mut turbo_model,
    direction: u32,
    index: u32,
    out: *mut turbo_tensor_info,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let m = unsafe { handle(m as *const Model, "turbo_model") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        let o = unsafe { &mut *out };
        check_size::<turbo_tensor_info>("turbo_tensor_info", o.struct_size)?;
        let list = match direction {
            TURBO_IO_INPUT => &m.info().inputs,
            TURBO_IO_OUTPUT => &m.info().outputs,
            other => return Err(Error::invalid_enum("io direction", other)),
        };
        let t = list.get(index as usize).ok_or_else(|| {
            Error::invalid_argument(format!("tensor index {index} is out of range; the model has {}", list.len()))
        })?;
        if t.shape.len() > TURBO_MAX_RANK {
            return Err(Error::internal(format!(
                "tensor `{}` has rank {} > {}",
                t.name,
                t.shape.len(),
                TURBO_MAX_RANK
            )));
        }
        let mut full = turbo_tensor_info {
            struct_size: o.struct_size,
            dtype: t.dtype.as_abi(),
            ndim: t.shape.len() as u32,
            reserved: 0,
            shape: [0; TURBO_MAX_RANK],
            name: [0; 64],
        };
        full.shape[..t.shape.len()].copy_from_slice(&t.shape);
        put_str(&mut full.name, &t.name);
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&full as *const turbo_tensor_info).cast::<u8>(),
                out.cast::<u8>(),
                o.struct_size as usize,
            )
        };
        Ok(())
    })
}

/// Label `index` as a view valid for the model's lifetime.
#[no_mangle]
pub unsafe extern "C" fn turbo_model_label(
    m: *mut turbo_model,
    index: u32,
    out: *mut turbo_text,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let m = unsafe { handle(m as *const Model, "turbo_model") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        let l = m.label(index)?;
        unsafe { *out = turbo_text { ptr: l.as_ptr().cast(), len: l.len() as u64 } };
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

/// Create a session. `desc` may be NULL for model defaults.
#[no_mangle]
pub unsafe extern "C" fn turbo_session_create(
    m: *mut turbo_model,
    desc: *const turbo_session_desc,
    out: *mut *mut turbo_session,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let out = unsafe { out_ptr(out, "turbo_session_create") }?;
        let m = unsafe { arc(m as *const Model, "turbo_model") }?;
        let mut sd = SessionDesc::default();
        if !desc.is_null() {
            let d = unsafe { read_sized::<turbo_session_desc>(desc, "turbo_session_desc") }?;
            if !d.next.is_null() {
                return Err(Error::invalid_argument("turbo_session_desc.next must be NULL"));
            }
            sd.max_batch = d.max_batch;
            sd.max_seq = d.max_seq;
            sd.options = unsafe { kvs(d.options, d.n_options, "turbo_session_desc.options") }?;
        }
        let s = m.create_session(&sd)?;
        *out = leak(s) as *mut turbo_session;
        Ok(())
    })
}

/// Release a session. Results keep it alive.
#[no_mangle]
pub unsafe extern "C" fn turbo_session_release(s: *mut turbo_session) {
    unsafe { reclaim(s as *mut Session) };
}

unsafe fn embed_options(opts: *const turbo_embed_options) -> Result<EmbedOptions> {
    if opts.is_null() {
        return Ok(EmbedOptions::default());
    }
    let o = unsafe { read_sized::<turbo_embed_options>(opts, "turbo_embed_options") }?;
    Ok(EmbedOptions {
        truncate: Truncate::from_abi(o.truncate).map_err(|e| e.with_field(2))?,
        max_tokens: o.max_tokens,
        prompt_role: PromptRole::from_abi(o.prompt_role).map_err(|e| e.with_field(4))?,
        normalize: Normalize::from_abi(o.normalize).map_err(|e| e.with_field(5))?,
        pooling: Pooling::from_abi(o.pooling).map_err(|e| e.with_field(6))?,
        output_dim: o.output_dim,
        output_dtype: OutputDType::from_abi(o.output_dtype).map_err(|e| e.with_field(8))?,
    })
}

/// Tokenize `count` texts into the session's inputs. `opts` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn turbo_session_write_text(
    s: *mut turbo_session,
    texts_ptr: *const turbo_text,
    count: u32,
    opts: *const turbo_embed_options,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let s = unsafe { handle(s as *const Session, "turbo_session") }?;
        let list = unsafe { texts(texts_ptr, count, "texts") }?;
        let o = unsafe { embed_options(opts) }?;
        s.write_text(&list, &o)
    })
}

/// Copy caller-prepared tokens into the session's inputs.
#[no_mangle]
pub unsafe extern "C" fn turbo_session_write_tokens(
    s: *mut turbo_session,
    batch: *const turbo_token_batch,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let s = unsafe { handle(s as *const Session, "turbo_session") }?;
        if batch.is_null() {
            return Err(Error::invalid_argument("batch is NULL"));
        }
        let b = unsafe { read_sized::<turbo_token_batch>(batch, "turbo_token_batch") }?;
        let row_stride = if b.row_stride == 0 { b.seq } else { b.row_stride };
        let need = TokenBatch::required_len(b.batch, b.seq, row_stride)?;
        if b.ids.is_null() {
            return Err(Error::invalid_argument("turbo_token_batch.ids is NULL").with_field(5));
        }
        if b.mask.is_null() {
            return Err(Error::invalid_argument("turbo_token_batch.mask is NULL").with_field(6));
        }
        // SAFETY: the caller promises `need` readable elements in each array.
        let ids = unsafe { std::slice::from_raw_parts(b.ids, need) };
        let mask = unsafe { std::slice::from_raw_parts(b.mask, need) };
        let types = if b.types.is_null() { None } else { Some(unsafe { std::slice::from_raw_parts(b.types, need) }) };
        let tb = TokenBatch { batch: b.batch, seq: b.seq, row_stride, ids, mask, types };
        s.write_tokens(&tb)
    })
}

/// Write a query and `count` documents for reranking. `opts` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn turbo_session_write_pairs(
    s: *mut turbo_session,
    query: *const turbo_text,
    docs: *const turbo_text,
    count: u32,
    opts: *const turbo_rerank_options,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let s = unsafe { handle(s as *const Session, "turbo_session") }?;
        if query.is_null() {
            return Err(Error::invalid_argument("query is NULL"));
        }
        let q = unsafe { text(&*query, "query") }?;
        let d = unsafe { texts(docs, count, "docs") }?;
        let mut ro = RerankOptions::default();
        if !opts.is_null() {
            let o = unsafe { read_sized::<turbo_rerank_options>(opts, "turbo_rerank_options") }?;
            ro.truncate = Truncate::from_abi(o.truncate).map_err(|e| e.with_field(2))?;
            ro.max_tokens = o.max_tokens;
            ro.top_n = o.top_n;
            ro.return_sorted = match o.return_sorted {
                0 => false,
                1 => true,
                v => {
                    return Err(Error::invalid_argument(format!("return_sorted must be 0 or 1, got {v}")).with_field(5))
                }
            };
            ro.raw_scores = match o.raw_scores {
                0 => false,
                1 => true,
                v => return Err(Error::invalid_argument(format!("raw_scores must be 0 or 1, got {v}")).with_field(6)),
            };
        }
        s.write_pairs(q, &d, &ro)
    })
}

/// Write `count` texts for classification. `opts` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn turbo_session_write_text_classify(
    s: *mut turbo_session,
    texts_ptr: *const turbo_text,
    count: u32,
    opts: *const turbo_classify_options,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let s = unsafe { handle(s as *const Session, "turbo_session") }?;
        let list = unsafe { texts(texts_ptr, count, "texts") }?;
        let mut co = ClassifyOptions::default();
        if !opts.is_null() {
            let o = unsafe { read_sized::<turbo_classify_options>(opts, "turbo_classify_options") }?;
            co.truncate = Truncate::from_abi(o.truncate).map_err(|e| e.with_field(2))?;
            co.max_tokens = o.max_tokens;
            co.aggregation = Aggregation::from_abi(o.aggregation).map_err(|e| e.with_field(4))?;
            co.raw_scores = match o.raw_scores {
                0 => false,
                1 => true,
                v => return Err(Error::invalid_argument(format!("raw_scores must be 0 or 1, got {v}")).with_field(5)),
            };
            if o.reserved != 0 {
                return Err(Error::invalid_argument("turbo_classify_options.reserved must be 0").with_field(6));
            }
        }
        s.write_text_classify(&list, &co)
    })
}

/// Bind a named input or output buffer (RUN models).
#[no_mangle]
pub unsafe extern "C" fn turbo_session_bind(
    s: *mut turbo_session,
    name: turbo_text,
    buf: *mut turbo_buffer,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let s = unsafe { handle(s as *const Session, "turbo_session") }?;
        let n = unsafe { text(&name, "name") }?;
        if n.is_empty() {
            return Err(Error::invalid_argument("name is empty"));
        }
        let b = unsafe { arc(buf as *const Buffer, "turbo_buffer") }?;
        s.bind(n, &b)
    })
}

/// Execute and lease the result. `opts` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn turbo_session_run(
    s: *mut turbo_session,
    opts: *const turbo_run_options,
    out: *mut *mut turbo_result,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let out = unsafe { out_ptr(out, "turbo_session_run") }?;
        let s = unsafe { arc(s as *const Session, "turbo_session") }?;
        let mut ro = RunOptions::default();
        if !opts.is_null() {
            let o = unsafe { read_sized::<turbo_run_options>(opts, "turbo_run_options") }?;
            ro.params = unsafe { kvs(o.params, o.n_params, "turbo_run_options.params") }?;
        }
        let r = s.run(&ro)?;
        *out = leak(r) as *mut turbo_result;
        Ok(())
    })
}

/// Session counters.
#[no_mangle]
pub unsafe extern "C" fn turbo_session_get_stats(
    s: *mut turbo_session,
    out: *mut turbo_session_stats,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let s = unsafe { handle(s as *const Session, "turbo_session") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        let o = unsafe { &mut *out };
        check_size::<turbo_session_stats>("turbo_session_stats", o.struct_size)?;
        let st = s.stats()?;
        let full = turbo_session_stats {
            struct_size: o.struct_size,
            reserved: 0,
            runs: st.runs,
            host_allocs: st.host_allocs.unwrap_or(u64::MAX),
            h2d_bytes: st.h2d_bytes,
            d2h_bytes: st.d2h_bytes,
            input_bytes: st.input_bytes,
            output_bytes: st.output_bytes,
            provider_allocs: st.provider_allocs.unwrap_or(u64::MAX),
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&full as *const turbo_session_stats).cast::<u8>(),
                out.cast::<u8>(),
                o.struct_size as usize,
            )
        };
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// Summary of a result.
#[no_mangle]
pub unsafe extern "C" fn turbo_result_get_info(
    r: *mut turbo_result,
    out: *mut turbo_result_info,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let r = unsafe { handle(r as *const ResultHandle, "turbo_result") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        let o = unsafe { &mut *out };
        check_size::<turbo_result_info>("turbo_result_info", o.struct_size)?;
        let first = r.output(0)?;
        let full = turbo_result_info {
            struct_size: o.struct_size,
            n_outputs: r.outputs().len() as u32,
            batch: first.shape.first().copied().unwrap_or(1) as u32,
            dim: if first.shape.len() >= 2 { first.shape[1..].iter().product::<u64>() as u32 } else { 1 },
            dtype: first.dtype().as_abi(),
            placement: first.placement().as_abi(),
            bytes: first.logical_bytes()?,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&full as *const turbo_result_info).cast::<u8>(),
                out.cast::<u8>(),
                o.struct_size as usize,
            )
        };
        Ok(())
    })
}

/// Description of output `index`, including its logical shape.
#[no_mangle]
pub unsafe extern "C" fn turbo_result_output_info(
    r: *mut turbo_result,
    index: u32,
    out: *mut turbo_tensor_info,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let r = unsafe { handle(r as *const ResultHandle, "turbo_result") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        let o = unsafe { &mut *out };
        check_size::<turbo_tensor_info>("turbo_tensor_info", o.struct_size)?;
        let t = r.output(index)?;
        let mut full = turbo_tensor_info {
            struct_size: o.struct_size,
            dtype: t.dtype().as_abi(),
            ndim: t.shape.len() as u32,
            reserved: 0,
            shape: [0; TURBO_MAX_RANK],
            name: [0; 64],
        };
        for (i, d) in t.shape.iter().enumerate().take(TURBO_MAX_RANK) {
            full.shape[i] = *d as i64;
        }
        put_str(&mut full.name, &t.name);
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&full as *const turbo_tensor_info).cast::<u8>(),
                out.cast::<u8>(),
                o.struct_size as usize,
            )
        };
        Ok(())
    })
}

/// Buffer view of output `index`. The view keeps the result lease alive.
#[no_mangle]
pub unsafe extern "C" fn turbo_result_buffer(
    r: *mut turbo_result,
    index: u32,
    out: *mut *mut turbo_buffer,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let out = unsafe { out_ptr(out, "turbo_result_buffer") }?;
        let r = unsafe { arc(r as *const ResultHandle, "turbo_result") }?;
        let b = r.buffer(index)?;
        *out = leak(b) as *mut turbo_buffer;
        Ok(())
    })
}

/// Blocking copy of output `index` into `dst` (`capacity` bytes). Writes the
/// byte count to `written` if non-NULL.
#[no_mangle]
pub unsafe extern "C" fn turbo_result_read(
    r: *mut turbo_result,
    index: u32,
    dst: *mut c_void,
    capacity: u64,
    written: *mut u64,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let r = unsafe { handle(r as *const ResultHandle, "turbo_result") }?;
        if dst.is_null() && capacity != 0 {
            return Err(Error::invalid_argument("dst is NULL with non-zero capacity"));
        }
        let cap = usize::try_from(capacity).map_err(|_| Error::invalid_argument("capacity exceeds usize"))?;
        // SAFETY: the caller promises `capacity` writable bytes at dst.
        let slice: &mut [u8] =
            if cap == 0 { &mut [] } else { unsafe { std::slice::from_raw_parts_mut(dst.cast(), cap) } };
        let n = r.read(index, slice)?;
        if !written.is_null() {
            unsafe { *written = n as u64 };
        }
        Ok(())
    })
}

/// Copy up to `capacity` spans into `dst` and report the total available in `count`.
#[no_mangle]
pub unsafe extern "C" fn turbo_result_spans(
    r: *mut turbo_result,
    dst: *mut turbo_span,
    capacity: u32,
    count: *mut u32,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let r = unsafe { handle(r as *const ResultHandle, "turbo_result") }?;
        if count.is_null() {
            return Err(Error::invalid_argument("count is NULL"));
        }
        let spans = r.spans();
        unsafe { *count = spans.len() as u32 };
        if capacity == 0 {
            return Ok(());
        }
        if dst.is_null() {
            return Err(Error::invalid_argument("dst is NULL with non-zero capacity"));
        }
        let n = spans.len().min(capacity as usize);
        let out = unsafe { std::slice::from_raw_parts_mut(dst, n) };
        for (o, s) in out.iter_mut().zip(spans) {
            *o = turbo_span {
                byte_start: s.byte_start,
                byte_end: s.byte_end,
                row: s.row,
                label: s.label,
                score: s.score,
                reserved: 0,
            };
        }
        Ok(())
    })
}

/// Release a result and return its lease (once every view is released too).
#[no_mangle]
pub unsafe extern "C" fn turbo_result_release(r: *mut turbo_result) {
    unsafe { reclaim(r as *mut ResultHandle) };
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

unsafe fn generate_desc(desc: *const turbo_generate_desc) -> Result<GenerateDesc> {
    if desc.is_null() {
        return Ok(GenerateDesc::default());
    }
    let d = unsafe { read_sized::<turbo_generate_desc>(desc, "turbo_generate_desc") }?;
    let stop =
        unsafe { texts(d.stop, d.n_stop, "turbo_generate_desc.stop") }?.into_iter().map(str::to_string).collect();
    let stop_tokens = if d.n_stop_tokens == 0 {
        Vec::new()
    } else if d.stop_tokens.is_null() {
        return Err(Error::invalid_argument("stop_tokens is NULL with n_stop_tokens > 0").with_field(17));
    } else {
        unsafe { std::slice::from_raw_parts(d.stop_tokens, d.n_stop_tokens as usize) }.to_vec()
    };
    let logit_bias = if d.n_logit_bias == 0 {
        Vec::new()
    } else if d.logit_bias.is_null() {
        return Err(Error::invalid_argument("logit_bias is NULL with n_logit_bias > 0").with_field(20));
    } else {
        unsafe { std::slice::from_raw_parts(d.logit_bias, d.n_logit_bias as usize) }
            .iter()
            .map(|b| (b.token, b.bias))
            .collect()
    };
    let tools =
        unsafe { texts(d.tools, d.n_tools, "turbo_generate_desc.tools") }?.into_iter().map(str::to_string).collect();
    let has_seed = match d.has_seed {
        0 => false,
        1 => true,
        v => return Err(Error::invalid_argument(format!("has_seed must be 0 or 1, got {v}")).with_field(12)),
    };
    let echo = match d.echo {
        0 => false,
        1 => true,
        v => return Err(Error::invalid_argument(format!("echo must be 0 or 1, got {v}")).with_field(22)),
    };
    Ok(GenerateDesc {
        max_new_tokens: d.max_new_tokens,
        min_new_tokens: d.min_new_tokens,
        n_sequences: d.n_sequences.max(1),
        temperature: d.temperature,
        top_k: d.top_k,
        top_p: d.top_p,
        min_p: d.min_p,
        repeat_penalty: d.repeat_penalty,
        presence_penalty: d.presence_penalty,
        frequency_penalty: d.frequency_penalty,
        seed: if has_seed { Some(d.seed) } else { None },
        stop,
        stop_tokens,
        logit_bias,
        logprobs: d.logprobs,
        structured_kind: StructuredKind::from_abi(d.structured_kind).map_err(|e| e.with_field(21))?,
        structured: unsafe { text(&d.structured, "turbo_generate_desc.structured") }?.to_string(),
        echo,
        tools,
        options: unsafe { kvs(d.options, d.n_options, "turbo_generate_desc.options") }?,
    })
}

/// Create a generation on a generative model. `desc` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn turbo_generation_create(
    m: *mut turbo_model,
    desc: *const turbo_generate_desc,
    out: *mut *mut turbo_generation,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let out = unsafe { out_ptr(out, "turbo_generation_create") }?;
        let m = unsafe { arc(m as *const Model, "turbo_model") }?;
        let gd = unsafe { generate_desc(desc) }?;
        let g = m.create_generation(&gd)?;
        *out = leak(g) as *mut turbo_generation;
        Ok(())
    })
}

/// Apply the chat template to `count` messages and tokenize.
#[no_mangle]
pub unsafe extern "C" fn turbo_generation_prompt(
    g: *mut turbo_generation,
    messages: *const turbo_message,
    count: u32,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let g = unsafe { handle(g as *const Generation, "turbo_generation") }?;
        if count == 0 {
            return Err(Error::invalid_argument("at least one message is required"));
        }
        if messages.is_null() {
            return Err(Error::invalid_argument("messages is NULL"));
        }
        let raw = unsafe { std::slice::from_raw_parts(messages, count as usize) };
        let mut list = Vec::with_capacity(raw.len());
        for (i, m) in raw.iter().enumerate() {
            list.push(Message {
                role: unsafe { text(&m.role, &format!("messages[{i}].role")) }?,
                content: unsafe { text(&m.content, &format!("messages[{i}].content")) }?,
            });
        }
        g.prompt(&list)
    })
}

/// Use `count` caller-supplied prompt token ids.
#[no_mangle]
pub unsafe extern "C" fn turbo_generation_prompt_tokens(
    g: *mut turbo_generation,
    ids: *const i32,
    count: u32,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let g = unsafe { handle(g as *const Generation, "turbo_generation") }?;
        if count == 0 {
            return Err(Error::invalid_argument("at least one prompt token is required"));
        }
        if ids.is_null() {
            return Err(Error::invalid_argument("ids is NULL"));
        }
        let slice = unsafe { std::slice::from_raw_parts(ids, count as usize) };
        g.prompt_tokens(slice)
    })
}

/// Produce the next chunk. Pointers in `out` stay valid until the next call on `g`.
#[no_mangle]
pub unsafe extern "C" fn turbo_generation_step(
    g: *mut turbo_generation,
    out: *mut turbo_generation_chunk,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let g = unsafe { handle(g as *const Generation, "turbo_generation") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        let o = unsafe { &mut *out };
        check_size::<turbo_generation_chunk>("turbo_generation_chunk", o.struct_size)?;
        let chunk = g.step()?;
        let full = turbo_generation_chunk {
            struct_size: o.struct_size,
            sequence: chunk.sequence,
            n_tokens: chunk.tokens.len() as u32,
            n_logprobs: chunk.logprobs.len() as u32,
            tokens: if chunk.tokens.is_empty() { std::ptr::null() } else { chunk.tokens.as_ptr() },
            text: turbo_text { ptr: chunk.text.as_ptr().cast(), len: chunk.text.len() as u64 },
            logprobs: if chunk.logprobs.is_empty() { std::ptr::null() } else { chunk.logprobs.as_ptr() },
            done: chunk.done as u32,
            finish_reason: chunk.finish_reason.as_abi(),
            prompt_tokens: chunk.prompt_tokens,
            generated_tokens: chunk.generated_tokens,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&full as *const turbo_generation_chunk).cast::<u8>(),
                out.cast::<u8>(),
                o.struct_size as usize,
            )
        };
        Ok(())
    })
}

/// Cancel; the next step reports `TURBO_FINISH_CANCELLED`.
#[no_mangle]
pub unsafe extern "C" fn turbo_generation_cancel(g: *mut turbo_generation, err: *mut turbo_error) -> i32 {
    boundary(err, || {
        let g = unsafe { handle(g as *const Generation, "turbo_generation") }?;
        g.cancel()
    })
}

/// Release a generation.
#[no_mangle]
pub unsafe extern "C" fn turbo_generation_release(g: *mut turbo_generation) {
    unsafe { reclaim(g as *mut Generation) };
}

/// Push-style generation over the pull iterator: creates a generation,
/// applies `messages`, and calls `callback` with every chunk until the
/// generation finishes or the callback returns `TURBO_STREAM_STOP`. The
/// chunk and its pointers are valid during the callback only. A stopped
/// generation is cancelled and released before this returns `TURBO_OK`; a
/// callback result other than `TURBO_STREAM_CONTINUE` or
/// `TURBO_STREAM_STOP` is `TURBO_E_INVALID_ARGUMENT`.
#[no_mangle]
pub unsafe extern "C" fn turbo_generate(
    m: *mut turbo_model,
    desc: *const turbo_generate_desc,
    messages: *const turbo_message,
    count: u32,
    callback: turbo_stream_fn,
    user_data: *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let m = unsafe { arc(m as *const Model, "turbo_model") }?;
        let Some(callback) = callback else {
            return Err(Error::invalid_argument("callback is NULL"));
        };
        if count == 0 {
            return Err(Error::invalid_argument("at least one message is required"));
        }
        if messages.is_null() {
            return Err(Error::invalid_argument("messages is NULL"));
        }
        let gd = unsafe { generate_desc(desc) }?;
        let raw = unsafe { std::slice::from_raw_parts(messages, count as usize) };
        let mut list = Vec::with_capacity(raw.len());
        for (i, msg) in raw.iter().enumerate() {
            list.push(Message {
                role: unsafe { text(&msg.role, &format!("messages[{i}].role")) }?,
                content: unsafe { text(&msg.content, &format!("messages[{i}].content")) }?,
            });
        }
        let g = m.create_generation(&gd)?;
        g.prompt(&list)?;
        loop {
            let chunk = g.step()?;
            let full = turbo_generation_chunk {
                struct_size: std::mem::size_of::<turbo_generation_chunk>() as u32,
                sequence: chunk.sequence,
                n_tokens: chunk.tokens.len() as u32,
                n_logprobs: chunk.logprobs.len() as u32,
                tokens: if chunk.tokens.is_empty() { std::ptr::null() } else { chunk.tokens.as_ptr() },
                text: turbo_text { ptr: chunk.text.as_ptr().cast(), len: chunk.text.len() as u64 },
                logprobs: if chunk.logprobs.is_empty() { std::ptr::null() } else { chunk.logprobs.as_ptr() },
                done: chunk.done as u32,
                finish_reason: chunk.finish_reason.as_abi(),
                prompt_tokens: chunk.prompt_tokens,
                generated_tokens: chunk.generated_tokens,
            };
            let done = chunk.done;
            // The callback runs while the chunk guard is held, so its
            // pointers stay valid; it must not call back into this generation.
            let verdict = unsafe { callback(user_data, &full) };
            drop(chunk);
            match verdict {
                TURBO_STREAM_CONTINUE => {}
                TURBO_STREAM_STOP => {
                    g.cancel()?;
                    return Ok(());
                }
                other => {
                    g.cancel()?;
                    return Err(Error::invalid_argument(format!(
                        "callback returned {other}; expected TURBO_STREAM_CONTINUE (0) or TURBO_STREAM_STOP (1)"
                    )));
                }
            }
            if done {
                return Ok(());
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Tokenizer and chunker
// ---------------------------------------------------------------------------

/// Load the tokenizer a bundle declares (`tokenizer.files["tokenizer.json"]`).
/// Tokenizers are thread-safe and independent of any device.
#[no_mangle]
pub unsafe extern "C" fn turbo_tokenizer_create(
    rt: *mut turbo_runtime,
    bundle_path: turbo_text,
    out: *mut *mut turbo_tokenizer,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let out = unsafe { out_ptr(out, "turbo_tokenizer_create") }?;
        let _ = unsafe { handle(rt as *const Runtime, "turbo_runtime") }?;
        let path = unsafe { text(&bundle_path, "bundle_path") }?;
        if path.is_empty() {
            return Err(Error::invalid_argument("bundle_path is empty"));
        }
        let bundle = turbo_core::Bundle::open(Path::new(path))?;
        let t = Tokenizer::from_bundle(&bundle)?;
        *out = leak(t) as *mut turbo_tokenizer;
        Ok(())
    })
}

/// Release a tokenizer.
#[no_mangle]
pub unsafe extern "C" fn turbo_tokenizer_release(t: *mut turbo_tokenizer) {
    unsafe { reclaim(t as *mut Tokenizer) };
}

/// Static facts about a tokenizer.
#[no_mangle]
pub unsafe extern "C" fn turbo_tokenizer_get_info(
    t: *mut turbo_tokenizer,
    out: *mut turbo_tokenizer_info,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let t = unsafe { handle(t as *const Tokenizer, "turbo_tokenizer") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        let o = unsafe { &mut *out };
        check_size::<turbo_tokenizer_info>("turbo_tokenizer_info", o.struct_size)?;
        let i = t.info();
        let mut full = turbo_tokenizer_info {
            struct_size: o.struct_size,
            vocab_size: i.vocab_size,
            max_seq: i.max_seq,
            specials_per_sequence: i.specials_per_sequence,
            pad_id: i.pad_id.unwrap_or(-1),
            bos_id: i.bos_id.unwrap_or(-1),
            eos_id: i.eos_id.unwrap_or(-1),
            unk_id: i.unk_id.unwrap_or(-1),
            kind: [0; 32],
            sha256: [0; 72],
        };
        put_str(&mut full.kind, &i.kind);
        put_str(&mut full.sha256, &i.sha256);
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&full as *const turbo_tokenizer_info).cast::<u8>(),
                out.cast::<u8>(),
                o.struct_size as usize,
            )
        };
        Ok(())
    })
}

unsafe fn encode_options(opts: *const turbo_encode_options) -> Result<EncodeOptions> {
    if opts.is_null() {
        return Ok(EncodeOptions::default());
    }
    let o = unsafe { read_sized::<turbo_encode_options>(opts, "turbo_encode_options") }?;
    Ok(EncodeOptions {
        add_special_tokens: match o.add_special_tokens {
            0 => false,
            1 => true,
            v => {
                return Err(Error::invalid_argument(format!("add_special_tokens must be 0 or 1, got {v}")).with_field(2))
            }
        },
        truncate: Truncate::from_abi(o.truncate).map_err(|e| e.with_field(3))?,
        max_tokens: o.max_tokens,
        pad_to: o.pad_to,
        prompt_role: PromptRole::from_abi(o.prompt_role).map_err(|e| e.with_field(6))?,
    })
}

/// Encode `count` texts into caller-owned row-major `[count, row_stride]`
/// arrays. Rows are padded with the pad id and mask 0 to `pad_to` (or to
/// `row_stride` when `pad_to` is 0). `types` and `lengths` may be NULL;
/// `lengths` receives each row's live token count. `opts` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn turbo_tokenizer_encode(
    t: *mut turbo_tokenizer,
    texts_ptr: *const turbo_text,
    count: u32,
    opts: *const turbo_encode_options,
    ids: *mut i32,
    mask: *mut i32,
    types: *mut i32,
    row_stride: u32,
    lengths: *mut u32,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let t = unsafe { handle(t as *const Tokenizer, "turbo_tokenizer") }?;
        let list = unsafe { texts(texts_ptr, count, "texts") }?;
        if list.is_empty() {
            return Err(Error::invalid_argument("count must be at least 1"));
        }
        let o = unsafe { encode_options(opts) }?;
        if row_stride == 0 {
            return Err(Error::invalid_argument("row_stride must be non-zero"));
        }
        if ids.is_null() || mask.is_null() {
            return Err(Error::invalid_argument("ids and mask must be non-NULL"));
        }
        let n = (count as usize)
            .checked_mul(row_stride as usize)
            .ok_or_else(|| Error::invalid_shape("count * row_stride overflows"))?;
        // SAFETY: the caller promises `n` writable elements in each array.
        let ids_s = unsafe { std::slice::from_raw_parts_mut(ids, n) };
        let mask_s = unsafe { std::slice::from_raw_parts_mut(mask, n) };
        let types_s = if types.is_null() { None } else { Some(unsafe { std::slice::from_raw_parts_mut(types, n) }) };
        let mut lens = vec![0u32; list.len()];
        t.encode_into(
            &list,
            &o,
            EncodeTarget {
                ids: ids_s,
                mask: mask_s,
                types: types_s,
                row_stride: row_stride as usize,
                lengths: &mut lens,
            },
        )?;
        if !lengths.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(lens.as_ptr(), lengths, lens.len()) };
        }
        Ok(())
    })
}

/// Decode `count` ids into `dst` (`capacity` bytes, not NUL-terminated).
/// Writes the byte length to `written`; if `capacity` is too small, returns
/// `TURBO_E_CAPACITY` with the required length in `written`.
#[no_mangle]
pub unsafe extern "C" fn turbo_tokenizer_decode(
    t: *mut turbo_tokenizer,
    ids: *const i32,
    count: u32,
    skip_special_tokens: u32,
    dst: *mut c_char,
    capacity: u64,
    written: *mut u64,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let t = unsafe { handle(t as *const Tokenizer, "turbo_tokenizer") }?;
        if count == 0 || ids.is_null() {
            return Err(Error::invalid_argument("ids is NULL or count is 0"));
        }
        let skip = match skip_special_tokens {
            0 => false,
            1 => true,
            v => return Err(Error::invalid_argument(format!("skip_special_tokens must be 0 or 1, got {v}"))),
        };
        let slice = unsafe { std::slice::from_raw_parts(ids, count as usize) };
        let s = t.decode(slice, skip)?;
        if !written.is_null() {
            unsafe { *written = s.len() as u64 };
        }
        if (s.len() as u64) > capacity {
            return Err(Error::capacity(format!("decoded text is {} bytes but capacity is {capacity}", s.len())));
        }
        if !s.is_empty() {
            if dst.is_null() {
                return Err(Error::invalid_argument("dst is NULL"));
            }
            unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), dst.cast::<u8>(), s.len()) };
        }
        Ok(())
    })
}

/// Number of tokens `text` produces, without truncation or prefix.
#[no_mangle]
pub unsafe extern "C" fn turbo_tokenizer_count(
    t: *mut turbo_tokenizer,
    text_in: turbo_text,
    add_special_tokens: u32,
    out: *mut u32,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let t = unsafe { handle(t as *const Tokenizer, "turbo_tokenizer") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        let s = unsafe { text(&text_in, "text") }?;
        let add = match add_special_tokens {
            0 => false,
            1 => true,
            v => return Err(Error::invalid_argument(format!("add_special_tokens must be 0 or 1, got {v}"))),
        };
        unsafe { *out = t.count(s, add)? };
        Ok(())
    })
}

/// Plan chunks over `text` with `tokenizer` counting content tokens. The
/// plan stores byte offsets only; the caller keeps the text.
#[no_mangle]
pub unsafe extern "C" fn turbo_chunk_plan_create(
    desc: *const turbo_chunk_desc,
    text_in: turbo_text,
    tokenizer: *mut turbo_tokenizer,
    out: *mut *mut turbo_chunk_plan,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let out = unsafe { out_ptr(out, "turbo_chunk_plan_create") }?;
        if desc.is_null() {
            return Err(Error::invalid_argument("turbo_chunk_desc is NULL"));
        }
        let d = unsafe { read_sized::<turbo_chunk_desc>(desc, "turbo_chunk_desc") }?;
        let t = unsafe { handle(tokenizer as *const Tokenizer, "turbo_tokenizer") }?;
        let s = unsafe { text(&text_in, "text") }?;
        let config = ChunkerConfig {
            max_tokens: d.max_tokens as usize,
            reserved_tokens: d.reserved_tokens as usize,
            overlap_tokens: d.overlap_tokens as usize,
            sentence_boundaries: true,
        };
        let plan = chunk_source("", s, &config, t).map_err(|e| match e {
            ChunkError::InvalidConfig(m) => Error::invalid_argument(m),
            other => Error::capacity(other.to_string()),
        })?;
        *out = leak(Arc::new(plan)) as *mut turbo_chunk_plan;
        Ok(())
    })
}

/// Number of chunks in a plan.
#[no_mangle]
pub unsafe extern "C" fn turbo_chunk_plan_count(p: *mut turbo_chunk_plan, out: *mut u32, err: *mut turbo_error) -> i32 {
    boundary(err, || {
        let p = unsafe { handle(p as *const ChunkPlan, "turbo_chunk_plan") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        unsafe { *out = p.chunks.len() as u32 };
        Ok(())
    })
}

/// Chunk `index` of a plan.
#[no_mangle]
pub unsafe extern "C" fn turbo_chunk_plan_get(
    p: *mut turbo_chunk_plan,
    index: u32,
    out: *mut turbo_chunk,
    err: *mut turbo_error,
) -> i32 {
    boundary(err, || {
        let p = unsafe { handle(p as *const ChunkPlan, "turbo_chunk_plan") }?;
        if out.is_null() {
            return Err(Error::invalid_argument("out is NULL"));
        }
        let c = p.chunks.get(index as usize).ok_or_else(|| {
            Error::invalid_argument(format!(
                "chunk index {index} is out of range; the plan has {} chunks",
                p.chunks.len()
            ))
        })?;
        unsafe {
            *out = turbo_chunk {
                byte_start: c.byte_range.start as u64,
                byte_end: c.byte_range.end as u64,
                paragraph: c.paragraph_index as u32,
                n_tokens: c.token_count as u32,
            }
        };
        Ok(())
    })
}

/// Release a chunk plan.
#[no_mangle]
pub unsafe extern "C" fn turbo_chunk_plan_release(p: *mut turbo_chunk_plan) {
    unsafe { reclaim(p as *mut ChunkPlan) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    fn err() -> turbo_error {
        turbo_error {
            struct_size: std::mem::size_of::<turbo_error>() as u32,
            code: 0,
            field: 0,
            message: [0; TURBO_ERROR_MESSAGE_LEN],
        }
    }

    fn msg(e: &turbo_error) -> String {
        unsafe { CStr::from_ptr(e.message.as_ptr()) }.to_string_lossy().into_owned()
    }

    #[test]
    fn error_message_is_nul_terminated_and_truncated_on_char_boundary() {
        let mut e = err();
        let long = "é".repeat(600);
        fail(&mut e, &Error::invalid_argument(long.clone()));
        let m = msg(&e);
        assert!(m.len() < TURBO_ERROR_MESSAGE_LEN);
        assert!(long.starts_with(&m));
        assert_eq!(e.code, TURBO_E_INVALID_ARGUMENT);
    }

    #[test]
    fn tiny_error_struct_gets_only_a_code() {
        let mut e = err();
        e.struct_size = 8;
        let rc = boundary(&mut e, || Err(Error::busy("x")));
        assert_eq!(rc, TURBO_E_BUSY);
        assert_eq!(e.code, TURBO_E_BUSY);
        assert_eq!(e.field, 0);
    }

    #[test]
    fn panics_are_caught() {
        let mut e = err();
        let rc = boundary(&mut e, || panic!("boom"));
        assert_eq!(rc, TURBO_E_PANIC);
        assert!(msg(&e).contains("boom"));
    }

    #[test]
    fn struct_size_larger_than_known_is_rejected() {
        let e = check_size::<turbo_embed_options>("x", 4096).unwrap_err();
        assert_eq!(e.code(), TURBO_E_INVALID_STRUCT_SIZE);
        // 8 is the end of `truncate`: a layout an older caller could hold.
        check_size::<turbo_embed_options>("x", 8).unwrap();
        // 6 ends inside `truncate`: never a layout.
        assert_eq!(check_size::<turbo_embed_options>("x", 6).unwrap_err().code(), TURBO_E_INVALID_STRUCT_SIZE);
    }

    #[test]
    fn status_names_are_static_c_strings() {
        let s = unsafe { CStr::from_ptr(turbo_status_name(TURBO_E_BUNDLE_INTEGRITY)) };
        assert_eq!(s.to_str().unwrap(), "TURBO_E_BUNDLE_INTEGRITY");
        let u = unsafe { CStr::from_ptr(turbo_status_name(-5)) };
        assert_eq!(u.to_str().unwrap(), "TURBO_E_UNKNOWN");
    }
}
