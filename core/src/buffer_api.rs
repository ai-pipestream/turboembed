//! C buffer allocation, import, access, and release entrypoints.

use super::*;

pub(super) fn dtype_size(dtype: u32) -> Option<u64> {
    match dtype {
        TURBO_DTYPE_I32 | TURBO_DTYPE_F32 => Some(4),
        TURBO_DTYPE_F16 | TURBO_DTYPE_BF16 => Some(2),
        _ => None,
    }
}

/// The caller's desc as a backend gets it: every field checked, and bytes
/// filled in from shape and dtype.
pub(super) unsafe fn buffer_desc(desc: *const turbo_buffer_desc) -> Result<turbo_buffer_desc> {
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
pub(super) fn new_buffer(
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

pub(super) fn handle_kind(kind: u32) -> Result<u32> {
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
            if buf.raw.is_null() {
                // Host memory the core owns (a rerank result's scores): no
                // backend object to ask, and a host pointer is all it has.
                if kind != TURBO_HANDLE_HOST_PTR {
                    let m = format!("kind: {kind}: a result in the core's host memory exports TURBO_HANDLE_HOST_PTR");
                    return Err(Error::new(status::UNSUPPORTED, m));
                }
                *out = turbo_native_handle {
                    struct_size: size_of::<turbo_native_handle>() as u32,
                    kind,
                    handle: buf.host as usize as u64,
                    aux: 0,
                    offset: 0,
                };
                return Ok(());
            }
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
