//! Host memory the device reads directly: the backend's buffers.

use std::ffi::c_void;

use super::*;
use crate::npu::ze::{self, Handle};
use crate::status::UNSUPPORTED;
use crate::{TURBO_HANDLE_HOST_PTR, TURBO_PLACE_HOST, turbo_buffer_desc, turbo_error, turbo_native_handle};

// ---- Buffers ---------------------------------------------------------------------------

/// Host memory the device reads directly, 64-byte aligned: the only
/// placement this backend has. The session's result vectors are one too,
/// so buffer_export serves them alike.
pub(crate) struct Buffer {
    /// zeMemFree for a device allocation. None when `ptr` is memory the
    /// caller owns. A test buffer is not a Level Zero allocation, and
    /// Drop does not call the driver for it.
    pub(super) free: Option<(&'static ze::Api, Handle)>,
    pub(super) ptr: *mut c_void,
    pub(super) bytes: u64,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub(super) fn new(ctx: &Context, bytes: u64, what: &str) -> Res<Buffer> {
        let ptr = ctx.alloc_host(bytes as usize, what)?;
        Ok(Buffer { free: Some((ctx.api, ctx.handle)), ptr, bytes })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some((api, context)) = self.free {
            unsafe { (api.mem_free)(context, self.ptr) };
        }
    }
}

/// # Safety
/// As turbo_backend.h says for buffer_alloc.
pub(crate) unsafe extern "C" fn buffer_alloc(
    ctx: *mut c_void,
    desc: *const turbo_buffer_desc,
    out: *mut *mut c_void,
    host: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let ctx = &*(ctx as *const Context);
            let desc = &*desc;
            if desc.placement != TURBO_PLACE_HOST {
                return Err(fail(UNSUPPORTED, "npu: TURBO_PLACE_HOST only: the device reads host memory"));
            }
            let b = Box::new(Buffer::new(ctx, desc.bytes, "a buffer")?);
            *host = b.ptr;
            *out = Box::into_raw(b) as *mut c_void;
            Ok(())
        })
    }
}

/// # Safety
/// `buf` is a buffer this backend allocated, released once.
pub(crate) unsafe extern "C" fn buffer_release(buf: *mut c_void) {
    quietly(|| drop(unsafe { Box::from_raw(buf as *mut Buffer) }));
}

/// # Safety
/// As turbo_backend.h says for buffer_export.
pub(crate) unsafe extern "C" fn buffer_export(
    buf: *mut c_void,
    kind: u32,
    out: *mut turbo_native_handle,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let b = &*(buf as *const Buffer);
            if kind != TURBO_HANDLE_HOST_PTR {
                return Err(fail(UNSUPPORTED, format!("npu: a buffer exports as TURBO_HANDLE_HOST_PTR, not {kind}")));
            }
            let out = &mut *out;
            out.kind = TURBO_HANDLE_HOST_PTR;
            out.handle = b.ptr as u64;
            out.aux = 0;
            out.offset = 0;
            Ok(())
        })
    }
}
