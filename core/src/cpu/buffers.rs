//! Contexts and buffers: host memory, allocated here or the caller's own.
//!
//! The CPU has no memory apart from the host's. HOST, PINNED and SHARED are
//! all pageable host memory here: there is no second device to pin for or
//! to share with. DEVICE is refused rather than read as host memory, so a
//! caller that asked for memory off the host is told it did not get it.

use std::alloc::Layout;
use std::ffi::c_void;

use crate::backend::refuse;
use crate::status::{INVALID_ARGUMENT, OUT_OF_MEMORY, UNSUPPORTED};
use crate::{
    TURBO_HANDLE_HOST_PTR, TURBO_PLACE_DEVICE, turbo_buffer_desc, turbo_error, turbo_log_fn, turbo_native_handle,
};

use super::only;

/// Every allocation starts on a 64-byte boundary: a cache line, and the
/// width of an AVX-512 register.
pub(super) const ALIGN: usize = 64;

/// The host keeps nothing per context and has nothing to warn about.
pub(super) struct Context;

pub(super) struct Buffer {
    pub(super) ptr: *mut u8,
    /// The allocation to free; None for memory the caller owns.
    pub(super) layout: Option<Layout>,
}

pub(super) unsafe extern "C" fn context_create(
    ordinal: u32,
    _log: turbo_log_fn,
    _log_user_data: *mut c_void,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    if let Some(rc) = unsafe { only(ordinal, err) } {
        return rc;
    }
    unsafe { *out = Box::into_raw(Box::new(Context)) as *mut c_void };
    0
}

pub(super) unsafe extern "C" fn context_release(ctx: *mut c_void) {
    drop(unsafe { Box::from_raw(ctx as *mut Context) });
}

/// DEVICE placement, refused: see above.
pub(super) unsafe fn host_only(desc: &turbo_buffer_desc, err: *mut turbo_error) -> Option<i32> {
    (desc.placement == TURBO_PLACE_DEVICE).then(|| unsafe {
        refuse(err, UNSUPPORTED, "placement: TURBO_PLACE_DEVICE: the cpu has no memory apart from the host's")
    })
}

pub(super) unsafe fn give(buf: Buffer, out: *mut *mut c_void, host: *mut *mut c_void) -> i32 {
    unsafe {
        *host = buf.ptr as *mut c_void;
        *out = Box::into_raw(Box::new(buf)) as *mut c_void;
    }
    0
}

pub(super) unsafe extern "C" fn buffer_alloc(
    _ctx: *mut c_void,
    desc: *const turbo_buffer_desc,
    out: *mut *mut c_void,
    host: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    let desc = unsafe { &*desc };
    if let Some(rc) = unsafe { host_only(desc, err) } {
        return rc;
    }
    let too_big =
        || unsafe { refuse(err, OUT_OF_MEMORY, &format!("{} bytes is more than the host can address", desc.bytes)) };
    let Ok(bytes) = usize::try_from(desc.bytes) else {
        return too_big();
    };
    // The allocator may not be asked for no bytes; the core never asks.
    if bytes == 0 {
        return unsafe { refuse(err, INVALID_ARGUMENT, "bytes: 0: a buffer holds at least one byte") };
    }
    let Ok(layout) = Layout::from_size_align(bytes, ALIGN) else {
        return too_big();
    };
    // Not zeroed: turbo.h promises no contents, and the caller writes them.
    match Buffer::alloc(layout) {
        Some(buf) => unsafe { give(buf, out, host) },
        None => unsafe { refuse(err, OUT_OF_MEMORY, &format!("{bytes} bytes of host memory")) },
    }
}

impl Buffer {
    /// Host memory of `layout`, not zeroed: turbo.h promises no contents.
    pub(super) fn alloc(layout: Layout) -> Option<Buffer> {
        // A zero-size layout is not the allocator's to take: buffer_alloc
        // refuses one, and a session's output is never empty.
        if layout.size() == 0 {
            return None;
        }
        let ptr = unsafe { std::alloc::alloc(layout) };
        (!ptr.is_null()).then_some(Buffer { ptr, layout: Some(layout) })
    }
}

/// The caller's own pointer plus offset. Nothing is copied.
pub(super) unsafe extern "C" fn buffer_import(
    _ctx: *mut c_void,
    desc: *const turbo_buffer_desc,
    handle: *const turbo_native_handle,
    out: *mut *mut c_void,
    host: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    let (desc, h) = unsafe { (&*desc, &*handle) };
    if let Some(rc) = unsafe { host_only(desc, err) } {
        return rc;
    }
    if h.kind != TURBO_HANDLE_HOST_PTR {
        let m = format!("kind: {} is not TURBO_HANDLE_HOST_PTR, the one kind the cpu imports", h.kind);
        return unsafe { refuse(err, UNSUPPORTED, &m) };
    }
    if h.handle == 0 {
        return unsafe { refuse(err, INVALID_ARGUMENT, "handle: a NULL host pointer") };
    }
    let end = h.handle.checked_add(h.offset).and_then(|a| a.checked_add(desc.bytes));
    if end.is_none_or(|e| usize::try_from(e).is_err()) {
        let m =
            format!("handle {:#x} + offset {} + {} bytes is past the address space", h.handle, h.offset, desc.bytes);
        return unsafe { refuse(err, INVALID_ARGUMENT, &m) };
    }
    let ptr = (h.handle as usize as *mut u8).wrapping_add(h.offset as usize);
    unsafe { give(Buffer { ptr, layout: None }, out, host) }
}

pub(super) unsafe extern "C" fn buffer_release(buf: *mut c_void) {
    drop(unsafe { Box::from_raw(buf as *mut Buffer) });
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(layout) = self.layout {
            unsafe { std::alloc::dealloc(self.ptr, layout) };
        }
    }
}

/// The buffer's own host address; no copy.
pub(super) unsafe extern "C" fn buffer_export(
    buf: *mut c_void,
    kind: u32,
    out: *mut turbo_native_handle,
    err: *mut turbo_error,
) -> i32 {
    if kind != TURBO_HANDLE_HOST_PTR {
        let m = format!("kind: {kind} is not TURBO_HANDLE_HOST_PTR, the one kind the cpu exports");
        return unsafe { refuse(err, UNSUPPORTED, &m) };
    }
    let (buf, out) = unsafe { (&*(buf as *const Buffer), &mut *out) };
    out.kind = TURBO_HANDLE_HOST_PTR;
    out.handle = buf.ptr as usize as u64;
    out.aux = 0;
    out.offset = 0;
    0
}
