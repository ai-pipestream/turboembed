//! Buffers in each placement, imported, exported and read back.

use std::alloc::Layout;
use std::ffi::c_void;

use super::*;
use crate::levelzero::ze::{self};
use crate::status::{INVALID_ARGUMENT, OUT_OF_MEMORY, UNSUPPORTED};
use crate::{
    TURBO_HANDLE_HOST_PTR, TURBO_HANDLE_ZE_USM, TURBO_PLACE_DEVICE, TURBO_PLACE_HOST, TURBO_PLACE_PINNED,
    TURBO_PLACE_SHARED, turbo_buffer_desc, turbo_error, turbo_native_handle,
};

// ---- Buffers ---------------------------------------------------------------------

/// Host buffers start on a 64-byte boundary, as on the CPU backend.
pub(super) const HOST_ALIGN: usize = 64;

pub(crate) struct Buffer {
    pub ctx: *const Context,
    pub ptr: *mut c_void,
    pub placement: u32,
    pub bytes: u64,
    /// Memory this context's driver allocated: exportable as a USM
    /// pointer.
    pub(super) usm: bool,
    pub(super) free: Free,
}

pub(super) enum Free {
    /// The caller's memory, or a session's output, which the session frees.
    Nothing,
    Driver,
    Host(Layout),
}

pub(super) fn placement_name(p: u32) -> &'static str {
    match p {
        TURBO_PLACE_HOST => "HOST",
        TURBO_PLACE_PINNED => "PINNED",
        TURBO_PLACE_DEVICE => "DEVICE",
        _ => "SHARED",
    }
}

pub(super) fn handle_name(k: u32) -> String {
    match k {
        TURBO_HANDLE_HOST_PTR => "TURBO_HANDLE_HOST_PTR".into(),
        TURBO_HANDLE_ZE_USM => "TURBO_HANDLE_ZE_USM".into(),
        crate::TURBO_HANDLE_CUDA_PTR => "TURBO_HANDLE_CUDA_PTR".into(),
        crate::TURBO_HANDLE_CL_MEM => "TURBO_HANDLE_CL_MEM".into(),
        crate::TURBO_HANDLE_MTL_BUFFER => "TURBO_HANDLE_MTL_BUFFER".into(),
        crate::TURBO_HANDLE_DMABUF_FD => "TURBO_HANDLE_DMABUF_FD".into(),
        k => format!("{k}"),
    }
}

pub(super) fn memory_type_name(t: u32) -> &'static str {
    match t {
        ze::MEMORY_TYPE_HOST => "host memory the driver allocated",
        ze::MEMORY_TYPE_DEVICE => "device memory",
        ze::MEMORY_TYPE_SHARED => "shared memory",
        _ => "memory this context's driver did not allocate",
    }
}

impl Buffer {
    /// A session's output: device memory the session owns.
    pub fn session_output(ctx: &Context, ptr: *mut c_void, bytes: u64) -> Buffer {
        Buffer { ctx, ptr, placement: TURBO_PLACE_DEVICE, bytes, usm: true, free: Free::Nothing }
    }
}

pub(super) unsafe fn give(buf: Buffer, out: *mut *mut c_void, host: *mut *mut c_void) {
    unsafe {
        *host = if buf.placement == TURBO_PLACE_DEVICE { std::ptr::null_mut() } else { buf.ptr };
        *out = Box::into_raw(Box::new(buf)) as *mut c_void;
    }
}

pub(crate) unsafe extern "C" fn buffer_alloc(
    ctx: *mut c_void,
    desc: *const turbo_buffer_desc,
    out: *mut *mut c_void,
    host: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let (c, desc) = (&*(ctx as *const Context), &*desc);
            let too_big = || fail(OUT_OF_MEMORY, format!("{} bytes is more than the host can address", desc.bytes));
            let bytes = usize::try_from(desc.bytes).map_err(|_| too_big())?;
            // Not zeroed: turbo.h promises no contents, and the caller
            // writes them.
            let (ptr, free) = match desc.placement {
                TURBO_PLACE_DEVICE => (c.alloc_device(bytes)?, Free::Driver),
                TURBO_PLACE_PINNED => (c.alloc_pinned(bytes)?, Free::Driver),
                TURBO_PLACE_SHARED => (c.alloc_shared(bytes)?, Free::Driver),
                _ => {
                    let layout = Layout::from_size_align(bytes.max(1), HOST_ALIGN).map_err(|_| too_big())?;
                    let p = std::alloc::alloc(layout);
                    if p.is_null() {
                        return Err(fail(OUT_OF_MEMORY, format!("{bytes} bytes of HOST memory")));
                    }
                    (p as *mut c_void, Free::Host(layout))
                }
            };
            let usm = matches!(free, Free::Driver);
            give(Buffer { ctx: c, ptr, placement: desc.placement, bytes: desc.bytes, usm, free }, out, host);
            Ok(())
        })
    }
}

/// The caller's memory, wrapped: nothing is copied, nothing is freed. A
/// TURBO_HANDLE_ZE_USM names memory this context's driver allocated, with
/// the context's handle as aux; a TURBO_HANDLE_HOST_PTR names host memory,
/// which for PINNED is host memory the driver allocated in this context
/// and for SHARED shared memory.
pub(crate) unsafe extern "C" fn buffer_import(
    ctx: *mut c_void,
    desc: *const turbo_buffer_desc,
    handle: *const turbo_native_handle,
    out: *mut *mut c_void,
    host: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let (c, desc, h) = (&*(ctx as *const Context), &*desc, &*handle);
            if h.kind != TURBO_HANDLE_ZE_USM && h.kind != TURBO_HANDLE_HOST_PTR {
                return Err(fail(
                    UNSUPPORTED,
                    format!(
                        "kind: {}: the levelzero backend imports TURBO_HANDLE_ZE_USM and TURBO_HANDLE_HOST_PTR",
                        handle_name(h.kind)
                    ),
                ));
            }
            if h.handle == 0 {
                return Err(fail(INVALID_ARGUMENT, "handle: a NULL pointer"));
            }
            let end = h.handle.checked_add(h.offset).filter(|e| e.checked_add(desc.bytes).is_some());
            let Some(start) = end.filter(|&e| usize::try_from(e + desc.bytes).is_ok()) else {
                return Err(fail(
                    INVALID_ARGUMENT,
                    format!(
                        "handle {:#x} + offset {} + {} bytes is past the address space",
                        h.handle, h.offset, desc.bytes
                    ),
                ));
            };
            let ptr = start as usize as *mut c_void;
            let pl = desc.placement;
            let (kind, dev) = c.memory_type(ptr);
            let not = |what: &str| fail(INVALID_ARGUMENT, what.to_owned());
            if h.kind == TURBO_HANDLE_ZE_USM {
                if h.aux != c.handle as usize as u64 {
                    return Err(not(
                        "aux: not this context's ze_context_handle_t; a USM pointer is valid only in its own context",
                    ));
                }
                let fits = match pl {
                    TURBO_PLACE_DEVICE => {
                        (kind == ze::MEMORY_TYPE_DEVICE && dev == c.device) || kind == ze::MEMORY_TYPE_SHARED
                    }
                    TURBO_PLACE_SHARED => kind == ze::MEMORY_TYPE_SHARED,
                    TURBO_PLACE_PINNED => kind == ze::MEMORY_TYPE_HOST,
                    _ => kind == ze::MEMORY_TYPE_HOST || kind == ze::MEMORY_TYPE_SHARED,
                };
                if !fits {
                    let m = if kind == ze::MEMORY_TYPE_DEVICE && dev != c.device {
                        "handle: memory of another device than this context's".to_owned()
                    } else {
                        format!("handle: the pointer is {}, not {} memory", memory_type_name(kind), placement_name(pl))
                    };
                    return Err(not(&m));
                }
            } else {
                let refused = match pl {
                    TURBO_PLACE_DEVICE => Some(
                        "placement: DEVICE: a TURBO_HANDLE_HOST_PTR is host memory; import device memory as TURBO_HANDLE_ZE_USM"
                            .to_owned(),
                    ),
                    TURBO_PLACE_PINNED if kind != ze::MEMORY_TYPE_HOST => Some(format!(
                        "placement: PINNED: the pointer is {}, not host memory the driver allocated (import it as HOST)",
                        memory_type_name(kind)
                    )),
                    TURBO_PLACE_SHARED if kind != ze::MEMORY_TYPE_SHARED => Some(format!(
                        "placement: SHARED: the pointer is {}, not shared memory",
                        memory_type_name(kind)
                    )),
                    _ if kind == ze::MEMORY_TYPE_DEVICE => {
                        Some("handle: the pointer is device memory, not host memory".to_owned())
                    }
                    _ => None,
                };
                if let Some(m) = refused {
                    return Err(not(&m));
                }
            }
            let usm = kind != ze::MEMORY_TYPE_UNKNOWN;
            give(Buffer { ctx: c, ptr, placement: pl, bytes: desc.bytes, usm, free: Free::Nothing }, out, host);
            Ok(())
        })
    }
}

pub(crate) unsafe extern "C" fn buffer_release(buf: *mut c_void) {
    quietly(|| {
        let b = unsafe { Box::from_raw(buf as *mut Buffer) };
        match b.free {
            Free::Nothing => {}
            Free::Host(layout) => unsafe { std::alloc::dealloc(b.ptr as *mut u8, layout) },
            Free::Driver => {
                let c = unsafe { &*b.ctx };
                c.free(b.ptr, &format!("a {} buffer of {} bytes", placement_name(b.placement), b.bytes));
            }
        }
    });
}

/// The buffer's own address, no copy: a USM pointer, with the context's
/// handle as aux, for memory this context's driver allocated; a host
/// pointer for HOST, PINNED and SHARED.
pub(crate) unsafe extern "C" fn buffer_export(
    buf: *mut c_void,
    kind: u32,
    out: *mut turbo_native_handle,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let (b, out) = (&*(buf as *const Buffer), &mut *out);
            let pl = placement_name(b.placement);
            match kind {
                TURBO_HANDLE_ZE_USM if !b.usm => Err(fail(
                    UNSUPPORTED,
                    format!(
                        "kind: TURBO_HANDLE_ZE_USM: this {pl} buffer is memory the driver did not allocate; export it as TURBO_HANDLE_HOST_PTR"
                    ),
                )),
                TURBO_HANDLE_HOST_PTR if b.placement == TURBO_PLACE_DEVICE => Err(fail(
                    UNSUPPORTED,
                    "kind: TURBO_HANDLE_HOST_PTR: a DEVICE buffer has no host address; export it as TURBO_HANDLE_ZE_USM",
                )),
                TURBO_HANDLE_ZE_USM | TURBO_HANDLE_HOST_PTR => {
                    out.kind = kind;
                    out.handle = b.ptr as usize as u64;
                    out.aux = if kind == TURBO_HANDLE_ZE_USM { (*b.ctx).handle as usize as u64 } else { 0 };
                    out.offset = 0;
                    Ok(())
                }
                _ => Err(fail(
                    UNSUPPORTED,
                    format!(
                        "kind: {}: the levelzero backend exports TURBO_HANDLE_ZE_USM and TURBO_HANDLE_HOST_PTR",
                        handle_name(kind)
                    ),
                )),
            }
        })
    }
}

/// A copy to the caller's host memory, waited for. Whatever wrote the
/// buffer has finished: a run synchronizes before it returns.
pub(crate) unsafe extern "C" fn buffer_read(
    buf: *mut c_void,
    dst: *mut c_void,
    bytes: u64,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let b = &*(buf as *const Buffer);
            if bytes > b.bytes {
                return Err(fail(INVALID_ARGUMENT, format!("{bytes} bytes from a buffer of {}", b.bytes)));
            }
            let c = &*b.ctx;
            let mut q = c.lock_queue()?;
            let copied = c.copy(&mut q, dst, b.ptr, bytes as usize);
            let synced = c.sync(&mut q);
            copied?;
            synced
        })
    }
}
