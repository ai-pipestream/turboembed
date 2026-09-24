//! The CPU backend: the host processor, through turbo_backend.h like any
//! other backend. This cut lists the device and its memory; no task runs
//! on it yet.

use std::alloc::Layout;
use std::ffi::{c_char, c_void};

use crate::backend::{TURBO_CAP_UNSUPPORTED, refuse, turbo_backend};
use crate::status::{INVALID_ARGUMENT, OUT_OF_MEMORY, UNSUPPORTED};
use crate::{
    TURBO_DEVICE_CPU, TURBO_HANDLE_HOST_PTR, TURBO_PLACE_DEVICE, turbo_buffer_desc, turbo_device_info, turbo_error,
    turbo_log_fn, turbo_native_handle, write_str,
};

pub static BACKEND: turbo_backend = turbo_backend {
    struct_size: size_of::<turbo_backend>() as u32,
    reserved: 0,
    name: c"cpu".as_ptr(),
    runtime_version: c"".as_ptr(),
    device_count,
    device_info,
    capability,
    context_create: Some(context_create),
    context_release: Some(context_release),
    buffer_alloc: Some(buffer_alloc),
    buffer_import: Some(buffer_import),
    buffer_release: Some(buffer_release),
    buffer_export: Some(buffer_export),
};

unsafe extern "C" fn device_count(out: *mut u32, _err: *mut turbo_error) -> i32 {
    unsafe { *out = 1 };
    0
}

/// The host is the one device this backend lists.
unsafe fn only(ordinal: u32, err: *mut turbo_error) -> Option<i32> {
    (ordinal != 0).then(|| unsafe { refuse(err, INVALID_ARGUMENT, &format!("cpu device {ordinal}: only 0 is listed")) })
}

unsafe extern "C" fn device_info(ordinal: u32, out: *mut turbo_device_info, err: *mut turbo_error) -> i32 {
    if let Some(rc) = unsafe { only(ordinal, err) } {
        return rc;
    }
    let out = unsafe { &mut *out };
    let host = Host::read();
    out.kind = TURBO_DEVICE_CPU;
    out.ordinal = 0;
    out.unified_memory = 1;
    out.memory_total = host.memory_total;
    out.memory_free = host.memory_free;
    // The instruction set, not the processor: arch[32] cannot hold a model
    // name. A CPU benchmark record is filed under arch and name together,
    // so a record for one x86_64 processor backs no other.
    write_str(&mut out.arch, std::env::consts::ARCH);
    write_str(&mut out.name, &host.name);
    write_str(&mut out.vendor, &host.vendor);
    write_str(&mut out.runtime_version, "");
    write_str(&mut out.driver_version, "");
    0
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn capability(
    ordinal: u32,
    _task: u32,
    _precision: u32,
    status: *mut u32,
    dtype: *mut u32,
    options_honored: *mut u32,
    reason: *mut c_char,
    reason_len: u32,
    err: *mut turbo_error,
) -> i32 {
    if let Some(rc) = unsafe { only(ordinal, err) } {
        return rc;
    }
    unsafe {
        *status = TURBO_CAP_UNSUPPORTED;
        *dtype = 0;
        *options_honored = 0;
        let r = std::slice::from_raw_parts_mut(reason, reason_len as usize);
        write_str(r, "the cpu backend has no embed kernels in this build");
    }
    0
}

// ---- Contexts and buffers ----------------------------------------------
//
// The CPU has no memory apart from the host's. HOST, PINNED and SHARED are
// all pageable host memory here: there is no second device to pin for or
// to share with. DEVICE is refused rather than read as host memory, so a
// caller that asked for memory off the host is told it did not get it.

/// Every allocation starts on a 64-byte boundary: a cache line, and the
/// width of an AVX-512 register.
const ALIGN: usize = 64;

/// The host keeps nothing per context and has nothing to warn about.
struct Context;

struct Buffer {
    ptr: *mut u8,
    /// The allocation to free; None for memory the caller owns.
    layout: Option<Layout>,
}

unsafe extern "C" fn context_create(
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

unsafe extern "C" fn context_release(ctx: *mut c_void) {
    drop(unsafe { Box::from_raw(ctx as *mut Context) });
}

/// DEVICE placement, refused: see above.
unsafe fn host_only(desc: &turbo_buffer_desc, err: *mut turbo_error) -> Option<i32> {
    (desc.placement == TURBO_PLACE_DEVICE).then(|| unsafe {
        refuse(err, UNSUPPORTED, "placement: TURBO_PLACE_DEVICE: the cpu has no memory apart from the host's")
    })
}

unsafe fn give(buf: Buffer, out: *mut *mut c_void, host: *mut *mut c_void) -> i32 {
    unsafe {
        *host = buf.ptr as *mut c_void;
        *out = Box::into_raw(Box::new(buf)) as *mut c_void;
    }
    0
}

unsafe extern "C" fn buffer_alloc(
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
    let Ok(layout) = Layout::from_size_align(bytes, ALIGN) else {
        return too_big();
    };
    // Not zeroed: turbo.h promises no contents, and the caller writes them.
    let ptr = unsafe { std::alloc::alloc(layout) };
    if ptr.is_null() {
        return unsafe { refuse(err, OUT_OF_MEMORY, &format!("{bytes} bytes of host memory")) };
    }
    unsafe { give(Buffer { ptr, layout: Some(layout) }, out, host) }
}

/// The caller's own pointer plus offset. Nothing is copied.
unsafe extern "C" fn buffer_import(
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

unsafe extern "C" fn buffer_release(buf: *mut c_void) {
    let buf = unsafe { Box::from_raw(buf as *mut Buffer) };
    if let Some(layout) = buf.layout {
        unsafe { std::alloc::dealloc(buf.ptr, layout) };
    }
}

/// The buffer's own host address; no copy.
unsafe extern "C" fn buffer_export(
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

/// What the operating system says about the processor and memory. A value
/// it does not report is empty or 0, which turbo.h reads as unknown.
#[derive(Default)]
struct Host {
    name: String,
    vendor: String,
    memory_total: u64,
    memory_free: u64,
}

impl Host {
    #[cfg(target_os = "linux")]
    fn read() -> Host {
        let mut h = Host::default();
        let field = |text: &str, key: &str| {
            text.lines()
                .find_map(|l| l.split_once(':').filter(|(k, _)| k.trim() == key).map(|(_, v)| v.trim().to_owned()))
        };
        if let Ok(cpu) = std::fs::read_to_string("/proc/cpuinfo") {
            // x86 names the model; Arm cores often do not, so fall back to
            // the implementer and part the kernel reports.
            h.name = field(&cpu, "model name").or_else(|| field(&cpu, "Model")).unwrap_or_default();
            h.vendor = field(&cpu, "vendor_id").or_else(|| field(&cpu, "CPU implementer")).unwrap_or_default();
        }
        if let Ok(mem) = std::fs::read_to_string("/proc/meminfo") {
            let kib = |key| field(&mem, key).and_then(|v| v.trim_end_matches("kB").trim().parse::<u64>().ok());
            h.memory_total = kib("MemTotal").map_or(0, |k| k * 1024);
            h.memory_free = kib("MemAvailable").map_or(0, |k| k * 1024);
        }
        h
    }

    #[cfg(not(target_os = "linux"))]
    fn read() -> Host {
        Host::default()
    }
}
