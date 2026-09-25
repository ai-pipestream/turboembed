//! Contexts and buffers on a listed GPU.
//!
//! A context is a Level Zero context on the device and one in-order
//! immediate command list, used under the context's lock: a run holds it
//! from its first launch to the list's synchronization. The encoder's
//! kernels are built from SPIR-V into a module the first time a model or
//! session needs them, and shared by everything on the context.
//!
//! Placements: DEVICE is zeMemAllocDevice memory, with no host address;
//! PINNED is host memory the driver allocated (zeMemAllocHost), which the
//! device reads directly; HOST is pageable, 64-byte aligned host memory;
//! SHARED is zeMemAllocShared memory, one address for both.

use std::alloc::Layout;
use std::cell::Cell;
use std::ffi::{CString, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::ze::{self, ComputeApi, Handle};
use crate::backend::{refuse, refuse_field};
use crate::status::{INVALID_ARGUMENT, OUT_OF_MEMORY, PANIC, RUNTIME, UNSUPPORTED};
use crate::{
    TURBO_HANDLE_HOST_PTR, TURBO_HANDLE_ZE_USM, TURBO_PLACE_DEVICE, TURBO_PLACE_HOST, TURBO_PLACE_PINNED,
    TURBO_PLACE_SHARED, turbo_buffer_desc, turbo_error, turbo_log_fn, turbo_native_handle, turbo_text,
};

/// The encoder's kernels, compiled from core/levelzero/encoder.cl.
static SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/levelzero_encoder.spv"));

// ---- Failures ------------------------------------------------------------------

/// A refusal on its way to the caller's turbo_error.
pub(crate) struct Fail {
    pub code: i32,
    pub field: u32,
    pub message: String,
}

pub(crate) type Res<T> = Result<T, Fail>;

pub(crate) fn fail(code: i32, message: impl Into<String>) -> Fail {
    Fail { code, field: 0, message: message.into() }
}

pub(crate) fn fail_field(code: i32, field: u32, message: impl Into<String>) -> Fail {
    Fail { code, field, message: message.into() }
}

/// A Level Zero status as a refusal: running out of memory is
/// OUT_OF_MEMORY, anything else RUNTIME with the call and its code.
pub(crate) fn ze(what: &str, rc: ze::Status) -> Res<()> {
    match rc {
        0 => Ok(()),
        ze::RESULT_ERROR_OUT_OF_DEVICE_MEMORY | ze::RESULT_ERROR_OUT_OF_HOST_MEMORY => {
            Err(fail(OUT_OF_MEMORY, format!("levelzero: {what}: out of memory (0x{rc:08x})")))
        }
        _ => Err(fail(RUNTIME, format!("levelzero: {what} failed with 0x{rc:08x}"))),
    }
}

/// Runs a table entry's body, turning a refusal into the caller's
/// turbo_error and a panic into TURBO_E_PANIC: nothing unwinds across the
/// table.
///
/// # Safety
/// `err` is NULL or valid for the call.
pub(crate) unsafe fn guarded(err: *mut turbo_error, f: impl FnOnce() -> Res<()>) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => unsafe { refuse_field(err, e.code, e.field, &e.message) },
        Err(_) => unsafe { refuse(err, PANIC, "a panic inside the levelzero backend") },
    }
}

/// A release entry's body: it returns nothing, and a panic stops here.
pub(crate) fn quietly(f: impl FnOnce()) {
    let _ = catch_unwind(AssertUnwindSafe(f));
}

// ---- Allocation counts -----------------------------------------------------------
//
// Every device allocation the backend makes goes through Context's alloc
// functions, which count it twice: in the process's total, which the tests
// read, and in the calling thread's own count, whose change across
// session_run is what a result reports. Driver allocations, host ones from
// zeMemAllocHost included, count as device allocations.

static DEVICE_TOTAL: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static DEVICE_HERE: Cell<u64> = const { Cell::new(0) };
}

/// Every device allocation this backend has made in the process.
#[cfg(feature = "internals")]
pub(crate) fn device_allocs_total() -> u64 {
    DEVICE_TOTAL.load(Ordering::Relaxed)
}

pub(crate) fn device_allocs_here() -> u64 {
    DEVICE_HERE.with(Cell::get)
}

fn counted_device() {
    DEVICE_TOTAL.fetch_add(1, Ordering::Relaxed);
    DEVICE_HERE.with(|c| c.set(c.get() + 1));
}

// ---- Contexts ----------------------------------------------------------------------

const LOG_WARNING: u32 = 1;
pub(crate) const LOG_DEBUG: u32 = 3;

/// A handle the driver lets any thread use.
#[derive(Clone, Copy)]
pub(crate) struct Shared(pub Handle);

unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

pub(crate) struct Context {
    pub api: &'static ComputeApi,
    pub ordinal: u32,
    pub device: Handle,
    pub handle: Handle,
    /// The in-order immediate command list, and the lock every use of it
    /// takes.
    pub queue: Mutex<Shared>,
    /// The encoder's module, built on first need; the failure is kept, so
    /// a device that cannot build it says why each time.
    module: Mutex<Option<Result<Shared, String>>>,
    /// The most local memory one work-group may have.
    pub max_local: u32,
    /// An append failed since the queue last synchronized.
    wedged: AtomicBool,
    log: turbo_log_fn,
    log_user_data: *mut c_void,
}

// The handles are the driver's, for any thread; the log function is the
// caller's, which turbo.h says may be called from any thread.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    pub fn say(&self, level: u32, message: &str) {
        if let Some(f) = self.log {
            let t = turbo_text { ptr: message.as_ptr() as *const _, len: message.len() as u64 };
            unsafe { f(self.log_user_data, level, t) };
        }
    }

    /// The encoder's module, built for this device the first time.
    pub fn module(&self) -> Res<Handle> {
        let mut m = self.module.lock().unwrap_or_else(|p| p.into_inner());
        let built = m.get_or_insert_with(|| self.build().map(Shared));
        built.as_ref().map(|s| s.0).map_err(|e| fail(RUNTIME, e.clone()))
    }

    fn build(&self) -> Result<Handle, String> {
        let flags = c"";
        let desc = ze::ModuleDesc {
            stype: ze::STRUCTURE_TYPE_MODULE_DESC,
            p_next: std::ptr::null(),
            format: ze::MODULE_FORMAT_IL_SPIRV,
            input_size: SPIRV.len(),
            input: SPIRV.as_ptr(),
            build_flags: flags.as_ptr(),
            constants: std::ptr::null(),
        };
        let (mut handle, mut log) = (std::ptr::null_mut(), std::ptr::null_mut());
        let rc = unsafe { (self.api.module_create)(self.handle, self.device, &desc, &mut handle, &mut log) };
        let text = self.build_log(log);
        if rc != 0 {
            return Err(format!("levelzero: building the encoder's kernels failed with 0x{rc:08x}: {text}"));
        }
        self.say(LOG_DEBUG, &format!("levelzero device {}: the encoder's kernels are built", self.ordinal));
        Ok(handle)
    }

    /// The build log's text, released after reading. Empty when there is
    /// none.
    fn build_log(&self, log: Handle) -> String {
        if log.is_null() {
            return String::new();
        }
        let mut n = 0usize;
        let mut text = String::new();
        if unsafe { (self.api.module_build_log_get_string)(log, &mut n, std::ptr::null_mut()) } == 0 && n > 0 {
            let mut buf = vec![0 as std::ffi::c_char; n];
            if unsafe { (self.api.module_build_log_get_string)(log, &mut n, buf.as_mut_ptr()) } == 0 {
                text = crate::backend::cstr(&buf);
            }
        }
        unsafe { (self.api.module_build_log_destroy)(log) };
        text
    }

    /// A kernel from the encoder's module, with its work-group size set.
    pub fn kernel(&self, name: &str, group: [u32; 3]) -> Res<Kernel> {
        let module = self.module()?;
        let cname = CString::new(name).map_err(|_| fail(RUNTIME, "a kernel name holds a NUL"))?;
        let desc = ze::KernelDesc {
            stype: ze::STRUCTURE_TYPE_KERNEL_DESC,
            p_next: std::ptr::null(),
            flags: 0,
            name: cname.as_ptr(),
        };
        let mut handle = std::ptr::null_mut();
        ze(&format!("zeKernelCreate({name})"), unsafe { (self.api.kernel_create)(module, &desc, &mut handle) })?;
        let k = Kernel { api: self.api, handle };
        ze(&format!("zeKernelSetGroupSize({name})"), unsafe {
            (self.api.kernel_set_group_size)(handle, group[0], group[1], group[2])
        })?;
        Ok(k)
    }

    /// `bytes` of device memory, counted.
    pub fn alloc_device(&self, bytes: usize) -> Res<*mut c_void> {
        let desc = ze::DeviceMemAllocDesc {
            stype: ze::STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            p_next: std::ptr::null(),
            flags: 0,
            ordinal: 0,
        };
        let mut ptr = std::ptr::null_mut();
        let rc = unsafe { (self.api.mem_alloc_device)(self.handle, &desc, bytes.max(1), 256, self.device, &mut ptr) };
        ze(&format!("{bytes} bytes of DEVICE memory"), rc)?;
        counted_device();
        Ok(ptr)
    }

    /// `bytes` of host memory the device reads directly, counted.
    pub fn alloc_pinned(&self, bytes: usize) -> Res<*mut c_void> {
        let desc =
            ze::HostMemAllocDesc { stype: ze::STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, p_next: std::ptr::null(), flags: 0 };
        let mut ptr = std::ptr::null_mut();
        let rc = unsafe { (self.api.mem_alloc_host)(self.handle, &desc, bytes.max(1), 64, &mut ptr) };
        ze(&format!("{bytes} bytes of PINNED memory"), rc)?;
        counted_device();
        Ok(ptr)
    }

    fn alloc_shared(&self, bytes: usize) -> Res<*mut c_void> {
        let d = ze::DeviceMemAllocDesc {
            stype: ze::STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            p_next: std::ptr::null(),
            flags: 0,
            ordinal: 0,
        };
        let h =
            ze::HostMemAllocDesc { stype: ze::STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, p_next: std::ptr::null(), flags: 0 };
        let mut ptr = std::ptr::null_mut();
        let rc = unsafe { (self.api.mem_alloc_shared)(self.handle, &d, &h, bytes.max(1), 64, self.device, &mut ptr) };
        ze(&format!("{bytes} bytes of SHARED memory"), rc)?;
        counted_device();
        Ok(ptr)
    }

    /// Frees memory from one of the alloc functions, warning through the
    /// log when the driver refuses.
    pub fn free(&self, ptr: *mut c_void, what: &str) {
        let rc = unsafe { (self.api.mem_free)(self.handle, ptr) };
        if rc != 0 {
            self.say(LOG_WARNING, &format!("levelzero backend: freeing {what}: zeMemFree failed with 0x{rc:08x}"));
        }
    }

    /// What the driver says an address is in this context, and the device
    /// it is on: MEMORY_TYPE_UNKNOWN for memory it did not allocate here.
    pub fn memory_type(&self, ptr: *const c_void) -> (u32, Handle) {
        let mut p = ze::MemoryAllocationProperties {
            stype: ze::STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
            ..Default::default()
        };
        let mut dev = std::ptr::null_mut();
        match unsafe { (self.api.mem_get_alloc_properties)(self.handle, ptr, &mut p, &mut dev) } {
            0 => (p.kind, dev),
            _ => (ze::MEMORY_TYPE_UNKNOWN, std::ptr::null_mut()),
        }
    }

    /// Appends a copy to the queue, which the caller holds.
    ///
    /// # Safety
    /// Both ranges stay valid until the queue has synchronized.
    pub unsafe fn copy(&self, queue: Handle, dst: *mut c_void, src: *const c_void, bytes: usize) -> Res<()> {
        let (no_event, no_waits) = (std::ptr::null_mut(), std::ptr::null_mut());
        let rc = unsafe { (self.api.command_list_append_memory_copy)(queue, dst, src, bytes, no_event, 0, no_waits) };
        self.appended("zeCommandListAppendMemoryCopy", rc)
    }

    /// An append's status. A failed append leaves the driver's immediate
    /// list unable to synchronize or be destroyed, so sync replaces it.
    fn appended(&self, what: &str, rc: ze::Status) -> Res<()> {
        if rc != 0 {
            self.wedged.store(true, Ordering::Relaxed);
        }
        ze(what, rc)
    }

    /// Waits for everything appended to the queue, which the caller holds.
    /// After a failed append the list never finishes: the device is waited
    /// for as a whole instead, and the list is set aside for a new one.
    pub fn sync(&self, queue: &mut Shared) -> Res<()> {
        if !self.wedged.swap(false, Ordering::Relaxed) {
            return ze("zeCommandListHostSynchronize", unsafe {
                (self.api.command_list_host_synchronize)(queue.0, u64::MAX)
            });
        }
        ze("zeContextSystemBarrier", unsafe { (self.api.context_system_barrier)(self.handle, self.device) })?;
        self.say(
            LOG_WARNING,
            &format!("levelzero device {}: an append failed; the queue is set aside for a new one", self.ordinal),
        );
        // Destroying the old list would wait for it forever; it is left.
        queue.0 = std::ptr::null_mut();
        let mut list = std::ptr::null_mut();
        ze("zeCommandListCreateImmediate", unsafe {
            (self.api.command_list_create_immediate)(self.handle, self.device, &queue_desc(), &mut list)
        })?;
        queue.0 = list;
        Ok(())
    }

    /// The queue, which is null only when replacing it failed.
    pub fn lock_queue(&self) -> Res<std::sync::MutexGuard<'_, Shared>> {
        let q = self.queue.lock().unwrap_or_else(|p| p.into_inner());
        if q.0.is_null() {
            return Err(fail(RUNTIME, format!("levelzero device {}: the context's queue is gone", self.ordinal)));
        }
        Ok(q)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let module = self.module.get_mut().unwrap_or_else(|p| p.into_inner());
        if let Some(Ok(m)) = module.take() {
            unsafe { (self.api.module_destroy)(m.0) };
        }
        let queue = self.queue.get_mut().unwrap_or_else(|p| p.into_inner());
        if !queue.0.is_null() {
            unsafe { (self.api.command_list_destroy)(queue.0) };
        }
        unsafe { (self.api.context_destroy)(self.handle) };
    }
}

/// A kernel object: its arguments are its own state, so each session
/// creates its own. Arguments are captured when a launch is appended.
pub(crate) struct Kernel {
    api: &'static ComputeApi,
    pub handle: Handle,
}

unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { (self.api.kernel_destroy)(self.handle) };
    }
}

/// A kernel argument.
#[derive(Clone, Copy)]
pub(crate) enum Arg {
    /// A device address.
    Ptr(u64),
    I32(i32),
    F32(f32),
    U64(u64),
    /// Local memory of this many bytes.
    Local(usize),
}

impl Kernel {
    /// The local memory the driver gives the kernel itself, beyond what its
    /// arguments ask for.
    pub fn local_bytes(&self) -> Res<u32> {
        let mut p = ze::KernelProperties { stype: ze::STRUCTURE_TYPE_KERNEL_PROPERTIES, ..Default::default() };
        ze("zeKernelGetProperties", unsafe { (self.api.kernel_get_properties)(self.handle, &mut p) })?;
        Ok(p.local_mem_size)
    }

    /// Sets the arguments and appends a launch of `groups` work-groups to
    /// the queue, which the caller holds. Allocates nothing.
    pub fn launch(&self, c: &Context, queue: Handle, name: &str, args: &[Arg], groups: [u32; 3]) -> Res<()> {
        for (i, a) in args.iter().enumerate() {
            let (n, p) = match a {
                Arg::Ptr(v) | Arg::U64(v) => (8, v as *const u64 as *const c_void),
                Arg::I32(v) => (4, v as *const i32 as *const c_void),
                Arg::F32(v) => (4, v as *const f32 as *const c_void),
                Arg::Local(n) => (*n, std::ptr::null()),
            };
            let rc = unsafe { (self.api.kernel_set_argument_value)(self.handle, i as u32, n, p) };
            if rc != 0 {
                return ze(name, rc);
            }
        }
        let g = ze::GroupCount { x: groups[0], y: groups[1], z: groups[2] };
        let (no_event, no_waits) = (std::ptr::null_mut(), std::ptr::null_mut());
        let rc = unsafe { (self.api.command_list_append_launch_kernel)(queue, self.handle, &g, no_event, 0, no_waits) };
        c.appended(name, rc)
    }
}

/// A context on the listed device `ordinal`.
///
/// # Safety
/// `out` is valid for the call.
pub(crate) unsafe fn create(
    driver: &'static super::Driver,
    dev: &'static super::Device,
    ordinal: u32,
    log: turbo_log_fn,
    log_user_data: *mut c_void,
    out: *mut *mut c_void,
) -> Res<()> {
    let api = &driver.api.compute;
    let desc = ze::ContextDesc { stype: ze::STRUCTURE_TYPE_CONTEXT_DESC, p_next: std::ptr::null(), flags: 0 };
    let mut handle = std::ptr::null_mut();
    ze("zeContextCreate", unsafe { (api.context_create)(dev.driver, &desc, &mut handle) })?;
    // From here the context's Drop releases what it holds.
    let mut ctx = Box::new(Context {
        api,
        ordinal,
        device: dev.handle,
        handle,
        queue: Mutex::new(Shared(std::ptr::null_mut())),
        module: Mutex::new(None),
        max_local: dev.max_local,
        wedged: AtomicBool::new(false),
        log,
        log_user_data,
    });
    let mut list = std::ptr::null_mut();
    ze("zeCommandListCreateImmediate", unsafe {
        (api.command_list_create_immediate)(handle, dev.handle, &queue_desc(), &mut list)
    })?;
    ctx.queue = Mutex::new(Shared(list));
    ctx.say(LOG_DEBUG, &format!("levelzero context on device {ordinal} ({}): one in-order queue", dev.name));
    unsafe { *out = Box::into_raw(ctx) as *mut c_void };
    Ok(())
}

/// An in-order immediate list on the device's first compute queue: work
/// runs in the order it is appended.
fn queue_desc() -> ze::CommandQueueDesc {
    ze::CommandQueueDesc {
        stype: ze::STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        p_next: std::ptr::null(),
        ordinal: 0,
        index: 0,
        flags: ze::COMMAND_QUEUE_FLAG_IN_ORDER,
        mode: ze::COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        priority: ze::COMMAND_QUEUE_PRIORITY_NORMAL,
    }
}

pub(crate) unsafe extern "C" fn context_release(ctx: *mut c_void) {
    quietly(|| drop(unsafe { Box::from_raw(ctx as *mut Context) }));
}

// ---- Buffers ---------------------------------------------------------------------

/// Host buffers start on a 64-byte boundary, as on the CPU backend.
const HOST_ALIGN: usize = 64;

pub(crate) struct Buffer {
    pub ctx: *const Context,
    pub ptr: *mut c_void,
    pub placement: u32,
    pub bytes: u64,
    /// Memory this context's driver allocated: exportable as a USM
    /// pointer.
    usm: bool,
    free: Free,
}

enum Free {
    /// The caller's memory, or a session's output, which the session frees.
    Nothing,
    Driver,
    Host(Layout),
}

fn placement_name(p: u32) -> &'static str {
    match p {
        TURBO_PLACE_HOST => "HOST",
        TURBO_PLACE_PINNED => "PINNED",
        TURBO_PLACE_DEVICE => "DEVICE",
        _ => "SHARED",
    }
}

fn handle_name(k: u32) -> String {
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

fn memory_type_name(t: u32) -> &'static str {
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

unsafe fn give(buf: Buffer, out: *mut *mut c_void, host: *mut *mut c_void) {
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
            let copied = c.copy(q.0, dst, b.ptr, bytes as usize);
            let synced = c.sync(&mut q);
            copied?;
            synced
        })
    }
}
