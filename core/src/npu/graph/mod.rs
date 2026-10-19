//! Contexts, buffers, models and embed sessions on a listed NPU.
//!
//! A context is a Level Zero context on the device and one in-order
//! immediate command list, used under the context's lock. A model is one
//! graph the driver's compiler built from the bundle's OpenVINO IR
//! (npu/ir.rs), initialized when it is loaded: that compile and
//! initialize are the load. A session owns host memory the device reads
//! and writes (zeMemAllocHost): one buffer per graph input, one for the
//! graph's hidden states, and the result vectors. A run binds the
//! session's buffers to the graph's arguments, executes it one frame of
//! fixed_batch rows at a time, and pools and normalizes on the host.
//! An INPUT_EMBEDDINGS graph does not take token ids. The host gathers
//! each token's word row and writes the attention bias the mask asks
//! for, and the graph owns everything after that gather. Argument
//! values live on the graph, so runs on one model are serialized under
//! the model's lock.

use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;

use super::ze::{self, GraphExt, Handle};
use super::{Device, Driver};
use crate::backend::{refuse, refuse_field};
use crate::status::{OUT_OF_MEMORY, PANIC, RUNTIME, UNSUPPORTED};
use crate::{turbo_error, turbo_log_fn, turbo_text};

mod accept;
mod args;
#[cfg(test)]
mod boundary_tests;
mod buffers;
mod host;
mod model;
#[cfg(test)]
mod pool_tests;
mod run;
mod session;
#[cfg(test)]
mod tests;
mod write;

pub(crate) use buffers::{buffer_alloc, buffer_export, buffer_release};
pub(crate) use model::{model_load, model_release};
pub(crate) use run::session_run;
pub(crate) use session::{session_create, session_create_tuned, session_open, session_release};
pub(crate) use write::{embed_write, embed_write_ranges};

// ---- Failures ----------------------------------------------------------------------

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

fn fail_field(code: i32, field: u32, message: impl Into<String>) -> Fail {
    Fail { code, field, message: message.into() }
}

/// How long a command list may be waited on. The wait holds the
/// context lock, so a device that never finishes must come back as an
/// error instead of blocking every model on the context.
const DEVICE_WAIT_NS: u64 = 30_000_000_000;

fn wait_for_device(api: &ze::Api, list: Handle, ordinal: u32, what: &str) -> Res<()> {
    let rc = unsafe { (api.command_list_host_synchronize)(list, DEVICE_WAIT_NS) };
    if rc == 0 {
        return Ok(());
    }
    Err(fail(
        RUNTIME,
        format!(
            "npu device {ordinal}: the device did not answer {what} within 30s (zeCommandListHostSynchronize 0x{rc:08x})"
        ),
    ))
}

/// A Level Zero status as a refusal: running out of memory is
/// OUT_OF_MEMORY, anything else RUNTIME with the call and its code.
pub(crate) fn ze_res(what: &str, rc: ze::Status) -> Res<()> {
    match rc {
        0 => Ok(()),
        ze::RESULT_ERROR_OUT_OF_DEVICE_MEMORY | ze::RESULT_ERROR_OUT_OF_HOST_MEMORY => {
            Err(fail(OUT_OF_MEMORY, format!("npu: {what}: out of memory (0x{rc:08x})")))
        }
        _ => Err(fail(RUNTIME, format!("npu: {what} failed with 0x{rc:08x}"))),
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
        Err(_) => unsafe { refuse(err, PANIC, "a panic inside the npu backend") },
    }
}

/// A release entry's body: it returns nothing, and a panic stops here.
fn quietly(f: impl FnOnce()) {
    let _ = catch_unwind(AssertUnwindSafe(f));
}

// ---- Contexts ------------------------------------------------------------------------

pub(crate) struct Context {
    api: &'static ze::Api,
    ext: GraphExt,
    device: Handle,
    handle: Handle,
    /// The in-order immediate command list, and the lock every use of it
    /// takes.
    list: Mutex<Shared>,
    compiler: ze::CompilerVersion,
    /// ze_graph_format_t bits and the highest OpenVINO opset the
    /// device's compiler takes, from the listing probe.
    formats_supported: u32,
    max_opset: u32,
    ordinal: u32,
    log: turbo_log_fn,
    log_user_data: *mut c_void,
}

/// A handle the driver lets any thread use.
#[derive(Clone, Copy)]
struct Shared(Handle);

unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

// The handles are the driver's, for any thread; the log function is the
// caller's, which turbo.h says may be called from any thread.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

const LOG_DEBUG: u32 = 3;

impl Context {
    fn say(&self, level: u32, message: &str) {
        if let Some(f) = self.log {
            let t = turbo_text { ptr: message.as_ptr() as *const _, len: message.len() as u64 };
            unsafe { f(self.log_user_data, level, t) };
        }
    }

    /// `bytes` of host memory the device reads directly.
    fn alloc_host(&self, bytes: usize, what: &str) -> Res<*mut c_void> {
        let desc =
            ze::HostMemAllocDesc { stype: ze::STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, p_next: std::ptr::null(), flags: 0 };
        let mut ptr = std::ptr::null_mut();
        let rc = unsafe { (self.api.mem_alloc_host)(self.handle, &desc, bytes.max(1), 64, &mut ptr) };
        ze_res(&format!("{bytes} bytes of host memory for {what}"), rc)?;
        Ok(ptr)
    }

    /// The immediate command list, held under its lock for the whole of
    /// `f`: an append and the synchronization that waits for it are one
    /// critical section, and the raw handle never leaves it.
    fn with_list<T>(&self, f: impl FnOnce(Handle) -> T) -> T {
        let guard = self.list.lock().unwrap_or_else(|p| p.into_inner());
        f(guard.0)
    }

    /// Runs the graph once over whatever its arguments point at, and
    /// returns when the device is done.
    fn execute(&self, graph: Handle) -> Res<()> {
        let execute = self
            .ext
            .append_graph_execute()
            .ok_or_else(|| fail(UNSUPPORTED, "npu: the driver's graph extension has no pfnAppendGraphExecute"))?;
        self.with_list(|list| {
            let null = std::ptr::null_mut();
            ze_res("pfnAppendGraphExecute", unsafe { execute(list, graph, null, null, 0, std::ptr::null_mut()) })?;
            wait_for_device(self.api, list, self.ordinal, "pfnAppendGraphExecute")
        })
    }
}

/// context_create: a Level Zero context and an immediate command list on
/// the listed device.
pub(crate) fn create(
    d: &'static Driver,
    dev: &'static Device,
    ordinal: u32,
    log: turbo_log_fn,
    log_user_data: *mut c_void,
    out: *mut *mut c_void,
) -> Res<()> {
    let desc = ze::ContextDesc { stype: ze::STRUCTURE_TYPE_CONTEXT_DESC, p_next: std::ptr::null(), flags: 0 };
    let mut handle = std::ptr::null_mut();
    ze_res("zeContextCreate", unsafe { (d.api.context_create)(dev.driver, &desc, &mut handle) })?;
    let qdesc = ze::CommandQueueDesc {
        stype: ze::STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        p_next: std::ptr::null(),
        ordinal: 0,
        index: 0,
        flags: ze::COMMAND_QUEUE_FLAG_IN_ORDER,
        mode: ze::COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        priority: ze::COMMAND_QUEUE_PRIORITY_NORMAL,
    };
    let mut list = std::ptr::null_mut();
    let rc = unsafe { (d.api.command_list_create_immediate)(handle, dev.handle, &qdesc, &mut list) };
    if let Err(e) = ze_res("zeCommandListCreateImmediate", rc) {
        unsafe { (d.api.context_destroy)(handle) };
        return Err(e);
    }
    let ctx = Box::new(Context {
        api: &d.api,
        ext: dev.ext,
        device: dev.handle,
        handle,
        list: Mutex::new(Shared(list)),
        compiler: dev.compiler,
        formats_supported: dev.formats_supported,
        max_opset: dev.max_opset,
        ordinal,
        log,
        log_user_data,
    });
    unsafe { *out = Box::into_raw(ctx) as *mut c_void };
    Ok(())
}

/// # Safety
/// `ctx` is a context this backend created, released once.
pub(crate) unsafe extern "C" fn context_release(ctx: *mut c_void) {
    quietly(|| {
        let ctx = unsafe { Box::from_raw(ctx as *mut Context) };
        // The box is the last owner: the mutex is consumed, not locked.
        let list = ctx.list.into_inner().unwrap_or_else(|p| p.into_inner()).0;
        unsafe {
            (ctx.api.command_list_destroy)(list);
            (ctx.api.context_destroy)(ctx.handle);
        }
    });
}
