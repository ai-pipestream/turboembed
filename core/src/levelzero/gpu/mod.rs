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

use std::cell::Cell;
use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use super::ze::{self, ComputeApi, Handle};
use crate::backend::{refuse, refuse_field};
use crate::status::{OUT_OF_MEMORY, PANIC, RUNTIME};
use crate::{turbo_error, turbo_log_fn};

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
        ze::RESULT_ERROR_DEVICE_LOST => Err(fail(RUNTIME, device_lost(what))),
        _ => Err(fail(RUNTIME, format!("levelzero: {what} failed with 0x{rc:08x}"))),
    }
}

/// DEVICE_LOST covers hangs, resets, removal and driver updates. The
/// timeout/reset observations in #1433 do not identify every loss's cause,
/// and a new context cannot recover a device that is still unavailable.
fn device_lost(what: &str) -> String {
    format!(
        "levelzero: {what} failed with 0x70000001 (device lost): possible device hang, reset, removal \
         or driver update; this status alone does not identify the cause. Check driver/kernel logs \
         (a job timeout is one possibility). Stop using any affected context, release its resources \
         and the context, and create a new one once the device is available"
    )
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
// zeMemAllocHost included, count as device allocations. Those not yet freed
// are counted too, for the tests.

static DEVICE_TOTAL: AtomicU64 = AtomicU64::new(0);
static DEVICE_LIVE: AtomicI64 = AtomicI64::new(0);

thread_local! {
    static DEVICE_HERE: Cell<u64> = const { Cell::new(0) };
}

/// Every device allocation this backend has made in the process.
#[cfg(feature = "internals")]
pub(crate) fn device_allocs_total() -> u64 {
    DEVICE_TOTAL.load(Ordering::Relaxed)
}

/// The device allocations this backend has made and not freed.
#[cfg(feature = "internals")]
pub(crate) fn device_allocs_live() -> i64 {
    DEVICE_LIVE.load(Ordering::Relaxed)
}

pub(crate) fn device_allocs_here() -> u64 {
    DEVICE_HERE.with(Cell::get)
}

fn counted_device() {
    DEVICE_TOTAL.fetch_add(1, Ordering::Relaxed);
    DEVICE_LIVE.fetch_add(1, Ordering::Relaxed);
    DEVICE_HERE.with(|c| c.set(c.get() + 1));
}

// ---- Contexts ----------------------------------------------------------------------

const LOG_WARNING: u32 = 1;
pub(crate) const LOG_INFO: u32 = 2;
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
    pub queue: Mutex<Queue>,
    /// The encoder's module, built on first need; the failure is kept, so
    /// a device that cannot build it says why each time.
    module: Mutex<Option<Result<Shared, String>>>,
    /// The most local memory one work-group may have.
    pub max_local: u32,
    /// Nanoseconds per tick of the device's timestamps.
    timer_ns: u64,
    /// With TURBO_LEVELZERO_PROFILE set: each append's name, how many
    /// times it ran, and its nanoseconds on the device.
    profile: Mutex<std::collections::BTreeMap<String, (u64, u64)>>,
    log: turbo_log_fn,
    log_user_data: *mut c_void,
    /// oneDNN on this context (the levelzero-onednn feature), opened on
    /// first need; None where it could not open.
    #[cfg(feature = "levelzero-onednn")]
    dnnl: std::sync::OnceLock<Option<super::onednn::Dnnl>>,
}

// The handles are the driver's, for any thread; the log function is the
// caller's, which turbo.h says may be called from any thread.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

mod buffers;
mod context;
mod kernel;

pub(crate) use buffers::{Buffer, buffer_alloc, buffer_export, buffer_import, buffer_read, buffer_release};
pub(crate) use context::Queue;
use context::{EVENTS, profiling};
pub(crate) use kernel::{Arg, Kernel};

impl Drop for Context {
    fn drop(&mut self) {
        if profiling() {
            let p = std::mem::take(self.profile.get_mut().unwrap_or_else(|p| p.into_inner()));
            let total: u64 = p.values().map(|v| v.1).sum();
            for (name, (n, ns)) in p.iter() {
                let share = 100.0 * *ns as f64 / total.max(1) as f64;
                let line = format!("levelzero profile: {name}: {n} times, {:.3} ms, {share:.1}%", *ns as f64 / 1e6);
                self.say(LOG_DEBUG, &line);
            }
        }
        // oneDNN's primitives free their memory through the Level Zero
        // context, so they go before it.
        #[cfg(feature = "levelzero-onednn")]
        drop(self.dnnl.take());
        let module = self.module.get_mut().unwrap_or_else(|p| p.into_inner());
        if let Some(Ok(m)) = module.take() {
            unsafe { (self.api.module_destroy)(m.0) };
        }
        let q = self.queue.get_mut().unwrap_or_else(|p| p.into_inner());
        if !q.list.is_null() {
            unsafe { (self.api.command_list_destroy)(q.list) };
        }
        for &e in &q.events {
            unsafe { (self.api.event_destroy)(e) };
        }
        if !q.pool.is_null() {
            unsafe { (self.api.event_pool_destroy)(q.pool) };
        }
        unsafe { (self.api.context_destroy)(self.handle) };
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
        queue: Mutex::new(Queue {
            list: std::ptr::null_mut(),
            pool: std::ptr::null_mut(),
            events: Vec::with_capacity(EVENTS as usize),
            used: 0,
            wedged: false,
            names: Vec::new(),
            pending_wait: std::ptr::null_mut(),
        }),
        module: Mutex::new(None),
        max_local: dev.max_local,
        timer_ns: dev.timer_ns,
        profile: Mutex::new(std::collections::BTreeMap::new()),
        log,
        log_user_data,
        #[cfg(feature = "levelzero-onednn")]
        dnnl: std::sync::OnceLock::new(),
    });
    let q = ctx.queue.get_mut().unwrap_or_else(|p| p.into_inner());
    let pd = ze::EventPoolDesc {
        stype: ze::STRUCTURE_TYPE_EVENT_POOL_DESC,
        p_next: std::ptr::null(),
        // Timestamp events: on this driver plain host-visible ones make each
        // append wait on the host, and a run takes twice as long.
        flags: ze::EVENT_POOL_FLAG_HOST_VISIBLE | ze::EVENT_POOL_FLAG_KERNEL_TIMESTAMP,
        count: EVENTS,
    };
    let no_devices = std::ptr::null_mut();
    ze("zeEventPoolCreate", unsafe { (api.event_pool_create)(handle, &pd, 0, no_devices, &mut q.pool) })?;
    for index in 0..EVENTS {
        let ed = ze::EventDesc {
            stype: ze::STRUCTURE_TYPE_EVENT_DESC,
            p_next: std::ptr::null(),
            index,
            signal: ze::EVENT_SCOPE_FLAG_HOST,
            wait: ze::EVENT_SCOPE_FLAG_HOST,
        };
        let mut e = std::ptr::null_mut();
        ze("zeEventCreate", unsafe { (api.event_create)(q.pool, &ed, &mut e) })?;
        q.events.push(e);
    }
    ze("zeCommandListCreateImmediate", unsafe {
        (api.command_list_create_immediate)(handle, dev.handle, &queue_desc(), &mut q.list)
    })?;
    ctx.say(LOG_DEBUG, &format!("levelzero context on device {ordinal} ({}): one in-order queue", dev.name));
    unsafe { *out = Box::into_raw(ctx) as *mut c_void };
    Ok(())
}

/// A failed append on a real queue, and what follows it: a large copy is
/// appended, then a launch the driver refuses (more local memory than the
/// device has). sync must return once the copy is done, the copy's bytes
/// must all have arrived, and the new queue must work. Built only with
/// `internals`, for tests/levelzero.rs.
#[cfg(feature = "internals")]
pub(crate) fn append_failure_recovers(
    driver: &'static super::Driver,
    dev: &'static super::Device,
) -> Result<(), String> {
    let text = |f: Fail| f.message;
    let mut out = std::ptr::null_mut();
    unsafe { create(driver, dev, 0, None, std::ptr::null_mut(), &mut out) }.map_err(text)?;
    let c = unsafe { Box::from_raw(out as *mut Context) };
    let n = 64usize << 20;
    let host = c.alloc_pinned(n * 4).map_err(text)?;
    let (src, dst) = (c.alloc_device(n * 4).map_err(text)?, c.alloc_device(n * 4).map_err(text)?);
    let values = unsafe { std::slice::from_raw_parts_mut(host as *mut u32, n) };
    for (i, v) in values.iter_mut().enumerate() {
        *v = i as u32 ^ 0x5a5a_5a5a;
    }
    let k = c.kernel("attention", [128, 1, 1]).map_err(text)?;
    let result: Result<(), String> = (|| {
        let mut q = c.lock_queue().map_err(text)?;
        unsafe { c.copy(&mut q, src, host, n * 4) }.map_err(text)?;
        c.sync(&mut q).map_err(text)?;
        let old = q.list;
        unsafe { c.copy(&mut q, dst, src, n * 4) }.map_err(text)?;
        // The general attention kernel's last argument is its local memory.
        let mut args = [Arg::U64(0); 9];
        args[8] = Arg::Local(c.max_local as usize + 4096);
        if k.launch(&c, &mut q, "the refused launch", &args, [1, 1, 1]).is_ok() {
            return Err("the driver took a launch with more local memory than the device has".into());
        }
        c.sync(&mut q).map_err(text)?;
        if q.list == old || q.list.is_null() {
            return Err("the failed queue was not replaced".into());
        }
        unsafe { c.copy(&mut q, host, dst, n * 4) }.map_err(text)?;
        c.sync(&mut q).map_err(text)?;
        Ok(())
    })();
    let back = unsafe { std::slice::from_raw_parts(host as *const u32, n) };
    let whole = back.iter().enumerate().all(|(i, &v)| v == i as u32 ^ 0x5a5a_5a5a);
    drop(k);
    for p in [host, src, dst] {
        c.free(p, "the test's memory");
    }
    result?;
    if !whole {
        return Err("the copy appended before the failure did not arrive whole".into());
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_loss_does_not_identify_a_queue_timeout() {
        // The same status can arise before a context or queue exists.
        for call in ["zeContextCreate", "zeCommandListHostSynchronize"] {
            let lost = ze(call, ze::RESULT_ERROR_DEVICE_LOST).unwrap_err();
            assert_eq!(lost.code, RUNTIME);
            for cause in ["hang", "reset", "removal", "driver update"] {
                assert!(lost.message.contains(cause), "missing {cause}: {}", lost.message);
            }
            assert!(!lost.message.contains("the driver reset this context's queue"));
            assert!(lost.message.contains("once the device is available"));
        }
    }

    #[test]
    fn device_loss_recovery_survives_the_error_abi() {
        let mut error = crate::new_error();
        let code = unsafe { guarded(&mut error, || ze("zeCommandListHostSynchronize", ze::RESULT_ERROR_DEVICE_LOST)) };
        assert_eq!(code, RUNTIME);
        assert_eq!(error.code, RUNTIME);
        let message = crate::backend::cstr(&error.message);
        assert_eq!(message, device_lost("zeCommandListHostSynchronize"));
        assert!(message.contains("once the device is available"));
    }

    /// A lost device gives possible causes and recovery; other failures keep
    /// their plain form (#1433).
    #[test]
    fn a_lost_device_says_the_context_is_gone() {
        let lost = ze("zeCommandListHostSynchronize", ze::RESULT_ERROR_DEVICE_LOST).unwrap_err();
        assert_eq!(lost.code, RUNTIME);
        for part in ["zeCommandListHostSynchronize failed with 0x70000001 (device lost)", "job timeout", "new one"] {
            assert!(lost.message.contains(part), "{part:?} not in {:?}", lost.message);
        }
        let other = ze("zeEventHostSynchronize", 0x7800_0004).unwrap_err();
        assert_eq!(other.message, "levelzero: zeEventHostSynchronize failed with 0x78000004");
        assert_eq!(ze("x", ze::RESULT_ERROR_OUT_OF_DEVICE_MEMORY).unwrap_err().code, OUT_OF_MEMORY);
        assert!(ze("x", 0).is_ok());
    }
}
