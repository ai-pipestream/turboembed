//! C runtime selection and context lifecycle entrypoints.

use super::*;

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_runtime_create(
    desc: *const turbo_runtime_desc,
    out: *mut *mut turbo_runtime,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let out = out_ptr(out, "out")?;
            let (log, user) = match desc.as_ref() {
                Some(d) => {
                    sized(d.struct_size, size_of::<turbo_runtime_desc>(), "turbo_runtime_desc")?;
                    (d.log, d.log_user_data as usize)
                }
                None => (None, 0),
            };
            let mut inner =
                Runtime { log, log_user_data: user, devices: Vec::new(), tune_cache: tuning::Cache::default() };
            inner.enumerate()?;
            let rt = turbo_runtime { magic: RUNTIME_MAGIC, inner: Arc::new(inner) };
            *out = Box::into_raw(Box::new(rt));
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_runtime_device_count(
    rt: *mut turbo_runtime,
    out: *mut u32,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let rt = runtime(rt)?;
            *out_ptr(out, "out")? = rt.inner.devices.len() as u32;
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_runtime_device_info(
    rt: *mut turbo_runtime,
    index: u32,
    out: *mut turbo_device_info,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let rt = runtime(rt)?;
            let out = out_ptr(out, "out")?;
            sized(out.struct_size, size_of::<turbo_device_info>(), "turbo_device_info")?;
            // Probed again, so memory_free is the device's now.
            let d = rt.inner.device(index)?;
            *out = probe(d.backend, d.ordinal)?;
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_runtime_capability(
    rt: *mut turbo_runtime,
    index: u32,
    task: u32,
    precision: u32,
    out: *mut turbo_capability,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let rt = runtime(rt)?;
            let out = out_ptr(out, "out")?;
            sized(out.struct_size, size_of::<turbo_capability>(), "turbo_capability")?;
            *out = rt.inner.capability(index, task, precision)?;
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says; `reason` is
/// NULL or holds `reason_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_runtime_select(
    rt: *mut turbo_runtime,
    task: u32,
    out: *mut u32,
    reason: *mut c_char,
    reason_len: u32,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let rt = &runtime(rt)?.inner;
            let out = out_ptr(out, "out")?;
            let (pick, why) = select(rt, task)?;
            if !reason.is_null() && reason_len > 0 {
                let len = array_len::<c_char>(reason_len as u64, "reason")?;
                write_str(std::slice::from_raw_parts_mut(reason, len), &why);
            }
            match pick {
                Some(i) => {
                    *out = i;
                    Ok(())
                }
                None => Err(Error::new(status::DEVICE_NOT_FOUND, why)),
            }
        })
    }
}

/// turbo.h's rule: the highest status at TURBO_PRECISION_MODEL, then the
/// lowest speed_ratio among equals (0 is no record, so it loses), never a
/// CPU. Returns the device, or none, and one line saying why.
pub(super) fn select(rt: &Runtime, task: u32) -> Result<(Option<u32>, String)> {
    if task != TURBO_TASK_EMBED && task != TURBO_TASK_RERANK && task != TURBO_TASK_EXTRACTIVE_QA {
        return Err(Error::new(INVALID_ENUM, format!("task: {task} is not a TURBO_TASK_* value")));
    }
    let mut best: Option<(u32, turbo_capability)> = None;
    let mut ties = 0u32;
    let mut skipped = Vec::new();
    for (i, d) in rt.devices.iter().enumerate() {
        let i = i as u32;
        let label = format!("device {i} ({} {})", d.backend.name(), cstr(&d.info.name));
        if d.info.kind == TURBO_DEVICE_CPU && task != TURBO_TASK_RERANK && task != TURBO_TASK_EXTRACTIVE_QA {
            skipped.push(format!("{label}: a CPU is never selected"));
            continue;
        }
        let cap = rt.capability(i, task, TURBO_PRECISION_MODEL)?;
        if cap.status == backend::TURBO_CAP_UNSUPPORTED {
            skipped.push(format!("{label}: {}", cstr(&cap.reason)));
            continue;
        }
        match &best {
            Some((_, b)) if !ranks_above(&cap, b) => {
                if !ranks_above(b, &cap) {
                    ties += 1;
                }
            }
            _ => {
                best = Some((i, cap));
                ties = 1;
            }
        }
    }
    Ok(match best {
        Some((i, cap)) => {
            let d = &rt.devices[i as usize];
            let status = if cap.status == backend::TURBO_CAP_SUPPORTED { "supported" } else { "experimental" };
            let tie = if ties > 1 { format!(", first in device order of {ties} that tie") } else { String::new() };
            (Some(i), format!("device {i} ({} {}): {status}{tie}", d.backend.name(), cstr(&d.info.name)))
        }
        None if skipped.is_empty() => (None, "no device is listed".to_owned()),
        None => (None, format!("no device offers the task: {}", skipped.join("; "))),
    })
}

/// Whether cell `a` ranks above cell `b`: a higher status, or the same
/// status and a lower speed_ratio, where 0 is no record and ranks below any.
/// Neither ranking above the other is a tie.
pub fn ranks_above(a: &turbo_capability, b: &turbo_capability) -> bool {
    a.status > b.status
        || (a.status == b.status && a.speed_ratio > 0.0 && (b.speed_ratio == 0.0 || a.speed_ratio < b.speed_ratio))
}

/// # Safety
/// `rt` is NULL or a handle from turbo_runtime_create, released once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_runtime_release(rt: *mut turbo_runtime) {
    if unsafe { runtime(rt) }.is_ok() {
        let mut b = unsafe { Box::from_raw(rt) };
        b.magic = 0;
    }
}

// ---- Context and buffers ---------------------------------------------------------

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_context_create(
    rt: *mut turbo_runtime,
    device: u32,
    out: *mut *mut turbo_context,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let rt = runtime(rt)?;
            let out = out_ptr(out, "out")?;
            let d = rt.inner.device(device)?;
            let b = d.backend;
            let create = backend::offered!(b, context_create)?;
            let release = backend::offered!(b, context_release)?;
            let mut raw = std::ptr::null_mut();
            let user = rt.inner.log_user_data as *mut c_void;
            backend::check(b, "context_create", |err| create(d.ordinal, rt.inner.log, user, &mut raw, err))?;
            let inner = Context { runtime: rt.inner.clone(), device, backend: b, raw, release };
            *out = Box::into_raw(Box::new(turbo_context { magic: CONTEXT_MAGIC, inner: Arc::new(inner) }));
            Ok(())
        })
    }
}

/// # Safety
/// `ctx` is NULL or a handle from turbo_context_create, released once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_context_release(ctx: *mut turbo_context) {
    if unsafe { context(ctx) }.is_ok() {
        let mut b = unsafe { Box::from_raw(ctx) };
        b.magic = 0;
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_context_device(ctx: *mut turbo_context, out: *mut u32, err: *mut turbo_error) -> i32 {
    unsafe {
        call(err, || {
            let ctx = context(ctx)?;
            *out_ptr(out, "out")? = ctx.inner.device;
            Ok(())
        })
    }
}
