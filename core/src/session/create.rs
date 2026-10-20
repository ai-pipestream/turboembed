//! Session creation, capacity resolution, and metadata.

use super::*;

// ---- Session ------------------------------------------------------------------

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_session_create(
    m: *mut turbo_model,
    desc: *const turbo_session_desc,
    out: *mut *mut turbo_session,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let m = &model_handle(m)?.inner;
            let out = out_ptr(out, "out")?;
            // NULL is 0 in every field, as for turbo_runtime_create. A
            // caller built against the struct before tuning has only that
            // much memory: it is read through the raw pointer, and the
            // fields past it are 0.
            let mut d = turbo_session_desc {
                struct_size: 0,
                max_batch: 0,
                max_seq: 0,
                precision: 0,
                tuning: 0,
                tuning_budget_ms: 0,
                cpu_threads: 0,
                max_ranges: 0,
            };
            if !desc.is_null() {
                let size = (desc as *const u32).read();
                if size as usize != TURBO_SESSION_DESC_SIZE_V1 && size as usize != TURBO_SESSION_DESC_SIZE_V2 {
                    sized(size, size_of::<turbo_session_desc>(), "turbo_session_desc")?;
                }
                std::ptr::copy_nonoverlapping(
                    desc.cast::<u8>(),
                    (&mut d as *mut turbo_session_desc).cast::<u8>(),
                    size as usize,
                );
            }
            let inner = create_session(m, d)?;
            *out = Box::into_raw(Box::new(turbo_session { magic: SESSION_MAGIC, inner: Arc::new(inner) }));
            Ok(())
        })
    }
}

/// The room a tuned session's timings lines have: a line per variant
/// timed, per GEMM and bin.
pub(super) const TIMINGS_LEN: usize = 16384;

pub(super) fn create_session(m: &Arc<Model>, d: turbo_session_desc) -> Result<SessionInner> {
    if d.precision > TURBO_PRECISION_EXACT {
        return Err(Error::new(INVALID_ENUM, format!("precision: {} is not a TURBO_PRECISION_* value", d.precision)));
    }
    let mi = &m.info;
    let backend_task = if mi.task == TURBO_TASK_RERANK { TURBO_TASK_EMBED } else { mi.task };
    // docs/bundle.md: max_batch is the largest batch the reference was
    // checked at, and a session larger than it is refused; max_seq is the
    // length the model was evaluated at.
    let limit = |field: u32, name: &str, asked: u32, model: u32| {
        if asked > model {
            return Err(Error::field(UNSUPPORTED_OPTION, field, format!("{name} {asked} is over the model's {model}")));
        }
        Ok(if asked == 0 { model } else { asked })
    };
    let max_batch = limit(1, "max_batch", d.max_batch, mi.max_batch)?;
    let max_seq = limit(2, "max_seq", d.max_seq, mi.max_seq)?;
    let rows = array_len::<i32>(max_batch as u64 * max_seq as u64, "session token rows")?;

    let tuning_mode = tuning::mode(d.tuning)?;
    let c = &m.context;
    let b = c.backend;
    // A backend with session_open is made every session through it; else
    // one with tuned sessions through session_create_tuned; else through
    // session_create.
    let open = backend::offered!(b, session_open).ok();
    let tuned = if open.is_some() { None } else { backend::offered!(b, session_create_tuned).ok() };
    let create = match (open, tuned) {
        (None, None) => Some(backend::offered!(b, session_create)?),
        _ => None,
    };
    let ranges = backend::offered!(b, embed_write_ranges).is_ok();
    let max_ranges = resolve_threads_and_ranges(b.name(), open.is_some(), ranges, &d, max_batch, max_seq)?;
    array_len::<f32>(max_ranges.max(max_batch) as u64 * mi.dim as u64, "session output")?;
    let release = backend::offered!(b, session_release)?;
    let run = backend::offered!(b, session_run)?;
    let cap = c.runtime.capability(c.device, mi.task, d.precision)?;
    if cap.status == backend::TURBO_CAP_UNSUPPORTED {
        let reason = backend::cstr(&cap.reason);
        // The task runs at another precision: the precision is the option refused.
        let other = (TURBO_PRECISION_MODEL..=TURBO_PRECISION_EXACT).filter(|&p| p != d.precision).any(|p| {
            c.runtime.capability(c.device, mi.task, p).is_ok_and(|o| o.status != backend::TURBO_CAP_UNSUPPORTED)
        });
        if other {
            return Err(Error::field(
                UNSUPPORTED_OPTION,
                3,
                format!("precision {}: the {} backend does not run it: {reason}", d.precision, b.name()),
            ));
        }
        return Err(Error::new(UNSUPPORTED_TASK, format!("the {} backend does not run embed: {reason}", b.name())));
    }
    let mut compute_dtype = 0;
    let mut raw = std::ptr::null_mut();
    let mut t: backend::turbo_backend_tuning = unsafe { std::mem::zeroed() };
    t.struct_size = size_of::<backend::turbo_backend_tuning>() as u32;
    t.mode = tuning_mode;
    let dir = tuning::cache_dir();
    t.budget_ms = tuning::budget(tuning_mode, d.tuning_budget_ms, dir.is_some())?;
    t.numerics_allowed = tuning::numerics_allowed(d.precision);
    // A tuned session's key, and the choice an earlier session with the
    // same key measured, which the backend takes as its incumbent.
    let rt = &c.runtime;
    let key = (tuning_mode != TURBO_AUTOTUNE_OFF && (tuned.is_some() || open.is_some())).then(|| {
        let dev = &rt.devices[c.device as usize].info;
        tuning::TuneKey {
            backend: b.name().to_owned(),
            device_name: backend::cstr(&dev.name),
            arch: backend::cstr(&dev.arch),
            driver_version: backend::cstr(&dev.driver_version),
            runtime_version: backend::cstr(&dev.runtime_version),
            library_build: tuning::LIBRARY_BUILD.to_owned(),
            manifest_sha256: backend::cstr(&mi.manifest_sha256),
            artifact_sha256: backend::cstr(&mi.artifact_sha256),
            precision: d.precision,
            compute_dtype: cap.dtype,
            numerics_allowed: t.numerics_allowed,
            tcap_bin: tuning::tcap_bin(max_batch.saturating_mul(max_seq)),
            max_seq_bin: tuning::max_seq_bin(max_seq),
        }
    });
    let cached = key
        .as_ref()
        .and_then(|k| rt.tune_cache.get(k, dir.as_deref()))
        .and_then(|e| std::ffi::CString::new(e.choices).ok());
    t.cached = cached.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
    let mut timings = vec![0 as c_char; TIMINGS_LEN];
    if key.is_some() {
        t.timings = timings.as_mut_ptr();
        t.timings_len = TIMINGS_LEN as u32;
    }
    match (open, tuned, create) {
        (Some(f), _, _) => {
            let desc = backend::turbo_backend_session_desc {
                struct_size: size_of::<backend::turbo_backend_session_desc>() as u32,
                task: backend_task,
                max_batch,
                max_seq,
                precision: d.precision,
                cpu_threads: d.cpu_threads,
                max_ranges,
                reserved: 0,
            };
            // turbo_backend.h: the out fields come in with the one-path
            // values, which a backend with one path leaves alone.
            t.tuned = TURBO_TUNED_DEFAULT;
            t.tune_ms = 0;
            t.choices[0] = 0;
            t.numerics_used = t.numerics_allowed;
            backend::check(b, "session_open", |err| unsafe {
                f(m.raw, &desc, &mut t, &mut compute_dtype, &mut raw, err)
            })?
        }
        (None, Some(f), _) => backend::check(b, "session_create_tuned", |err| unsafe {
            f(m.raw, backend_task, max_batch, max_seq, d.precision, &mut t, &mut compute_dtype, &mut raw, err)
        })?,
        (None, None, Some(f)) => backend::check(b, "session_create", |err| unsafe {
            f(m.raw, backend_task, max_batch, max_seq, d.precision, &mut compute_dtype, &mut raw, err)
        })?,
        (None, None, None) => unreachable!("offered above"),
    }
    // A table without tuned sessions made its choices itself, and has one path.
    if tuned.is_none() && open.is_none() {
        t.tuned = TURBO_TUNED_DEFAULT;
        t.tune_ms = 0;
        t.choices[0] = 0;
    }
    let choices_end = t.choices.iter().position(|&c| c == 0);
    if tuning::tuned_name(t.tuned).is_none() || choices_end.is_none() {
        unsafe { release(raw) };
        return Err(Error::new(
            INTERNAL,
            format!(
                "{} backend: tuned {} or its choices, not NUL-terminated, are not what it may report",
                b.name(),
                t.tuned
            ),
        ));
    }
    if !matches!(compute_dtype, TURBO_DTYPE_I8 | TURBO_DTYPE_F16 | TURBO_DTYPE_BF16 | TURBO_DTYPE_F32) {
        unsafe { release(raw) };
        return Err(Error::new(
            INTERNAL,
            format!("{} backend: compute dtype {compute_dtype} is not a TURBO_DTYPE_* a model computes in", b.name()),
        ));
    }
    // Only a measured choice with nothing forced and nothing widened is
    // kept: any other session's choice is not the key's to give.
    if let Some(k) = key {
        let choices = backend::cstr(&t.choices);
        let forced = tuning::forced_knobs(&choices);
        if t.tuned == TURBO_TUNED_MEASURED && !forced.is_empty() {
            rt.log(LOG_INFO, &format!("the measured kernels are not cached: the session forces {}", forced.join(",")));
        } else if t.tuned == TURBO_TUNED_MEASURED && t.numerics_used != t.numerics_allowed {
            rt.log(
                LOG_INFO,
                &format!(
                    "the measured kernels are not cached: they compute in numeric classes 0x{:x} beyond the \
                     precision's 0x{:x}",
                    t.numerics_used & !t.numerics_allowed,
                    t.numerics_allowed
                ),
            );
        }
        if t.tuned == TURBO_TUNED_MEASURED && forced.is_empty() && t.numerics_used == t.numerics_allowed {
            let timings = if t.timings.is_null() { String::new() } else { backend::cstr(&timings) };
            let e = tuning::TuneEntry::measured(k, &choices, t.tune_ms, &timings);
            if let Err(err) = rt.tune_cache.put(e, dir.as_deref()) {
                rt.log(
                    LOG_WARNING,
                    &format!("the tuning cache in {} was not written: {err}", dir.unwrap_or_default().display()),
                );
            }
        }
    }
    let info = turbo_session_info {
        struct_size: size_of::<turbo_session_info>() as u32,
        max_batch,
        max_seq,
        precision: d.precision,
        compute_dtype,
        reserved: 0,
        tuned: t.tuned,
        tune_ms: if t.tuned == TURBO_TUNED_MEASURED { t.tune_ms } else { 0 },
        choices: t.choices,
    };
    let result = Box::into_raw(Box::new(turbo_result {
        magic: 0,
        hold: None,
        info: result_template(m, &info),
        read: AtomicU64::new(0),
        output: std::ptr::null_mut(),
        host: std::ptr::null_mut(),
    }));
    let state = Mutex::new(State {
        written: None,
        ids: vec![0; rows],
        mask: vec![0; rows],
        rerank: None,
        scores: vec![0.0; max_batch as usize],
        pooled: vec![0.0; mi.dim as usize],
    });
    // Where the lock is a pthread mutex (macOS), std allocates it on first
    // use; take it once here so no write or run allocates it later.
    drop(state.try_lock());
    Ok(SessionInner {
        model: m.clone(),
        backend: b,
        raw,
        release,
        run,
        info,
        honored: cap.options_honored,
        max_ranges,
        state,
        holders: AtomicU32::new(0),
        result,
    })
}

/// turbo_session_desc.max_ranges with 0 resolved to max_batch and checked
/// against the session's shape. A backend without session_open is never
/// handed cpu_threads or max_ranges, so for it the core refuses a value of
/// either other than the one that backend works to anyway: 0 threads, and
/// max_batch ranges. A backend with session_open but no embed_write_ranges
/// pools no ranges, so a max_ranges other than max_batch is refused for it
/// too, rather than sizing an output no range write can fill. cpu_threads
/// is passed on as given otherwise, for the backend to check.
pub(super) fn resolve_threads_and_ranges(
    backend: &str,
    open: bool,
    ranges: bool,
    d: &turbo_session_desc,
    max_batch: u32,
    max_seq: u32,
) -> Result<u32> {
    let limit = max_batch as u64 * max_seq as u64;
    if d.max_ranges as u64 > limit {
        return Err(Error::field(
            UNSUPPORTED_OPTION,
            7,
            format!("max_ranges {} is over max_batch {max_batch} times max_seq {max_seq}, {limit}", d.max_ranges),
        ));
    }
    if !open && d.cpu_threads != 0 {
        return Err(Error::field(
            UNSUPPORTED_OPTION,
            6,
            format!(
                "cpu_threads {}: the {} backend takes its thread count from the environment alone",
                d.cpu_threads, backend
            ),
        ));
    }
    if !(open && ranges) && d.max_ranges != 0 && d.max_ranges != max_batch {
        return Err(Error::field(
            UNSUPPORTED_OPTION,
            7,
            format!("max_ranges {}: the {backend} backend pools no ranges", d.max_ranges),
        ));
    }
    Ok(if d.max_ranges == 0 { max_batch } else { d.max_ranges })
}

/// The fields of turbo_result_info that are the same for every run of the
/// session, filled in once so a run writes only its own.
pub(super) fn result_template(m: &Model, s: &turbo_session_info) -> turbo_result_info {
    let mut r: turbo_result_info = unsafe { std::mem::zeroed() };
    let c = &m.context;
    let device = &c.runtime.devices[c.device as usize].info;
    r.struct_size = size_of::<turbo_result_info>() as u32;
    r.task = m.info.task;
    r.dtype = TURBO_DTYPE_F32;
    r.compute_dtype = s.compute_dtype;
    r.device = c.device;
    r.stage_count =
        if m.info.task == TURBO_TASK_RERANK { TURBO_RERANK_STAGE_COUNT } else { TURBO_EMBED_STAGE_COUNT } as u32;
    r.backend = device.backend;
    r.arch = device.arch;
    r.runtime_version = device.runtime_version;
    r.manifest_sha256 = m.info.manifest_sha256;
    r.artifact_sha256 = m.info.artifact_sha256;
    r.tokenizer_sha256 = m.info.tokenizer_sha256;
    r
}

/// # Safety
/// `s` is NULL or a handle from turbo_session_create, released once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_session_release(s: *mut turbo_session) {
    if unsafe { session(s) }.is_ok() {
        let mut b = unsafe { Box::from_raw(s) };
        b.magic = 0;
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_session_get_info(
    s: *mut turbo_session,
    out: *mut turbo_session_info,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let s = session(s)?;
            if out.is_null() {
                return Err(Error::new(INVALID_ARGUMENT, "out is NULL"));
            }
            // A caller built against the struct before tuned has only that
            // much memory: it is written through the raw pointer, never as
            // the whole struct.
            let size = (out as *const u32).read();
            if size as usize != TURBO_SESSION_INFO_SIZE_V1 {
                sized(size, size_of::<turbo_session_info>(), "turbo_session_info")?;
            }
            let info = s.inner.info;
            std::ptr::copy_nonoverlapping(
                (&info as *const turbo_session_info).cast::<u8>().add(4),
                out.cast::<u8>().add(4),
                size as usize - 4,
            );
            Ok(())
        })
    }
}
