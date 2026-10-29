//! Session execution and held-result access and lifetime.

use super::*;

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_session_run(
    s: *mut turbo_session,
    out: *mut *mut turbo_result,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let handle = session(s)?;
            let s = &handle.inner;
            let mut state = s.lock()?;
            // The run takes the write before it checks its arguments.
            let w = state.written.take();
            let out = out_ptr(out, "out")?;
            let w = w.ok_or_else(|| Error::new(INVALID_STATE, "nothing is written: a run takes one write"))?;
            let mut r: turbo_backend_run = std::mem::zeroed();
            r.struct_size = size_of::<turbo_backend_run>() as u32;
            r.stage[TURBO_EMBED_STAGE_TOKENIZE] = if w.tokenized { TURBO_STAGE_HOST } else { TURBO_STAGE_UNUSED };
            let b = s.backend;
            if s.model.info.task == TURBO_TASK_RERANK {
                r = rerank::run_pairs(s, &mut state)?;
            } else {
                backend::check(b, "session_run", |err| (s.run)(s.raw, &mut r, err))?;
                check_run(b, &r, w.tokenized)?;
                if w.quantization != Quantization::F32 {
                    quantized(s, &mut state, &w, &mut r)?;
                }
            }

            let res = &mut *s.result;
            let i = &mut res.info;
            i.batch = w.batch;
            i.dim = w.output_dim;
            i.dtype = match w.quantization {
                Quantization::F32 => TURBO_DTYPE_F32,
                Quantization::Int8 => TURBO_DTYPE_I8,
                Quantization::UBinary => TURBO_DTYPE_U8,
            };
            i.placement = r.placement;
            i.bytes = w.batch as u64 * w.quantization.row_bytes(w.output_dim as usize) as u64;
            i.h2d_bytes = r.h2d_bytes;
            i.d2h_bytes = r.d2h_bytes;
            i.host_allocs = r.host_allocs + CORE_RUN_ALLOCS;
            i.device_allocs = r.device_allocs;
            i.stage = r.stage;
            res.output = r.output;
            res.host = r.host;
            res.read.store(0, Ordering::Relaxed);
            s.holders.store(1, Ordering::Release);
            res.hold = Some(s.clone());
            res.magic = RESULT_MAGIC;
            *out = s.result;
            drop(state);
            Ok(())
        })
    }
}

/// The run's f32 vectors as the write's codes, in the session's own host
/// memory, which the result then points at. When the backend offers
/// buffer_quantize and the vectors sit in DEVICE or SHARED memory, the
/// codes are made on the device and only those bytes cross the bus (or,
/// for SHARED, are copied on the host from the shared buffer). Otherwise
/// a DEVICE result is copied back as F32 first, into memory the first
/// such run makes, and the host codes them.
unsafe fn quantized(s: &SessionInner, state: &mut State, w: &Written, r: &mut turbo_backend_run) -> Result<()> {
    let dim = w.output_dim as usize;
    let n = w.batch as usize * dim;
    let code_bytes = w.batch as usize * w.quantization.row_bytes(dim);
    let q = match w.quantization {
        Quantization::Int8 => TURBO_QUANTIZE_INT8,
        Quantization::UBinary => TURBO_QUANTIZE_UBINARY,
        Quantization::F32 => unreachable!("quantized is not called for F32"),
    };
    let b = s.backend;
    let on_device = r.host.is_null();
    let shared = !on_device && r.placement == TURBO_PLACE_SHARED;
    let device_path = (on_device || shared) && backend::offered!(b, buffer_quantize).is_ok();
    if device_path {
        let quantize = backend::offered!(b, buffer_quantize).unwrap();
        let ranges_ptr =
            if w.quantization == Quantization::Int8 { state.int8_ranges.as_ptr() } else { std::ptr::null() };
        backend::check(b, "buffer_quantize", |err| unsafe {
            quantize(r.output, q, ranges_ptr, w.batch, w.output_dim, err)
        })?;
        let out = &mut state.codes[..code_bytes];
        if on_device {
            let read = backend::offered!(b, buffer_read).map_err(|_| {
                Error::new(
                    UNSUPPORTED,
                    format!("the vectors are in device memory, and the {} backend reads none", b.name()),
                )
            })?;
            let bytes = code_bytes as u64;
            backend::check(b, "buffer_read", |err| unsafe { read(r.output, out.as_mut_ptr().cast(), bytes, err) })?;
            r.d2h_bytes += bytes;
        } else {
            unsafe {
                std::ptr::copy_nonoverlapping(r.host as *const u8, out.as_mut_ptr(), code_bytes);
            }
        }
        r.output = std::ptr::null_mut();
        r.host = out.as_mut_ptr().cast();
        r.placement = TURBO_PLACE_HOST;
        return Ok(());
    }

    let State { codes, int8_ranges, staged, .. } = state;
    let src: &[f32] = if on_device {
        let read = backend::offered!(b, buffer_read).map_err(|_| {
            Error::new(
                UNSUPPORTED,
                format!("the vectors are in device memory, and the {} backend reads none", b.name()),
            )
        })?;
        if staged.len() < n {
            // As many values as the codes have bytes: the most a run makes.
            *staged = vec![0.0; codes.len()];
            r.host_allocs += 1;
        }
        let bytes = n as u64 * 4;
        backend::check(b, "buffer_read", |err| unsafe { read(r.output, staged.as_mut_ptr().cast(), bytes, err) })?;
        r.d2h_bytes += bytes;
        &staged[..n]
    } else {
        unsafe { std::slice::from_raw_parts(r.host as *const f32, n) }
    };
    let out = &mut codes[..code_bytes];
    quantize::rows(w.quantization, src, dim, &int8_ranges[..2 * dim], out);
    // Codes belong to the core, not to a backend buffer object. Native
    // handle export uses the host-only path when there is no such object.
    r.output = std::ptr::null_mut();
    r.host = out.as_mut_ptr().cast();
    r.placement = TURBO_PLACE_HOST;
    Ok(())
}

/// What the backend reported is something turbo.h can say.
pub(super) fn check_run(b: &turbo_backend, r: &turbo_backend_run, tokenized: bool) -> Result<()> {
    let bad = |what: String| Err(Error::new(INTERNAL, format!("{} backend, session_run: {what}", b.name())));
    if !(TURBO_PLACE_HOST..=TURBO_PLACE_SHARED).contains(&r.placement) {
        return bad(format!("placement {} is not a TURBO_PLACE_* value", r.placement));
    }
    if r.output.is_null() || r.host.is_null() != (r.placement == TURBO_PLACE_DEVICE) {
        return bad(format!("output {:?} at host address {:?} for placement {}", r.output, r.host, r.placement));
    }
    for (i, &st) in r.stage.iter().enumerate() {
        let ok = if i < TURBO_EMBED_STAGE_COUNT { st <= TURBO_STAGE_FUSED } else { st == TURBO_STAGE_UNUSED };
        if !ok {
            return bad(format!("stage[{i}] is {st}"));
        }
    }
    let tokenize = if tokenized { TURBO_STAGE_HOST } else { TURBO_STAGE_UNUSED };
    if r.stage[TURBO_EMBED_STAGE_TOKENIZE] != tokenize {
        return bad("it rewrote the tokenize stage, which is the core's".to_owned());
    }
    Ok(())
}

// ---- Result ---------------------------------------------------------------------

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_result_get_info(
    r: *mut turbo_result,
    out: *mut turbo_result_info,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let r = result(r)?;
            let out = out_ptr(out, "out")?;
            sized(out.struct_size, size_of::<turbo_result_info>(), "turbo_result_info")?;
            *out = r.info;
            out.d2h_bytes += r.read.load(Ordering::Relaxed);
            Ok(())
        })
    }
}

/// # Safety
/// `dst` is NULL or holds `capacity` bytes; other pointers are NULL or
/// valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_result_read(
    r: *mut turbo_result,
    dst: *mut c_void,
    capacity: u64,
    written: *mut u64,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let r = result(r)?;
            if dst.is_null() {
                return Err(Error::new(INVALID_ARGUMENT, "dst is NULL"));
            }
            let bytes = r.info.bytes;
            if capacity < bytes {
                return Err(Error::new(
                    INVALID_ARGUMENT,
                    format!("capacity {capacity} is under the result's {bytes} bytes"),
                ));
            }
            if r.host.is_null() {
                // Device memory: the backend copies it back.
                let b = r.hold.as_ref().expect("a held result holds its session").backend;
                let read = backend::offered!(b, buffer_read).map_err(|_| {
                    Error::new(
                        UNSUPPORTED,
                        format!("the vectors are in device memory, and the {} backend reads none", b.name()),
                    )
                })?;
                backend::check(b, "buffer_read", |err| read(r.output, dst, bytes, err))?;
            } else {
                std::ptr::copy_nonoverlapping(r.host as *const u8, dst as *mut u8, bytes as usize);
            }
            // turbo.h counts a read in d2h_bytes, whether or not the copy
            // crossed a bus: on a CPU it is a copy within host memory.
            r.read.fetch_add(bytes, Ordering::Relaxed);
            if let Some(w) = written.as_mut() {
                *w = bytes;
            }
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_result_buffer(
    r: *mut turbo_result,
    out: *mut *mut turbo_buffer,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let r = result(r)?;
            let out = out_ptr(out, "out")?;
            let s = r.hold.as_ref().expect("a held result holds its session");
            let i = &r.info;
            let desc = turbo_buffer_desc {
                struct_size: size_of::<turbo_buffer_desc>() as u32,
                placement: i.placement,
                dtype: i.dtype,
                ndim: 2,
                // Packed bits are a row's bytes wide, not its dimensions.
                shape: [i.batch as u64, if i.dtype == TURBO_DTYPE_U8 { i.dim.div_ceil(8) } else { i.dim } as u64],
                bytes: i.bytes,
            };
            s.holders.fetch_add(1, Ordering::AcqRel);
            let inner = Buffer {
                context: s.model.context.clone(),
                desc,
                raw: r.output,
                host: r.host,
                release: Release::Result(s.clone()),
            };
            *out = Box::into_raw(Box::new(turbo_buffer { magic: BUFFER_MAGIC, inner }));
            Ok(())
        })
    }
}

/// # Safety
/// `r` is NULL or a handle from turbo_session_run, released once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_result_release(r: *mut turbo_result) {
    if let Ok(res) = unsafe { result(r) } {
        let keep = res.hold.take();
        res.magic = 0;
        if let Some(s) = keep {
            s.let_go();
            // The last reference frees the session and this handle with
            // it, so nothing of it is touched after.
            drop(s);
        }
    }
}
