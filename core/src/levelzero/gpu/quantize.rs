//! On-device int8 and ubinary codes of a buffer's leading F32 values.

use std::ffi::c_void;
use std::ptr;

use super::*;
use crate::backend::turbo_backend_run;
use crate::status::{INTERNAL, INVALID_ARGUMENT};
use crate::{TURBO_PLACE_DEVICE, TURBO_PLACE_SHARED, TURBO_QUANTIZE_INT8, TURBO_QUANTIZE_UBINARY, turbo_error};

/// Quantize the first `rows * dim` F32 values of `buf` into codes at the
/// start of the same buffer. Codes match `core/src/quantize.rs`. A value
/// that is NaN or infinite is refused INTERNAL, naming its row and
/// dimension, never coded (#2517). When `run` is not null the device
/// memory taken here, the int8 ranges and the check's 4 bytes sent to the
/// device, and the check's 4 bytes read back, are added to it.
///
/// # Safety
/// `buf` is a buffer this backend gave; `int8_ranges` is null or `2 * dim`
/// host f32; `run` and `err` are null or valid for the call.
pub(crate) unsafe extern "C" fn buffer_quantize(
    buf: *mut c_void,
    quantization: u32,
    int8_ranges: *const f32,
    rows: u32,
    dim: u32,
    run: *mut turbo_backend_run,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        let device0 = device_allocs_here();
        let (mut sent, mut fetched) = (0u64, 0u64);
        let rc = guarded(err, || {
            if rows == 0 || dim == 0 {
                return Ok(());
            }
            if quantization != TURBO_QUANTIZE_INT8 && quantization != TURBO_QUANTIZE_UBINARY {
                return Err(fail(
                    INVALID_ARGUMENT,
                    format!("quantization {quantization} is not TURBO_QUANTIZE_INT8 or UBINARY"),
                ));
            }
            if quantization == TURBO_QUANTIZE_INT8 && int8_ranges.is_null() {
                return Err(fail(INVALID_ARGUMENT, "int8_ranges is NULL"));
            }
            let b = &*(buf as *const Buffer);
            if b.placement != TURBO_PLACE_DEVICE && b.placement != TURBO_PLACE_SHARED {
                return Err(fail(INVALID_ARGUMENT, "buffer_quantize needs DEVICE or SHARED memory"));
            }
            let floats = rows as u64 * dim as u64;
            let code_bytes =
                if quantization == TURBO_QUANTIZE_INT8 { floats } else { rows as u64 * (dim as u64).div_ceil(8) };
            if b.bytes < floats * 4 {
                return Err(fail(
                    INVALID_ARGUMENT,
                    format!("buffer of {} bytes is under the {floats} F32 values to quantize", b.bytes),
                ));
            }
            if floats > i32::MAX as u64 {
                return Err(fail(INVALID_ARGUMENT, format!("{floats} F32 values to quantize is over 2^31 - 1")));
            }
            let c = &*b.ctx;
            let codes = c.alloc_device(code_bytes as usize)?;
            let mut ranges_pin = std::ptr::null_mut();
            // The first value that is NaN or infinite, found before any is
            // coded: such a run is broken, and its codes would pass for a
            // vector's. It stays i32::MAX when every value is finite.
            let (mut at_dev, mut at_pin) = (std::ptr::null_mut(), std::ptr::null_mut());
            let result = (|| {
                at_dev = c.alloc_device(4)?;
                at_pin = c.alloc_pinned(4)?;
                at_pin.cast::<i32>().write(i32::MAX);
                let mut q = c.lock_queue()?;
                c.copy(&mut q, at_dev, at_pin, 4)?;
                let check = [Arg::Ptr(b.ptr as u64), Arg::I32(floats as i32), Arg::Ptr(at_dev as u64)];
                let k = c.kernel("first_non_finite", [256, 1, 1])?;
                k.launch(c, &mut q, "first_non_finite", &check, [floats.div_ceil(256).max(1) as u32, 1, 1])?;
                c.copy(&mut q, at_pin, at_dev, 4)?;
                (sent, fetched) = (4, 4);
                let slots;
                let args: &[Arg] = if quantization == TURBO_QUANTIZE_INT8 {
                    let rb = dim as usize * 2 * 4;
                    ranges_pin = c.alloc_pinned(rb)?;
                    ptr::copy_nonoverlapping(int8_ranges, ranges_pin.cast::<f32>(), dim as usize * 2);
                    sent += rb as u64;
                    slots = [
                        Arg::Ptr(b.ptr as u64),
                        Arg::Ptr(ranges_pin as u64),
                        Arg::Ptr(codes as u64),
                        Arg::I32(rows as i32),
                        Arg::I32(dim as i32),
                    ];
                    &slots
                } else {
                    slots = [
                        Arg::Ptr(b.ptr as u64),
                        Arg::Ptr(codes as u64),
                        Arg::I32(rows as i32),
                        Arg::I32(dim as i32),
                        Arg::I32(0),
                    ];
                    &slots[..4]
                };
                let name = if quantization == TURBO_QUANTIZE_INT8 { "quantize_int8" } else { "quantize_ubinary" };
                let k = c.kernel(name, [256, 1, 1])?;
                let n = if quantization == TURBO_QUANTIZE_INT8 { floats } else { code_bytes }.max(1);
                let groups = [n.div_ceil(256) as u32, 1, 1];
                k.launch(c, &mut q, name, args, groups)?;
                c.copy(&mut q, b.ptr, codes, code_bytes as usize)?;
                c.sync(&mut q)?;
                let at = at_pin.cast::<i32>().read();
                if at == i32::MAX {
                    return Ok(());
                }
                let (row, d) = (at as u32 / dim, at as u32 % dim);
                Err(fail(
                    INTERNAL,
                    format!("the model's output for row {row} is not finite at dimension {d}: the run is broken"),
                ))
            })();
            for (p, what) in [
                (ranges_pin, "int8_ranges for buffer_quantize"),
                (at_pin, "the finiteness check's host copy for buffer_quantize"),
                (at_dev, "the finiteness check for buffer_quantize"),
            ] {
                if !p.is_null() {
                    c.free(p, what);
                }
            }
            c.free(codes, "codes for buffer_quantize");
            result
        });
        if let Some(run) = run.as_mut() {
            run.device_allocs += device_allocs_here() - device0;
            if rc == 0 {
                run.h2d_bytes += sent;
                run.d2h_bytes += fetched;
            }
        }
        rc
    }
}
