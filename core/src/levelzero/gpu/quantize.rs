//! On-device int8 and ubinary codes of a buffer's leading F32 values.

use std::ffi::c_void;
use std::ptr;

use super::*;
use crate::status::INVALID_ARGUMENT;
use crate::{TURBO_PLACE_DEVICE, TURBO_PLACE_SHARED, TURBO_QUANTIZE_INT8, TURBO_QUANTIZE_UBINARY, turbo_error};

/// Quantize the first `rows * dim` F32 values of `buf` into codes at the
/// start of the same buffer. Codes match `core/src/quantize.rs`.
///
/// # Safety
/// `buf` is a buffer this backend gave; `int8_ranges` is null or `2 * dim`
/// host f32; `err` is null or valid for the call.
pub(crate) unsafe extern "C" fn buffer_quantize(
    buf: *mut c_void,
    quantization: u32,
    int8_ranges: *const f32,
    rows: u32,
    dim: u32,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
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
                if quantization == TURBO_QUANTIZE_INT8 { floats } else { rows as u64 * ((dim as u64 + 7) / 8) };
            if b.bytes < floats * 4 {
                return Err(fail(
                    INVALID_ARGUMENT,
                    format!("buffer of {} bytes is under the {floats} F32 values to quantize", b.bytes),
                ));
            }
            let c = &*b.ctx;
            let codes = c.alloc_device(code_bytes as usize)?;
            let mut ranges_pin = std::ptr::null_mut();
            let result = (|| {
                let mut q = c.lock_queue()?;
                let args = if quantization == TURBO_QUANTIZE_INT8 {
                    let rb = dim as usize * 2 * 4;
                    ranges_pin = c.alloc_pinned(rb)?;
                    ptr::copy_nonoverlapping(int8_ranges, ranges_pin.cast::<f32>(), dim as usize * 2);
                    vec![
                        Arg::Ptr(b.ptr as u64),
                        Arg::Ptr(ranges_pin as u64),
                        Arg::Ptr(codes as u64),
                        Arg::I32(rows as i32),
                        Arg::I32(dim as i32),
                    ]
                } else {
                    vec![Arg::Ptr(b.ptr as u64), Arg::Ptr(codes as u64), Arg::I32(rows as i32), Arg::I32(dim as i32)]
                };
                let name = if quantization == TURBO_QUANTIZE_INT8 { "quantize_int8" } else { "quantize_ubinary" };
                let k = c.kernel(name, [256, 1, 1])?;
                let n = if quantization == TURBO_QUANTIZE_INT8 { floats } else { code_bytes }.max(1);
                let groups = [((n + 255) / 256) as u32, 1, 1];
                k.launch(c, &mut q, name, &args, groups)?;
                c.copy(&mut q, b.ptr, codes, code_bytes as usize)?;
                c.sync(&mut q)
            })();
            if !ranges_pin.is_null() {
                c.free(ranges_pin, "int8_ranges for buffer_quantize");
            }
            c.free(codes, "codes for buffer_quantize");
            result
        })
    }
}
