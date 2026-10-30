//! The Metal backend: Apple GPUs through Metal. It is written in
//! Objective-C++ and the Metal Shading Language in core/metal/, compiled
//! by build.rs into a static library, and reached only through its
//! turbo_backend table, like any other backend. docs/metal.md says how to
//! build and test it.

use crate::backend::turbo_backend;

unsafe extern "C" {
    /// The table core/metal/backend.mm fills.
    pub(crate) static turbo_metal_backend: turbo_backend;
}

/// The backend's table.
pub fn backend() -> &'static turbo_backend {
    // A table the Objective-C++ side fills at compile time and never writes.
    unsafe { &turbo_metal_backend }
}

#[cfg(feature = "internals")]
unsafe extern "C" {
    fn turbo_metal_allocations(host: *mut u64, device: *mut u64);
    fn turbo_metal_in_place(model: *mut std::ffi::c_void) -> i32;
    fn turbo_metal_widened(model: *mut std::ffi::c_void) -> *const std::ffi::c_void;
    fn turbo_metal_use_choices(line: *const std::ffi::c_char);
    fn turbo_metal_track_peaks(on: i32);
    fn turbo_metal_peaks(out: *mut f32);
    fn turbo_metal_quantize(
        ordinal: u32,
        quantization: u32,
        x: *const f32,
        ranges: *const f32,
        rows: i32,
        dim: i32,
        out: *mut u8,
    ) -> i32;
}

/// The F16 tensors a FASTEST run stores between kernels, in the order
/// `peaks` gives their largest magnitudes.
#[cfg(feature = "internals")]
pub const PEAK_NAMES: [&str; 6] = ["Q", "K", "V", "attention context", "GELU output", "residual stream"];

/// Whether sessions made after this track the largest magnitude of each
/// F16 tensor their runs store. Built only with `internals`, for the
/// tests.
#[cfg(feature = "internals")]
pub fn track_peaks(on: bool) {
    unsafe { turbo_metal_track_peaks(on as i32) };
}

/// The last tracked run's largest magnitudes, as PEAK_NAMES orders them;
/// 0 for a tensor the run stored in F32. Built only with `internals`.
#[cfg(feature = "internals")]
pub fn peaks() -> [f32; 6] {
    let mut out = [0f32; 6];
    unsafe { turbo_metal_peaks(out.as_mut_ptr()) };
    out
}

/// A choices line the sessions made after this read in place of
/// TURBO_METAL_CHOICES, or None to read the variable again. Built only
/// with `internals`, for the tests.
#[cfg(feature = "internals")]
pub fn use_choices(choices: Option<&str>) {
    let line = choices.map(|c| std::ffi::CString::new(c).expect("a choices line without NUL"));
    unsafe { turbo_metal_use_choices(line.as_ref().map_or(std::ptr::null(), |c| c.as_ptr())) };
}

/// Every allocation the Metal backend has made in this process, host and
/// device, counted where it makes them. Built only with `internals`.
#[cfg(feature = "internals")]
pub fn allocations() -> (u64, u64) {
    let (mut host, mut device) = (0, 0);
    unsafe { turbo_metal_allocations(&mut host, &mut device) };
    (host, device)
}

/// Whether the model's weights are read where the core holds them, not
/// copied. Built only with `internals`.
///
/// # Safety
/// `model` is one this backend's model_load returned, not yet released.
#[cfg(feature = "internals")]
pub(crate) unsafe fn in_place(model: *mut std::ffi::c_void) -> bool {
    unsafe { turbo_metal_in_place(model) != 0 }
}

/// The Metal buffer holding the F32 copy of an F16 or BF16 model's
/// weights, once a session made it.
///
/// # Safety
/// `model` is one this backend's model_load returned, not yet released.
#[cfg(feature = "internals")]
pub(crate) unsafe fn widened(model: *mut std::ffi::c_void) -> Option<*const std::ffi::c_void> {
    let p = unsafe { turbo_metal_widened(model) };
    (!p.is_null()).then_some(p)
}

/// int8 or ubinary codes of host F32 rows on device `ordinal`, bit for bit
/// with [`crate::quantize`]. Built only with `internals`.
#[cfg(feature = "internals")]
pub fn quantize(
    ordinal: u32,
    quantization: u32,
    x: &[f32],
    ranges: &[f32],
    rows: i32,
    dim: i32,
    out: &mut [u8],
) -> Result<(), i32> {
    crate::quantize::check_device_call(quantization, x.len(), ranges.len(), rows, dim, out.len())
        .map_err(|_| crate::status::INVALID_ARGUMENT)?;
    let ranges_ptr = if quantization == crate::TURBO_QUANTIZE_INT8 { ranges.as_ptr() } else { std::ptr::null() };
    let rc =
        unsafe { turbo_metal_quantize(ordinal, quantization, x.as_ptr(), ranges_ptr, rows, dim, out.as_mut_ptr()) };
    if rc == 0 { Ok(()) } else { Err(rc) }
}
