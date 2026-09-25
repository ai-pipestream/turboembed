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
