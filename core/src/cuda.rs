//! The CUDA backend: NVIDIA GPUs through the CUDA runtime and cuBLAS. It
//! is written in C++ and CUDA in core/cuda/, compiled by build.rs into a
//! static library, and reached only through its turbo_backend table, like
//! any other backend. docs/cuda.md says how to build and test it.

use crate::backend::turbo_backend;

unsafe extern "C" {
    /// The table core/cuda/backend.cpp fills.
    pub(crate) static turbo_cuda_backend: turbo_backend;
}

/// The backend's table.
pub fn backend() -> &'static turbo_backend {
    // A table the C++ side fills at compile time and never writes.
    unsafe { &turbo_cuda_backend }
}

#[cfg(feature = "internals")]
unsafe extern "C" {
    fn turbo_cuda_allocations(host: *mut u64, device: *mut u64);
    fn turbo_cuda_arch_label(name: *const std::ffi::c_char, out: *mut std::ffi::c_char, len: usize);
    fn turbo_cuda_widened(model: *mut std::ffi::c_void) -> *const std::ffi::c_void;
}

/// Every allocation the CUDA backend has made in this process, host and
/// device, counted where it makes them. Built only with `internals`.
#[cfg(feature = "internals")]
pub fn allocations() -> (u64, u64) {
    let (mut host, mut device) = (0, 0);
    unsafe { turbo_cuda_allocations(&mut host, &mut device) };
    (host, device)
}

/// The arch label a device of this name is listed with. Built only with
/// `internals`.
#[cfg(feature = "internals")]
pub fn arch_label(name: &str) -> String {
    let name = std::ffi::CString::new(name).unwrap();
    let mut out = [0 as std::ffi::c_char; 32];
    unsafe { turbo_cuda_arch_label(name.as_ptr(), out.as_mut_ptr(), out.len()) };
    crate::backend::cstr(&out)
}

/// The device address of the F32 copy of an F16 or BF16 model's weights,
/// once a session made it.
///
/// # Safety
/// `model` is one this backend's model_load returned, not yet released.
#[cfg(feature = "internals")]
pub(crate) unsafe fn widened(model: *mut std::ffi::c_void) -> Option<*const std::ffi::c_void> {
    let p = unsafe { turbo_cuda_widened(model) };
    (!p.is_null()).then_some(p)
}
