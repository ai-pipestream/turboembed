//! The Hailo backend: embed on Hailo accelerators through HailoRT, from a
//! HEF. It is written in C++ in core/hailo/, compiled by build.rs into a
//! static library, and reached only through its turbo_backend table, like
//! any other backend. docs/hailo.md says how to build and test it.

use crate::backend::turbo_backend;

unsafe extern "C" {
    /// The table core/hailo/backend.cpp fills.
    pub(crate) static turbo_hailo_backend: turbo_backend;
}

/// The backend's table.
pub fn backend() -> &'static turbo_backend {
    // A table the C++ side fills at compile time and never writes.
    unsafe { &turbo_hailo_backend }
}
