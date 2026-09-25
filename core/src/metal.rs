//! The Metal backend: Apple GPUs, written in Objective-C++ under
//! core/metal/ and linked in when the metal feature is on. build.rs
//! compiles it; this names its table for the core.

use crate::backend::turbo_backend;

unsafe extern "C" {
    pub safe static turbo_metal_backend: turbo_backend;
}
