//! Shared-library packaging shim for the frozen `include/turboembed.h` ABI.
//!
//! The ABI implementation lives in the `turboembed` crate (Rust hooks) and
//! its cc-built native objects (`native/turboembed/src/stub.cpp`). This
//! crate only re-exports that surface as `libturboembed.so`: the KEEP table
//! below forces the linker to retain every public entry point from the
//! static archives, and `exports.map` (applied by `build.rs`) restricts the
//! dynamic symbol table to exactly the header's `turboembed_*` functions.
//!
//! Built with `--features ort-cuda` this is the NVIDIA package configuration
//! validated on Machine A; without features it still exports the same ABI
//! over the mock provider (useful for GPU-less consumer smoke checks, never
//! advertised as an accelerator).

use turboembed::ffi;

/// Raw function pointers are not `Sync`; this table is read-only link glue.
#[repr(transparent)]
pub struct AbiEntry(*const core::ffi::c_void);
unsafe impl Sync for AbiEntry {}

/// Force-link every public ABI entry point out of the native static archive.
/// Without these references a cdylib may drop the unreferenced stub.cpp
/// objects entirely.
#[used]
pub static TURBOEMBED_ABI_KEEP: [AbiEntry; 15] = [
    AbiEntry(ffi::turboembed_engine_create as *const _),
    AbiEntry(ffi::turboembed_engine_destroy as *const _),
    AbiEntry(ffi::turboembed_abi_version as *const _),
    AbiEntry(ffi::turboembed_status_name as *const _),
    AbiEntry(ffi::turboembed_device_name as *const _),
    AbiEntry(ffi::turboembed_last_error as *const _),
    AbiEntry(ffi::turboembed_list_models as *const _),
    AbiEntry(ffi::turboembed_model_list_free as *const _),
    AbiEntry(ffi::turboembed_load_model as *const _),
    AbiEntry(ffi::turboembed_embed_one as *const _),
    AbiEntry(ffi::turboembed_embed as *const _),
    AbiEntry(ffi::turboembed_embed_stream as *const _),
    AbiEntry(ffi::turboembed_embed_result_free as *const _),
    AbiEntry(ffi::turboembed_buffer_free as *const _),
    AbiEntry(ffi::turboembed_register_provider as *const _),
];
