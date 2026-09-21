//! `libturbo`: links `turbo-capi` so its `#[no_mangle] extern "C"` exports
//! become the shared library's symbol table. Nothing is defined here; the
//! functions live in `crates/turbo-capi` and the header in `include/turbo`.
#![deny(missing_docs)]

// The `extern crate` keeps the dependency linked even though nothing here
// names its items; the exported symbols are reachable through the linker.
extern crate turbo_capi;
