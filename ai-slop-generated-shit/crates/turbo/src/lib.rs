//! Safe Rust API for Turbo.
//!
//! This crate is the ergonomic surface over `turbo-core`. Handles are
//! reference counted and children retain parents; sessions and generations
//! are `Send + Sync` handles whose operations are single-owner, enforced at
//! run time (a second concurrent call returns `TURBO_E_BUSY`), so a shared
//! `Arc<Session>` compiles and the caller serializes its use; every option
//! is validated against the capability matrix before a provider is reached.

#![deny(missing_docs)]

pub use turbo_abi as abi;
pub use turbo_core::*;

use std::sync::Arc;

/// Providers this library ships built in: the mock (contract testing only)
/// and the static token-embedding provider. Hardware providers load as
/// libraries through [`Runtime::load_provider`].
pub fn builtin_providers() -> Vec<Arc<dyn Provider>> {
    let mut v = turbo_core::builtin_providers();
    v.push(Arc::new(turbo_provider_static::StaticProvider::new()));
    v
}

/// Create a runtime with [`builtin_providers`] (unless
/// `desc.no_default_providers` is set) plus any explicit provider libraries.
pub fn create_runtime(desc: RuntimeDesc) -> Result<Arc<Runtime>> {
    let builtin = if desc.no_default_providers { Vec::new() } else { builtin_providers() };
    Runtime::new(desc, builtin)
}
