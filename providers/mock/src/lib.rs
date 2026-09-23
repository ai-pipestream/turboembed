//! The mock provider as a loadable provider library.
//!
//! The provider itself lives in `turbo_core::mock`; this crate exports it
//! through the plugin ABI (`include/turbo/turbo_provider.h`) so the loading
//! path is exercised by the same provider the core builds in. Load it with
//! `turbo_runtime_load_provider` into a runtime created with
//! `TURBO_RUNTIME_NO_DEFAULT_PROVIDERS` (the built-in mock has the same id).
#![deny(missing_docs)]

use std::sync::Arc;

pub use turbo_core::mock::MockProvider;

turbo_core::export_provider!(c"mock", c"2.0.0-alpha.0", || Arc::new(MockProvider::new()));
