//! Mock provider crate. The provider itself lives in `turbo_core::mock`; this
//! crate becomes the loadable `libturbo_provider_mock` once the plugin vtable
//! lands (PLAN.md P1).
#![deny(missing_docs)]
pub use turbo_core::mock::MockProvider;
