//! Safe Rust API for Turbo.
//!
//! This crate is the ergonomic surface over `turbo-core`. Handles are
//! reference counted and children retain parents; sessions and generations
//! are single-owner (`Send`, not `Sync`); every option is validated against
//! the capability matrix before a provider is reached.

#![deny(missing_docs)]

pub use turbo_abi as abi;
pub use turbo_core::*;
