//! Turbo core: one contract, many providers.
//!
//! The core owns the provider registry, device discovery and selection, the
//! capability matrix, buffers, bundle verification, sessions with result
//! leases, streaming generation, and the error model. Providers implement the
//! traits in [`provider`] at task granularity and are free to run each task
//! as one fused device pipeline.
//!
//! The C ABI over this crate lives in `turbo-capi`; the ergonomic Rust API in
//! `turbo`. Both are thin.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod buffer;
pub mod bundle;
pub mod error;
pub mod handles;
pub mod mock;
pub mod provider;
pub mod runtime;
pub mod types;

pub use buffer::{BufferDesc, HostBuffer, NativeHandle, ProviderBuffer};
pub use bundle::{Bundle, Manifest};
pub use error::{status_name, Error, Result};
pub use handles::{Buffer, Context, Generation, Model, ResultHandle, Session};
pub use provider::{
    Capability, Chunk, ClassifyOptions, ContextDesc, DeviceInfo, EmbedOptions, GenerateDesc, Message, ModelDesc,
    ModelInfo, Options, Output, Provider, ProviderContext, ProviderGeneration, ProviderModel, ProviderResult,
    ProviderSession, RerankOptions, RunOptions, SessionDesc, SessionStats, Span, TensorInfo, TokenBatch,
};
pub use runtime::{DeviceEntry, DeviceSelector, LogLevel, LogSink, ProviderFailure, Runtime, RuntimeDesc};
pub use types::*;

use std::sync::Arc;

/// Library version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Providers compiled into this library. The mock is always present so the
/// contract can be exercised without hardware; it only serves mock bundles.
pub fn builtin_providers() -> Vec<Arc<dyn Provider>> {
    vec![Arc::new(mock::MockProvider::new())]
}

/// Create a runtime with the built-in providers.
pub fn create_runtime(desc: RuntimeDesc) -> Result<Arc<Runtime>> {
    Runtime::new(desc, builtin_providers())
}
