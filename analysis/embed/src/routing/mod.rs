//! Deterministic routing over handles the application has already loaded.
//!
//! Costs are supplied by the caller from timings on the serving host,
//! including its batching, device and thread configuration. Selection does
//! not open files, load models, allocate, or silently relax a budget.
mod error;
mod profile;
mod scoring;
mod select;
mod types;

pub use error::{RoutingError, TimedCost};
pub use profile::{LoadedModel, ModelProfile};
pub use scoring::QueryVector;
pub use select::{ModelRouter, Rationale, Route};
pub use types::{LatencyBudget, Request, Role, RoleCapability, RoutingPolicy, Task};
