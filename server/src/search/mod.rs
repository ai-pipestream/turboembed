//! Stateful hybrid search service (`TurboSearchService`, issue `#1221`)
//! uniting `turbo-chunker`, `turbo-vector-index`, and `turbo-query` over
//! gRPC with index lifecycle, aliases, snapshot persistence, and live
//! collection vocabulary drift monitoring.

pub(crate) mod clauses;
pub mod ingest;
pub mod query;
pub mod registry;
pub mod service;
pub mod snapshot;

pub use registry::SearchRegistry;
pub use service::SearchService;
