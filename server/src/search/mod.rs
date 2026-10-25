//! Hybrid search, index lifecycle, aliases, snapshots, and collection drift
//! over gRPC (`TurboSearchService`, issue `#1221`).

pub mod clauses;
pub mod ingest;
pub mod query;
pub mod registry;
pub mod segment;
pub mod service;
pub mod snapshot;

pub use registry::SearchRegistry;
pub use service::SearchService;
