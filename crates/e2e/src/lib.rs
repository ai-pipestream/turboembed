//! Client-side E2E harness for inferstream.
//!
//! Talks gRPC (OIP V2 + `inferstream.v1`) to a running arch server. The same
//! cases run against nvidia / intel / apple; aliases the catalog does not
//! resolve on that arch, or that the host did not put on `serve`, soft-skip.

pub mod catalog;
pub mod client;
pub mod golden;
pub mod matrix;
pub mod suite;
pub mod target;

pub use catalog::{CatalogIndex, SkipReason};
pub use matrix::Matrix;
pub use suite::{decide, run_suite, Decision, Outcome, Report, SuiteConfig, SuiteFilter};
pub use target::{infer_target_from_addr, Target};

#[derive(Debug, thiserror::Error)]
pub enum HarnessError {
    #[error("connect {endpoint}: {source}")]
    Connect {
        endpoint: String,
        source: tonic::transport::Error,
    },
    #[error("{0}")]
    Rpc(String),
    #[error("invalid address: {0}")]
    Addr(String),
    #[error("matrix: {0}")]
    Matrix(String),
    #[error("catalog: {0}")]
    Catalog(String),
}
