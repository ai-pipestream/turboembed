//! Client-side E2E harness for inferstream.
//!
//! Talks gRPC (OIP V2 + `inferstream.v1`) to a running arch server. The same
//! cases run against nvidia / intel / apple; aliases the catalog does not
//! resolve on that arch, or that the host did not put on `serve`, soft-skip.

pub mod catalog;
pub mod chunker;
mod client;
pub mod corpus;
pub mod fetch;
pub mod golden;
pub mod matrix;
pub mod parity;
pub mod suite;
pub mod target;

pub use catalog::{family_pooling, CatalogIndex, SkipReason};
pub use fetch::{
    candidate_aliases, ensure_plan, plan_corpus, plan_fetches, unclaimed_aliases, with_corpus,
    ArtifactKind, FetchPlan, FetchScope,
};
pub use matrix::Matrix;
pub use parity::{
    pair_threshold, run_parity, ParityConfig, ParityDump, ParityMode, CROSS_FP_MIN,
    CROSS_QUANT_MIN, DEFAULT_DRIFT_ALIASES, DEFAULT_PARITY_ALIASES, SAME_ARCH_MIN,
};
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
    #[error("fetch: {0}")]
    Fetch(String),
    #[error("parity: {0}")]
    Parity(String),
}
