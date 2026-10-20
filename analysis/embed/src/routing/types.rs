use std::time::Duration;
use turbo_vector_index::{Normalization, Space};

/// An output contract, independent of a model's marketing name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    StaticVectors,
    DenseQuery,
    DenseDocument,
    Sparse,
    TokenVectors,
    Rerank,
}

/// One timed role/width/normalization configuration of a loaded model.
/// Costs must cover inputs up to `context_limit`, including special tokens
/// and role prefixes, on the host and execution settings in use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoleCapability {
    pub role: Role,
    /// Vector width, vocabulary size for sparse output, or 1 for reranking.
    pub width: u32,
    pub normalization: Normalization,
    pub context_limit: u32,
    /// Positive wall time per item, including any output projection.
    pub cost_per_item: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Index,
    Query,
    Rerank,
    StaticVectors,
    SparseIndex,
    SparseQuery,
    TokenVectors,
}
impl Task {
    pub fn role(self) -> Role {
        match self {
            Self::Index => Role::DenseDocument,
            Self::Query => Role::DenseQuery,
            Self::Rerank => Role::Rerank,
            Self::StaticVectors => Role::StaticVectors,
            Self::SparseIndex | Self::SparseQuery => Role::Sparse,
            Self::TokenVectors => Role::TokenVectors,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LatencyBudget {
    /// Total time for `Request::items`, using the registered cost per item.
    MaxLatency(Duration),
    /// Positive finite throughput on the registered execution configuration.
    MinItemsPerSecond(f64),
}
impl LatencyBudget {
    pub(super) fn valid(self) -> bool {
        match self {
            Self::MaxLatency(d) => !d.is_zero(),
            Self::MinItemsPerSecond(r) => r.is_finite() && r > 0.0,
        }
    }
    pub(super) fn fits(self, cost: Duration, items: u32) -> bool {
        match self {
            Self::MaxLatency(limit) => cost.checked_mul(items).is_some_and(|total| total <= limit),
            Self::MinItemsPerSecond(rate) => rate <= 1.0 / cost.as_secs_f64(),
        }
    }
}

/// Preference among compatible configurations that meet the hard budget.
/// Neither policy assigns a quality score to unrelated models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingPolicy {
    Fastest,
    Widest,
}

#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    pub task: Task,
    pub budget: LatencyBudget,
    pub items: u32,
    /// Largest encoded input in this request, including prefixes/specials.
    pub tokens: u32,
    /// An exact requested output width. None allows the policy to choose.
    pub width: Option<u32>,
    /// Required output normalization; None allows the registered default.
    pub normalization: Option<Normalization>,
    /// Supply the destination index's space for retrieval. This constrains
    /// routing before cost comparison, so a cheaper unrelated model cannot win.
    pub space: Option<&'a Space>,
}
