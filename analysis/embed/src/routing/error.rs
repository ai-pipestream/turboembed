use super::{LatencyBudget, Role, Task};
use std::fmt;
use std::time::Duration;
use turbo_analysis::ModelRef;
use turbo_vector_index::{IndexError, Space};

#[derive(Clone, Debug, PartialEq)]
pub struct TimedCost {
    pub model: ModelRef,
    pub role: Role,
    pub width: u32,
    pub context_limit: u32,
    pub normalization: turbo_vector_index::Normalization,
    pub cost_per_item: Duration,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RoutingError {
    Invalid(String),
    /// All registered costs are retained, even when none support the task.
    Unavailable {
        task: Task,
        budget: LatencyBudget,
        costs: Vec<TimedCost>,
    },
    SpaceUnavailable {
        required: Box<Space>,
        costs: Vec<TimedCost>,
    },
    CrossSpace {
        expected: Box<Space>,
        actual: Box<Space>,
    },
    NotDenseQuery(Role),
    Index(IndexError),
}
impl fmt::Display for RoutingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid routing configuration: {s}"),
            Self::Unavailable { task, budget, costs } => {
                write!(f, "no loaded model meets {task:?} / {budget:?}; timed costs: {costs:?}")
            }
            Self::SpaceUnavailable { required, costs } => {
                write!(f, "no loaded configuration matches space {required:?}; timed costs: {costs:?}")
            }
            Self::CrossSpace { expected, actual } => {
                write!(f, "cross-space scoring: expected {expected:?}, got {actual:?}")
            }
            Self::NotDenseQuery(role) => write!(f, "{role:?} cannot score a dense query"),
            Self::Index(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for RoutingError {}
impl From<IndexError> for RoutingError {
    fn from(e: IndexError) -> Self {
        Self::Index(e)
    }
}
