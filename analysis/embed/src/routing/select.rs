use super::{LoadedModel, Request, RoleCapability, RoutingError, RoutingPolicy, TimedCost};
use std::cmp::Ordering;
use turbo_analysis::ModelRef;
use turbo_vector_index::Space;

/// Fixed preference, with deterministic tie breaking by cost, width, model
/// id/revision/digest and normalization. Registration order has no effect.
pub struct ModelRouter<H> {
    models: Vec<LoadedModel<H>>,
    policy: RoutingPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rationale {
    FastestWithinBudget,
    WidestWithinBudget,
}

/// Borrowed result: the exact handle, model, width, context and timed cost
/// selected. No strings or output vectors are allocated on a successful route.
pub struct Route<'a, H> {
    pub(super) loaded: &'a LoadedModel<H>,
    pub(super) capability: &'a RoleCapability,
    pub(super) space: &'a Space,
    rationale: Rationale,
}
impl<H> Route<'_, H> {
    pub fn handle(&self) -> &H {
        &self.loaded.handle
    }
    pub fn model(&self) -> &ModelRef {
        &self.loaded.profile.model
    }
    pub fn capability(&self) -> &RoleCapability {
        self.capability
    }
    pub fn space(&self) -> &Space {
        self.space
    }
    pub fn rationale(&self) -> Rationale {
        self.rationale
    }
}
impl<H> ModelRouter<H> {
    pub fn new(mut models: Vec<LoadedModel<H>>, policy: RoutingPolicy) -> Result<Self, RoutingError> {
        if models.is_empty() {
            return Err(RoutingError::Invalid("no loaded models".into()));
        }
        models.sort_by(|a, b| identity(&a.profile.model).cmp(&identity(&b.profile.model)));
        if models.windows(2).any(|w| w[0].profile.model == w[1].profile.model) {
            return Err(RoutingError::Invalid("duplicate model identity; combine timings in one profile".into()));
        }
        Ok(Self { models, policy })
    }
    pub fn models(&self) -> &[LoadedModel<H>] {
        &self.models
    }
    pub fn policy(&self) -> RoutingPolicy {
        self.policy
    }

    pub fn route(&self, request: Request<'_>) -> Result<Route<'_, H>, RoutingError> {
        if request.items == 0 || request.tokens == 0 || request.width == Some(0) || !request.budget.valid() {
            return Err(RoutingError::Invalid("items, tokens, width and finite budget must be positive".into()));
        }
        if let Some(s) = request.space {
            if s.dim == 0 || s.model.id.is_empty() || s.model.revision.is_empty() || s.model.sha256.is_empty() {
                return Err(RoutingError::Invalid("invalid requested space".into()));
            }
            if request.width.is_some_and(|w| w != s.dim) || request.normalization.is_some_and(|n| n != s.normalization)
            {
                return Err(RoutingError::Invalid("request contradicts its required space".into()));
            }
        }
        let mut best: Option<Route<'_, H>> = None;
        let mut compatible_space = request.space.is_none();
        for model in &self.models {
            for (c, space) in model.profile.capabilities.iter().zip(&model.profile.spaces) {
                if c.role != request.task.role()
                    || request.width.is_some_and(|w| w != c.width)
                    || request.normalization.is_some_and(|n| n != c.normalization)
                {
                    continue;
                }
                if request.space.is_some_and(|s| s != space) {
                    continue;
                }
                compatible_space = true;
                if request.tokens > c.context_limit || !request.budget.fits(c.cost_per_item, request.items) {
                    continue;
                }
                let better = best.as_ref().is_none_or(|b| compare(self.policy, c, b.capability).is_lt());
                if better {
                    best = Some(Route {
                        loaded: model,
                        capability: c,
                        space,
                        rationale: match self.policy {
                            RoutingPolicy::Fastest => Rationale::FastestWithinBudget,
                            RoutingPolicy::Widest => Rationale::WidestWithinBudget,
                        },
                    });
                }
            }
        }
        if let Some(route) = best {
            return Ok(route);
        }
        if !compatible_space {
            return Err(RoutingError::SpaceUnavailable {
                required: Box::new(request.space.expect("constrained space").clone()),
                costs: self.costs(),
            });
        }
        Err(RoutingError::Unavailable { task: request.task, budget: request.budget, costs: self.costs() })
    }
    fn costs(&self) -> Vec<TimedCost> {
        self.models
            .iter()
            .flat_map(|m| {
                m.profile.capabilities.iter().map(|c| TimedCost {
                    model: m.profile.model.clone(),
                    role: c.role,
                    width: c.width,
                    context_limit: c.context_limit,
                    normalization: c.normalization,
                    cost_per_item: c.cost_per_item,
                })
            })
            .collect()
    }
}
fn identity(m: &ModelRef) -> (&str, &str, &str) {
    (&m.id, &m.revision, &m.sha256)
}
fn compare(policy: RoutingPolicy, a: &RoleCapability, b: &RoleCapability) -> Ordering {
    match policy {
        RoutingPolicy::Fastest => a.cost_per_item.cmp(&b.cost_per_item).then_with(|| b.width.cmp(&a.width)),
        RoutingPolicy::Widest => b.width.cmp(&a.width).then_with(|| a.cost_per_item.cmp(&b.cost_per_item)),
    }
}
