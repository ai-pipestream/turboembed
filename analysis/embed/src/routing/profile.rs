use super::{Role, RoleCapability, RoutingError};
use turbo::bundle::Bundle;
use turbo::manifest::Normalize;
use turbo_analysis::ModelRef;
use turbo_vector_index::{Normalization, Space};

/// Validated runtime capabilities. The runtime, not the router, owns model
/// loading and timings. Immutable after registration to keep cost and output
/// identity paired. Use `from_bundle` for the core's current bundle formats.
#[derive(Clone, Debug)]
pub struct ModelProfile {
    pub(super) model: ModelRef,
    pub(super) capabilities: Vec<RoleCapability>,
    pub(super) spaces: Vec<Space>,
}
impl ModelProfile {
    /// Register facts from an already loaded runtime, including sparse or
    /// external runtimes. The caller must report the actual artifact identity,
    /// supported outputs and host timings, not estimates from model names.
    pub fn new(model: ModelRef, mut capabilities: Vec<RoleCapability>) -> Result<Self, RoutingError> {
        let fail = |s: &str| RoutingError::Invalid(s.into());
        if model.id.is_empty() || model.revision.is_empty() || model.sha256.is_empty() {
            return Err(fail("model identity must include id, revision and digest"));
        }
        if capabilities.is_empty() {
            return Err(fail("a loaded model needs at least one timed role"));
        }
        for c in &capabilities {
            if c.width == 0 || c.context_limit == 0 || c.cost_per_item.is_zero() {
                return Err(fail("width, context limit and timed cost must be positive"));
            }
            if c.role == Role::Rerank && (c.width != 1 || c.normalization != Normalization::None) {
                return Err(fail("rerank requires one unnormalized logit"));
            }
        }
        capabilities.sort_by_key(|c| (c.role, c.width, norm_key(c.normalization)));
        if capabilities
            .windows(2)
            .any(|w| w[0].role == w[1].role && w[0].width == w[1].width && w[0].normalization == w[1].normalization)
        {
            return Err(fail("duplicate role/width/normalization timing"));
        }
        let spaces = capabilities.iter().map(|c| Space::new(model.clone(), c.width, c.normalization)).collect();
        Ok(Self { model, capabilities, spaces })
    }

    /// Bind host timings to the capabilities of an opened core bundle.
    /// Advertise only configurations the loaded serving session can execute;
    /// smaller session context limits are allowed. No files are opened here.
    /// Sparse runtimes not yet represented in the core manifest can use `new`.
    pub fn from_bundle(bundle: &Bundle, capabilities: Vec<RoleCapability>) -> Result<Self, RoutingError> {
        let m = &bundle.manifest;
        for c in &capabilities {
            let supported = match m.task.as_str() {
                "TASK_EMBED" => {
                    let e = m
                        .embed
                        .as_ref()
                        .ok_or_else(|| RoutingError::Invalid("embed bundle has no embed configuration".into()))?;
                    let norm = match e.normalize {
                        Normalize::None => Normalization::None,
                        Normalize::L2 => Normalization::Unit,
                    };
                    let dense = matches!(c.role, Role::DenseQuery | Role::DenseDocument);
                    let static_role = c.role == Role::StaticVectors && m.static_embedding.is_some();
                    let tokens = c.role == Role::TokenVectors && m.architecture.is_some();
                    let width_ok = if tokens {
                        c.width == m.projection.as_ref().map_or(e.dim, |p| p.dim)
                    } else {
                        c.width == e.dim || e.output_dims.contains(&c.width)
                    };
                    (dense || static_role || tokens)
                        && width_ok
                        && c.context_limit <= e.max_seq
                        && (tokens || c.normalization == norm)
                }
                "TASK_RERANK" => {
                    c.role == Role::Rerank
                        && c.width == 1
                        && m.rerank.as_ref().is_some_and(|r| c.context_limit <= r.max_seq)
                }
                _ => false,
            };
            if !supported {
                return Err(RoutingError::Invalid(format!("bundle does not support {c:?}")));
            }
        }
        Self::new(crate::model_ref(bundle), capabilities)
    }
    pub fn model(&self) -> &ModelRef {
        &self.model
    }
    pub fn capabilities(&self) -> &[RoleCapability] {
        &self.capabilities
    }
}
pub(super) fn norm_key(n: Normalization) -> u8 {
    match n {
        Normalization::None => 0,
        Normalization::Unit => 1,
    }
}

/// The application's loaded model handle plus its immutable routing facts.
/// A handle may be an index into a session pool, an Arc, or a runtime object.
/// The router never downloads or opens the handle.
pub struct LoadedModel<H> {
    pub(super) handle: H,
    pub(super) profile: ModelProfile,
}
impl<H> LoadedModel<H> {
    pub fn new(handle: H, profile: ModelProfile) -> Self {
        Self { handle, profile }
    }
    pub fn handle(&self) -> &H {
        &self.handle
    }
    pub fn profile(&self) -> &ModelProfile {
        &self.profile
    }
}
