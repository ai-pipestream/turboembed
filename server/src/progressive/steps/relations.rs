use std::sync::Arc;

use turbo_analysis::analyzer::Builder;
use turbo_analysis::{EvidenceStep, Key};
use turbo_relation::{Gliner2Relations, Relations, ZeroShotRelationAnnotator, ZeroShotRelationConfig};

use crate::api::{Failure, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_INVALID_ARGUMENT};
use crate::state::State;

pub(super) fn open(state: Option<&State>, name: &str) -> Result<Arc<Gliner2Relations>, Failure> {
    let name = name.trim();
    if name.is_empty() {
        return Err(Failure::new(
            TURBO_E_INVALID_ARGUMENT,
            0,
            "turbo:zero-shot-relations requires relation_model to name a GLiNER2 package",
        ));
    }
    let state = state.ok_or_else(|| {
        Failure::new(TURBO_E_BUNDLE_NOT_FOUND, 0, "turbo:zero-shot-relations requires a loaded GLiNER2 package")
    })?;
    let (served, _) = state.peek(name)?;
    served.relations.get().cloned().ok_or_else(|| {
        Failure::new(TURBO_E_BUNDLE_NOT_FOUND, 0, format!("model {name} is not a GLiNER2 relation package"))
    })
}

pub(super) fn add(
    builder: Builder,
    model: Arc<Gliner2Relations>,
    config: ZeroShotRelationConfig,
) -> Result<Builder, Failure> {
    let annotator = ZeroShotRelationAnnotator::new(config, model)
        .map_err(|error| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, error.to_string()))?;
    Ok(builder.add(annotator).step(EvidenceStep::new(Key::<Relations>::DEFAULT)))
}
