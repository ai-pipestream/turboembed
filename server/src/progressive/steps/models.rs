//! The in-process form of a served request that names a GLiNER and a
//! GLiNER2 package.

use turbo_analysis::{Analyzer, Document};
use turbo_gliner::ExtractionSchema;
use turbo_relation::ZeroShotRelationConfig;

use super::{bad, compile_pipeline_with};
use crate::api::Failure;
use crate::vocabulary::ResolvedVocabulary;

/// The `Entities` instance the place-name gazetteer writes when GLiNER
/// writes the default one in the same request.
pub const PLACES: &str = "places";

/// The models [`compile_analyzer_with_models`] runs `turbo:gliner` and
/// `turbo:zero-shot-relations` with, in place of the loaded packages a
/// served request names.
#[derive(Default)]
pub struct Models {
    /// A GLiNER package's model and the label schema it extracts.
    pub gliner: Option<(std::sync::Arc<turbo_gliner::Model>, ExtractionSchema)>,
    /// A GLiNER2 package's relation model and its relation labels.
    pub relations: Option<(std::sync::Arc<turbo_relation::Gliner2Relations>, ZeroShotRelationConfig)>,
}

/// [`compile_analyzer`] with the model steps resolved against `models`: the
/// in-process form of a served request that names a GLiNER and a GLiNER2
/// package. A model step without its model, or a model without its step,
/// is refused as the served request refuses it.
pub fn compile_analyzer_with_models(
    initial_doc: &Document,
    step_names: &[&str],
    models: Models,
) -> Result<Analyzer, Failure> {
    let canonical: Vec<_> = step_names.iter().map(|&s| turbo_analysis::ids::canonical_layer_name(s)).collect();
    let wants = |step: &str| canonical.iter().any(|s| s.as_ref() == step);
    for (step, given) in
        [("turbo:gliner", models.gliner.is_some()), ("turbo:zero-shot-relations", models.relations.is_some())]
    {
        match (wants(step), given) {
            (true, false) => return Err(bad(format!("{step} requires its model"))),
            (false, true) => return Err(bad(format!("a model is given but the steps do not include {step}"))),
            _ => {}
        }
    }
    let vocabulary = ResolvedVocabulary::default();
    compile_pipeline_with(initial_doc, step_names, None, models.gliner, models.relations, "", None, &vocabulary)
}
