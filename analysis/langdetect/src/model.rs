//! A language detector model: OpenNLP's `LanguageDetectorModel`.

use turbo_analysis::ModelRef;
use turbo_maxent::{MaxentModel, ModelFile};

use crate::error::ModelError;

/// The classifier's entry in the model archive.
pub const MODEL_ENTRY: &str = "langdetect.model";
/// The one factory class whose behaviour this port reproduces.
pub const FACTORY: &str = "opennlp.tools.langdetect.LanguageDetectorFactory";
/// The component name OpenNLP checks.
pub const COMPONENT: &str = "LanguageDetectorME";

/// An opened `LanguageDetectorME` model: a maxent classifier whose
/// outcomes are language codes (ISO 639-3 in the models OpenNLP
/// publishes).
#[derive(Debug)]
pub struct LanguageModel {
    maxent: MaxentModel,
    model_ref: ModelRef,
}

impl LanguageModel {
    /// Reads the model from an opened model file, checking what OpenNLP
    /// checks: the component name and a classifier in `langdetect.model`.
    /// A manifest without a `factory` gets OpenNLP's default factory, as
    /// on the JVM; one naming any other factory is refused.
    pub fn open(file: ModelFile) -> Result<LanguageModel, ModelError> {
        if file.component_name() != COMPONENT {
            return Err(ModelError::Component(file.component_name().into()));
        }
        if let Some(factory) = file.manifest().get("factory")
            && factory != FACTORY
        {
            return Err(ModelError::Factory(factory.into()));
        }
        let maxent = file.maxent(MODEL_ENTRY)?;
        if let Err(outcome) = maxent.outcome_texts() {
            return Err(ModelError::OutcomeNotText(outcome.clone()));
        }
        let model_ref = ModelRef {
            id: format!("opennlp:{COMPONENT}:{}", file.language()),
            revision: file.opennlp_version().into(),
            sha256: file.sha256().to_hex(),
        };
        Ok(LanguageModel { maxent, model_ref })
    }

    /// The classifier.
    pub fn maxent(&self) -> &MaxentModel {
        &self.maxent
    }

    /// The language codes the model can answer, in outcome order
    /// (`getSupportedLanguages`).
    pub fn supported_languages(&self) -> impl Iterator<Item = &str> {
        (0..self.maxent.num_outcomes()).map(|i| self.code(i))
    }

    /// The language code of outcome `i`; `open` refused any model whose
    /// outcomes are not text.
    pub fn code(&self, i: usize) -> &str {
        self.maxent.outcome_str(i).expect("the outcomes were checked to be text when the model was opened")
    }

    /// What layers record as their model: `opennlp:LanguageDetectorME:`
    /// and the manifest's language (`und` for the published models), the
    /// OpenNLP version that trained it, and the SHA-256 of the model file.
    pub fn model_ref(&self) -> &ModelRef {
        &self.model_ref
    }
}
