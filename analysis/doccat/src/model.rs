//! Document categorizer model (`DoccatModel`), ported from
//! `opennlp.tools.doccat.DoccatModel`.

use std::sync::Arc;

use turbo_analysis::{ModelRef, Tag, TagSet};
use turbo_maxent::{MaxentModel, ModelFile};

use crate::error::DoccatError;
use crate::features::{BagOfWordsFeatureGenerator, BuiltinFeatureGenerator, DoccatContextGenerator};

/// `DoccatModel.COMPONENT_NAME`.
pub const COMPONENT_NAME: &str = "DocumentCategorizerME";
/// `DoccatModel.DOCCAT_MODEL_ENTRY_NAME`.
pub const MODEL_ENTRY: &str = "doccat.model";
/// Canonical class name of `DoccatFactory`.
pub const FACTORY_NAME: &str = "opennlp.tools.doccat.DoccatFactory";
/// Manifest property storing the comma-separated feature generator class names.
pub const FEATURE_GENERATORS_KEY: &str = "doccat.featureGenerators";

/// An opened or trained `DoccatModel` holding the `doccat.model` classifier,
/// the category `TagSet` in model outcome order, and the configured feature
/// generators from `DoccatFactory`.
#[derive(Clone, Debug)]
pub struct DoccatModel {
    language: String,
    maxent: Arc<MaxentModel>,
    categories: Arc<TagSet>,
    context_generator: DoccatContextGenerator,
    model_ref: ModelRef,
}

impl DoccatModel {
    /// Opens a `DoccatModel` from an OpenNLP `.bin` `ModelFile`.
    ///
    /// Validates `Component-Name == "DocumentCategorizerME"`, checks `factory`
    /// if present in the manifest, parses `doccat.featureGenerators` (or
    /// defaults to `BagOfWordsFeatureGenerator`), and loads `doccat.model`.
    pub fn open(file: ModelFile) -> Result<Self, DoccatError> {
        if file.component_name() != COMPONENT_NAME {
            return Err(DoccatError::Component(file.component_name().into()));
        }
        if let Some(factory) = file.manifest().get("factory")
            && factory != FACTORY_NAME
        {
            return Err(DoccatError::Factory(factory.into()));
        }
        let builtins = match file.manifest().get(FEATURE_GENERATORS_KEY) {
            Some(prop) => parse_feature_generators(prop)?,
            None => vec![BuiltinFeatureGenerator::BagOfWords(BagOfWordsFeatureGenerator::new())],
        };
        let maxent = file.maxent(MODEL_ENTRY)?;
        if let Err(outcome) = maxent.outcome_texts() {
            return Err(DoccatError::OutcomeNotText(outcome.clone()));
        }
        let labels: Vec<&str> = (0..maxent.num_outcomes())
            .map(|i| maxent.outcome_str(i).expect("outcomes were verified to be valid UTF-8"))
            .collect();
        let categories = TagSet::new("opennlp:doccat", &labels);
        let language = file.language().to_owned();
        let model_ref = ModelRef {
            id: format!("opennlp:{COMPONENT_NAME}:{language}"),
            revision: file.opennlp_version().into(),
            sha256: file.sha256().to_hex(),
        };
        Ok(Self {
            language,
            maxent: Arc::new(maxent),
            categories,
            context_generator: DoccatContextGenerator::from_builtins(builtins),
            model_ref,
        })
    }

    /// Constructs an in-memory `DoccatModel` from a `MaxentModel` and a
    /// `DoccatContextGenerator`.
    pub fn from_maxent(
        language: impl Into<String>,
        maxent: MaxentModel,
        context_generator: DoccatContextGenerator,
    ) -> Result<Self, DoccatError> {
        if let Err(outcome) = maxent.outcome_texts() {
            return Err(DoccatError::OutcomeNotText(outcome.clone()));
        }
        let labels: Vec<&str> = (0..maxent.num_outcomes())
            .map(|i| maxent.outcome_str(i).expect("outcomes were verified to be valid UTF-8"))
            .collect();
        let categories = TagSet::new("opennlp:doccat", &labels);
        let language = language.into();
        let model_ref = ModelRef {
            id: format!("opennlp:{COMPONENT_NAME}:{language}"),
            revision: "3.0.0".into(),
            sha256: String::new(),
        };
        Ok(Self { language, maxent: Arc::new(maxent), categories, context_generator, model_ref })
    }

    /// Replaces the `DoccatContextGenerator` (for example, when attaching
    /// custom `NGramFeatureGenerator` window sizes or `BagOfWordsFeatureGenerator`
    /// all-letter settings after loading a `.bin` archive whose manifest stores
    /// only class names).
    pub fn with_context_generator(mut self, context_generator: DoccatContextGenerator) -> Self {
        self.context_generator = context_generator;
        self
    }

    /// Returns the model's language code (`Language-Code` in `manifest.properties`).
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Returns the underlying `MaxentModel` (`doccat.model`).
    pub fn maxent(&self) -> &MaxentModel {
        &self.maxent
    }

    /// Returns the closed `TagSet` of category names in outcome order.
    pub fn categories(&self) -> &Arc<TagSet> {
        &self.categories
    }

    /// Returns the number of categories in the model (`getNumberOfCategories`).
    pub fn num_categories(&self) -> usize {
        self.categories.len()
    }

    /// Returns the category name at `index` (`getCategory`).
    pub fn category(&self, index: usize) -> Option<&str> {
        if index < self.categories.len() { Some(self.categories.label(Tag(index as u16))) } else { None }
    }

    /// Returns the outcome index of `category` (`getIndex`).
    pub fn index(&self, category: &str) -> Option<usize> {
        self.categories.tag(category).map(|t| t.0 as usize)
    }

    /// Returns the configured [`DoccatContextGenerator`].
    pub fn context_generator(&self) -> &DoccatContextGenerator {
        &self.context_generator
    }

    /// Returns the built-in feature generators configured in `DoccatFactory`.
    pub fn feature_generators(&self) -> &[BuiltinFeatureGenerator] {
        self.context_generator.builtins()
    }

    /// Returns the [`ModelRef`] recorded in layer provenance.
    pub fn model_ref(&self) -> &ModelRef {
        &self.model_ref
    }
}

/// Parses the comma-separated `doccat.featureGenerators` manifest property,
/// matching `DoccatFactory.getFeatureGenerators()`.
pub fn parse_feature_generators(prop: &str) -> Result<Vec<BuiltinFeatureGenerator>, DoccatError> {
    let mut out = Vec::new();
    for part in prop.split(',') {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            out.push(BuiltinFeatureGenerator::from_class_name(trimmed)?);
        }
    }
    if out.is_empty() {
        out.push(BuiltinFeatureGenerator::BagOfWords(BagOfWordsFeatureGenerator::new()));
    }
    Ok(out)
}
