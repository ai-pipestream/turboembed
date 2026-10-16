//! Training the document categorizer: `DocumentCategorizerME.train` over
//! the events of `DocumentCategorizerEventStream`
//! ([`turbo_doccat::DoccatEventStream`]), and the model file it returns
//! as a `DoccatModel`.
//!
//! The model is trained with the event trainer the parameters name and
//! built with the caller's language and the trainer's manifest entries;
//! `DoccatFactory` adds `factory` and `doccat.featureGenerators`, the
//! comma-separated class names of its feature generators.

use turbo_doccat::model::{COMPONENT_NAME, MODEL_ENTRY};
use turbo_doccat::{DoccatError, DoccatFactory, DoccatModel, doccat_events};
use turbo_formats::DocumentSample;
use turbo_maxent::ModelFile;
use turbo_train::{
    Event, ModelPackage, TrainedModel, TrainingConfiguration, TrainingParameters, TrainingReport, trainer,
};

use crate::error::Result;
use crate::package::{Stamp, add_artifacts};

/// A trained document categorizer model: the classifier, what the trainer
/// reports, and the `.bin` file.
#[derive(Debug, Clone)]
pub struct TrainedDoccatModel {
    pub model: TrainedModel,
    pub report: TrainingReport,
    pub package: ModelPackage,
}

impl TrainedDoccatModel {
    /// Opens the packaged model file as a [`DoccatModel`] with `factory`'s
    /// feature generators. The manifest records only their class names, so
    /// an n-gram generator with other than the default window, or an
    /// all-letter bag of words, is restored from `factory`.
    pub fn doccat_model(&self, factory: &DoccatFactory) -> std::result::Result<DoccatModel, DoccatError> {
        let file = ModelFile::open(self.package.to_bytes())?;
        Ok(DoccatModel::open(file)?.with_context_generator(factory.context_generator().clone()))
    }
}

/// `new DoccatModel(language, model, manifestInfoEntries, factory)`
/// serialized: the manifest with the trainer's entries and the factory's,
/// and the `doccat.model` entry holding `model_bytes`.
pub fn package_doccat_model(
    language: &str,
    factory: &DoccatFactory,
    report: &TrainingReport,
    model_bytes: Vec<u8>,
    stamp: Stamp<'_>,
) -> Result<ModelPackage> {
    let mut manifest = stamp.manifest(COMPONENT_NAME, language);
    manifest.set_training(report);
    for (k, v) in factory.manifest_entries() {
        manifest.set(&k, &v);
    }
    let mut package = ModelPackage::new(manifest);
    add_artifacts(&mut package, vec![(MODEL_ENTRY, model_bytes)])?;
    Ok(package)
}

/// `DocumentCategorizerME.train(language, samples, params, factory)`: the
/// events of [`doccat_events`], the event trainer the parameters name, and
/// the model file stamped with `stamp`.
pub fn train_doccat<'a, 's>(
    language: &str,
    samples: impl IntoIterator<Item = &'s DocumentSample<'a>>,
    params: &TrainingParameters,
    factory: &DoccatFactory,
    config: &mut TrainingConfiguration,
    stamp: Stamp<'_>,
) -> Result<TrainedDoccatModel>
where
    'a: 's,
{
    let events = doccat_events(samples, factory);
    train_doccat_from_events(language, &events, params, factory, config, stamp)
}

/// [`train_doccat`] from events already made.
pub fn train_doccat_from_events(
    language: &str,
    events: &[Event],
    params: &TrainingParameters,
    factory: &DoccatFactory,
    config: &mut TrainingConfiguration,
    stamp: Stamp<'_>,
) -> Result<TrainedDoccatModel> {
    let (model, report) = trainer::train_events(events, params, config)?;
    let bytes = model.write()?;
    let package = package_doccat_model(language, factory, &report, bytes, stamp)?;
    Ok(TrainedDoccatModel { model, report, package })
}
