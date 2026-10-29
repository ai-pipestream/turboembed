//! Training the language detector: `LanguageDetectorFactory`, the events
//! of `LanguageDetectorEventStream`, and the model file
//! `LanguageDetectorME.train` returns as a `LanguageDetectorModel`.
//!
//! `LanguageDetectorEventStream` makes one event per sample: the sample's
//! language code as the outcome, and as the context the distinct
//! character n-grams `DefaultLanguageDetectorContextGenerator.getContext`
//! returns for its text (n-grams of 1 to 3 UTF-16 units of the text after
//! the factory's five normalizers, lowercased), in the iteration order of
//! the `HashMap` that counted them
//! ([`turbo_langdetect::ContextGenerator::training_features`]). A text the
//! normalizers leave empty gives an event with no predicates; the stream
//! keeps it and the indexer drops it, as it drops every event left with
//! no predicate after the cutoff (its outcome is still numbered).
//!
//! `LanguageDetectorME.train` puts `DataIndexer=OnePass` into the
//! parameters unless they name an indexer, trains with the event trainer
//! the parameters name, and builds the model with the language `und` and
//! the trainer's manifest entries; the factory adds only `factory`.
//!
//! The JVM orders the n-grams of a `HashMap` bucket that became a red-black
//! tree (nine or more in one bucket of a table of 64 or more) with equal
//! hash codes by `System.identityHashCode`, which no port can follow.
//! Such an event has the JVM's predicates in another order: the indexer
//! sees the same predicates, but sorts and merges the event among the
//! others by its predicates in that order, so the trained parameters can
//! differ in the last bits. [`LanguageEventStream::inexact_events`]
//! counts those events, and [`TrainedLanguageModel::inexact_events`]
//! reports them for a training run.

use turbo_formats::LanguageSample;
use turbo_langdetect::ContextGenerator;
use turbo_train::{
    DataIndex, Event, IndexParams, JavaString, Key, ModelPackage, TrainedModel, TrainingConfiguration,
    TrainingParameters, TrainingReport, trainer,
};

use crate::error::Result;
use crate::package::{FACTORY_KEY, Stamp, add_artifacts};

pub mod cross_validate;
pub use cross_validate::{LanguageCrossValidation, LanguageDetectorCrossValidator, LanguageFold};

/// `LanguageDetectorModel.COMPONENT_NAME`.
pub const COMPONENT_NAME: &str = turbo_langdetect::model::COMPONENT;
/// `LanguageDetectorModel.LANGDETECT_MODEL_ENTRY_NAME`.
pub const MODEL_ENTRY: &str = turbo_langdetect::model::MODEL_ENTRY;
/// The factory class the manifest names.
pub const FACTORY: &str = turbo_langdetect::model::FACTORY;
/// The language `LanguageDetectorModel` records for every model.
pub const LANGUAGE: &str = "und";
/// `AbstractEventTrainer.DATA_INDEXER_ONE_PASS_VALUE`, the indexer
/// `LanguageDetectorME.train` uses unless the parameters name one.
pub const ONE_PASS: &str = "OnePass";

/// `new LanguageDetectorFactory()`: no settings and no artifacts. Its
/// context generator is `DefaultLanguageDetectorContextGenerator(1, 3)`
/// over the emoji, URL, social media, number and shrink normalizers.
#[derive(Debug, Clone, Copy, Default)]
pub struct LanguageDetectorFactory;

impl LanguageDetectorFactory {
    pub fn new() -> LanguageDetectorFactory {
        LanguageDetectorFactory
    }

    /// `getContextGenerator`.
    pub fn context_generator(&self) -> ContextGenerator {
        ContextGenerator::default()
    }

    /// `factory`, as `BaseModel` adds it; the factory has no entries of
    /// its own.
    pub fn manifest_entries(&self) -> Vec<(String, String)> {
        vec![(FACTORY_KEY.to_owned(), FACTORY.to_owned())]
    }
}

/// The parameters the command line trainer uses when it is given none,
/// `ModelUtil.createDefaultTrainingParameters()`: `Algorithm=MAXENT`,
/// `Iterations=100` and `Cutoff=5`, without a `TrainerType`. The
/// manifest of the published `langdetect-183.bin` records no training
/// parameters (only `TrainerType=Event` and the event hash); these are
/// most likely the parameters it was trained with.
pub fn default_training_parameters() -> TrainingParameters {
    let mut p = TrainingParameters::new();
    p.put(Key::ALGORITHM, turbo_train::Algorithm::Maxent.name());
    p.put(Key::ITERATIONS, turbo_train::params::DEFAULT_ITERATIONS);
    p.put(Key::CUTOFF, turbo_train::params::DEFAULT_CUTOFF);
    p
}

/// The parameters `LanguageDetectorME.train` hands the trainer: `params`
/// with `DataIndexer=OnePass` put if absent.
pub fn trainer_parameters(params: &TrainingParameters) -> TrainingParameters {
    let mut params = params.clone();
    params.put_if_absent(Key::DATA_INDEXER, ONE_PASS);
    params
}

/// `LanguageDetectorEventStream` over the factory's context generator.
#[derive(Debug, Default)]
pub struct LanguageEventStream {
    context: ContextGenerator,
    inexact: usize,
}

impl LanguageEventStream {
    pub fn new(factory: &LanguageDetectorFactory) -> LanguageEventStream {
        LanguageEventStream { context: factory.context_generator(), inexact: 0 }
    }

    /// The event of a text in `language`: the language as the outcome, the
    /// n-grams of `getContext(text)` as the predicates.
    pub fn event(&mut self, language: &str, text: &str) -> Event {
        self.context.generate(text);
        if !self.context.training_exact_order() {
            self.inexact += 1;
        }
        let predicates = self
            .context
            .training_features()
            .map(|gram| {
                JavaString::from_modified_utf8(gram).expect("the context generator writes well formed modified UTF-8")
            })
            .collect();
        Event::new(JavaString::new(language), predicates)
    }

    /// `createEvents(sample)`: appends the sample's one event to `out`.
    pub fn events(&mut self, sample: &LanguageSample<'_>, out: &mut Vec<Event>) {
        out.push(self.event(sample.language.as_str(), sample.context.as_str()));
    }

    /// How many of the events made so far hold their n-grams in an order
    /// other than the JVM's (see the module documentation).
    pub fn inexact_events(&self) -> usize {
        self.inexact
    }
}

/// The events of `LanguageDetectorEventStream` over `samples`, and how
/// many of them are not in the JVM's predicate order.
pub fn language_events<'a, 's>(
    samples: impl IntoIterator<Item = &'s LanguageSample<'a>>,
    factory: &LanguageDetectorFactory,
) -> (Vec<Event>, usize)
where
    'a: 's,
{
    let mut stream = LanguageEventStream::new(factory);
    let mut out = Vec::new();
    for sample in samples {
        stream.events(sample, &mut out);
    }
    let inexact = stream.inexact_events();
    (out, inexact)
}

/// A trained language detector model: the classifier, what the trainer
/// reports, the `.bin` file, and how many events were not in the JVM's
/// predicate order.
#[derive(Debug, Clone)]
pub struct TrainedLanguageModel {
    pub model: TrainedModel,
    pub report: TrainingReport,
    pub package: ModelPackage,
    pub inexact_events: usize,
}

/// `new LanguageDetectorModel(model, manifestInfoEntries, factory)`
/// serialized: the manifest with the language `und`, the trainer's
/// entries and `factory`, and the `langdetect.model` entry holding
/// `model_bytes`.
pub fn package_language_model(
    factory: &LanguageDetectorFactory,
    report: &TrainingReport,
    model_bytes: Vec<u8>,
    stamp: Stamp<'_>,
) -> Result<ModelPackage> {
    let mut manifest = stamp.manifest(COMPONENT_NAME, LANGUAGE);
    manifest.set_training(report);
    for (k, v) in factory.manifest_entries() {
        manifest.set(&k, &v);
    }
    let mut package = ModelPackage::new(manifest);
    add_artifacts(&mut package, vec![(MODEL_ENTRY, model_bytes)])?;
    Ok(package)
}

/// `LanguageDetectorME.train(samples, params, factory)`: the events of
/// [`language_events`], the event trainer the parameters name (with
/// `DataIndexer=OnePass` unless they name an indexer), and the model file
/// stamped with `stamp`.
pub fn train_language_model<'a, 's>(
    samples: impl IntoIterator<Item = &'s LanguageSample<'a>>,
    factory: &LanguageDetectorFactory,
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
    stamp: Stamp<'_>,
) -> Result<TrainedLanguageModel>
where
    'a: 's,
{
    let (events, inexact) = language_events(samples, factory);
    let mut trained = train_language_model_from_events(&events, factory, params, config, stamp)?;
    trained.inexact_events = inexact;
    Ok(trained)
}

/// [`train_language_model`] from events already made; the result counts
/// no inexact events, since events carry no record of their order.
pub fn train_language_model_from_events(
    events: &[Event],
    factory: &LanguageDetectorFactory,
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
    stamp: Stamp<'_>,
) -> Result<TrainedLanguageModel> {
    let params = trainer_parameters(params);
    let (model, report) = trainer::train_events(events, &params, config)?;
    let bytes = model.write()?;
    let package = package_language_model(factory, &report, bytes, stamp)?;
    Ok(TrainedLanguageModel { model, report, package, inexact_events: 0 })
}

/// The data index `LanguageDetectorME.train` builds for `events`: the
/// indexer and cutoff of [`trainer_parameters`]. Used to compare the
/// outcomes and predicates with the JVM's before training.
pub fn language_index(events: &[Event], params: &TrainingParameters) -> Result<DataIndex> {
    let params = trainer_parameters(params);
    Ok(DataIndex::index(events, &IndexParams::from_training_parameters(&params)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_formats::native::LanguageReader;

    fn samples(text: &str) -> Vec<LanguageSample<'_>> {
        LanguageReader::new(text).unwrap().map(|s| s.unwrap()).collect()
    }

    fn strings(e: &Event) -> Vec<String> {
        e.context().iter().map(JavaString::to_string_lossy).collect()
    }

    #[test]
    fn one_event_per_sample_with_the_first_map_s_order() {
        let f = LanguageDetectorFactory::new();
        let s = samples("eng\tabab\nfra\t\nnone\ndeu\t123\n");
        let (events, inexact) = language_events(&s, &f);
        assert_eq!(inexact, 0);
        assert_eq!(events.len(), 3);
        assert_eq!(*events[0].outcome(), "eng");
        // The order of the turbo-langdetect test of `getContext` for "abab".
        assert_eq!(strings(&events[0]), ["a", "ab", "b", "ba", "bab", "aba"]);
        // An empty text gives an event with no predicates.
        assert_eq!(*events[1].outcome(), "fra");
        assert!(events[1].context().is_empty());
        // The number normalizer leaves a space for the digits, and the
        // shrink normalizer trims it away.
        assert!(events[2].context().is_empty());
        assert_eq!(f.manifest_entries(), [("factory".to_owned(), FACTORY.to_owned())]);
    }

    #[test]
    fn an_astral_character_becomes_a_space_and_no_half_pair_is_a_predicate() {
        // The emoji normalizer turns every run of supplementary characters
        // into one space, emoji or not (U+10428 is a Deseret letter), so a
        // text read from a file never gives an n-gram holding half a pair.
        let f = LanguageDetectorFactory::new();
        let mut stream = LanguageEventStream::new(&f);
        let event = stream.event("dsrt", "a\u{10428}\u{10429}b");
        let mut grams = strings(&event);
        grams.sort();
        assert_eq!(grams, [" ", " b", "a", "a ", "a b", "b"]);
    }

    #[test]
    fn the_parameters_get_the_one_pass_indexer_unless_they_name_one() {
        let p = trainer_parameters(&default_training_parameters());
        assert_eq!(p.get_str(Key::DATA_INDEXER).unwrap(), Some(ONE_PASS));
        assert_eq!(p.get_str(Key::ALGORITHM).unwrap(), Some("MAXENT"));
        assert_eq!(p.get_int(Key::ITERATIONS).unwrap(), Some(100));
        assert_eq!(p.get_int(Key::CUTOFF).unwrap(), Some(5));
        assert_eq!(p.get(Key::TRAINER_TYPE), None);
        let mut two = default_training_parameters();
        two.put(Key::DATA_INDEXER, "TwoPass");
        assert_eq!(trainer_parameters(&two).get_str(Key::DATA_INDEXER).unwrap(), Some("TwoPass"));
    }

    #[test]
    fn the_model_file_is_und_with_the_factory() {
        let f = LanguageDetectorFactory::new();
        let report = TrainingReport { event_hash: Some(1), trainer_type: turbo_train::TrainerType::Event };
        let package = package_language_model(&f, &report, vec![1], Stamp::at(0)).unwrap();
        // The order of the published langdetect-183.bin.
        assert_eq!(package.entry_names(), ["manifest.properties", "langdetect.model"]);
        assert_eq!(package.manifest().get("factory"), Some(FACTORY));
        assert_eq!(package.manifest().component_name(), "LanguageDetectorME");
        assert_eq!(package.manifest().language(), "und");
    }
}
