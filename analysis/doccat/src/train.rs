//! What training a document categorizer model needs from this crate
//! (`DoccatFactory` and `DocumentCategorizerEventStream`). Training and
//! packaging the model file (`DocumentCategorizerME.train`) is
//! `turbo_train_components::doccat`, which depends on this crate.

use std::collections::BTreeMap;

use turbo_formats::DocumentSample;
use turbo_train::{DataIndex, Event, IndexParams, JavaString, TrainingParameters};

use crate::error::DoccatError;
use crate::features::{BagOfWordsFeatureGenerator, BuiltinFeatureGenerator, DoccatContextGenerator};
use crate::model::{FACTORY_NAME, FEATURE_GENERATORS_KEY};

/// `BaseModel.FACTORY_NAME`: the manifest entry naming the factory class.
const FACTORY_KEY: &str = "factory";

/// Factory configuring the feature generators of a `DoccatModel`, ported from
/// `opennlp.tools.doccat.DoccatFactory`.
#[derive(Clone, Debug)]
pub struct DoccatFactory {
    context_generator: DoccatContextGenerator,
}

impl Default for DoccatFactory {
    /// Default `DoccatFactory` with a single `BagOfWordsFeatureGenerator`.
    fn default() -> Self {
        Self::new()
    }
}

impl DoccatFactory {
    /// Creates a `DoccatFactory` with the default `BagOfWordsFeatureGenerator`.
    pub fn new() -> Self {
        Self { context_generator: DoccatContextGenerator::default() }
    }

    /// Creates a `DoccatFactory` with the given built-in feature generators.
    pub fn with_feature_generators(generators: Vec<BuiltinFeatureGenerator>) -> Self {
        Self { context_generator: DoccatContextGenerator::from_builtins(generators) }
    }

    /// Creates a `DoccatFactory` with a custom [`DoccatContextGenerator`].
    pub fn with_context_generator(context_generator: DoccatContextGenerator) -> Self {
        Self { context_generator }
    }

    /// Returns the built-in feature generators configured on this factory.
    pub fn feature_generators(&self) -> &[BuiltinFeatureGenerator] {
        self.context_generator.builtins()
    }

    /// Returns the [`DoccatContextGenerator`] used during training and inference.
    pub fn context_generator(&self) -> &DoccatContextGenerator {
        &self.context_generator
    }

    /// Returns the manifest entries written by `DoccatFactory.createManifestEntries()`:
    /// `factory = opennlp.tools.doccat.DoccatFactory` and, when built-in
    /// generators are present, `doccat.featureGenerators = <comma-separated class names>`.
    pub fn manifest_entries(&self) -> Vec<(String, String)> {
        let mut entries = vec![(FACTORY_KEY.to_owned(), FACTORY_NAME.to_owned())];
        if !self.context_generator.builtins().is_empty() {
            let classes: Vec<&str> =
                self.context_generator.builtins().iter().map(BuiltinFeatureGenerator::class_name).collect();
            entries.push((FEATURE_GENERATORS_KEY.to_owned(), classes.join(",")));
        } else {
            entries.push((
                FEATURE_GENERATORS_KEY.to_owned(),
                BuiltinFeatureGenerator::BagOfWords(BagOfWordsFeatureGenerator::new()).class_name().to_owned(),
            ));
        }
        entries
    }
}

/// Event stream converting document samples into training [`Event`]s, ported
/// from `opennlp.tools.doccat.DocumentCategorizerEventStream`.
#[derive(Clone, Debug, Default)]
pub struct DoccatEventStream {
    context_generator: DoccatContextGenerator,
    scratch: Vec<String>,
}

impl DoccatEventStream {
    /// Creates a `DoccatEventStream` using `factory`'s feature generators.
    pub fn new(factory: &DoccatFactory) -> Self {
        Self { context_generator: factory.context_generator().clone(), scratch: Vec::new() }
    }

    /// Creates one training [`Event`] for `category`, `tokens`, and
    /// `extra_information`.
    pub fn event(&mut self, category: &str, tokens: &[&str], extra_information: &BTreeMap<String, String>) -> Event {
        self.context_generator.extract_features(tokens, extra_information, &mut self.scratch);
        let predicates: Vec<JavaString> = self.scratch.iter().map(|s| JavaString::new(s.as_str())).collect();
        Event::new(JavaString::new(category), predicates)
    }

    /// Appends the training [`Event`] for a `turbo_formats::DocumentSample` to `out`.
    pub fn sample_event(&mut self, sample: &DocumentSample<'_>, out: &mut Vec<Event>) {
        let tokens: Vec<&str> = sample.tokens.iter().map(|f| f.as_str()).collect();
        let extra = BTreeMap::new();
        out.push(self.event(sample.category.as_str(), &tokens, &extra));
    }
}

/// Converts a sequence of `DocumentSample`s into training [`Event`]s using
/// `factory`.
pub fn doccat_events<'a, 's>(
    samples: impl IntoIterator<Item = &'s DocumentSample<'a>>,
    factory: &DoccatFactory,
) -> Vec<Event>
where
    'a: 's,
{
    let mut stream = DoccatEventStream::new(factory);
    let mut out = Vec::new();
    for sample in samples {
        stream.sample_event(sample, &mut out);
    }
    out
}

/// Builds the [`DataIndex`] that `DocumentCategorizerME.train` constructs for
/// `events` and `params`.
pub fn doccat_index(events: &[Event], params: &TrainingParameters) -> Result<DataIndex, DoccatError> {
    Ok(DataIndex::index(events, &IndexParams::from_training_parameters(params)?)?)
}
