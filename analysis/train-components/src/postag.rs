//! Training the part-of-speech tagger: `POSTaggerFactory`, the events of
//! `POSSampleEventStream`, and the model file `POSTaggerME.train` returns
//! as a `POSModel`.
//!
//! `POSSampleEventStream` makes one event per token of a sample: the
//! token's tag as the outcome, and as the context the features the
//! factory's generator builds for that token given the sample's own tags
//! as the tags before it (`ConfigurablePOSContextGenerator` with no cache,
//! which `getPOSContextGenerator()` returns for training, over the
//! factory's descriptor). The descriptor is the one the training tool was
//! given, or OpenNLP's default (`pos-default-features.xml`), which the
//! factory loads before the first context is asked for and which the
//! model file then carries as `generator.featuregen`.
//!
//! `POSModel`'s constructor checks a tag dictionary against the trained
//! model: every tag the dictionary lists must be an outcome, or
//! `validateArtifactMap` throws `InvalidFormatException`, which
//! `BaseModel.checkArtifactMap` wraps in an `IllegalArgumentException`,
//! and no model is written ([`Error::UnknownDictionaryTags`]).

use turbo_formats::PosSample;
use turbo_sequence::featuregen::descriptor;
use turbo_sequence::postag::{ConfigurablePosContext, DEFAULT_DESCRIPTOR};
use turbo_sequence::{ContextGenerator, Features, PatternMode};
use turbo_train::{
    Event, GisModel, JavaString, ModelPackage, TrainingConfiguration, TrainingParameters, TrainingReport, gis,
    write_gis,
};

use crate::error::{Error, Result};
use crate::package::{FACTORY_KEY, Stamp, add_artifacts, serializer_class_key};
use crate::tagdict::PosDictionary;

/// `POSModel.COMPONENT_NAME`.
pub const COMPONENT_NAME: &str = "POSTaggerME";
/// `POSModel.POS_MODEL_ENTRY_NAME`.
pub const MODEL_ENTRY: &str = "pos.model";
/// `POSModel.GENERATOR_DESCRIPTOR_ENTRY_NAME`.
pub const DESCRIPTOR_ENTRY: &str = "generator.featuregen";
/// `POSTaggerFactory.TAG_DICTIONARY_ENTRY_NAME`.
pub const TAG_DICTIONARY_ENTRY: &str = "tags.tagdict";
/// The factory class the manifest names.
pub const FACTORY: &str = "opennlp.tools.postag.POSTaggerFactory";
/// The class `BaseModel.serialize` records for the tag dictionary, a
/// `SerializableArtifact`.
pub const TAG_DICTIONARY_SERIALIZER: &str = "opennlp.tools.postag.POSTaggerFactory$POSDictionarySerializer";
/// `BeamSearch.BEAM_SIZE_PARAMETER`: the training parameter and the
/// manifest entry.
pub const BEAM_SIZE_KEY: &str = "BeamSize";
/// `POSTaggerME.DEFAULT_BEAM_SIZE`.
pub const DEFAULT_BEAM_SIZE: i32 = 3;

/// `POSTaggerFactory(featureGeneratorBytes, resources, posDictionary)` as
/// a training run uses it, with no resources: the descriptors a training
/// tool gives name no resource unless a `-resources` directory is given,
/// and a descriptor whose generators need one cannot be built here
/// ([`Error::Sequence`]).
///
/// With no descriptor (`null`) the factory loads OpenNLP's default one
/// before the first context generator is made, so a trained model always
/// carries a `generator.featuregen` entry.
#[derive(Debug, Clone)]
pub struct PosTaggerFactory {
    language: String,
    descriptor: Vec<u8>,
    tag_dictionary: Option<PosDictionary>,
}

impl PosTaggerFactory {
    /// `descriptor` is the feature generator XML, `None` for the default.
    /// It is built once here, so a descriptor the generators cannot be
    /// made from is refused before any training.
    pub fn new(language: &str, descriptor: Option<Vec<u8>>, tag_dictionary: Option<PosDictionary>) -> Result<Self> {
        let descriptor = descriptor.unwrap_or_else(|| DEFAULT_DESCRIPTOR.as_bytes().to_vec());
        let factory = PosTaggerFactory { language: language.to_owned(), descriptor, tag_dictionary };
        factory.context_generator()?;
        Ok(factory)
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    /// `getFeatureGenerator`: the descriptor bytes the model file carries.
    pub fn descriptor(&self) -> &[u8] {
        &self.descriptor
    }

    /// `getTagDictionary`.
    pub fn tag_dictionary(&self) -> Option<&PosDictionary> {
        self.tag_dictionary.as_ref()
    }

    /// `getPOSContextGenerator()`: the descriptor's generators with no
    /// cache, as the trainer uses them. Features are built over UTF-16
    /// code units, which is how OpenNLP builds them.
    pub fn context_generator(&self) -> Result<ConfigurablePosContext> {
        let generator = descriptor::build(&self.descriptor, None, PatternMode::Utf16Units)?;
        Ok(ConfigurablePosContext::new(generator, 0))
    }

    /// `createManifestEntries`, with `factory` as `BaseModel` adds it and
    /// the serializer class `BaseModel.serialize` records for the tag
    /// dictionary.
    pub fn manifest_entries(&self) -> Vec<(String, String)> {
        let mut entries = vec![(FACTORY_KEY.to_owned(), FACTORY.to_owned())];
        if self.tag_dictionary.is_some() {
            entries.push((serializer_class_key(TAG_DICTIONARY_ENTRY), TAG_DICTIONARY_SERIALIZER.to_owned()));
        }
        entries
    }
}

/// `POSSampleEventStream` over one factory's context generator.
pub struct PosEventStream {
    context: ConfigurablePosContext,
    features: Features,
}

impl PosEventStream {
    pub fn new(factory: &PosTaggerFactory) -> Result<PosEventStream> {
        Ok(PosEventStream { context: factory.context_generator()?, features: Features::new() })
    }

    /// `generateEvents(sentence, tags, cg)`: appends one event per token
    /// of the sample to `out`, in token order. The sample's tags stand in
    /// for the tags decided so far; the generators only read the tags of
    /// earlier tokens.
    ///
    /// The two slices of `&str` are built per sample: the generator is
    /// implemented over `&str`, and a sample's fields may own their text,
    /// so the slices cannot outlive the sample and be kept on the stream.
    /// Each token's event allocates far more than these two vectors.
    pub fn events(&mut self, sample: &PosSample<'_>, out: &mut Vec<Event>) {
        let tokens: Vec<&str> = sample.tokens().iter().map(|f| f.as_str()).collect();
        let tags: Vec<&str> = sample.tags().iter().map(|f| f.as_str()).collect();
        self.context.start_sequence();
        for i in 0..tokens.len() {
            self.features.clear();
            self.context.context(i, &tokens, &tags, &mut self.features);
            out.push(Event::new(tags[i], java_strings(&self.features)));
        }
    }
}

/// The features as the predicates the JVM would hold.
pub(crate) fn java_strings(features: &Features) -> Vec<JavaString> {
    features.iter().map(|f| JavaString::from_modified_utf8(f).expect("a feature is modified UTF-8")).collect()
}

/// The events of `POSSampleEventStream` over `samples`.
pub fn pos_events<'a, 's>(
    samples: impl IntoIterator<Item = &'s PosSample<'a>>,
    factory: &PosTaggerFactory,
) -> Result<Vec<Event>>
where
    'a: 's,
{
    let mut stream = PosEventStream::new(factory)?;
    let mut out = Vec::new();
    for sample in samples {
        stream.events(sample, &mut out);
    }
    Ok(out)
}

/// A trained part-of-speech model: the GIS model, what the trainer
/// reports, and the `.bin` file.
#[derive(Debug, Clone)]
pub struct TrainedPosModel {
    pub model: GisModel,
    pub report: TrainingReport,
    pub package: ModelPackage,
}

/// `mlParams.getIntParameter(BeamSize, DEFAULT_BEAM_SIZE)`.
pub fn beam_size(params: &TrainingParameters) -> Result<i32> {
    Ok(params.get_int(BEAM_SIZE_KEY)?.unwrap_or(DEFAULT_BEAM_SIZE))
}

/// `new POSModel(languageCode, posModel, beamSize, manifestInfoEntries,
/// posFactory)` serialized: the manifest with the trainer's entries, the
/// factory's and `BeamSize`; the factory's `tags.tagdict` when it has a
/// dictionary; `pos.model` holding `model_bytes`; and the descriptor as
/// `generator.featuregen`, in the JVM's zip order (the dictionary before
/// the manifest). The dictionary is not checked against the outcomes
/// here; [`train_pos_model_from_events`] does that with the trained model.
pub fn package_pos_model(
    factory: &PosTaggerFactory,
    report: &TrainingReport,
    model_bytes: Vec<u8>,
    beam_size: i32,
    stamp: Stamp<'_>,
) -> Result<ModelPackage> {
    let mut manifest = stamp.manifest(COMPONENT_NAME, factory.language());
    manifest.set_training(report);
    for (k, v) in factory.manifest_entries() {
        manifest.set(&k, &v);
    }
    manifest.set(BEAM_SIZE_KEY, &beam_size.to_string());
    let mut package = ModelPackage::new(manifest);
    let mut artifacts = Vec::new();
    if let Some(dictionary) = factory.tag_dictionary() {
        artifacts.push((TAG_DICTIONARY_ENTRY, dictionary.serialize()?));
    }
    artifacts.push((MODEL_ENTRY, model_bytes));
    artifacts.push((DESCRIPTOR_ENTRY, factory.descriptor().to_vec()));
    add_artifacts(&mut package, artifacts)?;
    Ok(package)
}

/// `POSTaggerME.train(languageCode, samples, mlParams, posFactory)` with
/// an event trainer: the events of [`pos_events`], the GIS trainer (with
/// the event checksum and `TrainerType` for the manifest), the beam size
/// of the parameters, and the model file stamped with `stamp`.
pub fn train_pos_model<'a, 's>(
    samples: impl IntoIterator<Item = &'s PosSample<'a>>,
    factory: &PosTaggerFactory,
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
    stamp: Stamp<'_>,
) -> Result<TrainedPosModel>
where
    'a: 's,
{
    let events = pos_events(samples, factory)?;
    train_pos_model_from_events(&events, factory, params, config, stamp)
}

/// [`train_pos_model`] from events already made. A tag dictionary with a
/// tag the model has no outcome for is refused, as `POSModel`'s
/// constructor refuses it.
pub fn train_pos_model_from_events(
    events: &[Event],
    factory: &PosTaggerFactory,
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
    stamp: Stamp<'_>,
) -> Result<TrainedPosModel> {
    let beam = beam_size(params)?;
    let (model, report) = gis::train_events(events, params, config)?;
    if let Some(dictionary) = factory.tag_dictionary() {
        let unknown = dictionary.tags_unknown_to(model.outcome_labels())?;
        if !unknown.is_empty() {
            return Err(Error::UnknownDictionaryTags(unknown));
        }
    }
    let bytes = write_gis(&model)?;
    let package = package_pos_model(factory, &report, bytes, beam, stamp)?;
    Ok(TrainedPosModel { model, report, package })
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_formats::native::WordTagReader;
    use turbo_train::Key;
    use turbo_unicode::WhitespaceMode;

    fn samples(text: &str) -> Vec<PosSample<'_>> {
        WordTagReader::new(text, WhitespaceMode::Unicode).unwrap().map(|s| s.unwrap()).collect()
    }

    fn strings(e: &Event) -> Vec<String> {
        e.context().iter().map(JavaString::to_string_lossy).collect()
    }

    #[test]
    fn the_default_descriptor_makes_one_event_per_token_with_the_samples_tags_as_history() {
        let f = PosTaggerFactory::new("en", None, None).unwrap();
        assert_eq!(f.descriptor(), DEFAULT_DESCRIPTOR.as_bytes());
        assert_eq!(f.manifest_entries(), [("factory".to_owned(), FACTORY.to_owned())]);
        let s = samples("The_DT cat_NN sat_VBD\n\nno tag_T\ntag_T\n");
        let events = pos_events(&s, &f).unwrap();
        let outcomes: Vec<&str> = events.iter().map(Event::outcome).collect();
        // The second line is a sample with no tokens, and so is the third
        // in place of a line with a token without `_`, as
        // `WordTagSampleStream` reads it; the fourth has one token.
        assert_eq!(s.len(), 4);
        assert_eq!(outcomes, ["DT", "NN", "VBD", "T"]);
        let second = strings(&events[1]);
        assert_eq!(second[0], "def");
        assert!(second.contains(&"w=cat".to_owned()) && second.contains(&"p1w=the".to_owned()), "{second:?}");
        // The previous tags are the sample's: the tag of "The", and both
        // previous tags for the third token.
        assert!(second.contains(&"t=DT".to_owned()), "{second:?}");
        let third = strings(&events[2]);
        assert!(third.contains(&"t2=DT,NN".to_owned()), "{third:?}");
        // The first token of the third line sees no history and the
        // sentence beginning, not the second line's tokens.
        let fourth = strings(&events[3]);
        assert!(fourth.contains(&"w=tag".to_owned()) && !fourth.iter().any(|x| x.starts_with("t=")), "{fourth:?}");
    }

    #[test]
    fn a_dictionary_adds_its_entry_and_serializer_and_must_fit_the_outcomes() {
        let d = PosDictionary::new(false, [("the".to_owned(), vec!["DT".to_owned()])]);
        let f = PosTaggerFactory::new("en", None, Some(d)).unwrap();
        assert_eq!(
            f.manifest_entries()[1],
            (serializer_class_key("tags.tagdict"), TAG_DICTIONARY_SERIALIZER.to_owned())
        );
        let report = TrainingReport { event_hash: Some(1), trainer_type: turbo_train::TrainerType::Event };
        let package = package_pos_model(&f, &report, vec![1], 3, Stamp::at(0)).unwrap();
        assert_eq!(package.entry_names(), ["tags.tagdict", "manifest.properties", "generator.featuregen", "pos.model"]);
        assert_eq!(package.manifest().get("BeamSize"), Some("3"));
        // Trained on tags that do not include DT, the dictionary is refused.
        let s = samples("a_X b_Y\n");
        let mut p = TrainingParameters::new();
        p.put(Key::ITERATIONS, 5);
        p.put(Key::CUTOFF, 0);
        let e = train_pos_model(&s, &f, &p, &mut TrainingConfiguration::default(), Stamp::at(0)).unwrap_err();
        assert!(matches!(&e, Error::UnknownDictionaryTags(t) if t == &["DT".to_owned()]), "{e}");
        assert_eq!(
            e.to_string(),
            "Tag dictionary contains tags which are unknown by the model! The unknown tags are: DT "
        );
    }

    #[test]
    fn the_beam_size_comes_from_the_parameters() {
        let mut p = TrainingParameters::new();
        assert_eq!(beam_size(&p).unwrap(), 3);
        p.put(BEAM_SIZE_KEY, 7);
        assert_eq!(beam_size(&p).unwrap(), 7);
        p.put_str(BEAM_SIZE_KEY, "x");
        assert!(beam_size(&p).is_err());
    }

    #[test]
    fn a_descriptor_that_cannot_be_built_is_refused() {
        let e = PosTaggerFactory::new("en", Some(b"<generators/>".to_vec()), None).unwrap_err();
        assert!(matches!(e, Error::Sequence(_)), "{e}");
    }
}
