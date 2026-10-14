//! Training the lemmatizer: `LemmatizerFactory`, the events of
//! `LemmaSampleEventStream`, and the model file `LemmatizerME.train`
//! returns as a `LemmatizerModel`.
//!
//! `LemmaSampleEventStream` first turns a sample's lemmas into classes,
//! `LemmatizerME.encodeLemmas`: the shortest edit script from each token
//! to its lemma (`turbo_lemmatizer::edit_script`), `_` when the script is
//! empty (a token or lemma with no characters). Then it makes one event
//! per token: the class as the outcome, and the features of
//! `DefaultLemmatizerContextGenerator` for the token, its tag and the
//! class of the token before it. The factory adds nothing to the model
//! file but `factory` and `BeamSize` in the manifest.
//!
//! A class is written over UTF-16 units, so a token or lemma with a
//! character outside the Basic Multilingual Plane can give a class that
//! holds half of a surrogate pair. The JVM trains on such a class; a Rust
//! string cannot hold it, so a sample that makes one is refused
//! ([`Error::LemmaClassSurrogate`]; issue #404 tracks holding the outcome
//! as a `JavaString`).

use turbo_formats::LemmaSample;
use turbo_lemmatizer::LemmaContext;
use turbo_lemmatizer::edit_script;
use turbo_sequence::Features;
use turbo_train::{
    Event, GisModel, ModelPackage, TrainingConfiguration, TrainingParameters, TrainingReport, gis, write_gis,
};

use crate::error::{Error, Result};
use crate::package::{FACTORY_KEY, Stamp, add_artifacts};
use crate::postag::{BEAM_SIZE_KEY, beam_size, java_strings};

/// `LemmatizerModel.COMPONENT_NAME`.
pub const COMPONENT_NAME: &str = "StatisticalLemmatizer";
/// `LemmatizerModel.LEMMATIZER_MODEL_ENTRY_NAME`.
pub const MODEL_ENTRY: &str = "lemmatizer.model";
/// The factory class the manifest names.
pub const FACTORY: &str = "opennlp.tools.lemmatizer.LemmatizerFactory";
/// `LemmatizerME.DEFAULT_BEAM_SIZE`.
pub const DEFAULT_BEAM_SIZE: i32 = 3;
/// The class of a token whose edit script is empty (`encodeLemmas`).
pub const EMPTY_CLASS: &str = "_";

/// `new LemmatizerFactory()`: it has no settings and no artifacts; the
/// language only names the model.
#[derive(Debug, Clone)]
pub struct LemmatizerFactory {
    language: String,
}

impl LemmatizerFactory {
    pub fn new(language: &str) -> LemmatizerFactory {
        LemmatizerFactory { language: language.to_owned() }
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    /// `getContextGenerator`.
    pub fn context_generator(&self) -> LemmaContext {
        LemmaContext
    }

    /// `factory`, as `BaseModel` adds it; the factory has no entries of
    /// its own.
    pub fn manifest_entries(&self) -> Vec<(String, String)> {
        vec![(FACTORY_KEY.to_owned(), FACTORY.to_owned())]
    }
}

/// `LemmatizerME.encodeLemmas(toks, lemmas)` for one token: the shortest
/// edit script, `_` when it is empty; `None` when the script holds half
/// of a surrogate pair.
pub fn lemma_class(token: &str, lemma: &str) -> Option<String> {
    let class = edit_script::encode(token, lemma)?;
    Some(if class.is_empty() { EMPTY_CLASS.to_owned() } else { class })
}

/// `LemmaSampleEventStream` over the default context generator.
pub struct LemmaEventStream {
    context: LemmaContext,
    features: Features,
    classes: Vec<String>,
}

impl LemmaEventStream {
    pub fn new(factory: &LemmatizerFactory) -> LemmaEventStream {
        LemmaEventStream { context: factory.context_generator(), features: Features::new(), classes: Vec::new() }
    }

    /// `createEvents(sample)`: appends one event per token to `out`, in
    /// token order, after encoding the sample's lemmas. A sample with no
    /// tokens gives no event.
    pub fn events(&mut self, sample: &LemmaSample<'_>, out: &mut Vec<Event>) -> Result<()> {
        self.classes.clear();
        for (token, lemma) in sample.tokens().iter().zip(sample.lemmas()) {
            let class = lemma_class(token.as_str(), lemma.as_str())
                .ok_or_else(|| Error::LemmaClassSurrogate { token: token.as_str().to_owned() })?;
            self.classes.push(class);
        }
        for (i, (token, tag)) in sample.tokens().iter().zip(sample.tags()).enumerate() {
            self.features.clear();
            let previous = if i == 0 { None } else { Some(self.classes[i - 1].as_str()) };
            self.context.features(&mut self.features, token.as_str(), tag.as_str(), previous);
            out.push(Event::new(self.classes[i].as_str(), java_strings(&self.features)));
        }
        Ok(())
    }
}

/// The events of `LemmaSampleEventStream` over `samples`.
pub fn lemma_events<'a, 's>(
    samples: impl IntoIterator<Item = &'s LemmaSample<'a>>,
    factory: &LemmatizerFactory,
) -> Result<Vec<Event>>
where
    'a: 's,
{
    let mut stream = LemmaEventStream::new(factory);
    let mut out = Vec::new();
    for sample in samples {
        stream.events(sample, &mut out)?;
    }
    Ok(out)
}

/// A trained lemmatizer model: the GIS model, what the trainer reports,
/// and the `.bin` file.
#[derive(Debug, Clone)]
pub struct TrainedLemmatizerModel {
    pub model: GisModel,
    pub report: TrainingReport,
    pub package: ModelPackage,
}

/// `new LemmatizerModel(languageCode, lemmatizerModel, beamSize,
/// manifestInfoEntries, factory)` serialized: the manifest with the
/// trainer's entries, `factory` and `BeamSize`, and the `lemmatizer.model`
/// entry holding `model_bytes`.
pub fn package_lemmatizer_model(
    factory: &LemmatizerFactory,
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
    add_artifacts(&mut package, vec![(MODEL_ENTRY, model_bytes)])?;
    Ok(package)
}

/// `LemmatizerME.train(languageCode, samples, params, factory)` with an
/// event trainer: the events of [`lemma_events`], the GIS trainer, the
/// beam size of the parameters, and the model file stamped with `stamp`.
pub fn train_lemmatizer_model<'a, 's>(
    samples: impl IntoIterator<Item = &'s LemmaSample<'a>>,
    factory: &LemmatizerFactory,
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
    stamp: Stamp<'_>,
) -> Result<TrainedLemmatizerModel>
where
    'a: 's,
{
    let events = lemma_events(samples, factory)?;
    train_lemmatizer_model_from_events(&events, factory, params, config, stamp)
}

/// [`train_lemmatizer_model`] from events already made.
pub fn train_lemmatizer_model_from_events(
    events: &[Event],
    factory: &LemmatizerFactory,
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
    stamp: Stamp<'_>,
) -> Result<TrainedLemmatizerModel> {
    let (model, report) = gis::train_events(events, params, config)?;
    let bytes = write_gis(&model)?;
    let package = package_lemmatizer_model(factory, &report, bytes, beam_size(params)?, stamp)?;
    Ok(TrainedLemmatizerModel { model, report, package })
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_formats::native::LemmaReader;
    use turbo_train::JavaString;

    fn samples(text: &str) -> Vec<LemmaSample<'_>> {
        LemmaReader::new(text).unwrap().map(|s| s.unwrap()).collect()
    }

    fn strings(e: &Event) -> Vec<String> {
        e.context().iter().map(JavaString::to_string_lossy).collect()
    }

    #[test]
    fn classes_are_edit_scripts_and_the_previous_class_is_the_samples() {
        let f = LemmatizerFactory::new("en");
        let s = samples("He\tPRP\the\nreckons\tVBZ\treckon\n\nDogs\tNNS\tdog\n");
        let events = lemma_events(&s, &f).unwrap();
        let outcomes: Vec<&str> = events.iter().map(Event::outcome).collect();
        assert_eq!(outcomes, ["O", "D0s", "D0s"]);
        let second = strings(&events[1]);
        assert_eq!(&second[..5], ["w0=reckons", "t0=VBZ", "p_1=O", "p_1=Ot0=VBZ", "p_1=Ow0=reckons"]);
        // Each sample starts from the beginning of a sentence.
        assert_eq!(strings(&events[2])[2], "p_1=bos");
        assert_eq!(f.manifest_entries(), [("factory".to_owned(), FACTORY.to_owned())]);
    }

    #[test]
    fn an_empty_script_is_the_underscore_class_and_a_half_pair_is_refused() {
        assert_eq!(lemma_class("", "x").unwrap(), "_");
        assert_eq!(lemma_class("x", "").unwrap(), "_");
        assert_eq!(lemma_class("x", "x").unwrap(), "O");
        assert_eq!(lemma_class("x", "X").unwrap(), "O");
        // Replacing the astral character by a letter writes its two units
        // into the class, deleting one half and replacing the other.
        assert!(lemma_class("\u{1F600}", "a").is_none());
        let f = LemmatizerFactory::new("en");
        let s = samples("\u{1F600}\tSYM\ta\n");
        let e = lemma_events(&s, &f).unwrap_err();
        assert!(matches!(&e, Error::LemmaClassSurrogate { token } if token == "\u{1F600}"), "{e}");
    }

    #[test]
    fn the_model_file_has_the_factory_and_the_beam_size() {
        let f = LemmatizerFactory::new("de");
        let report = TrainingReport { event_hash: Some(1), trainer_type: turbo_train::TrainerType::Event };
        let package = package_lemmatizer_model(&f, &report, vec![1], 5, Stamp::at(0)).unwrap();
        assert_eq!(package.entry_names(), ["manifest.properties", "lemmatizer.model"]);
        assert_eq!(package.manifest().get("BeamSize"), Some("5"));
        assert_eq!(package.manifest().get("factory"), Some(FACTORY));
        assert_eq!(package.manifest().component_name(), COMPONENT_NAME);
    }
}
