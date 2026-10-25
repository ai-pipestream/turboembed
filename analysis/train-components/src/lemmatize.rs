//! Trains lemma edit classes with the existing context generator and GIS
//! trainer. The default factory keeps the token and lemma case and writes
//! a versioned native factory name. `LemmatizerFactory::opennlp` selects
//! the original lowercase classes for JVM goldens and published models.
//!
//! Both formats use UTF-16 operations and `JavaString` outcomes, so an
//! operation can hold a surrogate half. The native format delimits its
//! indices; OpenNLP's one-digit decoder is kept only for compatibility.

use turbo_formats::LemmaSample;
use turbo_lemmatizer::edit_script;
use turbo_lemmatizer::{CASE_SENSITIVE_FACTORY, LemmaContext, LemmaEncoding};
use turbo_sequence::Features;
use turbo_train::{
    Event, GisModel, JavaString, ModelPackage, TrainingConfiguration, TrainingParameters, TrainingReport, gis,
    write_gis,
};

use crate::error::Result;
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

/// A case-preserving factory by default. Use `opennlp` explicitly for
/// the JVM format; the language names the model.
#[derive(Debug, Clone)]
pub struct LemmatizerFactory {
    language: String,
    encoding: LemmaEncoding,
}

impl LemmatizerFactory {
    /// Native case-preserving classes and delimited edit indices.
    pub fn new(language: &str) -> LemmatizerFactory {
        LemmatizerFactory { language: language.to_owned(), encoding: LemmaEncoding::CaseSensitiveV1 }
    }

    /// The original lowercase format for JVM goldens and reproduction.
    pub fn opennlp(language: &str) -> LemmatizerFactory {
        LemmatizerFactory { language: language.to_owned(), encoding: LemmaEncoding::OpenNlp }
    }

    pub fn encoding(&self) -> LemmaEncoding {
        self.encoding
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
        let factory = match self.encoding {
            LemmaEncoding::OpenNlp => FACTORY,
            LemmaEncoding::CaseSensitiveV1 => CASE_SENSITIVE_FACTORY,
        };
        vec![(FACTORY_KEY.to_owned(), factory.to_owned())]
    }
}

/// Compatibility helper: `LemmatizerME.encodeLemmas(toks, lemmas)` for one token: the shortest
/// edit script, `_` when it is empty.
pub fn lemma_class(token: &str, lemma: &str) -> JavaString {
    let class = edit_script::encode(token, lemma);
    if class.is_empty() { JavaString::new(EMPTY_CLASS) } else { class }
}

/// `LemmaSampleEventStream` over the default context generator.
pub struct LemmaEventStream {
    context: LemmaContext,
    encoding: LemmaEncoding,
    features: Features,
    classes: Vec<JavaString>,
}

impl LemmaEventStream {
    pub fn new(factory: &LemmatizerFactory) -> LemmaEventStream {
        LemmaEventStream {
            context: factory.context_generator(),
            encoding: factory.encoding(),
            features: Features::new(),
            classes: Vec::new(),
        }
    }

    /// `createEvents(sample)`: appends one event per token to `out`, in
    /// token order, after encoding the sample's lemmas. A sample with no
    /// tokens gives no event.
    pub fn events(&mut self, sample: &LemmaSample<'_>, out: &mut Vec<Event>) {
        self.classes.clear();
        for (token, lemma) in sample.tokens().iter().zip(sample.lemmas()) {
            self.classes.push(self.encoding.encode(token.as_str(), lemma.as_str()));
        }
        for (i, (token, tag)) in sample.tokens().iter().zip(sample.tags()).enumerate() {
            self.features.clear();
            let previous = if i == 0 { None } else { Some(&self.classes[i - 1]) };
            self.context.features(&mut self.features, token.as_str(), tag.as_str(), previous);
            out.push(Event::new(self.classes[i].clone(), java_strings(&self.features)));
        }
    }
}

/// The events of `LemmaSampleEventStream` over `samples`.
pub fn lemma_events<'a, 's>(
    samples: impl IntoIterator<Item = &'s LemmaSample<'a>>,
    factory: &LemmatizerFactory,
) -> Vec<Event>
where
    'a: 's,
{
    let mut stream = LemmaEventStream::new(factory);
    let mut out = Vec::new();
    for sample in samples {
        stream.events(sample, &mut out);
    }
    out
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
    let events = lemma_events(samples, factory);
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
        let f = LemmatizerFactory::opennlp("en");
        let s = samples("He\tPRP\the\nreckons\tVBZ\treckon\n\nDogs\tNNS\tdog\n");
        let events = lemma_events(&s, &f);
        let outcomes: Vec<&JavaString> = events.iter().map(Event::outcome).collect();
        assert_eq!(outcomes, ["O", "D0s", "D0s"].map(JavaString::new).iter().collect::<Vec<_>>());
        let second = strings(&events[1]);
        assert_eq!(&second[..5], ["w0=reckons", "t0=VBZ", "p_1=O", "p_1=Ot0=VBZ", "p_1=Ow0=reckons"]);
        // Each sample starts from the beginning of a sentence.
        assert_eq!(strings(&events[2])[2], "p_1=bos");
        assert_eq!(f.manifest_entries(), [("factory".to_owned(), FACTORY.to_owned())]);
    }

    #[test]
    fn an_empty_script_is_the_underscore_class_and_a_half_pair_is_an_outcome() {
        assert_eq!(lemma_class("", "x"), "_");
        assert_eq!(lemma_class("x", ""), "_");
        assert_eq!(lemma_class("x", "x"), "O");
        assert_eq!(lemma_class("x", "X"), "O");
        // Replacing the astral character by "_" writes its two units into
        // the class, replacing the low half and deleting the high half:
        // the class of the goldens' conllu-whitespace sample. The next
        // token's p_1= features hold the class as modified UTF-8.
        let class = lemma_class("\u{1F600}", "_");
        assert_eq!(class, JavaString::from_utf16(&[0x52, 0x31, 0xDE00, 0x5F, 0x44, 0x30, 0xD83D]));
        let f = LemmatizerFactory::opennlp("en");
        let s = samples("\u{1F600}\t_\t_\nx\t_\t_\n");
        let events = lemma_events(&s, &f);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].outcome(), &class);
        assert_eq!(*events[1].outcome(), "R0x_");
        let mut expected = JavaString::new("p_1=").as_bytes().to_vec();
        expected.extend_from_slice(class.as_bytes());
        assert_eq!(events[1].context()[2].as_bytes(), expected);
        assert_eq!(strings(&events[1])[3], "p_1=R1\u{FFFD}_D0\u{FFFD}t0=_");
    }

    #[test]
    fn the_model_file_has_the_factory_and_the_beam_size() {
        let f = LemmatizerFactory::opennlp("de");
        let report = TrainingReport { event_hash: Some(1), trainer_type: turbo_train::TrainerType::Event };
        let package = package_lemmatizer_model(&f, &report, vec![1], 5, Stamp::at(0)).unwrap();
        assert_eq!(package.entry_names(), ["manifest.properties", "lemmatizer.model"]);
        assert_eq!(package.manifest().get("BeamSize"), Some("5"));
        assert_eq!(package.manifest().get("factory"), Some(FACTORY));
        assert_eq!(package.manifest().component_name(), COMPONENT_NAME);
    }
}
