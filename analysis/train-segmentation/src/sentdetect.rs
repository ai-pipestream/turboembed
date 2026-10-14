//! Training the sentence detector: `SentenceDetectorFactory`, the events
//! of `SDEventStream`, and the model file `SentenceDetectorME.train`
//! returns as a `SentenceModel`.
//!
//! `SDEventStream` makes, for every sentence of a sample, one event per
//! end-of-sentence character the scanner finds in that sentence: the last
//! one is a split (`s`) and every earlier one is not (`n`), with the
//! features `DefaultSDContextGenerator` builds at that position of the
//! whole document. A sentence with no end character gives no event, so a
//! model never learns from a sentence that ends in a letter.

use turbo_formats::SentenceSample;
use turbo_maxent::Dictionary;
use turbo_sentdetect::context::{Limits, Lookup, SdContext, Walks};
use turbo_sentdetect::{EndOfSentence, Features, Position, WhitespaceMode};
use turbo_train::{
    Event, GisModel, JavaString, ModelPackage, TrainingConfiguration, TrainingParameters, TrainingReport, gis,
    write_gis,
};

use crate::dictionary::AbbreviationDictionary;
use crate::error::{Error, Result};
use crate::package::{DICTIONARY_SERIALIZER, FACTORY_KEY, Stamp, add_artifacts, serializer_class_key};

/// `SentenceModel.COMPONENT_NAME`.
pub const COMPONENT_NAME: &str = "SentenceDetectorME";
/// `SentenceModel.MAXENT_MODEL_ENTRY_NAME`.
pub const MODEL_ENTRY: &str = "sent.model";
/// `SentenceModel.ABBREVIATIONS_ENTRY_NAME`.
pub const ABBREVIATIONS_ENTRY: &str = "abbreviations.dictionary";
/// The factory class the manifest names.
pub const FACTORY: &str = "opennlp.tools.sentdetect.SentenceDetectorFactory";
/// `SentenceDetectorFactory.TOKEN_END_PROPERTY`.
pub const USE_TOKEN_END_KEY: &str = "useTokenEnd";
/// `SentenceDetectorFactory.EOS_CHARACTERS_PROPERTY`.
pub const EOS_CHARACTERS_KEY: &str = "eosCharacters";
/// `SentenceDetectorME.SPLIT`.
pub const SPLIT: &str = "s";
/// `SentenceDetectorME.NO_SPLIT`.
pub const NO_SPLIT: &str = "n";

/// `opennlp.tools.sentdetect.lang.Factory.getEOSCharacters`: the end
/// characters of a language when a factory is given none.
pub fn default_eos_characters(language: &str) -> &'static str {
    match language {
        "th" | "tha" => " \n",
        "pt" | "por" => ".?!;:()\u{AB}\u{BB}'\"",
        "jpn" => "\u{3002}\u{FF01}\u{FF1F}",
        _ => ".!?",
    }
}

/// `SentenceDetectorFactory(languageCode, useTokenEnd, abbreviationDictionary,
/// eosCharacters)` as a training run uses it.
///
/// With no end characters the factory asks the language factory for the
/// language's, and from then on holds them, so `getEOSCharacters` is never
/// null during training: the manifest always has `eosCharacters`, and the
/// context generator is always `DefaultSDContextGenerator`, built from
/// those characters. (The `SentenceContextGenerator` the language factory
/// has for Thai is reachable only through the branch for null end
/// characters, which training never takes.)
#[derive(Debug, Clone)]
pub struct SentenceDetectorFactory {
    language: String,
    use_token_end: bool,
    abbreviations: Option<AbbreviationDictionary>,
    eos_characters: Vec<u16>,
}

impl SentenceDetectorFactory {
    /// `eos_characters` are the `char[]`, as a string; `None` takes the
    /// language's. A surrogate code unit is refused: the scanner cannot
    /// match half a character ([`Error::SurrogateEos`]).
    pub fn new(
        language: &str,
        use_token_end: bool,
        abbreviations: Option<AbbreviationDictionary>,
        eos_characters: Option<&str>,
    ) -> Result<SentenceDetectorFactory> {
        let eos_characters: Vec<u16> =
            eos_characters.unwrap_or_else(|| default_eos_characters(language)).encode_utf16().collect();
        if let Some(&u) = eos_characters.iter().find(|&&u| (0xD800..0xE000).contains(&u)) {
            return Err(Error::SurrogateEos(u));
        }
        Ok(SentenceDetectorFactory { language: language.to_owned(), use_token_end, abbreviations, eos_characters })
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    /// `isUseTokenEnd`.
    pub fn use_token_end(&self) -> bool {
        self.use_token_end
    }

    /// `getAbbreviationDictionary`.
    pub fn abbreviations(&self) -> Option<&AbbreviationDictionary> {
        self.abbreviations.as_ref()
    }

    /// `getEOSCharacters`, as code units.
    pub fn eos_characters(&self) -> &[u16] {
        &self.eos_characters
    }

    /// `getEOSCharacters` as the manifest writes them: `String.valueOf(char[])`.
    pub fn eos_string(&self) -> String {
        String::from_utf16(&self.eos_characters).expect("no surrogate is accepted")
    }

    /// `getEndOfSentenceScanner`.
    pub fn end_of_sentence(&self) -> EndOfSentence {
        EndOfSentence::new(&self.eos_characters).expect("no surrogate is accepted")
    }

    /// `createManifestEntries`, with `factory` as `BaseModel` adds it and
    /// the serializer class `BaseModel.serialize` records for the
    /// dictionary, in the order they are set.
    pub fn manifest_entries(&self) -> Vec<(String, String)> {
        let mut entries = vec![
            (FACTORY_KEY.to_owned(), FACTORY.to_owned()),
            (USE_TOKEN_END_KEY.to_owned(), self.use_token_end.to_string()),
            (EOS_CHARACTERS_KEY.to_owned(), self.eos_string()),
        ];
        if self.abbreviations.is_some() {
            entries.push((serializer_class_key(ABBREVIATIONS_ENTRY), DICTIONARY_SERIALIZER.to_owned()));
        }
        entries
    }
}

/// `SDEventStream` for one factory: the end characters, the dictionary
/// and the buffers its events are built in.
pub struct SdEventStream<'f> {
    eos: EndOfSentence,
    dictionary: Option<&'f Dictionary>,
    whitespace: WhitespaceMode,
    candidates: Vec<Position>,
    features: Features,
    lookup: Lookup,
}

impl<'f> SdEventStream<'f> {
    /// The stream of `factory.getSDContextGenerator()` and
    /// `factory.getEndOfSentenceScanner()`, with `whitespace` where the
    /// Java calls `StringUtil.isWhitespace`.
    pub fn new(factory: &'f SentenceDetectorFactory, whitespace: WhitespaceMode) -> SdEventStream<'f> {
        SdEventStream {
            eos: factory.end_of_sentence(),
            dictionary: factory.abbreviations().map(AbbreviationDictionary::lookup),
            whitespace,
            candidates: Vec::new(),
            features: Features::new(),
            lookup: Lookup::default(),
        }
    }

    /// `createEvents(sample)`: appends the sample's events to `out`, in the
    /// order of its sentences and of the candidates within each.
    pub fn events(&mut self, sample: &SentenceSample<'_>, out: &mut Vec<Event>) {
        let document = sample.document();
        let scanned = self.eos.positions(document, &mut self.candidates);
        let context = SdContext::new(&self.eos, self.dictionary, self.whitespace, Limits::NONE);
        let mut walks = Walks::new();
        let mut last_unit = 0;
        for span in sample.sentences() {
            // The scanner runs over the sentence's text; its candidates are
            // the document's candidates inside the span, as the span is
            // on character boundaries.
            let from = self.candidates.partition_point(|p| p.byte < span.start);
            let to = self.candidates.partition_point(|p| p.byte < span.end);
            for (i, &at) in self.candidates[from..to].iter().enumerate() {
                // The context generator's memory of its whitespace
                // searches wants candidates in text order; sentences out
                // of order start it over.
                if at.unit < last_unit {
                    walks = Walks::new();
                }
                last_unit = at.unit;
                context.context(scanned, at, &mut walks, &mut self.features, &mut self.lookup);
                let outcome = if i + 1 == to - from { SPLIT } else { NO_SPLIT };
                out.push(Event::new(outcome, java_strings(&self.features)));
            }
        }
    }
}

/// The features as the predicates the JVM would hold.
fn java_strings(features: &Features) -> Vec<JavaString> {
    features.iter().map(|f| JavaString::from_modified_utf8(f).expect("a feature is modified UTF-8")).collect()
}

/// The events of `SDEventStream` over `samples`.
pub fn sd_events<'a, 's>(
    samples: impl IntoIterator<Item = &'s SentenceSample<'a>>,
    factory: &SentenceDetectorFactory,
    whitespace: WhitespaceMode,
) -> Vec<Event>
where
    'a: 's,
{
    let mut stream = SdEventStream::new(factory, whitespace);
    let mut out = Vec::new();
    for sample in samples {
        stream.events(sample, &mut out);
    }
    out
}

/// A trained sentence detector model: the GIS model, what the trainer
/// reports, and the `.bin` file.
#[derive(Debug, Clone)]
pub struct TrainedSentenceModel {
    pub model: GisModel,
    pub report: TrainingReport,
    pub package: ModelPackage,
}

/// `new SentenceModel(languageCode, sentModel, manifestInfoEntries,
/// sdFactory)` serialized: the manifest with the trainer's and the
/// factory's entries, the `sent.model` entry holding `model_bytes` and,
/// with a dictionary, its `abbreviations.dictionary` entry, in the JVM's
/// zip order.
pub fn package_sentence_model(
    factory: &SentenceDetectorFactory,
    report: &TrainingReport,
    model_bytes: Vec<u8>,
    stamp: Stamp<'_>,
) -> Result<ModelPackage> {
    let mut manifest = stamp.manifest(COMPONENT_NAME, factory.language());
    manifest.set_training(report);
    for (k, v) in factory.manifest_entries() {
        manifest.set(&k, &v);
    }
    let mut package = ModelPackage::new(manifest);
    let mut artifacts = Vec::new();
    if let Some(dictionary) = factory.abbreviations() {
        artifacts.push((ABBREVIATIONS_ENTRY, dictionary.serialize()?));
    }
    artifacts.push((MODEL_ENTRY, model_bytes));
    add_artifacts(&mut package, artifacts)?;
    Ok(package)
}

/// `SentenceDetectorME.train(languageCode, samples, sdFactory, mlParams)`:
/// the events of [`sd_events`], the event trainer `mlParams` names (the
/// GIS trainer, with the event checksum and `TrainerType` for the
/// manifest), and the model file stamped with `stamp`.
pub fn train_sentence_model<'a, 's>(
    samples: impl IntoIterator<Item = &'s SentenceSample<'a>>,
    factory: &SentenceDetectorFactory,
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
    whitespace: WhitespaceMode,
    stamp: Stamp<'_>,
) -> Result<TrainedSentenceModel>
where
    'a: 's,
{
    let events = sd_events(samples, factory, whitespace);
    train_sentence_model_from_events(&events, factory, params, config, stamp)
}

/// [`train_sentence_model`] from events already made.
pub fn train_sentence_model_from_events(
    events: &[Event],
    factory: &SentenceDetectorFactory,
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
    stamp: Stamp<'_>,
) -> Result<TrainedSentenceModel> {
    let (model, report) = gis::train_events(events, params, config)?;
    let bytes = write_gis(&model)?;
    let package = package_sentence_model(factory, &report, bytes, stamp)?;
    Ok(TrainedSentenceModel { model, report, package })
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_formats::native::SentenceReader;

    fn samples(text: &str) -> Vec<SentenceSample<'_>> {
        SentenceReader::new(text).unwrap().map(|s| s.unwrap()).collect()
    }

    #[test]
    fn the_last_candidate_of_a_sentence_splits_and_the_others_do_not() {
        let f = SentenceDetectorFactory::new("en", true, None, None).unwrap();
        let s = samples("Dr. Who left.\nNo end here\nTwo? Yes!\n");
        let events = sd_events(&s, &f, WhitespaceMode::Unicode);
        let outcomes: Vec<&str> = events.iter().map(Event::outcome).collect();
        // "Dr." and "left." in the first sentence; none in the second;
        // "Two?" and "Yes!" in the third.
        assert_eq!(outcomes, ["n", "s", "n", "s"]);
        // No space before "Dr.", so no `sp`: `sn`, `eos=.`, `x=Dr`, ...
        assert_eq!(events[0].context()[1].to_string_lossy(), "eos=.");
        assert_eq!(events[0].context()[2].to_string_lossy(), "x=Dr");
        assert_eq!(events[2].context()[1].to_string_lossy(), "eos=?");
        assert_eq!(events[3].context().last().unwrap().to_string_lossy(), "n=");
    }

    #[test]
    fn the_language_decides_the_end_characters_and_the_manifest_always_lists_them() {
        let pt = SentenceDetectorFactory::new("pt", true, None, None).unwrap();
        assert_eq!(pt.eos_string(), ".?!;:()\u{AB}\u{BB}'\"");
        let th = SentenceDetectorFactory::new("tha", false, None, None).unwrap();
        assert_eq!(th.eos_string(), " \n");
        let given = SentenceDetectorFactory::new("pt", true, None, Some(".!")).unwrap();
        assert_eq!(given.eos_string(), ".!");
        assert_eq!(
            given.manifest_entries(),
            [("factory", FACTORY), ("useTokenEnd", "true"), ("eosCharacters", ".!")]
                .map(|(k, v)| (k.to_owned(), v.to_owned()))
        );
        assert!(matches!(
            SentenceDetectorFactory::new("en", true, None, Some("\u{1F600}")),
            Err(Error::SurrogateEos(0xD83D))
        ));
    }

    #[test]
    fn a_dictionary_adds_abbreviation_features_and_a_serializer_entry() {
        let d = AbbreviationDictionary::new(false, vec![vec!["Dr.".to_string()]]).unwrap();
        let f = SentenceDetectorFactory::new("en", true, Some(d), None).unwrap();
        assert_eq!(
            f.manifest_entries()[3],
            ("serializer-class-abbreviations.dictionary".to_owned(), DICTIONARY_SERIALIZER.to_owned())
        );
        let s = samples("See Dr. Who.\n");
        let events = sd_events(&s, &f, WhitespaceMode::Unicode);
        assert!(events[0].context().iter().any(|c| c.as_bytes() == b"xabbrev"));
    }
}
