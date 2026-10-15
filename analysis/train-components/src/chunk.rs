//! Training the phrase chunker: `ChunkerFactory`, the events of
//! `ChunkerEventStream`, and the model file `ChunkerME.train` returns as a
//! `ChunkerModel`.
//!
//! `ChunkerEventStream` makes one event per token of a sample: the token's
//! chunk tag (`B-NP`, `I-NP`, `O`) as the outcome, and as the context the
//! 41 features of `DefaultChunkerContextGenerator` for that token, with
//! the sample's own chunk tags as the tags decided before it. The
//! generator is `turbo_sequence::chunker::DefaultChunkerContext`, the one
//! the chunker runs with (`turbo-phrase-chunker`), so training and
//! inference share it. `ChunkerFactory` has no settings and no artifacts:
//! the model file holds the manifest, with `factory` and `BeamSize`, and
//! `chunker.model`.
//!
//! `ChunkerME.train` asks `TrainerFactory.getTrainerType` which kind of
//! trainer the `Algorithm` parameter names. An event trainer (`MAXENT`,
//! the default, `MAXENT_QN` or `PERCEPTRON`) is trained on the events
//! ([`train_chunker_model`]). `NAIVEBAYES` is an event trainer too, which
//! `ChunkerME.train` takes and this port does not have: it passes the
//! trainer type check and fails with `turbo_train::Error::NotPorted`
//! ([`Error::Train`]). The sequence perceptron
//! (`PERCEPTRON_SEQUENCE`) is an event model sequence trainer, which
//! `ChunkerME.train` does not take: it throws
//! `IllegalArgumentException("Trainer type is not supported:
//! EVENT_MODEL_SEQUENCE_TRAINER")` ([`Error::UnsupportedTrainerType`]),
//! as it does for an algorithm it cannot resolve (`null`). Its branch for a
//! sequence trainer (`SEQUENCE_TRAINER`) is reached only by a trainer
//! class loaded as an extension, which OpenNLP has none of built in and
//! this crate does not load.

use turbo_formats::ChunkSample;
use turbo_sequence::Features;
use turbo_sequence::chunker::{DefaultChunkerContext, TokenTag};
use turbo_train::params::trainer_type;
use turbo_train::trainer::{self, TrainedModel};
use turbo_train::{Event, ModelPackage, TrainerType, TrainingConfiguration, TrainingParameters, TrainingReport};

use crate::error::{Error, Result};
use crate::package::{FACTORY_KEY, Stamp, add_artifacts};
use crate::postag::{BEAM_SIZE_KEY, java_strings};

/// `ChunkerModel.COMPONENT_NAME`.
pub const COMPONENT_NAME: &str = "ChunkerME";
/// `ChunkerModel.CHUNKER_MODEL_ENTRY_NAME`.
pub const MODEL_ENTRY: &str = turbo_sequence::chunker::MODEL_ENTRY_NAME;
/// The factory class the manifest names.
pub const FACTORY: &str = "opennlp.tools.chunker.ChunkerFactory";
/// `ChunkerME.DEFAULT_BEAM_SIZE`.
pub const DEFAULT_BEAM_SIZE: i32 = 10;

/// `new ChunkerFactory()`: it has no settings and no artifacts; the
/// language only names the model.
#[derive(Debug, Clone)]
pub struct ChunkerFactory {
    language: String,
}

impl ChunkerFactory {
    pub fn new(language: &str) -> ChunkerFactory {
        ChunkerFactory { language: language.to_owned() }
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    /// `getContextGenerator`: `DefaultChunkerContextGenerator`.
    pub fn context_generator(&self) -> DefaultChunkerContext {
        DefaultChunkerContext
    }

    /// `factory`, as `BaseModel` adds it; the factory has no entries of
    /// its own.
    pub fn manifest_entries(&self) -> Vec<(String, String)> {
        vec![(FACTORY_KEY.to_owned(), FACTORY.to_owned())]
    }
}

/// `ChunkerEventStream` over the factory's context generator.
pub struct ChunkerEventStream {
    context: DefaultChunkerContext,
    features: Features,
}

impl ChunkerEventStream {
    pub fn new(factory: &ChunkerFactory) -> ChunkerEventStream {
        ChunkerEventStream { context: factory.context_generator(), features: Features::new() }
    }

    /// `createEvents(sample)`: appends one event per token of the sample
    /// to `out`, in token order. The sample's chunk tags stand in for the
    /// tags decided so far; the generator reads only the two before the
    /// token.
    ///
    /// The tokens and the two tag vectors are built per sample: the
    /// generator reads `&str`, and a sample's fields may own their text,
    /// so they cannot outlive the sample and be kept on the stream. Each
    /// token's event allocates far more than these vectors.
    pub fn events(&mut self, sample: &ChunkSample<'_>, out: &mut Vec<Event>) {
        let words: Vec<TokenTag<'_>> = sample
            .tokens()
            .iter()
            .zip(sample.tags())
            .map(|(token, tag)| TokenTag { token: token.as_str(), tag: tag.as_str() })
            .collect();
        let preds: Vec<&str> = sample.chunks().iter().map(|c| c.as_str()).collect();
        for (i, pred) in preds.iter().enumerate() {
            self.features.clear();
            self.context.features(&mut self.features, i, &words, &preds);
            out.push(Event::new(*pred, java_strings(&self.features)));
        }
    }
}

/// The events of `ChunkerEventStream` over `samples`.
pub fn chunk_events<'a, 's>(
    samples: impl IntoIterator<Item = &'s ChunkSample<'a>>,
    factory: &ChunkerFactory,
) -> Vec<Event>
where
    'a: 's,
{
    let mut stream = ChunkerEventStream::new(factory);
    let mut out = Vec::new();
    for sample in samples {
        stream.events(sample, &mut out);
    }
    out
}

/// A trained chunker model: the model the event trainer made, what the
/// trainer reports, and the `.bin` file.
#[derive(Debug, Clone)]
pub struct TrainedChunkerModel {
    pub model: TrainedModel,
    pub report: TrainingReport,
    pub package: ModelPackage,
}

/// `mlParams.getIntParameter(BeamSize, ChunkerME.DEFAULT_BEAM_SIZE)`.
pub fn beam_size(params: &TrainingParameters) -> Result<i32> {
    Ok(params.get_int(BEAM_SIZE_KEY)?.unwrap_or(DEFAULT_BEAM_SIZE))
}

/// The trainer type `ChunkerME.train` trains with: an event trainer, or
/// [`Error::UnsupportedTrainerType`] for the one `getTrainerType` gives
/// otherwise, `None` for an algorithm it cannot resolve.
pub fn check_trainer_type(params: &TrainingParameters) -> Result<()> {
    match trainer_type(params)? {
        Some(TrainerType::Event) => Ok(()),
        other => Err(Error::UnsupportedTrainerType(other)),
    }
}

/// `new ChunkerModel(languageCode, chunkerModel, beamSize,
/// manifestInfoEntries, factory)` serialized: the manifest with the
/// trainer's entries, `factory` and `BeamSize`, and the `chunker.model`
/// entry holding `model_bytes`, after the manifest in the zip as the JVM's
/// artifact map orders them.
pub fn package_chunker_model(
    factory: &ChunkerFactory,
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

/// `ChunkerME.train(languageCode, samples, mlParams, factory)`: the events
/// of [`chunk_events`], the event trainer the `Algorithm` parameter names
/// (with the event checksum and `TrainerType` for the manifest), the beam
/// size of the parameters, and the model file stamped with `stamp`.
///
/// # Errors
///
/// [`Error::UnsupportedTrainerType`] for a parameter set that names no
/// event trainer, checked before any event is made, as the JVM checks it
/// before it reads a sample; what the trainer fails on, among it
/// `turbo_train::Error::NotPorted` for `NAIVEBAYES`.
pub fn train_chunker_model<'a, 's>(
    samples: impl IntoIterator<Item = &'s ChunkSample<'a>>,
    factory: &ChunkerFactory,
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
    stamp: Stamp<'_>,
) -> Result<TrainedChunkerModel>
where
    'a: 's,
{
    beam_size(params)?;
    check_trainer_type(params)?;
    let events = chunk_events(samples, factory);
    train_chunker_model_from_events(&events, factory, params, config, stamp)
}

/// [`train_chunker_model`] from events already made.
pub fn train_chunker_model_from_events(
    events: &[Event],
    factory: &ChunkerFactory,
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
    stamp: Stamp<'_>,
) -> Result<TrainedChunkerModel> {
    let beam = beam_size(params)?;
    check_trainer_type(params)?;
    let (model, report) = trainer::train_events(events, params, config)?;
    let bytes = model.write()?;
    let package = package_chunker_model(factory, &report, bytes, beam, stamp)?;
    Ok(TrainedChunkerModel { model, report, package })
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_formats::native::ChunkReader;
    use turbo_train::{Algorithm, JavaString, Key};

    fn samples(text: &str) -> Vec<ChunkSample<'_>> {
        ChunkReader::new(text).unwrap().map(|s| s.unwrap()).collect()
    }

    fn strings(e: &Event) -> Vec<String> {
        e.context().iter().map(JavaString::to_string_lossy).collect()
    }

    const TEXT: &str = "He PRP B-NP\nreckons VBZ B-VP\nthe DT B-NP\ndeficit NN I-NP\n. . O\n\n\
        Dogs NNS B-NP\nbark VBP B-VP\n";

    #[test]
    fn one_event_per_token_with_the_samples_chunk_tags_as_history() {
        let f = ChunkerFactory::new("en");
        assert_eq!(f.manifest_entries(), [("factory".to_owned(), FACTORY.to_owned())]);
        let s = samples(TEXT);
        let events = chunk_events(&s, &f);
        let outcomes: Vec<String> = events.iter().map(|e| e.outcome().to_string()).collect();
        assert_eq!(outcomes, ["B-NP", "B-VP", "B-NP", "I-NP", "O", "B-NP", "B-VP"]);
        assert!(events.iter().all(|e| e.context().len() == 41));
        let third = strings(&events[2]);
        assert_eq!(&third[..5], ["w_2=He", "w_1=reckons", "w0=the", "w1=deficit", "w2=."]);
        assert_eq!(third[19], "p_2=B-NP");
        assert_eq!(third[20], "p_1=B-VP");
        assert_eq!(third[21], "p_2=B-NPp_1=B-VP");
        assert_eq!(third[40], "p_1=B-VPw0=thew1=deficit");
        // The last token sees the end of the sentence on both sides of
        // the window ahead.
        let fifth = strings(&events[4]);
        assert_eq!((fifth[3].as_str(), fifth[4].as_str()), ("w1=eos", "w2=eos"));
        assert_eq!((fifth[10].as_str(), fifth[11].as_str()), ("t1=eos", "t2=eos"));
        // Each sample starts from the beginning of a sentence.
        let sixth = strings(&events[5]);
        assert_eq!(&sixth[..2], ["w_2=bos", "w_1=bos"]);
        assert_eq!(&sixth[19..22], ["p_2=bos", "p_1=bos", "p_2=bosp_1=bos"]);
    }

    #[test]
    fn the_history_is_the_samples_tags_whatever_they_are() {
        // An I- tag after O, and a tag of another kind, are what the
        // features name; the event stream does not check the tags.
        let s = samples("a DT O\nb NN I-NP\nc NN X\n");
        let events = chunk_events(&s, &ChunkerFactory::new("en"));
        assert_eq!(*events[1].outcome(), "I-NP");
        assert_eq!(strings(&events[2])[21], "p_2=Op_1=I-NP");
        assert_eq!(*events[2].outcome(), "X");
    }

    #[test]
    fn the_model_file_has_the_factory_and_the_beam_size() {
        let f = ChunkerFactory::new("de");
        let report = TrainingReport { event_hash: Some(1), trainer_type: TrainerType::Event };
        let package = package_chunker_model(&f, &report, vec![1], 5, Stamp::at(0)).unwrap();
        // The JVM's artifact map puts chunker.model before the manifest
        // (the fixture chunker-fixture.bin has them in this order).
        assert_eq!(package.entry_names(), ["chunker.model", "manifest.properties"]);
        assert_eq!(package.manifest().get("BeamSize"), Some("5"));
        assert_eq!(package.manifest().get("factory"), Some(FACTORY));
        assert_eq!(package.manifest().get("TrainerType"), Some("Event"));
        assert_eq!(package.manifest().component_name(), COMPONENT_NAME);
    }

    #[test]
    fn the_beam_size_comes_from_the_parameters() {
        let mut p = TrainingParameters::new();
        assert_eq!(beam_size(&p).unwrap(), 10);
        p.put(BEAM_SIZE_KEY, 4);
        assert_eq!(beam_size(&p).unwrap(), 4);
        p.put_str(BEAM_SIZE_KEY, "x");
        assert!(beam_size(&p).is_err());
    }

    #[test]
    fn the_ported_event_trainers_train_naive_bayes_is_not_ported_and_the_sequence_perceptron_is_refused() {
        let s = samples(TEXT);
        let f = ChunkerFactory::new("en");
        let mut p = TrainingParameters::new();
        p.put(Key::ITERATIONS, 5);
        p.put(Key::CUTOFF, 0);
        for (algorithm, model_type) in
            [(None, "GIS"), (Some(Algorithm::MaxentQn), "QN"), (Some(Algorithm::Perceptron), "Perceptron")]
        {
            if let Some(a) = algorithm {
                p.put_str(Key::ALGORITHM, a.name());
            }
            let t = train_chunker_model(&s, &f, &p, &mut TrainingConfiguration::default(), Stamp::at(0)).unwrap();
            let file = turbo_maxent::ModelFile::open(t.package.to_bytes()).unwrap();
            let entry = file.entry(MODEL_ENTRY).unwrap();
            assert_eq!(&entry[2..2 + model_type.len()], model_type.as_bytes(), "{algorithm:?}");
            assert_eq!(file.manifest().get("BeamSize"), Some("10"));
            assert_eq!(t.report.trainer_type, TrainerType::Event);
            let maxent = file.maxent(MODEL_ENTRY).unwrap();
            let outcomes: Vec<&str> = maxent.outcome_texts().unwrap();
            assert_eq!(outcomes.len(), 4, "{outcomes:?}");
        }
        // `ChunkerME.train` trains with `NAIVEBAYES` too, an event trainer
        // this port does not have: the trainer type passes, training fails.
        p.put_str(Key::ALGORITHM, Algorithm::NaiveBayes.name());
        let e = train_chunker_model(&s, &f, &p, &mut TrainingConfiguration::default(), Stamp::at(0)).unwrap_err();
        assert!(matches!(e, Error::Train(turbo_train::Error::NotPorted(Algorithm::NaiveBayes))), "{e}");
        p.put_str(Key::ALGORITHM, Algorithm::PerceptronSequence.name());
        let e = train_chunker_model(&s, &f, &p, &mut TrainingConfiguration::default(), Stamp::at(0)).unwrap_err();
        assert!(matches!(e, Error::UnsupportedTrainerType(Some(TrainerType::EventModelSequence))), "{e}");
        assert_eq!(e.to_string(), "Trainer type is not supported: EVENT_MODEL_SEQUENCE_TRAINER");
        p.put_str(Key::ALGORITHM, "org.example.NoSuchTrainer");
        let e = train_chunker_model(&s, &f, &p, &mut TrainingConfiguration::default(), Stamp::at(0)).unwrap_err();
        assert_eq!(e.to_string(), "Trainer type is not supported: null");
    }
}
