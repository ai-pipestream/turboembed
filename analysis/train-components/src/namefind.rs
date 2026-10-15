//! Training the name finder: `TokenNameFinderFactory`, the events of
//! `NameFinderEventStream`, and the model file `NameFinderME.train`
//! returns as a `TokenNameFinderModel`.
//!
//! `NameFinderME.train(languageCode, type, samples, params, factory)`
//! first puts its defaults into the parameters where they are absent
//! (`putIfAbsent`): `Algorithm=MAXENT`, `Cutoff=0` and `Iterations=300`,
//! which are not `TrainingParameters.defaultParams()` (100 iterations,
//! cutoff 5). The trainer tool with no parameters file trains with an
//! empty `TrainingParameters`, so these are the parameters a model is
//! trained with by default ([`training_parameters`]).
//!
//! With an event trainer (`MAXENT`, `MAXENT_QN`, `PERCEPTRON`) the events
//! are `NameFinderEventStream`'s over the factory's context generator and
//! codec. The stream adds an `AdditionalContextFeatureGenerator` in a
//! window of eight tokens on each side after the factory's generators, so
//! the generators are those `NameFinderME` builds for a model of the same
//! descriptor. For each sample it:
//!
//! 1. clears the generators' adaptive data when the sample says so (the
//!    name format's empty line, the start of a document);
//! 2. gives every name the stream's type when one was given (`overrideType`);
//! 3. encodes the names with the codec, a name without a type as
//!    `default`;
//! 4. makes one event per token: the token's outcome, and the context the
//!    generators build given the sample's own outcomes as the outcomes
//!    before it, then `po=`, `pow=`, `powf=` and `ppo=`
//!    (`DefaultNameContextGenerator`);
//! 5. passes the tokens and outcomes to the generators' adaptive data
//!    (`PreviousMapFeatureGenerator` remembers each token's outcome until
//!    the next clear).
//!
//! The sequence perceptron (`PERCEPTRON_SEQUENCE`), which trains from
//! `NameSampleSequenceStream` instead, is not ported
//! ([`Error::NameSequenceTraining`]).
//!
//! OpenNLP behaviour kept here:
//!
//! - The codecs index the outcome array directly. With BIO an empty name
//!   after the last token (`<START:x> <END>` at the end of a line) and
//!   with BILOU an empty name at the first token (whose unit goes at
//!   index -1) throw `ArrayIndexOutOfBoundsException`, and the sample
//!   gives no events ([`Error::UnencodableName`]). Any other empty name
//!   marks a token: BIO marks its start token as a start, and BILOU
//!   marks the token before it as a unit, over whatever outcome that
//!   token had.
//! - `TokenNameFinderModel` checks the trained model's outcomes with the
//!   codec's `areOutcomesCompatible`, and throws
//!   `IllegalArgumentException("Model not compatible with name finder!")`
//!   when they fail it ([`Error::IncompatibleOutcomes`]): BIO needs a
//!   `start` outcome and a start for every type that continues, BILOU a
//!   start or unit, a last for every type that starts, a start for every
//!   type that ends and a start or last for every type that continues.
//!   With BILOU an empty name right after a name leaves that name's type
//!   a start without a last. Samples without a single name give only
//!   `other`, which the trainer refuses first ("Training data must
//!   contain more than one outcome").
//! - With no descriptor the factory loads `ner-default-features.xml` when
//!   the context generator is first made, so a trained model always
//!   carries a `generator.featuregen` entry.
//! - The model file's entries are in the order of the JVM's artifact
//!   `HashMap`: `manifest.properties`, `nameFinder.model`,
//!   `generator.featuregen`.

use turbo_formats::NameSample;
use turbo_sequence::codec::Span;
use turbo_sequence::featuregen::{AdditionalContext, Window, descriptor};
use turbo_sequence::namefind::{CODEC_PROPERTY, DEFAULT_DESCRIPTOR, DefaultNameContext, NameCodec};
use turbo_sequence::{BoxedGenerator, ContextGenerator, Features, PatternMode};
use turbo_train::params::trainer_type;
use turbo_train::trainer::{self, TrainedModel};
use turbo_train::{
    Algorithm, Event, JavaString, Key, ModelPackage, TrainerType, TrainingConfiguration, TrainingParameters,
    TrainingReport,
};

use crate::error::{Error, Result};
use crate::package::{FACTORY_KEY, Stamp, add_artifacts};
use crate::postag::java_strings;

/// `TokenNameFinderModel.COMPONENT_NAME`.
pub const COMPONENT_NAME: &str = "NameFinderME";
/// `TokenNameFinderModel.MAXENT_MODEL_ENTRY_NAME`.
pub const MODEL_ENTRY: &str = turbo_sequence::namefind::MODEL_ENTRY_NAME;
/// `TokenNameFinderModel.GENERATOR_DESCRIPTOR_ENTRY_NAME`.
pub const DESCRIPTOR_ENTRY: &str = "generator.featuregen";
/// The factory class the manifest names.
pub const FACTORY: &str = "opennlp.tools.namefind.TokenNameFinderFactory";
/// `BeamSearch.BEAM_SIZE_PARAMETER`: the training parameter and the
/// manifest entry.
pub const BEAM_SIZE_KEY: &str = "BeamSize";
/// `NameFinderME.DEFAULT_BEAM_SIZE`.
pub const DEFAULT_BEAM_SIZE: i32 = 3;
/// The `Cutoff` `NameFinderME.train` puts in when absent.
pub const DEFAULT_CUTOFF: i32 = 0;
/// The `Iterations` `NameFinderME.train` puts in when absent.
pub const DEFAULT_ITERATIONS: i32 = 300;
/// The message of the `IllegalArgumentException` `TokenNameFinderModel`
/// throws for outcomes the codec cannot decode.
pub const INCOMPATIBLE_MESSAGE: &str = "Model not compatible with name finder!";

/// `TokenNameFinderFactory(featureGeneratorBytes, resources, seqCodec)` as
/// a training run uses it, with no resources: a descriptor whose
/// generators need one (a dictionary, a word cluster) cannot be built
/// here ([`Error::Sequence`]).
///
/// With no descriptor (`null`) the factory loads OpenNLP's default one
/// before the first context generator is made, so a trained model always
/// carries a `generator.featuregen` entry.
#[derive(Debug, Clone)]
pub struct TokenNameFinderFactory {
    language: String,
    descriptor: Vec<u8>,
    codec: NameCodec,
}

impl TokenNameFinderFactory {
    /// `descriptor` is the feature generator XML, `None` for the default.
    /// It is built once here, so a descriptor the generators cannot be
    /// made from is refused before any training.
    pub fn new(language: &str, descriptor: Option<Vec<u8>>, codec: NameCodec) -> Result<Self> {
        let descriptor = descriptor.unwrap_or_else(|| DEFAULT_DESCRIPTOR.as_bytes().to_vec());
        let factory = TokenNameFinderFactory { language: language.to_owned(), descriptor, codec };
        factory.feature_generators()?;
        Ok(factory)
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    /// `getFeatureGenerator`: the descriptor bytes the model file carries.
    pub fn descriptor(&self) -> &[u8] {
        &self.descriptor
    }

    /// `createSequenceCodec`.
    pub fn codec(&self) -> NameCodec {
        self.codec
    }

    /// `createFeatureGenerators`: the descriptor's generators. Features are
    /// built over UTF-16 code units, which is how OpenNLP builds them.
    fn feature_generators(&self) -> Result<BoxedGenerator> {
        Ok(descriptor::build(&self.descriptor, None, PatternMode::Utf16Units)?)
    }

    /// `createContextGenerator()`: `DefaultNameContextGenerator` over the
    /// descriptor's generators.
    pub fn context_generator(&self) -> Result<DefaultNameContext> {
        Ok(DefaultNameContext::new(vec![self.feature_generators()?], PatternMode::Utf16Units))
    }

    /// The manifest entries the model adds for the factory: `factory`, as
    /// `BaseModel` adds it, and the codec's class, as
    /// `TokenNameFinderModel` adds it.
    pub fn manifest_entries(&self) -> Vec<(String, String)> {
        vec![
            (FACTORY_KEY.to_owned(), FACTORY.to_owned()),
            (CODEC_PROPERTY.to_owned(), self.codec.class_name().to_owned()),
        ]
    }
}

/// `NameFinderEventStream` over one factory's context generator and codec.
pub struct NameEventStream {
    context: DefaultNameContext,
    codec: NameCodec,
    kind: Option<String>,
    features: Features,
}

impl NameEventStream {
    /// The stream for `NameFinderME.train(language, kind, ...)`: `kind`
    /// replaces every name's type when given.
    pub fn new(factory: &TokenNameFinderFactory, kind: Option<&str>) -> Result<NameEventStream> {
        let mut context = factory.context_generator()?;
        context.add_feature_generator(Box::new(Window::new(Box::new(AdditionalContext::new()), 8, 8)));
        Ok(NameEventStream {
            context,
            codec: factory.codec(),
            kind: kind.map(str::to_owned),
            features: Features::new(),
        })
    }

    /// `createEvents(sample)`: appends one event per token of the sample
    /// to `out`, in token order, and passes the sample to the generators'
    /// adaptive data. A name the codec would write outside the sentence
    /// is [`Error::UnencodableName`], and the sample adds no event; the
    /// generators' adaptive data was cleared first if the sample asked
    /// for it, as in the JVM.
    ///
    /// A sample read from the name format has no additional context, so
    /// the window of additional context never adds a feature.
    pub fn events(&mut self, sample: &NameSample<'_>, out: &mut Vec<Event>) -> Result<()> {
        if sample.clear_adaptive_data() {
            self.context.clear_adaptive_data();
        }
        let length = sample.tokens().len();
        let spans: Vec<Span> = sample
            .names()
            .iter()
            .map(|n| Span::new(n.start as usize, n.end as usize, self.kind.as_deref().or(n.kind)))
            .collect();
        if let Some(index) = unencodable(self.codec, &spans, length) {
            return Err(Error::UnencodableName { index, length });
        }
        let outcomes = self.codec.encode(&spans, length);
        let outcomes: Vec<&str> = outcomes.iter().map(String::as_str).collect();
        let tokens: Vec<&str> = sample.tokens().iter().map(|f| f.as_str()).collect();
        self.context.set_additional_context(&[]);
        self.context.start_sequence();
        for i in 0..length {
            self.features.clear();
            self.context.context(i, &tokens, &outcomes, &mut self.features);
            out.push(Event::new(outcomes[i], java_strings(&self.features)));
        }
        self.context.update_adaptive_data(&tokens, &outcomes);
        Ok(())
    }
}

/// The index of the first outcome the codec would write outside a
/// sentence of `length` tokens, in the order it writes them: with BIO an
/// empty span at `length`, with BILOU an empty span at 0 (index -1). The
/// spans end at or before `length`, which `NameSample` checks.
fn unencodable(codec: NameCodec, spans: &[Span], length: usize) -> Option<i64> {
    spans.iter().find_map(|s| match codec {
        NameCodec::Bio => (s.start >= length).then_some(s.start as i64),
        NameCodec::Bilou => (s.len() <= 1 && s.end == 0).then_some(-1),
    })
}

/// The events of `NameFinderEventStream` over `samples`. The first sample
/// the codec cannot encode stops the stream, as the exception stops the
/// JVM's training.
pub fn name_events<'a, 's>(
    samples: impl IntoIterator<Item = &'s NameSample<'a>>,
    factory: &TokenNameFinderFactory,
    kind: Option<&str>,
) -> Result<Vec<Event>>
where
    'a: 's,
{
    let mut stream = NameEventStream::new(factory, kind)?;
    let mut out = Vec::new();
    for sample in samples {
        stream.events(sample, &mut out)?;
    }
    Ok(out)
}

/// The parameters `NameFinderME.train` trains with: `params` with
/// `Algorithm=MAXENT`, `Cutoff=0` and `Iterations=300` put in where
/// absent. A key is matched without regard to case, as Java's
/// `TrainingParameters` matches it.
pub fn training_parameters(params: &TrainingParameters) -> TrainingParameters {
    let mut p = params.clone();
    p.put_if_absent(Key::ALGORITHM, Algorithm::Maxent.name());
    p.put_if_absent(Key::CUTOFF, DEFAULT_CUTOFF);
    p.put_if_absent(Key::ITERATIONS, DEFAULT_ITERATIONS);
    p
}

/// `mlParams.getIntParameter(BeamSize, DEFAULT_BEAM_SIZE)`.
pub fn beam_size(params: &TrainingParameters) -> Result<i32> {
    Ok(params.get_int(BEAM_SIZE_KEY)?.unwrap_or(DEFAULT_BEAM_SIZE))
}

/// A trained name finder model: the model the trainer made, what the
/// trainer reports, the parameters it was trained with (the defaults
/// put in), and the `.bin` file.
#[derive(Debug, Clone)]
pub struct TrainedNameModel {
    pub model: TrainedModel,
    pub report: TrainingReport,
    pub params: TrainingParameters,
    pub package: ModelPackage,
}

/// `new TokenNameFinderModel(languageCode, nameFinderModel, beamSize,
/// factory.getFeatureGenerator(), resources, manifestInfoEntries,
/// seqCodec, factory)` serialized: the manifest with the trainer's
/// entries, `factory`, `BeamSize` and `sequenceCodecImplName`;
/// `nameFinder.model` holding `model_bytes`; and the descriptor as
/// `generator.featuregen`, in the JVM's zip order. The outcomes are not
/// checked against the codec here; [`train_name_model_from_events`] does
/// that with the trained model.
pub fn package_name_model(
    factory: &TokenNameFinderFactory,
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
    add_artifacts(&mut package, vec![(MODEL_ENTRY, model_bytes), (DESCRIPTOR_ENTRY, factory.descriptor().to_vec())])?;
    Ok(package)
}

/// `NameFinderME.train(languageCode, kind, samples, params, factory)` with
/// an event trainer: the events of [`name_events`], the trainer the
/// parameters name once the defaults are in ([`training_parameters`]),
/// the beam size of the parameters, and the model file stamped with
/// `stamp`.
pub fn train_name_model<'a, 's>(
    samples: impl IntoIterator<Item = &'s NameSample<'a>>,
    factory: &TokenNameFinderFactory,
    kind: Option<&str>,
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
    stamp: Stamp<'_>,
) -> Result<TrainedNameModel>
where
    'a: 's,
{
    let params = training_parameters(params);
    event_trainer(&params)?;
    let events = name_events(samples, factory, kind)?;
    train_name_model_from_events(&events, factory, &params, config, stamp)
}

/// Refuses the parameters when they name a trainer that does not train
/// from events, which `NameFinderME.train` would feed with
/// `NameSampleSequenceStream`.
fn event_trainer(params: &TrainingParameters) -> Result<()> {
    match trainer_type(params)? {
        Some(TrainerType::Event) | None => Ok(()),
        Some(other) => Err(Error::NameSequenceTraining(other)),
    }
}

/// [`train_name_model`] from events already made, with the defaults put
/// into `params` first. Outcomes the codec cannot decode are refused, as
/// `TokenNameFinderModel`'s constructor refuses them.
pub fn train_name_model_from_events(
    events: &[Event],
    factory: &TokenNameFinderFactory,
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
    stamp: Stamp<'_>,
) -> Result<TrainedNameModel> {
    let params = training_parameters(params);
    event_trainer(&params)?;
    let beam = beam_size(&params)?;
    let (model, report) = trainer::train_events(events, &params, config)?;
    let outcomes: Vec<String> = model.as_writable().outcome_labels().iter().map(JavaString::to_string_lossy).collect();
    let outcomes: Vec<&str> = outcomes.iter().map(String::as_str).collect();
    if !factory.codec().are_outcomes_compatible(&outcomes) {
        return Err(Error::IncompatibleOutcomes);
    }
    let bytes = model.write()?;
    let package = package_name_model(factory, &report, bytes, beam, stamp)?;
    Ok(TrainedNameModel { model, report, params, package })
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::Span as Bytes;
    use turbo_formats::native::NameReader;
    use turbo_formats::{Field, Name};
    use turbo_train::Value;
    use turbo_unicode::WhitespaceMode;

    fn samples(text: &str) -> Vec<NameSample<'_>> {
        NameReader::new(text, WhitespaceMode::Unicode).unwrap().map(|s| s.unwrap()).collect()
    }

    fn strings(e: &Event) -> Vec<String> {
        e.context().iter().map(JavaString::to_string_lossy).collect()
    }

    fn outcomes(events: &[Event]) -> Vec<String> {
        events.iter().map(|e| e.outcome().to_string_lossy()).collect()
    }

    fn bio() -> TokenNameFinderFactory {
        TokenNameFinderFactory::new("en", None, NameCodec::Bio).unwrap()
    }

    #[test]
    fn the_default_factory_carries_the_default_descriptor_and_names_its_codec() {
        let f = bio();
        assert_eq!(f.descriptor(), DEFAULT_DESCRIPTOR.as_bytes());
        assert_eq!(
            f.manifest_entries(),
            [
                ("factory".to_owned(), FACTORY.to_owned()),
                ("sequenceCodecImplName".to_owned(), "opennlp.tools.namefind.BioCodec".to_owned())
            ]
        );
        let f = TokenNameFinderFactory::new("en", None, NameCodec::Bilou).unwrap();
        assert_eq!(f.manifest_entries()[1].1, "opennlp.tools.namefind.BilouCodec");
    }

    #[test]
    fn each_token_is_an_event_with_the_samples_outcomes_as_history() {
        let s = samples("<START:person> Pierre Vinken <END> said .\n");
        let events = name_events(&s, &bio(), None).unwrap();
        assert_eq!(outcomes(&events), ["person-start", "person-cont", "other", "other"]);
        let first = strings(&events[0]);
        // The default descriptor's features, then the previous outcomes.
        assert!(first.contains(&"w=pierre".to_owned()) && first.contains(&"S=begin".to_owned()), "{first:?}");
        assert_eq!(&first[first.len() - 4..], ["po=other", "pow=other,Pierre", "powf=other,ic", "ppo=other"]);
        let third = strings(&events[2]);
        assert_eq!(
            &third[third.len() - 4..],
            ["po=person-cont", "pow=person-cont,said", "powf=person-cont,lc", "ppo=person-start"]
        );
    }

    #[test]
    fn the_previous_map_remembers_outcomes_until_an_empty_line_clears_it() {
        // `PreviousMapFeatureGenerator` writes `pd=` and the outcome the
        // token had the last time it was seen since the last clear.
        let s = samples("<START:x> Bo <END> ran .\nBo ran .\n\nBo ran .\n");
        let events = name_events(&s, &bio(), None).unwrap();
        let pd = |i: usize| strings(&events[i]).into_iter().filter(|f| f.starts_with("pd=")).collect::<Vec<_>>();
        assert_eq!(pd(0), ["pd=null"]);
        assert_eq!(pd(3), ["pd=x-start"]);
        assert_eq!(pd(4), ["pd=other"]);
        assert_eq!(pd(6), ["pd=null"], "the empty line clears the map");
    }

    #[test]
    fn a_type_replaces_every_names_type_and_an_untyped_name_is_default() {
        let tokens = vec![Field::made("a".into(), Bytes::default()), Field::made("b".into(), Bytes::default())];
        let s =
            vec![NameSample::new(tokens, vec![Name::new(0, 1, None).unwrap()], false, Bytes::default(), 1).unwrap()];
        assert_eq!(outcomes(&name_events(&s, &bio(), None).unwrap()), ["default-start", "other"]);
        let s = samples("<START:person> a <END> <START:location> b c <END>\n");
        assert_eq!(outcomes(&name_events(&s, &bio(), Some("t")).unwrap()), ["t-start", "t-start", "t-cont"]);
        let bilou = TokenNameFinderFactory::new("en", None, NameCodec::Bilou).unwrap();
        assert_eq!(
            outcomes(&name_events(&s, &bilou, None).unwrap()),
            ["person-unit", "location-start", "location-last"]
        );
    }

    #[test]
    fn a_name_the_codec_would_write_outside_the_sentence_gives_no_event() {
        // BIO: an empty name after the last token.
        let s = samples("a <START:x> <END>\n");
        let mut stream = NameEventStream::new(&bio(), None).unwrap();
        let mut out = Vec::new();
        let e = stream.events(&s[0], &mut out).unwrap_err();
        assert!(matches!(e, Error::UnencodableName { index: 1, length: 1 }), "{e}");
        assert_eq!(e.to_string(), "Index 1 out of bounds for length 1");
        assert!(out.is_empty());
        // BILOU: an empty name at the first token.
        let s = samples("<START:x> <END> a\n");
        let bilou = TokenNameFinderFactory::new("en", None, NameCodec::Bilou).unwrap();
        let e = name_events(&s, &bilou, None).unwrap_err();
        assert_eq!(e.to_string(), "Index -1 out of bounds for length 1");
        // The same samples encode with the other codec: BIO marks the
        // first token, BILOU the token before the empty name.
        assert_eq!(outcomes(&name_events(&s, &bio(), None).unwrap()), ["x-start"]);
        let s = samples("a <START:x> <END>\n");
        assert_eq!(outcomes(&name_events(&s, &bilou, None).unwrap()), ["x-unit"]);
    }

    #[test]
    fn the_defaults_go_in_only_where_absent_whatever_the_case_of_the_key() {
        let p = training_parameters(&TrainingParameters::new());
        assert_eq!(p.get(Key::ALGORITHM), Some(&Value::from("MAXENT")));
        assert_eq!(p.get_int(Key::CUTOFF).unwrap(), Some(0));
        assert_eq!(p.get_int(Key::ITERATIONS).unwrap(), Some(300));
        let mut given = TrainingParameters::new();
        given.put("iterations", 7);
        given.put_str(Key::ALGORITHM, "PERCEPTRON");
        let p = training_parameters(&given);
        assert_eq!(p.get_int(Key::ITERATIONS).unwrap(), Some(7));
        assert_eq!(p.algorithm().unwrap(), Some("PERCEPTRON"));
        assert_eq!(p.len(), 3);
        assert_eq!(beam_size(&p).unwrap(), 3);
        given.put(BEAM_SIZE_KEY, 5);
        assert_eq!(beam_size(&given).unwrap(), 5);
    }

    #[test]
    fn a_trained_model_is_packaged_in_the_jvms_entry_order() {
        let s = samples("<START:person> Ann <END> met <START:person> Bo <END> .\nIt rained .\n");
        let mut p = TrainingParameters::new();
        p.put(Key::ITERATIONS, 10);
        let ours = train_name_model(&s, &bio(), None, &p, &mut TrainingConfiguration::default(), Stamp::at(0)).unwrap();
        assert_eq!(ours.package.entry_names(), ["manifest.properties", "nameFinder.model", "generator.featuregen"]);
        let file = turbo_maxent::ModelFile::open(ours.package.to_bytes()).unwrap();
        let m = file.manifest();
        assert_eq!(m.get("Component-Name"), Some("NameFinderME"));
        assert_eq!(m.get("BeamSize"), Some("3"));
        assert_eq!(m.get("factory"), Some(FACTORY));
        assert_eq!(m.get("sequenceCodecImplName"), Some("opennlp.tools.namefind.BioCodec"));
        assert_eq!(m.get("TrainerType"), Some("Event"));
        assert_eq!(&*file.entry(DESCRIPTOR_ENTRY).unwrap(), DEFAULT_DESCRIPTOR.as_bytes());
        assert_eq!(ours.model.algorithm(), Algorithm::Maxent);
        assert_eq!(ours.params.get_int(Key::CUTOFF).unwrap(), Some(0));
    }

    #[test]
    fn outcomes_the_codec_cannot_decode_are_refused() {
        // Samples without a name give only `other`, one outcome, which GIS
        // refuses before the codec is asked.
        let s = samples("It rained .\nThen it stopped .\n");
        let mut config = TrainingConfiguration::default();
        let e = train_name_model(&s, &bio(), None, &TrainingParameters::new(), &mut config, Stamp::at(0)).unwrap_err();
        assert!(matches!(e, Error::Train(_)), "{e}");
        // With BILOU an empty name right after a name turns that name's
        // last token into a unit, so the model has `a-start` and no
        // `a-last`, which BILOU cannot decode.
        let s = samples("<START:a> x y <END> <START:b> <END> z\n");
        let bilou = TokenNameFinderFactory::new("en", None, NameCodec::Bilou).unwrap();
        assert_eq!(outcomes(&name_events(&s, &bilou, None).unwrap()), ["a-start", "b-unit", "other"]);
        let mut p = TrainingParameters::new();
        p.put(Key::ITERATIONS, 5);
        let e = train_name_model(&s, &bilou, None, &p, &mut config, Stamp::at(0)).unwrap_err();
        assert!(matches!(e, Error::IncompatibleOutcomes), "{e}");
        assert_eq!(e.to_string(), INCOMPATIBLE_MESSAGE);
    }

    #[test]
    fn the_sequence_perceptron_is_refused_before_any_event() {
        let s = samples("<START:x> a <END>\n");
        let mut p = TrainingParameters::new();
        p.put_str(Key::ALGORITHM, "PERCEPTRON_SEQUENCE");
        let e =
            train_name_model(&s, &bio(), None, &p, &mut TrainingConfiguration::default(), Stamp::at(0)).unwrap_err();
        assert!(matches!(e, Error::NameSequenceTraining(_)), "{e}");
    }

    #[test]
    fn a_descriptor_that_needs_a_resource_is_refused() {
        let xml = br#"<featureGenerators><generator class="opennlp.tools.util.featuregen.WordClusterFeatureGeneratorFactory"><str name="dict">x</str></generator></featureGenerators>"#;
        let e = TokenNameFinderFactory::new("en", Some(xml.to_vec()), NameCodec::Bio).unwrap_err();
        assert!(matches!(e, Error::Sequence(_)), "{e}");
    }
}
