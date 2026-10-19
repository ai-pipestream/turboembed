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
    let s = vec![NameSample::new(tokens, vec![Name::new(0, 1, None).unwrap()], false, Bytes::default(), 1).unwrap()];
    assert_eq!(outcomes(&name_events(&s, &bio(), None).unwrap()), ["default-start", "other"]);
    let s = samples("<START:person> a <END> <START:location> b c <END>\n");
    assert_eq!(outcomes(&name_events(&s, &bio(), Some("t")).unwrap()), ["t-start", "t-start", "t-cont"]);
    let bilou = TokenNameFinderFactory::new("en", None, NameCodec::Bilou).unwrap();
    assert_eq!(outcomes(&name_events(&s, &bilou, None).unwrap()), ["person-unit", "location-start", "location-last"]);
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
    let e = train_name_model(&s, &bio(), None, &p, &mut TrainingConfiguration::default(), Stamp::at(0)).unwrap_err();
    assert!(matches!(e, Error::NameSequenceTraining(_)), "{e}");
}

#[test]
fn a_descriptor_that_needs_a_resource_is_refused() {
    let xml = br#"<featureGenerators><generator class="opennlp.tools.util.featuregen.WordClusterFeatureGeneratorFactory"><str name="dict">x</str></generator></featureGenerators>"#;
    let e = TokenNameFinderFactory::new("en", Some(xml.to_vec()), NameCodec::Bio).unwrap_err();
    assert!(matches!(e, Error::Sequence(_)), "{e}");
}
