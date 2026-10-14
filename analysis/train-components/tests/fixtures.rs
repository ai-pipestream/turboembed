//! The fixture models the JVM trained are trained again here, and the
//! parts the JVM wrote come out byte for byte: the manifest, the
//! dictionary entry and the model entry.

use turbo_goldens::data_dir;
use turbo_maxent::ModelFile;
use turbo_train::package::MANIFEST_ENTRY;
use turbo_train::{Key, TrainingConfiguration, TrainingParameters};
use turbo_train_components::sentdetect::{self, SentenceDetectorFactory, train_sentence_model};
use turbo_train_components::tokenize::{self, TokenizerFactory, train_tokenizer_model};
use turbo_train_components::{AbbreviationDictionary, Stamp};
use turbo_unicode::WhitespaceMode;

fn fixture(name: &str) -> ModelFile {
    ModelFile::read(data_dir().join("fixtures").join(name)).unwrap()
}

fn timestamp(file: &ModelFile) -> i64 {
    file.manifest().get("Timestamp").unwrap().parse().unwrap()
}

/// `analysis/sentdetect/tests/fixtures/src/TrainSentenceModel.java`.
fn sentence_parameters() -> TrainingParameters {
    let mut p = TrainingParameters::new();
    p.put_str(Key::ALGORITHM, "MAXENT");
    p.put_str(Key::TRAINER_TYPE, "Event");
    p.put(Key::ITERATIONS, 100);
    p.put(Key::CUTOFF, 1);
    p
}

fn sentences() -> String {
    std::fs::read_to_string(data_dir().join("inputs/train/sentences.txt")).unwrap()
}

#[test]
fn en_plain_is_trained_again_from_its_sentences() {
    let theirs = fixture("en-plain.bin");
    let text = sentences();
    let samples: Vec<_> = turbo_formats::native::SentenceReader::new(&text).unwrap().map(|s| s.unwrap()).collect();
    let factory = SentenceDetectorFactory::new("en", false, None, None).unwrap();
    let ours = train_sentence_model(
        &samples,
        &factory,
        &sentence_parameters(),
        &mut TrainingConfiguration::default(),
        WhitespaceMode::Unicode,
        Stamp::at(timestamp(&theirs)),
    )
    .unwrap();
    let ours = ModelFile::open(ours.package.to_bytes()).unwrap();
    for entry in [MANIFEST_ENTRY, sentdetect::MODEL_ENTRY] {
        assert_eq!(&*ours.entry(entry).unwrap(), &*theirs.entry(entry).unwrap(), "{entry}");
    }
    assert_eq!(ours.archive().len(), 2);
}

#[test]
fn en_abbrev_is_trained_again_with_its_dictionary() {
    let theirs = fixture("en-abbrev.bin");
    let text = sentences();
    let samples: Vec<_> = turbo_formats::native::SentenceReader::new(&text).unwrap().map(|s| s.unwrap()).collect();
    // The entries in the order TrainSentenceModel.java puts them.
    let entries = [
        vec!["Mr."],
        vec!["Mrs."],
        vec!["Dr."],
        vec!["Prof."],
        vec!["St."],
        vec!["Inc."],
        vec!["U.S."],
        vec!["e.g."],
        vec!["i.e."],
        vec!["etc."],
        vec!["a.m."],
        vec!["p.m."],
        vec!["et", "al."],
    ];
    let dictionary =
        AbbreviationDictionary::new(false, entries.into_iter().map(|e| e.into_iter().map(str::to_string).collect()))
            .unwrap();
    let factory = SentenceDetectorFactory::new("en", true, Some(dictionary), None).unwrap();
    let ours = train_sentence_model(
        &samples,
        &factory,
        &sentence_parameters(),
        &mut TrainingConfiguration::default(),
        WhitespaceMode::Unicode,
        Stamp::at(timestamp(&theirs)),
    )
    .unwrap();
    let names: Vec<&str> = std::iter::once(MANIFEST_ENTRY).chain(ours.package.entries().map(|(n, _)| n)).collect();
    let their_names: Vec<String> = theirs.archive().names().map(|n| String::from_utf8(n.to_vec()).unwrap()).collect();
    assert_eq!(names, their_names, "the zip entries in the JVM's order");
    let ours = ModelFile::open(ours.package.to_bytes()).unwrap();
    for entry in [MANIFEST_ENTRY, sentdetect::ABBREVIATIONS_ENTRY, sentdetect::MODEL_ENTRY] {
        assert_eq!(&*ours.entry(entry).unwrap(), &*theirs.entry(entry).unwrap(), "{entry}");
    }
}

#[test]
fn the_tokenizer_fixture_is_trained_again_from_its_six_samples() {
    // analysis/maxent/tests/fixtures/src/MakeModelFixtures.java.
    let theirs = fixture("tokenizer-abbreviations.bin");
    let xml =
        b"<dictionary case_sensitive=\"false\"><entry><token>Dr.</token></entry><entry><token>e.g.</token></entry>\
        <entry><token>U.S.</token></entry></dictionary>";
    let dictionary = AbbreviationDictionary::read(xml).unwrap();
    let text = "Dr. Smith<SPLIT>, e.g. here<SPLIT>.\nIt<SPLIT>'s ok<SPLIT>.\nThe U.S. team won<SPLIT>!\n\
        Call me<SPLIT>, please<SPLIT>.\n(<SPLIT>yes<SPLIT>)\n\"<SPLIT>Hi<SPLIT>\"<SPLIT>, she said<SPLIT>.\n";
    let samples: Vec<_> =
        turbo_formats::native::TokenReader::new(text, WhitespaceMode::Unicode).unwrap().map(|s| s.unwrap()).collect();
    let mut p = TrainingParameters::new();
    p.put_str(Key::ALGORITHM, "MAXENT");
    p.put(Key::ITERATIONS, 50);
    p.put(Key::CUTOFF, 0);
    let factory = TokenizerFactory::new("en", Some(dictionary), false, Some("^[A-Za-z0-9]+$")).unwrap();
    let ours = train_tokenizer_model(
        &samples,
        &factory,
        &p,
        &mut TrainingConfiguration::default(),
        WhitespaceMode::Unicode,
        Stamp::at(timestamp(&theirs)),
    )
    .unwrap();
    let names: Vec<&str> = std::iter::once(MANIFEST_ENTRY).chain(ours.package.entries().map(|(n, _)| n)).collect();
    let their_names: Vec<String> = theirs.archive().names().map(|n| String::from_utf8(n.to_vec()).unwrap()).collect();
    assert_eq!(names, their_names, "the zip entries in the JVM's order");
    let ours = ModelFile::open(ours.package.to_bytes()).unwrap();
    for entry in [MANIFEST_ENTRY, tokenize::ABBREVIATIONS_ENTRY, tokenize::MODEL_ENTRY] {
        assert_eq!(&*ours.entry(entry).unwrap(), &*theirs.entry(entry).unwrap(), "{entry}");
    }
}
