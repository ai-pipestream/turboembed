//! The fixture models the JVM trained are trained again here, and the
//! parts the JVM wrote come out byte for byte: the manifest, the
//! dictionary entries, the descriptor and the model entry. The fixtures
//! the Rust trainer wrote (`pos-fixture-rust.bin`, `lemma-fixture-rust.bin`,
//! `chunker-fixture-rust.bin`, `langdetect-fixture-rust.bin`) are trained
//! here too and must be the checked in files byte for byte; the golden
//! generator loads them in the JVM and tags, lemmatizes, chunks or detects
//! with them (`analysis/postag/tests/golden.rs`,
//! `analysis/lemmatizer/tests/golden.rs`,
//! `analysis/phrase-chunker/tests/golden.rs`,
//! `analysis/langdetect/tests/fixture.rs`).
//!
//! With `TURBO_TRAIN_OUT` set to a directory, the Rust fixtures trained
//! here are written there, which is how they were made.

use std::path::{Path, PathBuf};

use turbo_formats::native::{ChunkReader, LanguageReader, LemmaReader, WordTagReader};
use turbo_goldens::data_dir;
use turbo_maxent::{JavaString, ModelFile};
use turbo_train::package::MANIFEST_ENTRY;
use turbo_train::{Algorithm, Key, TimeZone, TrainingConfiguration, TrainingParameters};
use turbo_train_components::chunk::{self, ChunkerFactory, train_chunker_model};
use turbo_train_components::langdetect::{self, LanguageDetectorFactory, train_language_model};
use turbo_train_components::lemmatize::{self, LemmatizerFactory, train_lemmatizer_model};
use turbo_train_components::postag::{self, PosTaggerFactory, train_pos_model};
use turbo_train_components::sentdetect::{self, SentenceDetectorFactory, train_sentence_model};
use turbo_train_components::tokenize::{self, TokenizerFactory, train_tokenizer_model};
use turbo_train_components::{AbbreviationDictionary, PosDictionary, Stamp};
use turbo_unicode::WhitespaceMode;

fn fixture(name: &str) -> ModelFile {
    ModelFile::read(data_dir().join("fixtures").join(name)).unwrap()
}

fn timestamp(file: &ModelFile) -> i64 {
    file.manifest().get("Timestamp").unwrap().parse().unwrap()
}

fn entry_names(file: &ModelFile) -> Vec<String> {
    file.archive().names().map(|n| String::from_utf8(n.to_vec()).unwrap()).collect()
}

/// A file of `analysis/goldens/data`, where the tagger fixture's training
/// data (`train-fixtures`), descriptor and tag dictionary (`fixtures`) are.
fn goldens_data(path: &str) -> PathBuf {
    data_dir().join(path)
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn read_text(path: &Path) -> String {
    String::from_utf8(read(path)).unwrap()
}

/// `FixtureModels.java` in `analysis/goldens-tool`, and the settings
/// of the Rust fixtures.
fn tagger_parameters(iterations: i32) -> TrainingParameters {
    let mut p = TrainingParameters::new();
    p.put_str(Key::ALGORITHM, "MAXENT");
    p.put(Key::ITERATIONS, iterations);
    p.put(Key::CUTOFF, 1);
    p.put(Key::THREADS, 1);
    p
}

/// The instant the Rust fixtures are stamped with.
const RUST_FIXTURE_TIMESTAMP: i64 = 1_760_400_000_000;

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
    assert_eq!(ours.package.entry_names(), entry_names(&theirs), "the zip entries in the JVM's order");
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
    assert_eq!(ours.package.entry_names(), entry_names(&theirs), "the zip entries in the JVM's order");
    let ours = ModelFile::open(ours.package.to_bytes()).unwrap();
    for entry in [MANIFEST_ENTRY, tokenize::ABBREVIATIONS_ENTRY, tokenize::MODEL_ENTRY] {
        assert_eq!(&*ours.entry(entry).unwrap(), &*theirs.entry(entry).unwrap(), "{entry}");
    }
}

#[test]
fn pos_fixture_is_trained_again_with_its_descriptor_and_tag_dictionary() {
    let theirs = fixture("pos-fixture.bin");
    let text = read_text(&goldens_data("train-fixtures/train-pos.txt"));
    let samples: Vec<_> = WordTagReader::new(&text, WhitespaceMode::Unicode).unwrap().map(|s| s.unwrap()).collect();
    let descriptor = read(&goldens_data("fixtures/featuregen/pos-default.xml"));
    let dictionary = PosDictionary::read(&read(&goldens_data("fixtures/tags.tagdict"))).unwrap();
    let factory = PosTaggerFactory::new("en", Some(descriptor), Some(dictionary)).unwrap();
    // The fixture was written by the fork at 3.0.0-SNAPSHOT, in UTC.
    let stamp = Stamp { timestamp_ms: timestamp(&theirs), version: "3.0.0-SNAPSHOT", zone: TimeZone::UTC };
    let ours =
        train_pos_model(&samples, &factory, &tagger_parameters(50), &mut TrainingConfiguration::default(), stamp)
            .unwrap();
    assert_eq!(
        ours.package.entry_names(),
        ["tags.tagdict", "manifest.properties", "generator.featuregen", "pos.model"],
        "the tag dictionary before the manifest, as the JVM's artifact map orders them"
    );
    assert_eq!(ours.package.entry_names(), entry_names(&theirs));
    let ours = ModelFile::open(ours.package.to_bytes()).unwrap();
    for entry in [MANIFEST_ENTRY, postag::TAG_DICTIONARY_ENTRY, postag::DESCRIPTOR_ENTRY, postag::MODEL_ENTRY] {
        assert_eq!(&*ours.entry(entry).unwrap(), &*theirs.entry(entry).unwrap(), "{entry}");
    }
}

#[test]
fn chunker_fixture_is_trained_again_from_its_eight_sentences() {
    // FixtureModels.java: ChunkerME.train over ChunkSampleStream with the
    // perceptron, 50 iterations, cutoff 1, one thread.
    let theirs = fixture("chunker-fixture.bin");
    let text = read_text(&goldens_data("train-fixtures/train-chunk.txt"));
    let samples: Vec<_> = ChunkReader::new(&text).unwrap().map(|s| s.unwrap()).collect();
    assert_eq!(samples.len(), 8);
    let mut p = TrainingParameters::new();
    p.put_str(Key::ALGORITHM, Algorithm::Perceptron.name());
    p.put(Key::ITERATIONS, 50);
    p.put(Key::CUTOFF, 1);
    p.put(Key::THREADS, 1);
    let stamp = Stamp { timestamp_ms: timestamp(&theirs), version: "3.0.0-SNAPSHOT", zone: TimeZone::UTC };
    let ours =
        train_chunker_model(&samples, &ChunkerFactory::new("en"), &p, &mut TrainingConfiguration::default(), stamp)
            .unwrap();
    assert_eq!(ours.package.entry_names(), ["chunker.model", "manifest.properties"]);
    assert_eq!(ours.package.entry_names(), entry_names(&theirs), "the zip entries in the JVM's order");
    let ours = ModelFile::open(ours.package.to_bytes()).unwrap();
    for entry in [MANIFEST_ENTRY, chunk::MODEL_ENTRY] {
        assert_eq!(&*ours.entry(entry).unwrap(), &*theirs.entry(entry).unwrap(), "{entry}");
    }
}

/// The tagged inputs the Rust fixtures are trained from.
fn tagged(file: &str) -> String {
    read_text(&data_dir().join("inputs/tagged").join(file))
}

/// Writes a Rust fixture under `TURBO_TRAIN_OUT` when set, and checks it is
/// the checked in file.
fn check_rust_fixture(name: &str, bin: &[u8]) {
    if let Ok(dir) = std::env::var("TURBO_TRAIN_OUT") {
        std::fs::write(Path::new(&dir).join(name), bin).unwrap();
    }
    let committed = read(&data_dir().join("fixtures").join(name));
    assert_eq!(bin, committed, "analysis/goldens/data/fixtures/{name} is not the .bin written here");
}

#[test]
fn the_rust_pos_fixture_is_the_bin_written_here() {
    let text = tagged("wordtag.txt");
    let samples: Vec<_> = WordTagReader::new(&text, WhitespaceMode::Unicode).unwrap().map(|s| s.unwrap()).collect();
    assert_eq!(samples.len(), 20);
    let dictionary = PosDictionary::read(&read(&data_dir().join("fixtures/tags.tagdict"))).unwrap();
    let factory = PosTaggerFactory::new("en", None, Some(dictionary)).unwrap();
    let ours = train_pos_model(
        &samples,
        &factory,
        &tagger_parameters(100),
        &mut TrainingConfiguration::default(),
        Stamp::at(RUST_FIXTURE_TIMESTAMP),
    )
    .unwrap();
    let bin = ours.package.to_bytes();
    check_rust_fixture("pos-fixture-rust.bin", &bin);
    let file = ModelFile::open(bin).unwrap();
    assert_eq!(entry_names(&file), ["tags.tagdict", "manifest.properties", "generator.featuregen", "pos.model"]);
    assert_eq!(file.manifest().get("Component-Name"), Some("POSTaggerME"));
    assert_eq!(file.manifest().get("BeamSize"), Some("3"));
    assert_eq!(
        file.manifest().get("serializer-class-tags.tagdict"),
        Some("opennlp.tools.postag.POSTaggerFactory$POSDictionarySerializer")
    );
    // The dictionary's tags are all outcomes, so the JVM's check passes.
    let model = file.maxent(postag::MODEL_ENTRY).unwrap();
    let outcomes: Vec<&str> = model.outcome_texts().unwrap();
    for tag in ["DT", "NN", "VB", "VBP", "VBD", "VBN", "NNP"] {
        assert!(outcomes.contains(&tag), "{tag} is not an outcome of {outcomes:?}");
    }
    let dictionary = turbo_sequence::postag::TagDictionary::read(&file.entry(postag::TAG_DICTIONARY_ENTRY).unwrap());
    assert!(dictionary.is_ok(), "{dictionary:?}");
}

#[test]
fn the_rust_lemma_fixture_is_the_bin_written_here() {
    let text = tagged("lemmas.txt");
    let samples: Vec<_> = LemmaReader::new(&text).unwrap().map(|s| s.unwrap()).collect();
    assert_eq!(samples.len(), 20);
    let factory = LemmatizerFactory::new("en");
    let ours = train_lemmatizer_model(
        &samples,
        &factory,
        &tagger_parameters(100),
        &mut TrainingConfiguration::default(),
        Stamp::at(RUST_FIXTURE_TIMESTAMP),
    )
    .unwrap();
    let bin = ours.package.to_bytes();
    check_rust_fixture("lemma-fixture-rust.bin", &bin);
    let file = ModelFile::open(bin).unwrap();
    assert_eq!(entry_names(&file), ["manifest.properties", "lemmatizer.model"]);
    assert_eq!(file.manifest().get("Component-Name"), Some("StatisticalLemmatizer"));
    assert_eq!(file.manifest().get("factory"), Some("opennlp.tools.lemmatizer.LemmatizerFactory"));
    let model = file.maxent(lemmatize::MODEL_ENTRY).unwrap();
    let outcomes: Vec<&JavaString> = model.outcomes().collect();
    // The identity class, the class of a plural noun losing its s, and the
    // classes of `Smile\u{1F600}` to `smile` and `ok\u{1F44D}` to `ok`,
    // which delete the two units of the emoji and so hold each half of its
    // surrogate pair on its own.
    for class in ["O", "D0s"] {
        assert!(outcomes.iter().any(|o| *o == class), "{class} is not an outcome of {outcomes:?}");
    }
    let smile = JavaString::from_utf16(&[0x44, 0x31, 0xDE00, 0x44, 0x30, 0xD83D]);
    let ok = JavaString::from_utf16(&[0x44, 0x31, 0xDC4D, 0x44, 0x30, 0xD83D]);
    assert!(outcomes.contains(&&smile) && outcomes.contains(&&ok), "{outcomes:?}");
    assert_eq!(model.outcome_texts(), Err(&smile));
}

#[test]
fn the_rust_chunker_fixture_is_the_bin_written_here() {
    let text = tagged("chunks.txt");
    let samples: Vec<_> = ChunkReader::new(&text).unwrap().map(|s| s.unwrap()).collect();
    assert_eq!(samples.len(), CHUNK_SAMPLES);
    let factory = ChunkerFactory::new("en");
    let ours = train_chunker_model(
        &samples,
        &factory,
        &tagger_parameters(100),
        &mut TrainingConfiguration::default(),
        Stamp::at(RUST_FIXTURE_TIMESTAMP),
    )
    .unwrap();
    let bin = ours.package.to_bytes();
    check_rust_fixture("chunker-fixture-rust.bin", &bin);
    let file = ModelFile::open(bin).unwrap();
    assert_eq!(entry_names(&file), ["chunker.model", "manifest.properties"]);
    assert_eq!(file.manifest().get("Component-Name"), Some("ChunkerME"));
    assert_eq!(file.manifest().get("factory"), Some("opennlp.tools.chunker.ChunkerFactory"));
    assert_eq!(file.manifest().get("BeamSize"), Some("10"));
    // turbo-phrase-chunker opens it.
    let model = turbo_phrase_chunker::ChunkerModel::open(file).unwrap();
    let outcomes: Vec<&str> = model.maxent().outcome_texts().unwrap();
    for tag in ["O", "B-NP", "I-NP", "B-VP", "I-VP", "B-PP", "B-ADJP", "B-ADVP", "B-SBAR", "B-PRT"] {
        assert!(outcomes.contains(&tag), "{tag} is not an outcome of {outcomes:?}");
    }
}

/// The samples of `inputs/tagged/chunks.txt`.
const CHUNK_SAMPLES: usize = 30;

#[test]
fn the_rust_language_fixture_is_the_bin_written_here() {
    // The parameters langdetect-183 was most likely trained with: those
    // of the command line trainer when it is given none.
    let text = read_text(&data_dir().join("inputs/langdetect/samples.txt"));
    let samples: Vec<_> = LanguageReader::new(&text).unwrap().map(|s| s.unwrap()).collect();
    assert_eq!(samples.len(), 56);
    let ours = train_language_model(
        &samples,
        &LanguageDetectorFactory::new(),
        &langdetect::default_training_parameters(),
        &mut TrainingConfiguration::default(),
        Stamp::at(RUST_FIXTURE_TIMESTAMP),
    )
    .unwrap();
    assert_eq!(ours.inexact_events, 0);
    let bin = ours.package.to_bytes();
    check_rust_fixture("langdetect-fixture-rust.bin", &bin);
    let file = ModelFile::open(bin).unwrap();
    assert_eq!(entry_names(&file), ["manifest.properties", "langdetect.model"]);
    assert_eq!(file.manifest().get("Component-Name"), Some("LanguageDetectorME"));
    assert_eq!(file.manifest().get("Language"), Some("und"));
    assert_eq!(file.manifest().get("factory"), Some("opennlp.tools.langdetect.LanguageDetectorFactory"));
    assert_eq!(file.manifest().get("TrainerType"), Some("Event"));
    let model = file.maxent(langdetect::MODEL_ENTRY).unwrap();
    let outcomes: Vec<&str> = model.outcome_texts().unwrap();
    for code in ["eng", "deu", "fra", "spa", "ita", "nld", "rus", "ell", "tur"] {
        assert!(outcomes.contains(&code), "{code} is not an outcome of {outcomes:?}");
    }
    // turbo-langdetect opens it.
    let detector = turbo_langdetect::LanguageModel::open(file).unwrap();
    assert_eq!(detector.supported_languages().count(), 9);
}
