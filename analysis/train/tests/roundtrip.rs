//! Train, write, load, score: the models the JVM trained for the fixtures
//! are trained again here from the same events and parameters, and the
//! bytes written must be the fixtures' bytes.
//!
//! - `analysis/maxent/tests/fixtures/models/gis.model` and `qn.model`: nine
//!   hand-written events, `MAXENT` and `MAXENT_QN`, 100 iterations, cutoff 0
//!   (`analysis/maxent/tests/fixtures/src/MakeModelFixtures.java`).
//! - `analysis/goldens/data/fixtures/en-plain.bin`: the sentence detector
//!   model `TrainSentenceModel.java` trained with cutoff 1 and 100
//!   iterations from `analysis/sentdetect/tests/fixtures/sentences.txt`,
//!   whose events the JVM dumped in the training golden
//!   `sentdetect-twopass-cutoff1/train/sentences`. The model trained here is
//!   packaged as a `.bin` with the fixture's manifest entries and instant,
//!   opened with `turbo-maxent` and `turbo-sentdetect`, and must split the
//!   training text exactly as the fixture does, probabilities included.
//!   It is also the checked in fixture
//!   `analysis/goldens/data/fixtures/en-plain-rust.bin`, byte for byte,
//!   which the golden generator loads in the JVM with `SentenceModel` and
//!   runs beside `en-plain.bin` (`analysis/goldens/data/sentdetect/en-plain-rust`,
//!   checked by `analysis/sentdetect/tests/golden.rs`).
//!
//! - The same sentence events trained with `MAXENT_QN` (the defaults, cutoff
//!   1), packaged the same way. The `sent.model` bytes are those the JVM's
//!   `QNTrainer` wrote for the same events and parameters (the training
//!   golden `sentdetect-twopass-cutoff1-qn/train/sentences` records their
//!   SHA-256), and the `.bin` is the checked in fixture
//!   `analysis/goldens/data/fixtures/en-plain-qn-rust.bin`, which the golden
//!   generator loads in the JVM with `SentenceModel` and runs on the sentence
//!   fixture inputs (`analysis/goldens/data/sentdetect/en-plain-qn-rust`,
//!   checked by `analysis/sentdetect/tests/golden.rs`). It splits the
//!   training text into 210 sentences, as `SentenceDetectorME` does with it.
//!
//! With `TURBO_TRAIN_OUT` set to a directory, the `.bin` files written here
//! are saved there as `en-plain-rust.bin` and `en-plain-qn-rust.bin`, which
//! is how the fixtures were made.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo_goldens::{Golden, data_dir};
use turbo_maxent::{MaxentModel, ModelFile};
use turbo_sentdetect::{SentenceDetector, SentenceModel, WhitespaceMode};
use turbo_train::gis;
use turbo_train::package::{MANIFEST_ENTRY, SENTENCE_MODEL_ENTRY, sentence_model_package};
use turbo_train::trainer::train_events;
use turbo_train::{Algorithm, Event, JavaString, Key, TrainingConfiguration, TrainingParameters, write_gis};

mod common;

fn maxent_fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../maxent/tests/fixtures/models").join(name)
}

/// `MakeModelFixtures.events()`.
fn fixture_events() -> Vec<Event> {
    let event = |outcome: &str, context: &[&str]| Event::from_strs(outcome, context);
    let (nul, accent, emoji) = ("w=\u{0}", "w=été", "w=😀");
    vec![
        event("yes", &["w=a", "len=1", "cap"]),
        event("yes", &["w=a", "len=2"]),
        event("yes", &["w=b", "cap", accent]),
        event("no", &["w=c", "len=1"]),
        event("no", &["w=c", "len=3", nul]),
        event("no", &["w=d", "len=3"]),
        event("maybe", &["w=b", "len=2", emoji]),
        event("maybe", &["w=d", "cap", emoji]),
        event("maybe", &["w=a", "len=3", nul, accent]),
    ]
}

/// Trains the fixture events as `MakeModelFixtures` does with `algorithm`
/// and checks the bytes are the fixture `file`'s, and that the model read
/// back gives the probabilities `expected.txt` lists for it.
fn check_fixture_model(algorithm: Algorithm, file: &str) {
    let mut params = TrainingParameters::new();
    params.put(Key::ALGORITHM, algorithm.name());
    params.put(Key::ITERATIONS, 100);
    params.put(Key::CUTOFF, 0);
    let (model, _) = train_events(&fixture_events(), &params, &mut TrainingConfiguration::default()).unwrap();
    assert_eq!(model.algorithm(), algorithm);
    let bytes = model.write().unwrap();
    let fixture = std::fs::read(maxent_fixture(file)).unwrap();
    assert_eq!(bytes, fixture, "{file}");
    // The fixture's expected probabilities hold for the model read back.
    let read = MaxentModel::read(&bytes).unwrap();
    let expected = std::fs::read_to_string(maxent_fixture("expected.txt")).unwrap();
    let mut checked = 0;
    for line in expected.lines().filter(|l| l.starts_with(&format!("{file}\t"))) {
        let mut fields = line.split('\t');
        fields.next();
        let context: Vec<String> = fields.next().unwrap().split(' ').filter(|f| !f.is_empty()).map(unescape).collect();
        let probabilities: Vec<f64> = fields.next().unwrap().split(' ').map(|p| p.parse().unwrap()).collect();
        let mut out = vec![0.0; read.num_outcomes()];
        let context: Vec<&str> = context.iter().map(String::as_str).collect();
        read.eval(&context, &mut out);
        assert_eq!(out, probabilities, "{line}");
        checked += 1;
    }
    assert!(checked > 5);
}

#[test]
fn the_gis_fixture_model_is_trained_byte_for_byte() {
    check_fixture_model(Algorithm::Maxent, "gis.model");
}

#[test]
fn the_qn_fixture_model_is_trained_byte_for_byte() {
    check_fixture_model(Algorithm::MaxentQn, "qn.model");
}

/// `expected.txt` escapes a feature's non-ASCII characters as `\uXXXX`.
fn unescape(field: &str) -> String {
    let mut units = Vec::new();
    let mut rest = field;
    while let Some(i) = rest.find("\\u") {
        units.extend(rest[..i].encode_utf16());
        units.push(u16::from_str_radix(&rest[i + 2..i + 6], 16).unwrap());
        rest = &rest[i + 6..];
    }
    units.extend(rest.encode_utf16());
    String::from_utf16(&units).unwrap()
}

#[test]
fn the_sentence_fixture_model_is_trained_written_loaded_and_scores_the_same() {
    let path = data_dir().join("train/sentdetect-twopass-cutoff1/train/sentences.golden");
    let g = Golden::read_with(&path, &common::train_shapes()).unwrap_or_else(|e| panic!("{e}"));
    let events = common::events(&g);
    let text = g.input_text(data_dir()).unwrap();

    // TrainSentenceModel.java: cutoff 1, 100 iterations, the default
    // algorithm and indexer.
    let mut params = TrainingParameters::new();
    params.put(Key::ITERATIONS, 100);
    params.put(Key::CUTOFF, 1);
    let (model, report) = gis::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
    let bytes = write_gis(&model).unwrap();

    let fixture_path = data_dir().join("fixtures/en-plain.bin");
    let fixture = ModelFile::read(&fixture_path).unwrap();
    assert_eq!(&*fixture.entry(SENTENCE_MODEL_ENTRY).unwrap(), &bytes[..], "the model entry");
    assert_eq!(fixture.manifest().get("Training-Eventhash"), Some(report.event_hash.unwrap().to_string().as_str()));

    // The .bin with the fixture's instant and factory settings has the
    // fixture's manifest, byte for byte.
    let timestamp: i64 = fixture.manifest().get("Timestamp").unwrap().parse().unwrap();
    let package = sentence_model_package("en", timestamp, &report, false, Some(".!?"), bytes);
    let bin = package.to_bytes();
    if let Ok(dir) = std::env::var("TURBO_TRAIN_OUT") {
        std::fs::write(Path::new(&dir).join("en-plain-rust.bin"), &bin).unwrap();
    }
    let committed = std::fs::read(data_dir().join("fixtures/en-plain-rust.bin")).unwrap();
    assert_eq!(bin, committed, "analysis/goldens/data/fixtures/en-plain-rust.bin is not the .bin written here");
    let ours = ModelFile::open(bin).unwrap();
    assert_eq!(&*ours.entry(MANIFEST_ENTRY).unwrap(), &*fixture.entry(MANIFEST_ENTRY).unwrap());
    assert_eq!(ours.manifest().iter().collect::<Vec<_>>(), fixture.manifest().iter().collect::<Vec<_>>());

    // Both load as sentence models and split the training text alike.
    let theirs = Arc::new(SentenceModel::open(fixture).unwrap());
    let ours = Arc::new(SentenceModel::open(ours).unwrap());
    for mode in [WhitespaceMode::Unicode, WhitespaceMode::Legacy] {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        SentenceDetector::new(theirs.clone(), mode).detect(&text, &mut a);
        SentenceDetector::new(ours.clone(), mode).detect(&text, &mut b);
        assert!(a.len() > 100, "{mode:?}: {} sentences", a.len());
        assert_eq!(a, b, "{mode:?}");
    }

    // The predicates of the model trained here are the fixture's.
    let fixture_maxent = ModelFile::read(&fixture_path).unwrap().maxent(SENTENCE_MODEL_ENTRY).unwrap();
    let mut labels: Vec<JavaString> = model.pred_labels().to_vec();
    labels.sort();
    let mut theirs: Vec<JavaString> =
        fixture_maxent.predicates().map(|p| JavaString::from_modified_utf8(p).unwrap()).collect();
    theirs.sort();
    assert_eq!(labels, theirs);
}

/// The SHA-256 and length a training golden recorded for the written model.
fn golden_model(dir: &str, layer: &str) -> (String, usize) {
    let path = data_dir().join(format!("train/{dir}/train/sentences.golden"));
    let g = Golden::read_with(&path, &common::train_shapes()).unwrap_or_else(|e| panic!("{e}"));
    let written = common::layer(&g, layer);
    (written.strs("sha256").unwrap().get(0).to_string(), written.u32s("length").unwrap()[0] as usize)
}

#[test]
fn the_qn_sentence_model_is_the_jvms_and_its_bin_is_the_fixture() {
    let path = data_dir().join("train/sentdetect-twopass-cutoff1/train/sentences.golden");
    let g = Golden::read_with(&path, &common::train_shapes()).unwrap_or_else(|e| panic!("{e}"));
    let events = common::events(&g);
    let text = g.input_text(data_dir()).unwrap();

    let mut params = TrainingParameters::new();
    params.put(Key::ALGORITHM, Algorithm::MaxentQn.name());
    params.put(Key::ITERATIONS, 100);
    params.put(Key::CUTOFF, 1);
    let (model, report) = train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
    assert_eq!(model.algorithm(), Algorithm::MaxentQn);
    let bytes = model.write().unwrap();
    assert_eq!(&bytes[..4], b"\0\x02QN");
    // The bytes QNTrainer and GenericModelWriter produced on the JVM for
    // the same events and parameters.
    let (sha, length) = golden_model("sentdetect-twopass-cutoff1-qn", "turbo:qn-model");
    assert_eq!((turbo_maxent::sha256(&bytes).to_hex(), bytes.len()), (sha, length), "the JVM's model bytes");

    // The fixture's instant, so the manifest differs from en-plain.bin's
    // only where the model does.
    let fixture = ModelFile::read(data_dir().join("fixtures/en-plain.bin")).unwrap();
    assert_eq!(fixture.manifest().get("Training-Eventhash"), Some(report.event_hash.unwrap().to_string().as_str()));
    let timestamp: i64 = fixture.manifest().get("Timestamp").unwrap().parse().unwrap();
    let bin = sentence_model_package("en", timestamp, &report, false, Some(".!?"), bytes).to_bytes();
    if let Ok(dir) = std::env::var("TURBO_TRAIN_OUT") {
        std::fs::write(Path::new(&dir).join("en-plain-qn-rust.bin"), &bin).unwrap();
    }
    let committed = std::fs::read(data_dir().join("fixtures/en-plain-qn-rust.bin")).unwrap();
    assert_eq!(bin, committed, "analysis/goldens/data/fixtures/en-plain-qn-rust.bin is not the .bin written here");
    let ours = ModelFile::open(bin).unwrap();
    assert_eq!(ours.manifest().iter().collect::<Vec<_>>(), fixture.manifest().iter().collect::<Vec<_>>());
    let maxent = ours.maxent(SENTENCE_MODEL_ENTRY).unwrap();
    assert_eq!(maxent.num_predicates(), model.as_writable().num_predicates());

    // It splits the training text into 210 sentences, as SentenceDetectorME
    // does with the same file.
    let ours = Arc::new(SentenceModel::open(ours).unwrap());
    let mut spans = Vec::new();
    SentenceDetector::new(ours, WhitespaceMode::Unicode).detect(&text, &mut spans);
    assert_eq!(spans.len(), 210);
}
