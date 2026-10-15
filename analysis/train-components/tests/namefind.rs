//! The name finder's training against the JVM: the goldens of
//! `analysis/goldens/data/namefind/` (`NameEventGolden` and
//! `NameFindGolden` of the generator).
//!
//! - The `train-` runs: `NameFinderEventStream`'s events over the name
//!   inputs with each codec, a type given to every name and the other
//!   event trainers, what the streams threw, and the model
//!   `NameFinderME.train` trained from the same input with its default
//!   parameters put in. This crate's events are the JVM's exactly; its
//!   model has the JVM's manifest, entries and parameters, every weight
//!   within 1e-9 of the JVM's (they are equal), and its `nameFinder.model`
//!   entry is the JVM's byte for byte.
//! - The `ner-fixture-rust` run: the JVM's `NameFinderME` over the model
//!   this crate wrote from `inputs/tagged/names.txt`
//!   (`ner-fixture-rust.bin`), which shows the file loads in the JVM;
//!   `turbo-sequence`'s name finder over the same file finds the same
//!   names with the same probabilities.

mod common;

use std::collections::HashMap;
use std::path::Path;

use turbo_formats::native::NameReader;
use turbo_formats::{FormatError, NameSample};
use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, Shape, data_dir};
use turbo_maxent::{JavaString, ModelFile, Properties};
use turbo_sequence::PatternMode;
use turbo_sequence::namefind::{NameCodec, NameFinder};
use turbo_train::event::checksum;
use turbo_train::package::MANIFEST_ENTRY;
use turbo_train::{Event, Key, TrainingConfiguration, TrainingParameters};
use turbo_train_components::namefind::{self, NameEventStream, TokenNameFinderFactory, train_name_model};
use turbo_train_components::{Error, Stamp};
use turbo_unicode::WhitespaceMode;

const fn col(name: &'static str, ty: ColumnType) -> ColumnSpec {
    ColumnSpec::new(name, ty)
}

const STR: ColumnType = ColumnType::Str;
const U32: ColumnType = ColumnType::U32;
const F64: ColumnType = ColumnType::F64;

const LAYERS: &[(&str, &[ColumnSpec])] = &[
    ("turbo:events", &[col("outcome", STR), col("context", STR), col("values", STR)]),
    ("turbo:stream-errors", &[col("after", U32), col("class", STR), col("message", STR)]),
    ("turbo:event-checksum", &[col("crc32c", U32)]),
    ("turbo:name-training", &[col("class", STR), col("message", STR)]),
    ("turbo:name-parameters", &[col("key", STR), col("class", STR), col("value", STR)]),
    ("turbo:manifest", &[col("key", STR), col("value", STR)]),
    ("turbo:package-entries", &[col("name", STR), col("length", U32)]),
    ("turbo:name-model", &[col("sha256", STR), col("length", U32)]),
    ("turbo:name-weights", &[col("predicate", STR), col("outcome", STR), col("value", F64)]),
    (
        "turbo:name-found",
        &[col("sentence", U32), col("start", U32), col("end", U32), col("type", STR), col("prob", F64)],
    ),
    ("turbo:name-probs", &[col("sentence", U32), col("token", U32), col("prob", F64)]),
];

fn read(path: &Path) -> Golden {
    let shapes: Vec<Shape> = LAYERS
        .iter()
        .map(|(id, columns)| Shape { id: id.to_string(), anchor: GoldenAnchor::Document, columns })
        .collect();
    Golden::read_with(path, &shapes).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// What the JVM threw: the class and the message.
type Thrown = (String, String);

/// The class and message the JVM throws where this crate returns `e`.
fn thrown(e: &Error) -> Thrown {
    let class = match e {
        Error::Format(f) => return format_thrown(f),
        Error::UnencodableName { .. } => "java.lang.ArrayIndexOutOfBoundsException",
        Error::IncompatibleOutcomes => "java.lang.IllegalArgumentException",
        other => panic!("no Java exception for {other}"),
    };
    (class.to_owned(), e.to_string())
}

fn format_thrown(e: &FormatError) -> Thrown {
    (e.java_class().unwrap_or_else(|| panic!("{e} has no Java class")).to_owned(), e.to_string())
}

/// The samples of the input and the errors the reader gave in their
/// place, by the number of samples read before each.
fn samples(text: &str) -> (Vec<NameSample<'_>>, Vec<(usize, FormatError)>) {
    let (mut samples, mut errors) = (Vec::new(), Vec::new());
    for item in NameReader::new(text, WhitespaceMode::Unicode).unwrap() {
        match item {
            Ok(s) => samples.push(s),
            Err(e) => errors.push((samples.len(), e)),
        }
    }
    (samples, errors)
}

fn codec(g: &Golden) -> NameCodec {
    match common::config(g, "codec").unwrap() {
        "bio" => NameCodec::Bio,
        "bilou" => NameCodec::Bilou,
        other => panic!("{other}"),
    }
}

/// The events of the samples in the order the JVM reads them, with every
/// exception the reader or the event stream gave and the number of events
/// read before it; the first exception is the one that stops training.
fn events(g: &Golden, text: &str) -> (Vec<Event>, Vec<(u32, Thrown)>) {
    let factory = TokenNameFinderFactory::new("en", None, codec(g)).unwrap();
    let mut stream = NameEventStream::new(&factory, common::config(g, "type")).unwrap();
    let (samples, reader_errors) = samples(text);
    let mut reader_errors = reader_errors.into_iter().peekable();
    let (mut out, mut errors) = (Vec::new(), Vec::new());
    let after = |out: &Vec<Event>| u32::try_from(out.len()).unwrap();
    for (i, s) in samples.iter().enumerate() {
        while let Some((_, e)) = reader_errors.next_if(|(at, _)| *at == i) {
            errors.push((after(&out), format_thrown(&e)));
        }
        if let Err(e) = stream.events(s, &mut out) {
            errors.push((after(&out), thrown(&e)));
        }
    }
    for (_, e) in reader_errors {
        errors.push((after(&out), format_thrown(&e)));
    }
    (out, errors)
}

fn stream_errors(g: &Golden) -> Vec<(u32, Thrown)> {
    let l = common::layer(g, "turbo:stream-errors");
    let (after, class, message) = (l.u32s("after").unwrap(), l.strs("class").unwrap(), l.strs("message").unwrap());
    (0..l.len()).map(|i| (after[i], (class.get(i).to_owned(), message.get(i).to_owned()))).collect()
}

fn manifest_entries(package: &turbo_train::ModelPackage) -> Vec<(String, String)> {
    let text = package.manifest().to_properties();
    let properties = Properties::load(text.as_bytes()).unwrap();
    let mut entries: Vec<(String, String)> =
        properties.iter().filter(|(k, _)| *k != "Timestamp").map(|(k, v)| (k.to_owned(), v.to_owned())).collect();
    entries.sort();
    entries
}

/// The weights of a model file's `nameFinder.model` entry by predicate
/// and outcome label, as the JVM's `GenericModelReader` reads them back:
/// the perceptron's writer leaves out its zero weights.
fn weights(file: &ModelFile) -> HashMap<(JavaString, JavaString), f64> {
    let model = file.maxent(namefind::MODEL_ENTRY).unwrap();
    let mut out = HashMap::new();
    for name in model.predicates() {
        let p = model.predicate_index(name).unwrap();
        let label = JavaString::from_modified_utf8(name).unwrap();
        for (o, v) in model.pattern(p).iter().zip(model.parameters(p)) {
            out.insert((label.clone(), model.outcome(*o as usize).clone()), *v);
        }
    }
    out
}

/// What `NameFinderME.train` made of the input, against what the golden
/// records; the number of weights compared.
fn check_training(path: &Path, g: &Golden, text: &str, first_error: Option<&Thrown>) -> usize {
    let l = common::layer(g, "turbo:name-training");
    let expected: Thrown = (l.strs("class").unwrap().get(0).to_owned(), l.strs("message").unwrap().get(0).to_owned());
    let mut params = TrainingParameters::new();
    if let Some(a) = common::config(g, "algorithm") {
        params.put_str(Key::ALGORITHM, a);
    }
    // The parameters after training: the defaults put in, and `sort`,
    // which the trainer puts into the caller's parameters before it reads
    // the events (false for the perceptron, which does not sort), where
    // this crate's trainers put it into a copy.
    let mut used = namefind::training_parameters(&params);
    used.put_bool(Key::SORT, used.algorithm().unwrap() != Some("PERCEPTRON"));
    let l = common::layer(g, "turbo:name-parameters");
    let (keys, classes, values) = (l.strs("key").unwrap(), l.strs("class").unwrap(), l.strs("value").unwrap());
    let mut theirs: Vec<(String, String, String)> =
        (0..l.len()).map(|i| (keys.get(i).to_owned(), classes.get(i).to_owned(), values.get(i).to_owned())).collect();
    theirs.sort();
    let mut ours: Vec<(String, String, String)> =
        used.iter().map(|(k, v)| (k.to_owned(), format!("java.lang.{}", v.java_class()), v.java_string())).collect();
    ours.sort();
    assert_eq!(theirs, ours, "{}: parameters", path.display());

    if let Some(e) = first_error {
        assert_eq!(&expected, e, "{}: the first exception stops training", path.display());
        return 0;
    }
    let (samples, _) = samples(text);
    let factory = TokenNameFinderFactory::new("en", None, codec(g)).unwrap();
    let trained = train_name_model(
        &samples,
        &factory,
        common::config(g, "type"),
        &params,
        &mut TrainingConfiguration::default(),
        Stamp::at(1_700_000_000_000),
    );
    let trained = match trained {
        Err(e) => {
            assert_eq!(expected, thrown(&e), "{}: training", path.display());
            return 0;
        }
        Ok(t) => t,
    };
    assert_eq!(expected, (String::new(), String::new()), "{}: the JVM trained no model", path.display());

    let l = common::layer(g, "turbo:manifest");
    let (keys, values) = (l.strs("key").unwrap(), l.strs("value").unwrap());
    let theirs: Vec<(String, String)> =
        (0..l.len()).map(|i| (keys.get(i).to_owned(), common::lossless_text(values.get(i)))).collect();
    assert_eq!(theirs, manifest_entries(&trained.package), "{}: manifest", path.display());

    let file = ModelFile::open(trained.package.to_bytes()).unwrap();
    let l = common::layer(g, "turbo:package-entries");
    let (names, lengths) = (l.strs("name").unwrap(), l.u32s("length").unwrap());
    let theirs: Vec<&str> = (0..l.len()).map(|i| names.get(i)).collect();
    assert_eq!(theirs, trained.package.entry_names(), "{}: entry order", path.display());
    for (i, name) in theirs.iter().enumerate() {
        // Both manifests' timestamps have thirteen digits, so the lengths
        // agree although the instants do not.
        assert_eq!(lengths[i] as usize, file.entry(name).unwrap().len(), "{}: length of {name}", path.display());
    }
    let entry = file.entry(namefind::MODEL_ENTRY).unwrap();
    let l = common::layer(g, "turbo:name-model");
    assert_eq!(l.strs("sha256").unwrap().get(0), sha256_hex(&entry), "{}: nameFinder.model", path.display());
    assert_eq!(&*file.entry(MANIFEST_ENTRY).unwrap(), trained.package.manifest().to_properties().as_bytes());

    let ours = weights(&file);
    let l = common::layer(g, "turbo:name-weights");
    let (predicates, outcomes, values) =
        (l.strs("predicate").unwrap(), l.strs("outcome").unwrap(), l.f64s("value").unwrap());
    assert_eq!(l.len(), ours.len(), "{}: number of weights", path.display());
    for (i, &value) in values.iter().enumerate() {
        let key = (common::java_string(predicates.get(i)), common::java_string(outcomes.get(i)));
        let ours = *ours.get(&key).unwrap_or_else(|| panic!("{}: no weight {key:?}", path.display()));
        assert!((ours - value).abs() <= 1e-9, "{}: {key:?} {ours} against {value}", path.display());
        assert_eq!(ours.to_bits(), value.to_bits(), "{}: {key:?}", path.display());
    }
    l.len()
}

fn sha256_hex(bytes: &[u8]) -> String {
    turbo_maxent::sha256::sha256(bytes).to_hex()
}

#[test]
fn name_events_and_trained_models_are_the_jvms_over_every_training_golden() {
    let dir = data_dir().join("namefind");
    let mut runs = std::collections::BTreeSet::new();
    let (mut files, mut trained, mut weights, mut with_errors) = (0, 0, 0, 0);
    for run in std::fs::read_dir(&dir).unwrap() {
        let run = run.unwrap().path();
        if !run.file_name().unwrap().to_str().unwrap().starts_with("train-") {
            continue;
        }
        for path in common::golden_files(&run) {
            let g = read(&path);
            assert!(g.header.error.is_none(), "{}: the generator failed", path.display());
            runs.insert(common::config(&g, "run").unwrap().to_owned());
            let text = g.input_text(data_dir()).unwrap();
            let (events, errors) = events(&g, &text);
            let expected = common::events(&g);
            assert_eq!(expected, events, "{}: {}", path.display(), common::describe_difference(&expected, &events));
            assert_eq!(common::checksum(&g), checksum(&events), "{}: checksum", path.display());
            assert_eq!(stream_errors(&g), errors, "{}: stream errors", path.display());
            with_errors += usize::from(!errors.is_empty());
            let n = check_training(&path, &g, &text, errors.first().map(|(_, e)| e));
            trained += usize::from(n > 0);
            weights += n;
            files += 1;
        }
    }
    let expected: std::collections::BTreeSet<String> =
        ["train-bio", "train-bilou", "train-type", "train-perceptron", "train-qn"]
            .into_iter()
            .map(String::from)
            .collect();
    assert_eq!(runs, expected);
    // Each run over tagged/names and the four name inputs of the formats set.
    assert_eq!(files, 25);
    // Every run trains on tagged/names, name-basic and name-crlf; on
    // name-edge an empty name throws in the codec, and on name-errors the
    // reader throws, which stops training.
    assert_eq!((trained, with_errors), (15, 10));
    assert!(weights >= 40_000, "{weights} weights");
}

#[test]
fn the_jvm_finds_with_the_rust_fixture_what_turbo_sequence_finds() {
    let fixture = ModelFile::read(data_dir().join("fixtures/ner-fixture-rust.bin")).unwrap();
    let model = fixture.maxent(namefind::MODEL_ENTRY).unwrap();
    let paths = common::golden_files(&data_dir().join("namefind/ner-fixture-rust"));
    assert_eq!(paths.len(), 5);
    let mut found_names = 0;
    for path in paths {
        let g = read(&path);
        assert!(g.header.error.is_none(), "{}: the generator failed", path.display());
        let text = g.input_text(data_dir()).unwrap();
        let (samples, _) = samples(&text);
        let mut finder = NameFinder::new(&fixture, &model, PatternMode::Utf16Units).unwrap();
        let (mut names, mut probs) = (Vec::new(), Vec::new());
        for (k, s) in samples.iter().enumerate() {
            if s.clear_adaptive_data() {
                finder.clear_adaptive_data();
            }
            let tokens: Vec<&str> = s.tokens().iter().map(|t| t.as_str()).collect();
            for f in finder.find(&tokens) {
                let kind = f.span.kind.clone().unwrap_or_default();
                names.push((k as u32, f.span.start as u32, f.span.end as u32, kind, f.prob.to_bits()));
            }
            for (t, p) in finder.probs().iter().enumerate() {
                probs.push((k as u32, t as u32, p.to_bits()));
            }
        }
        let l = common::layer(&g, "turbo:name-found");
        let (sentence, start, end, kind, prob) = (
            l.u32s("sentence").unwrap(),
            l.u32s("start").unwrap(),
            l.u32s("end").unwrap(),
            l.strs("type").unwrap(),
            l.f64s("prob").unwrap(),
        );
        let theirs: Vec<(u32, u32, u32, String, u64)> =
            (0..l.len()).map(|i| (sentence[i], start[i], end[i], kind.get(i).to_owned(), prob[i].to_bits())).collect();
        assert_eq!(theirs, names, "{}: names", path.display());
        let l = common::layer(&g, "turbo:name-probs");
        let (sentence, token, prob) = (l.u32s("sentence").unwrap(), l.u32s("token").unwrap(), l.f64s("prob").unwrap());
        let theirs: Vec<(u32, u32, u64)> = (0..l.len()).map(|i| (sentence[i], token[i], prob[i].to_bits())).collect();
        assert_eq!(theirs, probs, "{}: probabilities", path.display());
        found_names += names.len();
    }
    assert!(found_names >= 40, "{found_names} names found");
}
