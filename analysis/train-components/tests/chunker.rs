//! `ChunkerME.train` against the JVM: the `chunker-` runs of
//! `analysis/goldens/data/train-events/` (`ChunkerTrainGolden`), each a
//! set of training parameters over the chunk inputs of the `formats` and
//! `tagged` sets, read as `ChunkSampleStream` reads them. For each input
//! the samples' events, what the stream threw and the event checksum are
//! compared; then what `ChunkerME.train` threw, or the model it trained:
//! the algorithm, the outcomes in order, the predicates in the order the
//! model writes them, every parameter (the JVM's double, bit for bit; the
//! issue asks for 1e-9), the `chunker.model` entry's length and SHA-256,
//! the manifest without its timestamp and the zip entries.

mod common;

use std::path::Path;

use turbo_formats::native::ChunkReader;
use turbo_formats::{ChunkSample, FormatError};
use turbo_goldens::{Golden, data_dir};
use turbo_maxent::Properties;
use turbo_train::event::checksum;
use turbo_train::package::MANIFEST_ENTRY;
use turbo_train::params::Key;
use turbo_train::{Event, TrainingConfiguration, TrainingParameters};
use turbo_train_components::chunk::{self, ChunkerFactory, train_chunker_model};
use turbo_train_components::{Error, Stamp, postag};
use turbo_zoo::Sha256;

/// When the models here are said to be trained, in milliseconds: thirteen
/// digits, as the JVM's timestamp has, so the manifests have one length.
const TIMESTAMP_MS: i64 = 1_700_000_000_000;

/// The runs of `ChunkerTrainGolden.all`, each with its parameters as the
/// generator puts them: integers as integers, the rest as strings.
fn parameters(run: &str) -> TrainingParameters {
    let mut p = TrainingParameters::new();
    match run {
        "chunker-defaults" => return TrainingParameters::defaults(),
        "chunker-perceptron" => {
            p.put(Key::ALGORITHM, "PERCEPTRON");
            p.put(Key::ITERATIONS, 100);
            p.put(Key::CUTOFF, 1);
            p.put(postag::BEAM_SIZE_KEY, 5);
        }
        "chunker-qn" => {
            p.put(Key::ALGORITHM, "MAXENT_QN");
            p.put(Key::CUTOFF, 1);
        }
        "chunker-perceptron-sequence" => {
            p.put(Key::ALGORITHM, "PERCEPTRON_SEQUENCE");
            p.put(Key::ITERATIONS, 10);
            p.put(Key::CUTOFF, 1);
        }
        other => panic!("no chunker run {other}"),
    }
    p
}

/// What the sample stream threw, after how many events.
#[derive(Debug, PartialEq, Eq)]
struct StreamError {
    after: u32,
    class: String,
    message: String,
}

/// The samples of the input, the events of each in turn, and what the
/// stream threw between them; the generator reads on after a throw.
fn samples_and_events(text: &str) -> (Vec<ChunkSample<'_>>, Vec<Event>, Vec<StreamError>) {
    let factory = ChunkerFactory::new("en");
    let mut stream = chunk::ChunkerEventStream::new(&factory);
    let (mut samples, mut events, mut errors) = (Vec::new(), Vec::new(), Vec::new());
    for read in ChunkReader::new(text).unwrap() {
        match read {
            Ok(sample) => {
                stream.events(&sample, &mut events);
                samples.push(sample);
            }
            Err(e) => errors.push(stream_error(events.len(), &e)),
        }
    }
    (samples, events, errors)
}

fn stream_error(after: usize, e: &FormatError) -> StreamError {
    let class = e.java_class().unwrap_or_else(|| panic!("{e} has no Java class")).to_owned();
    StreamError { after: u32::try_from(after).unwrap(), class, message: e.to_string() }
}

fn stream_errors(g: &Golden) -> Vec<StreamError> {
    let l = common::layer(g, "turbo:stream-errors");
    let (after, class, message) = (l.u32s("after").unwrap(), l.strs("class").unwrap(), l.strs("message").unwrap());
    (0..l.len())
        .map(|i| StreamError {
            after: after[i],
            class: class.get(i).to_owned(),
            message: common::lossless_text(message.get(i)),
        })
        .collect()
}

/// The class and the message of the exception `ChunkerME.train` throws
/// where the port fails.
fn java_exception(e: &Error) -> (&'static str, String) {
    match e {
        Error::UnsupportedTrainerType(_) => ("java.lang.IllegalArgumentException", e.to_string()),
        Error::Train(inner @ turbo_train::Error::InsufficientTrainingData(_)) => {
            ("opennlp.tools.util.InsufficientTrainingDataException", inner.to_string())
        }
        other => panic!("{other} is not an error ChunkerME.train throws"),
    }
}

/// The manifest entries of a package, sorted by key, without `Timestamp`.
fn manifest_entries(package: &turbo_train::ModelPackage) -> Vec<(String, String)> {
    let text = package.manifest().to_properties();
    let properties = Properties::load(text.as_bytes()).unwrap();
    let mut entries: Vec<(String, String)> =
        properties.iter().filter(|(k, _)| *k != "Timestamp").map(|(k, v)| (k.to_owned(), v.to_owned())).collect();
    entries.sort();
    entries
}

fn labels(g: &Golden, id: &str) -> Vec<String> {
    let l = common::layer(g, id);
    let labels = l.strs("label").unwrap();
    (0..l.len()).map(|i| common::lossless_text(labels.get(i))).collect()
}

/// What one golden held: a trained model, or what training threw.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    /// The parameters compared, each the JVM's double.
    Trained {
        parameters: usize,
    },
    Refused,
}

fn check(path: &Path, g: &Golden) -> Outcome {
    let run = common::config(g, "run").unwrap();
    assert_eq!(common::config(g, "source"), Some("chunk"), "{}", path.display());
    let text = g.input_text(data_dir()).unwrap();
    let (samples, events, errors) = samples_and_events(&text);
    let expected = common::events(g);
    assert_eq!(expected, events, "{}: {}", path.display(), common::describe_difference(&expected, &events));
    assert_eq!(common::checksum(g), checksum(&events), "{}: checksum", path.display());
    assert_eq!(stream_errors(g), errors, "{}: stream errors", path.display());

    // Every training parameter the run put, as the generator records it.
    let params = parameters(run);
    let mut recorded: Vec<(&str, String)> = g
        .header
        .config
        .iter()
        .filter(|c| !["run", "source", "language"].contains(&c.key.as_str()))
        .map(|c| (c.key.as_str(), c.value.clone()))
        .collect();
    let mut put: Vec<(&str, String)> = params.iter().map(|(k, v)| (k, v.to_string())).collect();
    recorded.sort();
    put.sort();
    assert_eq!(recorded, put, "{}: training parameters", path.display());
    let mut config = TrainingConfiguration::default();
    let trained =
        train_chunker_model(&samples, &ChunkerFactory::new("en"), &params, &mut config, Stamp::at(TIMESTAMP_MS));
    if let Some(l) = common::layer_opt(g, "turbo:train-error") {
        let err =
            trained.err().unwrap_or_else(|| panic!("{}: the JVM refused to train, the port trained", path.display()));
        let (class, message) = (l.strs("class").unwrap().get(0), l.strs("message").unwrap().get(0));
        let (made_class, made_message) = java_exception(&err);
        assert_eq!((made_class, made_message.as_str()), (class, message), "{}", path.display());
        return Outcome::Refused;
    }
    let trained = trained.unwrap_or_else(|e| panic!("{}: {e}", path.display()));

    let entry = common::layer(g, "turbo:model-entry");
    assert_eq!(entry.strs("type").unwrap().get(0), trained.model.algorithm().name(), "{}", path.display());
    let model = trained.model.as_writable();
    let outcomes: Vec<String> = model.outcome_labels().iter().map(|o| o.to_string_lossy()).collect();
    assert_eq!(labels(g, "turbo:model-outcomes"), outcomes, "{}: outcomes", path.display());
    let predicates: Vec<String> = model.pred_labels().iter().map(|p| p.to_string_lossy()).collect();
    assert_eq!(labels(g, "turbo:model-predicates"), predicates, "{}: predicates", path.display());

    let l = common::layer(g, "turbo:model-parameters");
    let (predicate, outcome, value) =
        (l.u32s("predicate").unwrap(), l.u32s("outcome").unwrap(), l.f64s("value").unwrap());
    let mut made = Vec::new();
    for p in 0..model.num_predicates() {
        for (&o, &v) in model.pattern(p).iter().zip(model.parameters(p)) {
            made.push((p as u32, o, v));
        }
    }
    assert_eq!(made.len(), l.len(), "{}: parameters", path.display());
    for (i, &(p, o, v)) in made.iter().enumerate() {
        assert_eq!((predicate[i], outcome[i]), (p, o), "{}: parameter {i}", path.display());
        let jvm = value[i];
        assert_eq!(
            v.to_bits(),
            jvm.to_bits(),
            "{}: parameter {i} is {v:e}, the JVM's {jvm:e}, {:e} apart",
            path.display(),
            (v - jvm).abs()
        );
    }

    // The entry the model writes, and the file: the manifest (the JVM's
    // without its timestamp) and the zip entries with their lengths.
    let bytes = trained.package.entries().find(|(n, _)| *n == chunk::MODEL_ENTRY).unwrap().1.to_vec();
    assert_eq!(entry.strs("sha256").unwrap().get(0), Sha256::of(&bytes).to_string(), "{}: entry", path.display());
    assert_eq!(entry.u32s("length").unwrap()[0] as usize, bytes.len(), "{}: entry length", path.display());
    let l = common::layer(g, "turbo:manifest");
    let (keys, values) = (l.strs("key").unwrap(), l.strs("value").unwrap());
    let manifest: Vec<(String, String)> =
        (0..l.len()).map(|i| (keys.get(i).to_owned(), common::lossless_text(values.get(i)))).collect();
    assert_eq!(manifest, manifest_entries(&trained.package), "{}: manifest", path.display());
    let l = common::layer(g, "turbo:package-entries");
    let (names, lengths) = (l.strs("name").unwrap(), l.u32s("length").unwrap());
    let names: Vec<&str> = (0..l.len()).map(|i| names.get(i)).collect();
    assert_eq!(names, trained.package.entry_names(), "{}: entry order", path.display());
    let manifest_length = trained.package.manifest().to_properties().len();
    let at = names.iter().position(|n| *n == MANIFEST_ENTRY).unwrap();
    // the generator's JVM dates in UTC, as `Stamp::at` does, and both
    // timestamps have thirteen digits
    assert_eq!(lengths[at] as usize, manifest_length, "{}: manifest length", path.display());
    Outcome::Trained { parameters: made.len() }
}

#[test]
fn chunkers_trained_here_are_the_jvms_over_every_chunker_golden() {
    let dir = data_dir().join("train-events");
    let files: Vec<_> = common::golden_files(&dir)
        .into_iter()
        .filter(|p| {
            let run = p.strip_prefix(&dir).unwrap().components().next().unwrap();
            run.as_os_str().to_str().unwrap().starts_with("chunker-")
        })
        .collect();
    let mut runs = std::collections::BTreeMap::<String, (usize, usize, usize)>::new();
    for path in &files {
        let g = common::read(path);
        assert!(g.header.error.is_none(), "{}: the generator failed", path.display());
        let run = common::config(&g, "run").unwrap().to_owned();
        let counts = runs.entry(run).or_default();
        match check(path, &g) {
            Outcome::Trained { parameters } => {
                counts.0 += 1;
                counts.2 += parameters;
            }
            Outcome::Refused => counts.1 += 1,
        }
    }
    // Cutoff 5 leaves `chunk-basic` no predicate; `chunk-empty-first` has
    // no sample, as `ChunkSampleStream` ends at an empty first line; and
    // `ChunkerME.train` takes no sequence trainer. Every parameter counted
    // is the JVM's double.
    let made: Vec<(&str, usize, usize, usize)> = runs.iter().map(|(run, c)| (run.as_str(), c.0, c.1, c.2)).collect();
    assert_eq!(
        made,
        [
            ("chunker-defaults", 3, 2, 1347),
            ("chunker-perceptron", 4, 1, 94545),
            ("chunker-perceptron-sequence", 0, 5, 0),
            ("chunker-qn", 4, 1, 94545)
        ]
    );
}
