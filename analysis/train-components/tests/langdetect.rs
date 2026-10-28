//! `LanguageDetectorME.train` against the JVM's goldens under
//! `analysis/goldens/data/train-langdetect/<run>/`: the events of every
//! input (outcome and predicates, in the JVM's order), their checksum, the
//! index, the GIS training iteration by iteration, the weights bit for bit,
//! the written model, the manifest and the zip entries. The runs are
//! `default` (the command line trainer's parameters, cutoff 5) and
//! `cutoff1`. Bit for bit holds for events in the JVM's predicate order:
//! an event with equal-hash n-grams in a tree bucket can differ in the
//! last bits (`inexact_events`), and the test checks the goldens hold
//! none.

mod common;

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use common::{describe_difference, golden_files, java_string, layer, layer_opt};
use turbo_formats::LanguageSample;
use turbo_formats::native::LanguageReader;
use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, Shape, data_dir};
use turbo_maxent::Properties;
use turbo_train::error::NO_EVENTS;
use turbo_train::event::checksum;
use turbo_train::gis::train_model;
use turbo_train::monitor::finished_message;
use turbo_train::package::MANIFEST_ENTRY;
use turbo_train::{
    DataIndex, Event, GisModel, GisParams, JavaString, Key, LogLikelihoodThresholdBreached, ProgressMonitor,
    StopCriteria, TrainingConfiguration, TrainingMeasure, TrainingParameters, write_gis,
};
use turbo_train_components::Stamp;
use turbo_train_components::langdetect::{
    self, LanguageDetectorFactory, MODEL_ENTRY, language_events, language_index, train_language_model,
    trainer_parameters,
};

/// The layers only these goldens have: the events, a row each, and their
/// predicates, a row each, since an n-gram can hold a space or a tab.
const LANGUAGE_LAYERS: &[(&str, &[ColumnSpec])] = &[
    (
        "turbo:language-events",
        &[ColumnSpec::new("outcome", ColumnType::Str), ColumnSpec::new("predicates", ColumnType::U32)],
    ),
    (
        "turbo:language-predicates",
        &[ColumnSpec::new("event", ColumnType::U32), ColumnSpec::new("predicate", ColumnType::Str)],
    ),
];

fn read(path: &Path) -> Golden {
    let mut shapes = common::shapes();
    shapes.extend(LANGUAGE_LAYERS.iter().map(|(id, columns)| Shape {
        id: id.to_string(),
        anchor: GoldenAnchor::Document,
        columns,
    }));
    Golden::read_with(path, &shapes).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn config<'g>(at: &str, g: &'g Golden, key: &str) -> &'g str {
    common::config(g, key).unwrap_or_else(|| panic!("{at}: no config {key}"))
}

/// The parameters the run called `LanguageDetectorME.train` with, from its
/// config records; `DataIndexer` is the one `train` puts in.
fn parameters(at: &str, g: &Golden) -> TrainingParameters {
    let mut p = TrainingParameters::new();
    p.put(Key::ALGORITHM, config(at, g, "Algorithm"));
    p.put(Key::ITERATIONS, config(at, g, "Iterations").parse::<i32>().unwrap());
    p.put(Key::CUTOFF, config(at, g, "Cutoff").parse::<i32>().unwrap());
    assert_eq!(config(at, g, "DataIndexer"), langdetect::ONE_PASS, "{at}");
    assert!(common::config(g, "TrainerType").is_none(), "{at}: the defaults name no trainer type");
    p
}

/// The events the golden lists, read back through [`java_string`].
fn golden_events(g: &Golden) -> Vec<Event> {
    let events = layer(g, "turbo:language-events");
    let (outcomes, sizes) = (events.strs("outcome").unwrap(), events.u32s("predicates").unwrap());
    let predicates = layer(g, "turbo:language-predicates");
    let (owner, names) = (predicates.u32s("event").unwrap(), predicates.strs("predicate").unwrap());
    let mut contexts: Vec<Vec<JavaString>> = vec![Vec::new(); events.len()];
    for i in 0..predicates.len() {
        contexts[owner[i] as usize].push(java_string(names.get(i)));
    }
    (0..events.len())
        .map(|i| {
            assert_eq!(contexts[i].len(), sizes[i] as usize, "event {i}");
            Event::new(java_string(outcomes.get(i)), std::mem::take(&mut contexts[i]))
        })
        .collect()
}

#[derive(Default, Debug)]
struct Coverage {
    goldens: usize,
    trained: usize,
    indexer_errors: usize,
    one_outcome: usize,
    stopped_early: usize,
    empty_events: usize,
    tab_predicates: usize,
    /// Events whose n-grams `getContext` gives in another order than
    /// `predict` counts them in (a bucket that became a tree).
    reordered: usize,
    merged: usize,
    runs: [usize; 2],
}

#[derive(Default)]
struct Recorded {
    rows: Vec<(i32, i32, i32, f64)>,
    last: Option<String>,
}

/// A monitor writing into a shared [`Recorded`].
struct Recorder(Rc<RefCell<Recorded>>);

impl ProgressMonitor for Recorder {
    fn finished_iteration(&mut self, iteration: i32, correct: i32, total: i32, measure: TrainingMeasure, value: f64) {
        assert_eq!(measure, TrainingMeasure::LogLikelihood);
        self.0.borrow_mut().rows.push((iteration, correct, total, value));
    }

    fn finished_training(&mut self, iterations: i32, stop: Option<&dyn StopCriteria>) {
        self.0.borrow_mut().last = Some(match stop {
            Some(s) => s.message_if_satisfied(),
            None => finished_message(iterations),
        });
    }

    fn is_training_finished(&self) -> bool {
        self.0.borrow().last.is_some()
    }

    fn display(&mut self, _clear: bool) {}
}

/// `TrainGolden.digest`: the SHA-256 over every parameter's bits.
fn digest(parameters: &[f64]) -> String {
    let mut bytes = Vec::with_capacity(parameters.len() * 8);
    for p in parameters {
        bytes.extend_from_slice(&p.to_bits().to_be_bytes());
    }
    turbo_maxent::sha256(&bytes).to_hex()
}

fn joined<T: ToString>(items: impl Iterator<Item = T>) -> String {
    items.map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
}

/// The index against the `turbo:indexer-*` layers.
fn check_index(at: &str, g: &Golden, index: &DataIndex, cov: &mut Coverage) {
    let outcomes = layer(g, "turbo:indexer-outcomes").strs("label").unwrap();
    assert_eq!(index.outcome_labels(), outcomes.iter().collect::<Vec<_>>(), "{at}: outcome labels");
    let preds = layer(g, "turbo:indexer-predicates");
    let labels: Vec<JavaString> = preds.strs("label").unwrap().iter().map(java_string).collect();
    assert_eq!(index.pred_labels(), labels, "{at}: predicate labels");
    assert_eq!(index.pred_counts(), preds.u32s("count").unwrap(), "{at}: predicate counts");
    let indexed = layer(g, "turbo:indexer-events");
    assert_eq!(index.num_unique_events(), indexed.len(), "{at}: unique events");
    let (outcome, context, values, seen) = (
        indexed.u32s("outcome").unwrap(),
        indexed.strs("context").unwrap(),
        indexed.strs("values").unwrap(),
        indexed.u32s("seen").unwrap(),
    );
    for i in 0..indexed.len() {
        assert_eq!(index.outcome(i), outcome[i], "{at}: event {i} outcome");
        assert_eq!(joined(index.context(i).iter()), context.get(i), "{at}: event {i} context");
        assert!(index.values(i).is_none() && values.get(i).is_empty(), "{at}: event {i} values");
        assert_eq!(index.seen(i), seen[i], "{at}: event {i} seen");
        cov.merged += usize::from(seen[i] > 1);
    }
}

/// The GIS training against the `turbo:gis-*` layers: the parameters after
/// every iteration by digest and after the first and the last bit for bit,
/// the log likelihood and counts the monitor heard, the end of training
/// and the written model.
fn check_gis(at: &str, g: &Golden, index: &DataIndex, params: &TrainingParameters, ran: usize) -> GisModel {
    let trainer = trainer_parameters(params);
    let gis = GisParams::from_training_parameters(&trainer).unwrap();
    assert_eq!(gis.threads, 1, "{at}");
    let stop = LogLikelihoodThresholdBreached::from_parameters(&trainer).unwrap();
    let iterations = layer(g, "turbo:gis-iterations");
    assert_eq!(iterations.len(), ran, "{at}");
    let digests = iterations.strs("parameters").unwrap();
    let recorded = Rc::new(RefCell::new(Recorded::default()));
    let mut config = TrainingConfiguration::new(Some(Box::new(Recorder(recorded.clone()))), None);
    let (mut first, mut seen) = (Vec::new(), 0);
    let model = train_model(index, &gis, &mut config, &stop, &mut |i, parameters| {
        seen = i as usize;
        assert_eq!(digest(parameters), digests.get(i as usize - 1), "{at}: parameters after iteration {i}");
        if i == 1 {
            first = parameters.to_vec();
        }
    })
    .unwrap_or_else(|e| panic!("{at}: {e}"));
    assert_eq!(seen, ran, "{at}: iterations run");

    let recorded = recorded.borrow();
    let (correct, events, likelihood) = (
        iterations.u32s("correct").unwrap(),
        iterations.u32s("events").unwrap(),
        iterations.f64s("likelihood").unwrap(),
    );
    assert_eq!(recorded.rows.len(), ran, "{at}");
    for (i, (it, c, total, ll)) in recorded.rows.iter().enumerate() {
        assert_eq!(*it as usize, i + 1, "{at}");
        assert_eq!((*c as u32, *total as u32), (correct[i], events[i]), "{at}: counts after iteration {}", i + 1);
        assert_eq!(ll.to_bits(), likelihood[i].to_bits(), "{at}: log likelihood after iteration {}", i + 1);
    }
    let training = layer(g, "turbo:gis-training");
    assert_eq!(recorded.last.as_deref(), Some(training.strs("message").unwrap().get(0)), "{at}");

    // Every weight after the first and the last iteration, in the model's
    // layout: equal bits, so within any tolerance of the JVM's.
    let values = layer(g, "turbo:gis-parameters");
    let (it, pred, oid, value) = (
        values.u32s("iteration").unwrap(),
        values.u32s("predicate").unwrap(),
        values.u32s("outcome").unwrap(),
        values.f64s("value").unwrap(),
    );
    let last = model.all_parameters();
    assert_eq!(values.len(), if ran > 1 { 2 * last.len() } else { last.len() }, "{at}: parameter rows");
    let start = |p: usize| (0..p).map(|q| model.pattern(q).len()).sum::<usize>();
    for r in 0..values.len() {
        let flat = r % last.len();
        let (expected_it, parameters) = if r < last.len() { (1, &first[..]) } else { (ran as u32, last) };
        assert_eq!(it[r], expected_it, "{at}: parameter row {r} iteration");
        let p = pred[r] as usize;
        assert_eq!(model.pattern(p)[flat - start(p)], oid[r], "{at}: parameter row {r} outcome");
        assert_eq!(parameters[flat].to_bits(), value[r].to_bits(), "{at}: parameter row {r} value");
    }

    let bytes = write_gis(&model).unwrap_or_else(|e| panic!("{at}: {e}"));
    let written = layer(g, "turbo:gis-model");
    assert_eq!(bytes.len() as u32, written.u32s("length").unwrap()[0], "{at}: model length");
    assert_eq!(turbo_maxent::sha256(&bytes).to_hex(), written.strs("sha256").unwrap().get(0), "{at}: model sha256");
    model
}

/// `LanguageDetectorME.train` end to end: the model entry is the bytes
/// `turbo:gis-model` describes, and the manifest (without `Timestamp`)
/// and the zip entries are the JVM's.
fn check_package(at: &str, g: &Golden, samples: &[LanguageSample<'_>], params: &TrainingParameters) {
    let trained = train_language_model(
        samples,
        &LanguageDetectorFactory::new(),
        params,
        &mut TrainingConfiguration::default(),
        Stamp::at(1_700_000_000_000),
    )
    .unwrap_or_else(|e| panic!("{at}: {e}"));
    let package = &trained.package;
    let (_, model) = package.entries().find(|(n, _)| *n == MODEL_ENTRY).unwrap();
    let written = layer(g, "turbo:gis-model");
    assert_eq!(turbo_maxent::sha256(model).to_hex(), written.strs("sha256").unwrap().get(0), "{at}: model entry");

    let l = layer(g, "turbo:manifest");
    let (keys, values) = (l.strs("key").unwrap(), l.strs("value").unwrap());
    let expected: Vec<(String, String)> =
        (0..l.len()).map(|i| (keys.get(i).to_owned(), values.get(i).to_owned())).collect();
    let text = package.manifest().to_properties();
    let properties = Properties::load(text.as_bytes()).unwrap();
    let mut ours: Vec<(String, String)> =
        properties.iter().filter(|(k, _)| *k != "Timestamp").map(|(k, v)| (k.to_owned(), v.to_owned())).collect();
    ours.sort();
    assert_eq!(expected, ours, "{at}: manifest");

    // The entries in order; the manifest's length too, both timestamps
    // having thirteen digits and both dated in UTC.
    let l = layer(g, "turbo:package-entries");
    let (names, lengths) = (l.strs("name").unwrap(), l.u32s("length").unwrap());
    assert_eq!((0..l.len()).map(|i| names.get(i)).collect::<Vec<_>>(), package.entry_names(), "{at}: entries");
    for (i, name) in package.entry_names().into_iter().enumerate() {
        let len = match name {
            MANIFEST_ENTRY => text.len(),
            _ => package.entries().find(|(n, _)| *n == name).unwrap().1.len(),
        };
        assert_eq!(lengths[i] as usize, len, "{at}: the length of {name}");
    }
    // The model file opens as a language detector with the trained outcomes.
    let file = turbo_maxent::ModelFile::open(package.to_bytes()).unwrap();
    let opened = turbo_langdetect::LanguageModel::open(file).unwrap_or_else(|e| panic!("{at}: {e}"));
    let outcomes = layer(g, "turbo:indexer-outcomes").strs("label").unwrap();
    assert_eq!(opened.supported_languages().collect::<Vec<_>>(), outcomes.iter().collect::<Vec<_>>(), "{at}");
}

fn check(path: &Path, cov: &mut Coverage) {
    let at = path.display().to_string();
    let g = read(path);
    assert!(g.header.error.is_none(), "{at}");
    let run = config(&at, &g, "run");
    let params = parameters(&at, &g);
    let cutoff: i32 = config(&at, &g, "Cutoff").parse().unwrap();
    match (run, cutoff) {
        ("default", 5) => cov.runs[0] += 1,
        ("cutoff1", 1) => cov.runs[1] += 1,
        other => panic!("{at}: run {other:?}"),
    }
    let text = g.input_text(data_dir()).unwrap_or_else(|e| panic!("{at}: {e}"));
    let samples: Vec<LanguageSample<'_>> =
        LanguageReader::new(&text).unwrap().map(|s| s.unwrap_or_else(|e| panic!("{at}: {e}"))).collect();
    let (events, inexact) = language_events(&samples, &LanguageDetectorFactory::new());
    assert_eq!(inexact, 0, "{at}");
    let expected = golden_events(&g);
    assert!(events == expected, "{at}: {}", describe_difference(&expected, &events));
    cov.empty_events += events.iter().filter(|e| e.context().is_empty()).count();
    cov.tab_predicates +=
        events.iter().flat_map(|e| e.context()).filter(|p| p.to_string_lossy().contains('\t')).count();
    let mut generator = LanguageDetectorFactory::new().context_generator();
    cov.reordered += samples
        .iter()
        .filter(|s| {
            generator.generate(s.context.as_str());
            !generator.training_features().eq(generator.features())
        })
        .count();
    cov.goldens += 1;

    let index = language_index(&events, &params);
    if let Some(error) = layer_opt(&g, "turbo:indexer-error") {
        assert_eq!(error.strs("class").unwrap().get(0), "opennlp.tools.util.InsufficientTrainingDataException", "{at}");
        assert_eq!(error.strs("message").unwrap().get(0), NO_EVENTS, "{at}");
        assert!(
            matches!(
                index,
                Err(turbo_train_components::Error::Train(turbo_train::Error::InsufficientTrainingData(m))) if m == NO_EVENTS
            ),
            "{at}: {index:?}"
        );
        cov.indexer_errors += 1;
        return;
    }
    let index = index.unwrap_or_else(|e| panic!("{at}: {e}"));
    assert_eq!(checksum(&events), layer(&g, "turbo:event-checksum").u32s("crc32c").unwrap()[0], "{at}: checksum");
    check_index(&at, &g, &index, cov);

    let training = layer(&g, "turbo:gis-training");
    let ran = training.u32s("iterations").unwrap()[0] as usize;
    if ran == 0 {
        assert!(index.outcome_labels().len() <= 1, "{at}");
        assert_eq!(
            training.strs("message").unwrap().get(0),
            "opennlp.tools.util.InsufficientTrainingDataException: Training data must contain more than one outcome",
            "{at}"
        );
        cov.one_outcome += 1;
        return;
    }
    check_gis(&at, &g, &index, &params, ran);
    cov.stopped_early += usize::from(ran < 100);
    check_package(&at, &g, &samples, &params);
    cov.trained += 1;
}

#[test]
fn language_detector_training_is_the_jvms_over_every_golden() {
    let paths = golden_files(&data_dir().join("train-langdetect"));
    let mut cov = Coverage::default();
    for path in &paths {
        check(path, &mut cov);
    }
    eprintln!("{cov:?}");
    // Two runs over the two inputs written for these goldens and the
    // language format input.
    assert_eq!(cov.goldens, 6, "{cov:?}");
    assert_eq!(cov.runs, [3, 3], "{cov:?}");
    // The language format input has too few events for cutoff 5.
    assert_eq!((cov.trained, cov.indexer_errors), (5, 1), "{cov:?}");
    assert!(cov.empty_events > 0 && cov.tab_predicates > 0 && cov.merged > 0, "{cov:?}");
    // The six lines of `inputs/langdetect/tree-bucket.txt`, in both runs,
    // are in the order training reads, not prediction's.
    assert_eq!(cov.reordered, 12, "{cov:?}");
}
