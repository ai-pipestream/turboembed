//! The indexers against the JVM: `analysis/goldens/data/train/` holds, for
//! events the JVM's sentence detector, tokenizer and a real valued source
//! made from the input texts, what `OnePassDataIndexer`,
//! `TwoPassDataIndexer` and `OnePassRealValueDataIndexer` produced with
//! each run's cutoff and sorting (`analysis/goldens-tool`, `TrainGolden`).
//! This test reads the events back through the event stream parsers,
//! indexes them, and compares every label, count, context and seen count,
//! and the event checksum, with the golden's. It then trains the GIS model
//! the run trained and compares the log likelihood, the counts and every
//! parameter after every iteration, the end of training, and the bytes of
//! the written model.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use turbo_goldens::{Golden, data_dir};
use turbo_jvm::text::float_to_string;
use turbo_train::error::NO_EVENTS;
use turbo_train::event::{checksum, parse_event, parse_real_value_event};
use turbo_train::gis::train_model;
use turbo_train::monitor::finished_message;
use turbo_train::{
    DataIndex, Error, GisModel, GisParams, IndexerKind, JavaString, Key, LogLikelihoodThresholdBreached,
    ProgressMonitor, StopCriteria, TrainingConfiguration, TrainingMeasure, TrainingParameters, write_gis,
};

mod common;
use common::{config, events, java_string, layer, params, train_shapes};

fn golden_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let p = e.unwrap().path();
        if p.is_dir() {
            golden_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "golden") {
            out.push(p);
        }
    }
}

fn joined<T: ToString>(items: impl Iterator<Item = T>) -> String {
    items.map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
}

/// What the run has so the test can tell the goldens cover every shape.
#[derive(Default, Debug)]
struct Coverage {
    goldens: usize,
    errors: usize,
    one_outcome: usize,
    unsorted: usize,
    merged: usize,
    with_values: usize,
    dropped_events: usize,
    lone_surrogates: usize,
    trained: usize,
    stopped_early: usize,
    kinds: [usize; 3],
}

fn check(path: &Path, g: &Golden, cov: &mut Coverage) {
    let at = path.display();
    let params = params(g);
    cov.kinds[match params.kind {
        IndexerKind::OnePass => 0,
        IndexerKind::TwoPass => 1,
        IndexerKind::OnePassRealValue => 2,
    }] += 1;
    if !params.sort {
        cov.unsorted += 1;
    }
    let events = events(g);
    // An event without a lone surrogate round trips through its line
    // format; one with a lone surrogate cannot be written as text.
    for e in &events {
        if e.context().iter().any(|c| c.to_str().is_none()) {
            cov.lone_surrogates += 1;
            continue;
        }
        let line = e.real_value_line();
        let back = if e.values().is_some() { parse_real_value_event(&line, 1) } else { parse_event(&line, 1) };
        assert_eq!(back.unwrap(), *e, "{at}");
    }
    let index = DataIndex::index(&events, &params);
    assert!(g.header.error.is_none(), "{at}");
    if let Some(error) = g.layers.iter().find(|l| l.id == "turbo:indexer-error") {
        assert_eq!(error.strs("class").unwrap().get(0), "opennlp.tools.util.InsufficientTrainingDataException", "{at}");
        assert_eq!(error.strs("message").unwrap().get(0), NO_EVENTS, "{at}");
        assert_eq!(index, Err(Error::InsufficientTrainingData(NO_EVENTS)), "{at}");
        cov.errors += 1;
        return;
    }
    let index = index.unwrap_or_else(|e| panic!("{at}: {e}"));

    let crc = layer(g, "turbo:event-checksum").u32s("crc32c").unwrap();
    assert_eq!(checksum(&events), crc[0], "{at}: event checksum");

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
        let v = index.values(i).map_or(String::new(), |v| joined(v.iter().map(|x| float_to_string(*x))));
        assert_eq!(v, values.get(i), "{at}: event {i} values");
        assert_eq!(index.seen(i), seen[i], "{at}: event {i} seen");
        if seen[i] > 1 {
            cov.merged += 1;
        }
        if !values.get(i).is_empty() {
            cov.with_values += 1;
        }
    }
    assert_eq!(index.has_values(), params.kind.keeps_values(), "{at}");
    if (index.num_events() as usize) < events.len() {
        cov.dropped_events += 1;
    }

    // The trainer refuses an index with one outcome before it starts.
    let training = layer(g, "turbo:gis-training");
    let iterations = training.u32s("iterations").unwrap()[0];
    if iterations == 0 {
        assert!(index.outcome_labels().len() <= 1, "{at}");
        assert_eq!(
            training.strs("message").unwrap().get(0),
            "opennlp.tools.util.InsufficientTrainingDataException: Training data must contain more than one outcome",
            "{at}"
        );
        cov.one_outcome += 1;
    } else {
        assert!(index.outcome_labels().len() > 1, "{at}");
        check_gis(&at.to_string(), g, &index, iterations as usize, cov);
    }
}

/// What `TrainGolden.Recorder` kept: `(correct, total, log likelihood)`
/// per iteration, and the message of the end of training.
#[derive(Default)]
struct Recorded {
    rows: Vec<(i32, i32, f64)>,
    last: Option<String>,
}

/// A monitor writing into a shared [`Recorded`].
struct Recorder(Rc<RefCell<Recorded>>);

impl ProgressMonitor for Recorder {
    fn finished_iteration(&mut self, iteration: i32, correct: i32, total: i32, measure: TrainingMeasure, value: f64) {
        let mut r = self.0.borrow_mut();
        assert_eq!(iteration as usize, r.rows.len() + 1);
        assert_eq!(measure, TrainingMeasure::LogLikelihood);
        r.rows.push((correct, total, value));
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

/// The offset of predicate `p`'s parameters in [`GisModel::all_parameters`].
fn pattern_start(model: &GisModel, p: usize) -> usize {
    (0..p).map(|q| model.pattern(q).len()).sum()
}

/// The GIS training of the run against its `turbo:gis-*` layers: the
/// parameters after every iteration (bit for bit after the first and the
/// last, by digest in between), the log likelihood and the counts reported
/// after every iteration, the end of training, and the written model.
fn check_gis(at: &str, g: &Golden, index: &DataIndex, ran: usize, cov: &mut Coverage) {
    let mut params = TrainingParameters::new();
    params.put(Key::ITERATIONS, config(g, "Iterations").parse::<i32>().unwrap());
    let gis = GisParams::from_training_parameters(&params).unwrap();
    let stop = LogLikelihoodThresholdBreached::from_parameters(&params).unwrap();

    let iterations = layer(g, "turbo:gis-iterations");
    assert_eq!(iterations.len(), ran, "{at}");
    let digests = iterations.strs("parameters").unwrap();
    let mut first = Vec::new();
    let mut seen = 0;
    let recorded = Rc::new(RefCell::new(Recorded::default()));
    let mut config = TrainingConfiguration::new(Some(Box::new(Recorder(recorded.clone()))), None);
    let model = train_model(index, &gis, &mut config, &stop, &mut |i, parameters| {
        seen = i as usize;
        assert_eq!(digest(parameters), digests.get(i as usize - 1), "{at}: parameters after iteration {i}");
        if i == 1 {
            first = parameters.to_vec();
        }
    });
    assert_eq!(seen, ran, "{at}: iterations run");

    let recorded = recorded.borrow();
    let (correct, events, likelihood) = (
        iterations.u32s("correct").unwrap(),
        iterations.u32s("events").unwrap(),
        iterations.f64s("likelihood").unwrap(),
    );
    assert_eq!(recorded.rows.len(), ran, "{at}");
    for (i, (c, total, ll)) in recorded.rows.iter().enumerate() {
        assert_eq!(*c as u32, correct[i], "{at}: correct after iteration {}", i + 1);
        assert_eq!(*total as u32, events[i], "{at}: events after iteration {}", i + 1);
        assert_eq!(ll.to_bits(), likelihood[i].to_bits(), "{at}: log likelihood after iteration {}", i + 1);
    }
    let training = layer(g, "turbo:gis-training");
    assert_eq!(recorded.last.as_deref(), Some(training.strs("message").unwrap().get(0)), "{at}");
    if ran < gis.iterations as usize {
        cov.stopped_early += 1;
    }

    // Every parameter after the first and the last iteration, in the
    // layout of the model: predicates in order, each one's pattern in order.
    let values = layer(g, "turbo:gis-parameters");
    let (it, pred, oid, value) = (
        values.u32s("iteration").unwrap(),
        values.u32s("predicate").unwrap(),
        values.u32s("outcome").unwrap(),
        values.f64s("value").unwrap(),
    );
    let last = model.all_parameters();
    assert_eq!(first.len(), last.len(), "{at}");
    assert_eq!(values.len(), if ran > 1 { 2 * last.len() } else { last.len() }, "{at}: parameter rows");
    for r in 0..values.len() {
        let flat = r % last.len();
        let (expected_it, parameters) = if r < last.len() { (1, &first[..]) } else { (ran as u32, last) };
        assert_eq!(it[r], expected_it, "{at}: parameter row {r} iteration");
        let p = pred[r] as usize;
        let o = flat - pattern_start(&model, p);
        assert_eq!(model.pattern(p)[o], oid[r], "{at}: parameter row {r} outcome");
        assert_eq!(parameters[flat].to_bits(), value[r].to_bits(), "{at}: parameter row {r} value");
    }

    // The written model: the bytes GenericModelWriter wrote.
    let bytes = write_gis(&model).unwrap_or_else(|e| panic!("{at}: {e}"));
    let written = layer(g, "turbo:gis-model");
    assert_eq!(bytes.len() as u32, written.u32s("length").unwrap()[0], "{at}: model length");
    assert_eq!(turbo_maxent::sha256(&bytes).to_hex(), written.strs("sha256").unwrap().get(0), "{at}: model sha256");
    let read = turbo_maxent::MaxentModel::read(&bytes).unwrap_or_else(|e| panic!("{at}: {e}"));
    assert_eq!(read.num_predicates(), model.num_predicates(), "{at}");
    cov.trained += 1;
}

#[test]
fn indexers_match_the_jvm_on_every_training_golden() {
    let shapes = train_shapes();
    let mut paths = Vec::new();
    golden_files(&data_dir().join("train"), &mut paths);
    paths.sort();
    let mut cov = Coverage::default();
    for path in &paths {
        let g = Golden::read_with(path, &shapes).unwrap_or_else(|e| panic!("{e}"));
        g.input_text(data_dir()).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        check(path, &g, &mut cov);
        cov.goldens += 1;
    }
    eprintln!("{cov:?}");
    assert!(cov.goldens >= 100, "{cov:?}");
    assert!(cov.errors > 0 && cov.one_outcome > 0 && cov.unsorted > 0, "{cov:?}");
    assert!(cov.merged > 0 && cov.with_values > 0 && cov.dropped_events > 0, "{cov:?}");
    assert!(cov.lone_surrogates > 0, "{cov:?}");
    assert!(cov.trained > 0 && cov.stopped_early > 0, "{cov:?}");
    assert!(cov.kinds.iter().all(|&n| n > 0), "{cov:?}");
}

/// The run that trained the fixture model en-plain.bin: TwoPass, cutoff 1,
/// on the sentences it was trained from. Its event checksum is the
/// `Training-Eventhash` in the model's manifest.
#[test]
fn the_sentence_fixture_run_has_the_fixture_models_event_hash() {
    let path = data_dir().join("train/sentdetect-twopass-cutoff1/train/sentences.golden");
    let g = Golden::read_with(&path, &train_shapes()).unwrap_or_else(|e| panic!("{e}"));
    let events = events(&g);
    let crc = layer(&g, "turbo:event-checksum").u32s("crc32c").unwrap()[0];
    assert_eq!(checksum(&events), crc);
    let model = turbo_maxent::ModelFile::read(data_dir().join("fixtures/en-plain.bin")).unwrap();
    assert_eq!(model.manifest().get("Training-Eventhash"), Some(crc.to_string().as_str()));
    let index = DataIndex::index(&events, &params(&g)).unwrap();
    let maxent = model.maxent("sent.model").unwrap();
    assert_eq!(index.outcome_labels(), maxent.outcomes().collect::<Vec<_>>());
    assert_eq!(index.pred_labels().len(), maxent.num_predicates());
}
