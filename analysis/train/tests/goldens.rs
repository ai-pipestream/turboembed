//! The indexers against the JVM: `analysis/goldens/data/train/` holds, for
//! events the JVM's sentence detector, tokenizer, a real valued source and
//! a part-of-speech shaped source with many outcomes made from the input
//! texts, what `OnePassDataIndexer`, `TwoPassDataIndexer` and
//! `OnePassRealValueDataIndexer` produced with each run's cutoff and
//! sorting (`analysis/goldens-tool`, `TrainGolden`). This test reads the
//! events back through the event stream parsers, indexes them, and
//! compares every label, count, context and seen count, and the event
//! checksum, with the golden's. It then trains the GIS model the run
//! trained, with the run's `Threads` and smoothing settings, and compares
//! the log likelihood, the counts and every parameter after every
//! iteration, the end of training, and the bytes of the written model.
//! The goldens of the QN runs (`QnTrainGolden`) hold the events and the
//! trainer's layers only; for them the index is built from the events and
//! the QN trainer is compared the same way, with the objective value,
//! change rate and training accuracy per iteration and the reason it
//! stopped.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use turbo_goldens::{Golden, data_dir};
use turbo_jvm::text::float_to_string;
use turbo_train::error::NO_EVENTS;
use turbo_train::error::ONE_OUTCOME;
use turbo_train::event::{checksum, parse_event, parse_real_value_event};
use turbo_train::gis::train_model;
use turbo_train::monitor::finished_message;
use turbo_train::{
    DataIndex, Error, Event, GisModel, GisParams, IndexParams, IndexerKind, IterDeltaAccuracyUnderTolerance,
    JavaString, Key, LogLikelihoodThresholdBreached, PerceptronModel, PerceptronParams, ProgressMonitor, QnStop,
    Sequence, SequenceSource, Smoothing, StopCriteria, TrainingConfiguration, TrainingMeasure, TrainingParameters,
    write_gis, write_perceptron, write_qn,
};
use turbo_train::{perceptron, qn};

mod common;
use common::{
    PERCEPTRON_TRAINER, QN_TRAINER, SEQUENCE_PERCEPTRON_TRAINER, config, config_opt, events, java_string, layer,
    params, perceptron_params, qn_params, train_shapes,
};

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
    /// Trained runs with two threads, with `Smoothing`, with
    /// `GaussianSmoothing`, and with more than two outcomes.
    threads: usize,
    smoothing: usize,
    gaussian: usize,
    many_outcomes: usize,
    qn: usize,
    qn_stops: [usize; 5],
    qn_threads: usize,
    qn_l1: usize,
    qn_l2: usize,
    qn_plain: usize,
    qn_elastic: usize,
    qn_many_outcomes: usize,
    qn_discarding: usize,
    perceptron: usize,
    perceptron_one_outcome: usize,
    perceptron_stopped_early: usize,
    perceptron_raw: usize,
    perceptron_skipped: usize,
    perceptron_decrease: usize,
    perceptron_unwritable: usize,
    perceptron_values: usize,
    perceptron_many_outcomes: usize,
    sequence: usize,
    sequence_raw: usize,
    sequence_dropped: usize,
    sequence_many_outcomes: usize,
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

/// What the generators' `Recorder` kept: `(iteration, correct, total,
/// measure)` per iteration the monitor heard of, and the message of the
/// end of training.
#[derive(Default)]
struct Recorded {
    rows: Vec<(i32, i32, i32, f64)>,
    last: Option<String>,
}

/// A monitor writing into a shared [`Recorded`], expecting one measure.
struct Recorder(Rc<RefCell<Recorded>>, TrainingMeasure);

impl ProgressMonitor for Recorder {
    fn finished_iteration(&mut self, iteration: i32, correct: i32, total: i32, measure: TrainingMeasure, value: f64) {
        assert_eq!(measure, self.1);
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

/// The offset of predicate `p`'s parameters in [`GisModel::all_parameters`].
fn pattern_start(model: &GisModel, p: usize) -> usize {
    (0..p).map(|q| model.pattern(q).len()).sum()
}

/// The GIS training of the run against its `turbo:gis-*` layers: the
/// parameters after every iteration (bit for bit after the first and the
/// last, by digest in between), the log likelihood and the counts reported
/// after every iteration, the end of training, and the written model. The
/// trainer's settings are the run's config records: `Iterations` always,
/// and `Threads`, `Smoothing`, `SmoothingObservation`, `GaussianSmoothing`
/// and `GaussianSmoothingSigma` when the run set them, as strings, as a
/// parameters file gives them.
fn check_gis(at: &str, g: &Golden, index: &DataIndex, ran: usize, cov: &mut Coverage) {
    let mut params = TrainingParameters::new();
    params.put(Key::ITERATIONS, config(g, "Iterations").parse::<i32>().unwrap());
    for key in [
        Key::THREADS,
        Key::SMOOTHING,
        Key::SMOOTHING_OBSERVATION,
        Key::GAUSSIAN_SMOOTHING,
        Key::GAUSSIAN_SMOOTHING_SIGMA,
    ] {
        if let Some(value) = config_opt(g, key) {
            params.put(key, value);
        }
    }
    let gis = GisParams::from_training_parameters(&params).unwrap();
    // The JVM adds the chunks' log likelihoods in the order its threads
    // finish. Two chunks give the same sum either way, so the goldens of a
    // two thread run are reproducible and compared bit for bit like the
    // others; a run with more threads would not be (TrainGolden).
    assert!(gis.threads <= 2, "{at}: a golden with {} threads is not reproducible", gis.threads);
    if gis.threads == 2 {
        cov.threads += 1;
    }
    match gis.smoothing {
        Smoothing::None => {}
        Smoothing::Simple { .. } => cov.smoothing += 1,
        Smoothing::Gaussian { .. } => cov.gaussian += 1,
    }
    if index.outcome_labels().len() > 2 {
        cov.many_outcomes += 1;
    }
    let stop = LogLikelihoodThresholdBreached::from_parameters(&params).unwrap();

    let iterations = layer(g, "turbo:gis-iterations");
    assert_eq!(iterations.len(), ran, "{at}");
    let digests = iterations.strs("parameters").unwrap();
    let mut first = Vec::new();
    let mut seen = 0;
    let recorded = Rc::new(RefCell::new(Recorded::default()));
    let mut config =
        TrainingConfiguration::new(Some(Box::new(Recorder(recorded.clone(), TrainingMeasure::LogLikelihood))), None);
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
        assert_eq!(*it as usize, i + 1, "{at}: the monitor heard of iteration {it} as number {}", i + 1);
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
        if g.header.annotator.class == QN_TRAINER {
            check_qn(path, &g, &mut cov);
            cov.qn += 1;
        } else if g.header.annotator.class == PERCEPTRON_TRAINER {
            check_perceptron(path, &g, &mut cov);
            cov.perceptron += 1;
        } else if g.header.annotator.class == SEQUENCE_PERCEPTRON_TRAINER {
            check_sequence_perceptron(path, &g, &mut cov);
            cov.sequence += 1;
        } else {
            check(path, &g, &mut cov);
            cov.goldens += 1;
        }
    }
    eprintln!("{cov:?}");
    assert!(cov.goldens >= 1000, "{cov:?}");
    assert!(cov.errors > 0 && cov.one_outcome > 0 && cov.unsorted > 0, "{cov:?}");
    assert!(cov.merged > 0 && cov.with_values > 0 && cov.dropped_events > 0, "{cov:?}");
    assert!(cov.lone_surrogates > 0, "{cov:?}");
    assert!(cov.trained > 0 && cov.stopped_early > 0, "{cov:?}");
    assert!(cov.kinds.iter().all(|&n| n > 0), "{cov:?}");
    assert!(cov.threads > 0 && cov.smoothing > 0 && cov.gaussian > 0 && cov.many_outcomes > 0, "{cov:?}");
    assert!(cov.qn >= 100 && cov.qn_threads > 0 && cov.qn_discarding > 0, "{cov:?}");
    assert!(cov.qn_l1 > 0 && cov.qn_l2 > 0 && cov.qn_elastic > 0 && cov.qn_plain > 0, "{cov:?}");
    assert!(cov.qn_many_outcomes > 0, "{cov:?}");
    // The change rate and the evaluation limit stop runs here; the
    // iteration limit is tested in qn.rs. No run reaches the gradient norm
    // or the minimum step before the change rate does.
    assert!(cov.qn_stops[0] > 0 && cov.qn_stops[3] > 0, "{cov:?}");
    assert!(cov.perceptron >= 100 && cov.perceptron_one_outcome > 0 && cov.perceptron_stopped_early > 0, "{cov:?}");
    assert!(cov.perceptron_raw > 0 && cov.perceptron_skipped > 0 && cov.perceptron_decrease > 0, "{cov:?}");
    assert!(cov.perceptron_unwritable > 0 && cov.perceptron_values > 0 && cov.perceptron_many_outcomes > 0, "{cov:?}");
    assert!(cov.sequence >= 100 && cov.sequence_raw > 0 && cov.sequence_dropped > 0, "{cov:?}");
    assert!(cov.sequence_many_outcomes > 0, "{cov:?}");
}

/// The events indexed as the run's config says, or `None` when the golden
/// records the indexer's refusal (checked against the Rust indexer).
fn indexed(at: &str, g: &Golden, events: &[Event]) -> Option<DataIndex> {
    let index = DataIndex::index(events, &params(g));
    assert!(g.header.error.is_none(), "{at}");
    if let Some(error) = g.layers.iter().find(|l| l.id == "turbo:indexer-error") {
        assert_eq!(error.strs("message").unwrap().get(0), NO_EVENTS, "{at}");
        assert_eq!(index, Err(Error::InsufficientTrainingData(NO_EVENTS)), "{at}");
        return None;
    }
    Some(index.unwrap_or_else(|e| panic!("{at}: {e}")))
}

/// The `turbo:perceptron-iterations` digests, the `turbo:perceptron-parameters`
/// rows after the first and the last iteration (the model's layout: every
/// predicate's parameter for every outcome), and the `turbo:perceptron-model`
/// bytes or the writer's failure, against a trained model and what the
/// observer saw of it.
fn check_perceptron_layers(
    at: &str,
    g: &Golden,
    model: &PerceptronModel,
    first: &[f64],
    ran: usize,
    cov: &mut Coverage,
) {
    let n = model.num_outcomes();
    let values = layer(g, "turbo:perceptron-parameters");
    let (it, pred, oid, val) = (
        values.u32s("iteration").unwrap(),
        values.u32s("predicate").unwrap(),
        values.u32s("outcome").unwrap(),
        values.f64s("value").unwrap(),
    );
    let last = model.all_parameters();
    assert_eq!(last.len(), n * model.num_predicates(), "{at}");
    assert_eq!(values.len(), if ran > 1 { 2 * last.len() } else { last.len() }, "{at}: parameter rows");
    for r in 0..values.len() {
        let flat = r % last.len();
        let (expected_it, parameters) = if r < last.len() { (1, first) } else { (ran as u32, last) };
        assert_eq!(it[r], expected_it, "{at}: parameter row {r} iteration");
        assert_eq!((pred[r] as usize, oid[r] as usize), (flat / n, flat % n), "{at}: parameter row {r} ids");
        assert_eq!(parameters[flat].to_bits(), val[r].to_bits(), "{at}: parameter row {r} value");
    }

    let written = layer(g, "turbo:perceptron-model");
    let error = written.strs("error").unwrap().get(0);
    match write_perceptron(model) {
        Err(Error::NoParameters) => {
            // PerceptronModelWriter.computeOutcomePatterns on no predicates.
            assert_eq!(error, "java.lang.ArrayIndexOutOfBoundsException", "{at}: the writer failed on the JVM");
            assert_eq!(written.u32s("length").unwrap()[0], 0, "{at}");
            assert!(model.all_parameters().iter().all(|&x| x == 0.0), "{at}: a parameter is not zero");
            cov.perceptron_unwritable += 1;
        }
        Err(e) => panic!("{at}: {e}"),
        Ok(bytes) => {
            assert_eq!(error, "", "{at}: the JVM writer failed: {error}");
            assert_eq!(bytes.len() as u32, written.u32s("length").unwrap()[0], "{at}: model length");
            assert_eq!(
                turbo_maxent::sha256(&bytes).to_hex(),
                written.strs("sha256").unwrap().get(0),
                "{at}: model sha256"
            );
            let read = turbo_maxent::MaxentModel::read(&bytes).unwrap_or_else(|e| panic!("{at}: {e}"));
            assert!(read.num_predicates() <= model.num_predicates(), "{at}");
        }
    }
}

/// A perceptron golden: the events indexed as the run's config says, then
/// `PerceptronTrainer` against the `turbo:perceptron-*` layers: what the
/// monitor heard and the end of training, the parameters after every
/// iteration (bit for bit after the first and the last, by digest in
/// between), and the written model.
fn check_perceptron(path: &Path, g: &Golden, cov: &mut Coverage) {
    let at = path.display().to_string();
    let events = events(g);
    let Some(index) = indexed(&at, g, &events) else { return };
    let training = layer(g, "turbo:perceptron-training");
    let ran = training.u32s("iterations").unwrap()[0] as usize;
    let message = training.strs("message").unwrap().get(0);
    let tp = perceptron_params(g);
    if ran == 0 {
        assert!(index.outcome_labels().len() <= 1, "{at}");
        assert_eq!(
            message,
            "opennlp.tools.util.InsufficientTrainingDataException: Training data must contain more than one outcome",
            "{at}"
        );
        assert_eq!(
            perceptron::train(&index, &tp, &mut TrainingConfiguration::default()).err(),
            Some(Error::InsufficientTrainingData(ONE_OUTCOME)),
            "{at}"
        );
        cov.perceptron_one_outcome += 1;
        return;
    }
    let params = PerceptronParams::from_training_parameters(&tp).unwrap_or_else(|e| panic!("{at}: {e}"));
    let stop = IterDeltaAccuracyUnderTolerance::from_parameters(&tp).unwrap();
    if !params.use_average {
        cov.perceptron_raw += 1;
    }
    if params.use_skipped_averaging {
        cov.perceptron_skipped += 1;
    }
    if params.step_size_decrease.is_some() {
        cov.perceptron_decrease += 1;
    }
    if ran < params.iterations as usize {
        cov.perceptron_stopped_early += 1;
    }
    if index.has_values() {
        cov.perceptron_values += 1;
    }
    if index.outcome_labels().len() > 2 {
        cov.perceptron_many_outcomes += 1;
    }

    let iterations = layer(g, "turbo:perceptron-iterations");
    assert_eq!(iterations.len(), ran, "{at}");
    let digests = iterations.strs("parameters").unwrap();
    let recorded = Rc::new(RefCell::new(Recorded::default()));
    let mut config =
        TrainingConfiguration::new(Some(Box::new(Recorder(recorded.clone(), TrainingMeasure::Accuracy))), None);
    let mut first = Vec::new();
    let mut seen = 0;
    let model = perceptron::train_model(
        &index,
        &params,
        &mut config,
        &stop,
        Some(&mut |it, parameters| {
            let i = it.iteration as usize;
            seen = i;
            assert!(i <= ran, "{at}: iteration {i} after the JVM stopped at {ran}");
            assert_eq!(digest(parameters), digests.get(i - 1), "{at}: parameters after iteration {i}");
            if i == 1 {
                first = parameters.to_vec();
            }
        }),
    )
    .unwrap_or_else(|e| panic!("{at}: {e}"));
    assert_eq!(seen, ran, "{at}: iterations run");

    // The monitor: iterations 1 to 9 and every tenth, then the end.
    let recorded = recorded.borrow();
    let heard = layer(g, "turbo:perceptron-monitor");
    let (iteration, correct, total, accuracy) = (
        heard.u32s("iteration").unwrap(),
        heard.u32s("correct").unwrap(),
        heard.u32s("events").unwrap(),
        heard.f64s("accuracy").unwrap(),
    );
    assert_eq!(recorded.rows.len(), heard.len(), "{at}: iterations the monitor heard of");
    for (r, (it, c, t, acc)) in recorded.rows.iter().enumerate() {
        assert_eq!(*it as u32, iteration[r], "{at}: monitor row {r} iteration");
        assert_eq!(*c as u32, correct[r], "{at}: monitor row {r} correct");
        assert_eq!(*t as u32, total[r], "{at}: monitor row {r} events");
        assert_eq!(acc.to_bits(), accuracy[r].to_bits(), "{at}: monitor row {r} accuracy");
    }
    assert_eq!(recorded.last.as_deref(), Some(message), "{at}: end of training");

    check_perceptron_layers(&at, g, &model, &first, ran, cov);
}

/// `PerceptronTrainGolden.EventSequences`: the events in sequences of five,
/// tagged by evaluating the model on each event's own context.
struct EventSequences(Vec<Sequence>);

impl SequenceSource for EventSequences {
    fn sequences(&self) -> &[Sequence] {
        &self.0
    }

    fn update_context(&self, sequence: &Sequence, model: &PerceptronModel) -> Vec<Event> {
        let mut probabilities = vec![0.0; model.num_outcomes()];
        sequence
            .events
            .iter()
            .map(|e| {
                model.eval(e.context(), e.values(), &mut probabilities);
                // getBestOutcome: the first of the largest.
                let mut best = 0;
                for (i, &p) in probabilities.iter().enumerate() {
                    if p > probabilities[best] {
                        best = i;
                    }
                }
                Event::with_values(
                    model.outcome_labels()[best].clone(),
                    e.context().to_vec(),
                    e.values().map(<[f32]>::to_vec),
                )
                .unwrap()
            })
            .collect()
    }
}

/// A sequence perceptron golden: the run's events cut into sequences as
/// the generator cut them, then `SimplePerceptronSequenceTrainer` against
/// the `turbo:perceptron-*` layers (it has no monitor).
fn check_sequence_perceptron(path: &Path, g: &Golden, cov: &mut Coverage) {
    let at = path.display().to_string();
    let events = events(g);
    let cutoff: i32 = config(g, "Cutoff").parse().unwrap();
    let source = EventSequences(events.chunks(5).map(|c| Sequence { events: c.to_vec() }).collect());
    let Some(index) = indexed(&at, g, &events) else {
        // The sequence trainer indexes the same events with OnePass and the
        // run's cutoff, and meets the same refusal.
        assert_eq!(
            perceptron::train_sequence_model(&source, 1, cutoff, true, None).err(),
            Some(Error::InsufficientTrainingData(NO_EVENTS)),
            "{at}"
        );
        return;
    };
    if index.outcome_labels().len() > 2 {
        cov.sequence_many_outcomes += 1;
    }
    let training = layer(g, "turbo:perceptron-training");
    let ran = training.u32s("iterations").unwrap()[0] as usize;
    let message = training.strs("message").unwrap().get(0);
    let iterations: i32 = config(g, "Iterations").parse().unwrap();
    let use_average: bool = config(g, "UseAverage").parse().unwrap();
    if ran == 0 {
        assert_eq!(message, "java.lang.ArrayIndexOutOfBoundsException", "{at}");
        let index = DataIndex::index(&events, &IndexParams::new(IndexerKind::OnePass, cutoff, false)).unwrap();
        assert_eq!(
            perceptron::train_sequence_model(&source, iterations, cutoff, use_average, None).err(),
            Some(Error::SequenceEventsDropped { indexed: index.num_events(), streamed: events.len() }),
            "{at}"
        );
        cov.sequence_dropped += 1;
        return;
    }
    assert_eq!(message, "", "{at}");
    assert_eq!(ran, iterations as usize, "{at}: the sequence trainer runs every iteration");
    if !use_average {
        cov.sequence_raw += 1;
    }

    let digests = layer(g, "turbo:perceptron-iterations").strs("parameters").unwrap();
    assert_eq!(digests.len(), ran, "{at}");
    let mut first = Vec::new();
    let mut seen = 0;
    let model = perceptron::train_sequence_model(
        &source,
        iterations,
        cutoff,
        use_average,
        Some(&mut |it, parameters| {
            let i = it.iteration as usize;
            seen = i;
            assert_eq!(digest(parameters), digests.get(i - 1), "{at}: parameters after iteration {i}");
            if i == 1 {
                first = parameters.to_vec();
            }
        }),
    )
    .unwrap_or_else(|e| panic!("{at}: {e}"));
    assert_eq!(seen, ran, "{at}: iterations run");
    check_perceptron_layers(&at, g, &model, &first, ran, cov);
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

/// A QN golden: the events indexed as the run's config says, then the
/// QN trainer against the `turbo:qn-*` layers: the parameters after every
/// iteration (bit for bit after the first and the last, by digest in
/// between), the objective value, change rate and training accuracy of
/// every iteration, why training stopped, and the written model.
fn check_qn(path: &Path, g: &Golden, cov: &mut Coverage) {
    let at = path.display().to_string();
    let events = events(g);
    let index = DataIndex::index(&events, &params(g));
    assert!(g.header.error.is_none(), "{at}");
    if let Some(error) = g.layers.iter().find(|l| l.id == "turbo:indexer-error") {
        assert_eq!(error.strs("message").unwrap().get(0), NO_EVENTS, "{at}");
        assert_eq!(index, Err(Error::InsufficientTrainingData(NO_EVENTS)), "{at}");
        return;
    }
    let index = index.unwrap_or_else(|e| panic!("{at}: {e}"));
    let qn = qn_params(g);
    if qn.threads > 1 {
        cov.qn_threads += 1;
    }
    match (qn.l1_cost > 0.0, qn.l2_cost > 0.0) {
        (true, false) => cov.qn_l1 += 1,
        (false, true) => cov.qn_l2 += 1,
        (true, true) => cov.qn_elastic += 1,
        (false, false) => cov.qn_plain += 1,
    }
    if index.outcome_labels().len() > 2 {
        cov.qn_many_outcomes += 1;
    }

    let training = layer(g, "turbo:qn-training");
    let ran = training.u32s("iterations").unwrap()[0] as usize;
    let stop = training.strs("stop").unwrap().get(0);
    if ran == 0 {
        assert_eq!(stop, "refused", "{at}");
        assert!(index.outcome_labels().len() <= 1, "{at}");
        assert_eq!(
            training.strs("message").unwrap().get(0),
            "opennlp.tools.util.InsufficientTrainingDataException: Training data must contain more than one outcome",
            "{at}"
        );
        return;
    }
    assert_eq!(training.strs("message").unwrap().get(0), "", "{at}");
    if ran > qn.updates as usize {
        cov.qn_discarding += 1;
    }

    let iterations = layer(g, "turbo:qn-iterations");
    assert_eq!(iterations.len(), ran, "{at}");
    let (value, rate, accuracy, digests) = (
        iterations.f64s("value").unwrap(),
        iterations.f64s("rate").unwrap(),
        iterations.f64s("accuracy").unwrap(),
        iterations.strs("parameters").unwrap(),
    );
    let (n, f) = (index.outcome_labels().len(), index.pred_labels().len());
    let mut first = Vec::new();
    let mut seen = 0;
    let (model, stopped) = qn::train_model(&index, &qn, &mut |it, point| {
        let i = it.iteration as usize;
        seen = i;
        assert!(i <= ran, "{at}: iteration {i} after the JVM stopped at {ran}");
        assert_eq!(it.value.to_bits(), value[i - 1].to_bits(), "{at}: value after iteration {i}");
        assert_eq!(it.change_rate.to_bits(), rate[i - 1].to_bits(), "{at}: change rate after iteration {i}");
        let acc = qn::training_accuracy(&index, point);
        assert_eq!(acc.to_bits(), accuracy[i - 1].to_bits(), "{at}: accuracy after iteration {i}");
        let parameters = qn::model_parameters(point, n, f, &qn);
        assert_eq!(digest(&parameters), digests.get(i - 1), "{at}: parameters after iteration {i}");
        if i == 1 {
            first = parameters;
        }
    })
    .unwrap_or_else(|e| panic!("{at}: {e}"));
    assert_eq!(seen, ran, "{at}: iterations run");
    assert_eq!(stopped.name(), stop, "{at}: stop");
    cov.qn_stops[match stopped {
        QnStop::ChangeRate => 0,
        QnStop::GradientNorm => 1,
        QnStop::StepSize => 2,
        QnStop::FunctionEvaluations => 3,
        QnStop::Iterations => 4,
    }] += 1;

    // Every parameter after the first and the last iteration, each
    // predicate's per outcome.
    let values = layer(g, "turbo:qn-parameters");
    let (it, pred, oid, val) = (
        values.u32s("iteration").unwrap(),
        values.u32s("predicate").unwrap(),
        values.u32s("outcome").unwrap(),
        values.f64s("value").unwrap(),
    );
    let last = model.all_parameters();
    assert_eq!(last.len(), n * f, "{at}");
    assert_eq!(values.len(), if ran > 1 { 2 * last.len() } else { last.len() }, "{at}: parameter rows");
    for r in 0..values.len() {
        let flat = r % last.len();
        let (expected_it, parameters) = if r < last.len() { (1, &first[..]) } else { (ran as u32, last) };
        assert_eq!(it[r], expected_it, "{at}: parameter row {r} iteration");
        assert_eq!((pred[r] as usize, oid[r] as usize), (flat / n, flat % n), "{at}: parameter row {r} ids");
        assert_eq!(parameters[flat].to_bits(), val[r].to_bits(), "{at}: parameter row {r} value");
    }

    let bytes = write_qn(&model).unwrap_or_else(|e| panic!("{at}: {e}"));
    let written = layer(g, "turbo:qn-model");
    assert_eq!(bytes.len() as u32, written.u32s("length").unwrap()[0], "{at}: model length");
    assert_eq!(turbo_maxent::sha256(&bytes).to_hex(), written.strs("sha256").unwrap().get(0), "{at}: model sha256");
    let read = turbo_maxent::MaxentModel::read(&bytes).unwrap_or_else(|e| panic!("{at}: {e}"));
    assert_eq!(read.num_predicates(), model.num_predicates(), "{at}");
}
