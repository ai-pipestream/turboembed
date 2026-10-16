use turbo_train::error::{Error, ONE_OUTCOME};
use turbo_train::event::Event;
use turbo_train::gis::{GisParams, Smoothing, train, train_events, train_model};
use turbo_train::index::{DataIndex, IndexParams, IndexerKind};
use turbo_train::java_string::JavaString;
use turbo_train::monitor::{DefaultProgressMonitor, LogLikelihoodThresholdBreached, TrainingConfiguration};
use turbo_train::params::{Key, TrainingParameters};

fn event(outcome: &str, context: &str) -> Event {
    Event::from_strs(outcome, &context.split(' ').collect::<Vec<_>>())
}

fn events() -> Vec<Event> {
    vec![
        event("yes", "w=a len=1 cap"),
        event("yes", "w=a len=2"),
        event("yes", "w=b cap"),
        event("no", "w=c len=1"),
        event("no", "w=c len=3"),
        event("no", "w=d len=3"),
        event("maybe", "w=b len=2"),
        event("maybe", "w=d cap"),
    ]
}

#[test]
fn params_come_from_the_training_parameters() {
    let mut p = TrainingParameters::new();
    assert_eq!(GisParams::from_training_parameters(&p).unwrap(), GisParams::default());
    p.put(Key::ITERATIONS, 7);
    p.put(Key::THREADS, "3");
    p.put(Key::SMOOTHING, true);
    let g = GisParams::from_training_parameters(&p).unwrap();
    assert_eq!(g, GisParams { iterations: 7, threads: 3, smoothing: Smoothing::Simple { observation: 0.1 } });
    p.put(Key::GAUSSIAN_SMOOTHING, "true");
    assert_eq!(GisParams::from_training_parameters(&p), Err(Error::BothSmoothings));
    p.put(Key::SMOOTHING, false);
    p.put(Key::GAUSSIAN_SMOOTHING_SIGMA, 0.5);
    assert_eq!(GisParams::from_training_parameters(&p).unwrap().smoothing, Smoothing::Gaussian { sigma: 0.5 });
    p.put(Key::THREADS, 0);
    assert_eq!(GisParams::from_training_parameters(&p), Err(Error::Threads(0)));
}

#[test]
fn one_outcome_is_refused_before_training() {
    let events = [event("s", "a"), event("s", "b")];
    let index = DataIndex::index(&events, &IndexParams::new(IndexerKind::OnePass, 0, true)).unwrap();
    let mut config = TrainingConfiguration::default();
    assert_eq!(
        train(&index, &TrainingParameters::new(), &mut config).err(),
        Some(Error::InsufficientTrainingData(ONE_OUTCOME))
    );
}

#[test]
fn iterations_must_parse_before_the_events_are_looked_at() {
    // AbstractTrainer.validate runs before the indexer: with no events
    // and Iterations that does not parse, the JVM reports Iterations.
    let mut p = TrainingParameters::new();
    p.put(Key::ITERATIONS, "many");
    let mut config = TrainingConfiguration::default();
    assert!(matches!(train_events(&[], &p, &mut config), Err(Error::ParameterFormat { .. })));
    // Threads is read in doTrain, after the outcome check.
    let events = [event("s", "a"), event("s", "b")];
    let mut p = TrainingParameters::new();
    p.put(Key::THREADS, 0);
    p.put(Key::CUTOFF, 0);
    assert_eq!(train_events(&events, &p, &mut config).err(), Some(Error::InsufficientTrainingData(ONE_OUTCOME)));
}

#[test]
fn negative_cutoff_minus_one_keeps_singletons() {
    let mut p = TrainingParameters::defaults();
    p.put(Key::CUTOFF, -1);
    p.put(Key::ITERATIONS, 5);
    let mut config = TrainingConfiguration::default();
    let (model, _) = train_events(&events(), &p, &mut config).unwrap();
    assert_eq!(model.num_predicates(), 8);
}

#[test]
fn an_event_without_values_in_a_real_value_index_is_refused() {
    // GISTrainer reads values[ei][ci] while incorporating the index and
    // throws NullPointerException on the event without values.
    let events = [
        Event::with_values("yes", vec![JavaString::new("a")], Some(vec![2.0])).unwrap(),
        Event::with_values("no", vec![JavaString::new("b")], None).unwrap(),
    ];
    let index = DataIndex::index(&events, &IndexParams::new(IndexerKind::OnePassRealValue, 0, true)).unwrap();
    let mut config = TrainingConfiguration::default();
    assert_eq!(train(&index, &TrainingParameters::new(), &mut config).err(), Some(Error::MissingValues { event: 1 }));
}

#[test]
fn training_fits_the_events_and_reports_every_iteration() {
    let mut p = TrainingParameters::defaults();
    p.put(Key::CUTOFF, 0);
    p.put(Key::ITERATIONS, 30);
    let mut config = TrainingConfiguration::new(Some(Box::new(DefaultProgressMonitor::new())), None);
    let (model, report) = train_events(&events(), &p, &mut config).unwrap();
    assert_eq!(model.outcome_labels(), ["yes", "no", "maybe"]);
    assert_eq!(model.num_predicates(), 8);
    // "cap" was seen with "yes" and "maybe" only.
    let cap = model.pred_labels().iter().position(|l| *l == "cap").unwrap();
    assert_eq!(model.pattern(cap), [0, 2]);
    assert_eq!(model.parameters(cap).len(), 2);
    assert_eq!(report.event_hash, Some(turbo_train::event::checksum(&events())));
    assert_eq!(report.manifest_entries()[1], ("TrainerType", "Event".to_string()));
    let monitor = config.monitor.unwrap();
    assert!(monitor.is_training_finished());
    assert!(model.all_parameters().iter().all(|&x| x != 0.0));
}

#[test]
fn the_observer_sees_the_parameters_after_each_iteration() {
    let index = DataIndex::index(&events(), &IndexParams::new(IndexerKind::OnePass, 0, true)).unwrap();
    let mut seen = Vec::new();
    let params = GisParams { iterations: 5, ..GisParams::default() };
    let stop = LogLikelihoodThresholdBreached::default();
    let model = train_model(&index, &params, &mut TrainingConfiguration::default(), &stop, &mut |i, p| {
        seen.push((i, p.to_vec()));
    })
    .unwrap();
    assert_eq!(seen.iter().map(|(i, _)| *i).collect::<Vec<_>>(), [1, 2, 3, 4, 5]);
    assert_eq!(seen[4].1, model.all_parameters());
    // A run of two iterations is the first two of a run of five.
    let params = GisParams { iterations: 2, ..GisParams::default() };
    let two = train_model(&index, &params, &mut TrainingConfiguration::default(), &stop, &mut |_, _| {}).unwrap();
    assert_eq!(two.all_parameters(), &seen[1].1[..]);
}

#[test]
fn a_thread_count_under_one_is_refused() {
    let index = DataIndex::index(&events(), &IndexParams::new(IndexerKind::OnePass, 0, true)).unwrap();
    let stop = LogLikelihoodThresholdBreached::default();
    for threads in [0, -3] {
        let params = GisParams { threads, ..GisParams::default() };
        let result = train_model(&index, &params, &mut TrainingConfiguration::default(), &stop, &mut |_, _| {});
        assert_eq!(result.err(), Some(Error::Threads(threads)));
    }
}

#[test]
fn chunking_by_threads_changes_only_the_order_of_sums() {
    let index = DataIndex::index(&events(), &IndexParams::new(IndexerKind::OnePass, 0, true)).unwrap();
    let stop = LogLikelihoodThresholdBreached::default();
    let one = train_model(&index, &GisParams::default(), &mut TrainingConfiguration::default(), &stop, &mut |_, _| {})
        .unwrap();
    let params = GisParams { threads: 3, ..GisParams::default() };
    let three = train_model(&index, &params, &mut TrainingConfiguration::default(), &stop, &mut |_, _| {}).unwrap();
    for (a, b) in one.all_parameters().iter().zip(three.all_parameters()) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }
}

#[test]
fn simple_smoothing_gives_every_predicate_every_outcome() {
    let index = DataIndex::index(&events(), &IndexParams::new(IndexerKind::OnePass, 0, true)).unwrap();
    let params = GisParams { iterations: 3, smoothing: Smoothing::Simple { observation: 0.1 }, ..GisParams::default() };
    let stop = LogLikelihoodThresholdBreached::default();
    let model = train_model(&index, &params, &mut TrainingConfiguration::default(), &stop, &mut |_, _| {}).unwrap();
    for p in 0..model.num_predicates() {
        assert_eq!(model.pattern(p), [0, 1, 2]);
    }
}
