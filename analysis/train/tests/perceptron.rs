use turbo_train::error::{Error, ONE_OUTCOME};
use turbo_train::event::Event;
use turbo_train::index::{DataIndex, IndexParams, IndexerKind};
use turbo_train::java_string::JavaString;
use turbo_train::monitor::{
    DefaultProgressMonitor, IterDeltaAccuracyUnderTolerance, ProgressMonitor, StopCriteria, TrainingConfiguration,
    TrainingMeasure,
};
use turbo_train::params::{Key, TrainerType, TrainingParameters};
use turbo_train::perceptron::{
    PerceptronModel, PerceptronParams, Sequence, SequenceSource, train, train_events, train_model,
    train_sequence_model, train_sequences,
};

fn argmax(x: &[f64]) -> usize {
    let mut max = 0;
    for (i, &v) in x.iter().enumerate().skip(1) {
        if x[max] < v {
            max = i;
        }
    }
    max
}

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

fn index() -> DataIndex {
    DataIndex::index(&events(), &IndexParams::new(IndexerKind::OnePass, 0, false)).unwrap()
}

#[test]
fn params_come_from_the_training_parameters_and_are_validated() {
    let mut p = TrainingParameters::new();
    assert_eq!(PerceptronParams::from_training_parameters(&p).unwrap(), PerceptronParams::default());
    p.put(Key::USE_AVERAGE, false);
    p.put(Key::STEP_SIZE_DECREASE, 0.5);
    p.put(Key::TOLERANCE, "0.1");
    p.put(Key::ITERATIONS, 3);
    assert_eq!(
        PerceptronParams::from_training_parameters(&p).unwrap(),
        PerceptronParams {
            iterations: 3,
            use_average: false,
            use_skipped_averaging: false,
            step_size_decrease: Some(0.5),
            tolerance: 0.1
        }
    );
    // Skipped averaging forces averaging on.
    p.put(Key::USE_SKIPPED_AVERAGING, true);
    let got = PerceptronParams::from_training_parameters(&p).unwrap();
    assert!(got.use_average && got.use_skipped_averaging);
    p.put(Key::STEP_SIZE_DECREASE, 101.0);
    assert_eq!(PerceptronParams::from_training_parameters(&p), Err(Error::StepSizeDecrease(101.0)));
    p.put(Key::STEP_SIZE_DECREASE, -1.0);
    assert_eq!(PerceptronParams::from_training_parameters(&p).unwrap().step_size_decrease, None);
    p.put(Key::TOLERANCE, -0.5);
    assert_eq!(PerceptronParams::from_training_parameters(&p), Err(Error::Tolerance(-0.5)));
}

#[test]
fn one_outcome_is_refused_before_training() {
    let events = [event("s", "a"), event("s", "b")];
    let index = DataIndex::index(&events, &IndexParams::new(IndexerKind::OnePass, 0, false)).unwrap();
    assert_eq!(
        train(&index, &TrainingParameters::new(), &mut TrainingConfiguration::default()).err(),
        Some(Error::InsufficientTrainingData(ONE_OUTCOME))
    );
}

#[test]
fn training_fits_the_events_and_reports_some_iterations() {
    let mut p = TrainingParameters::defaults();
    p.put(Key::CUTOFF, 0);
    p.put(Key::ITERATIONS, 100);
    p.put(Key::TOLERANCE, 0.0);
    // The raw parameters: the mean over the iterations can leave an
    // event wrongly tagged that the last iteration's parameters tag right.
    p.put(Key::USE_AVERAGE, false);
    let mut config = TrainingConfiguration::new(Some(Box::new(DefaultProgressMonitor::new())), None);
    let (model, report) = train_events(&events(), &p, &mut config).unwrap();
    assert_eq!(model.outcome_labels(), ["yes", "no", "maybe"]);
    assert_eq!(model.num_predicates(), 8);
    assert_eq!(model.pattern(0), [0, 1, 2]);
    assert_eq!(report.trainer_type, TrainerType::Event);
    let mut out = vec![0.0; 3];
    for e in events() {
        model.eval(e.context(), None, &mut out);
        let best = &model.outcome_labels()[argmax(&out)];
        assert_eq!(best, e.outcome(), "{e:?}: {out:?}");
    }
    assert!(config.monitor.unwrap().is_training_finished());
}

#[test]
#[should_panic(expected = "the output must hold one probability per outcome")]
fn eval_panics_when_output_slice_length_differs_from_num_outcomes() {
    let stop = IterDeltaAccuracyUnderTolerance::default();
    let model = train_model(&index(), &PerceptronParams::default(), &mut TrainingConfiguration::default(), &stop, None)
        .unwrap();
    let mut out = [0.0; 2];
    model.eval(&[JavaString::new("w=a")], None, &mut out);
}

#[test]
#[should_panic(expected = "one value per context element")]
fn eval_panics_when_values_slice_is_shorter_than_context() {
    let stop = IterDeltaAccuracyUnderTolerance::default();
    let model = train_model(&index(), &PerceptronParams::default(), &mut TrainingConfiguration::default(), &stop, None)
        .unwrap();
    let mut out = [0.0; 3];
    model.eval(&[JavaString::new("w=a"), JavaString::new("cap")], Some(&[1.0]), &mut out);
}

#[test]
fn the_monitor_hears_of_the_first_nine_and_every_tenth_iteration() {
    use std::cell::RefCell;
    use std::rc::Rc;
    struct Heard(Rc<RefCell<Vec<i32>>>);
    impl ProgressMonitor for Heard {
        fn finished_iteration(&mut self, i: i32, _: i32, _: i32, m: TrainingMeasure, _: f64) {
            assert_eq!(m, TrainingMeasure::Accuracy);
            self.0.borrow_mut().push(i);
        }
        fn finished_training(&mut self, _: i32, _: Option<&dyn StopCriteria>) {}
        fn is_training_finished(&self) -> bool {
            false
        }
        fn display(&mut self, _: bool) {}
    }
    let heard = Rc::new(RefCell::new(Vec::new()));
    let params = PerceptronParams { iterations: 25, tolerance: 0.0, ..PerceptronParams::default() };
    let mut config = TrainingConfiguration::new(Some(Box::new(Heard(Rc::clone(&heard)))), None);
    let stop = IterDeltaAccuracyUnderTolerance { tolerance: 0.0 };
    let mut seen = Vec::new();
    train_model(&index(), &params, &mut config, &stop, Some(&mut |it, _| seen.push(it.iteration))).unwrap();
    // The observer hears every iteration; the monitor only the first
    // nine and every tenth.
    assert_eq!(seen, (1..=25).collect::<Vec<_>>());
    assert_eq!(*heard.borrow(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]);
}

#[test]
fn a_shorter_run_is_a_prefix_and_averaging_is_the_mean() {
    let index = index();
    let stop = IterDeltaAccuracyUnderTolerance { tolerance: 0.0 };
    let raw = PerceptronParams { iterations: 6, use_average: false, tolerance: 0.0, ..PerceptronParams::default() };
    let mut raws = Vec::new();
    let model = train_model(
        &index,
        &raw,
        &mut TrainingConfiguration::default(),
        &stop,
        Some(&mut |_, p| {
            raws.push(p.to_vec());
        }),
    )
    .unwrap();
    assert_eq!(model.all_parameters(), &raws[5][..]);
    let short = PerceptronParams { iterations: 2, ..raw };
    let two = train_model(&index, &short, &mut TrainingConfiguration::default(), &stop, None).unwrap();
    assert_eq!(two.all_parameters(), &raws[1][..]);
    // The averaged model is the mean of the raw parameters.
    let averaged = PerceptronParams { use_average: true, ..raw };
    let mut means = Vec::new();
    let model = train_model(
        &index,
        &averaged,
        &mut TrainingConfiguration::default(),
        &stop,
        Some(&mut |_, p| {
            means.push(p.to_vec());
        }),
    )
    .unwrap();
    for k in 0..6 {
        let mut sum = vec![0.0; raws[0].len()];
        for r in &raws[..=k] {
            for (s, x) in sum.iter_mut().zip(r) {
                *s += x;
            }
        }
        let mean: Vec<f64> = sum.iter().map(|s| s / (k + 1) as f64).collect();
        assert_eq!(means[k], mean, "after iteration {}", k + 1);
    }
    assert_eq!(model.all_parameters(), &means[5][..]);
    // Skipped averaging changes nothing.
    let skipped = PerceptronParams { use_skipped_averaging: true, ..averaged };
    let same = train_model(&index, &skipped, &mut TrainingConfiguration::default(), &stop, None).unwrap();
    assert_eq!(same, model);
}

#[test]
fn the_tolerance_stops_after_three_steady_iterations() {
    let index = index();
    let params = PerceptronParams { iterations: 100, ..PerceptronParams::default() };
    let stop = IterDeltaAccuracyUnderTolerance::default();
    let mut config = TrainingConfiguration::new(Some(Box::new(DefaultProgressMonitor::new())), None);
    let mut accuracies = Vec::new();
    train_model(&index, &params, &mut config, &stop, Some(&mut |it, _| accuracies.push(it.accuracy))).unwrap();
    let n = accuracies.len();
    assert!((4..100).contains(&n), "{n}");
    let last = accuracies[n - 1];
    assert!(accuracies[n - 4..n - 1].iter().all(|a| (a - last).abs() < stop.tolerance));
}

#[test]
fn the_step_decreases_by_the_fraction() {
    let index = index();
    let stop = IterDeltaAccuracyUnderTolerance { tolerance: 0.0 };
    let params = PerceptronParams {
        iterations: 1,
        use_average: false,
        step_size_decrease: Some(0.5),
        tolerance: 0.0,
        ..PerceptronParams::default()
    };
    let model = train_model(&index, &params, &mut TrainingConfiguration::default(), &stop, None).unwrap();
    // One iteration with step 0.5: every parameter is a multiple of 0.5.
    assert!(model.all_parameters().iter().all(|p| (p * 2.0).fract() == 0.0));
    assert!(model.all_parameters().iter().any(|&p| p == 0.5 || p == -0.5));
}

/// Sequences of two events, tagged by the model on each event's context.
struct Pairs(Vec<Sequence>);

impl SequenceSource for Pairs {
    fn sequences(&self) -> &[Sequence] {
        &self.0
    }

    fn update_context(&self, sequence: &Sequence, model: &PerceptronModel) -> Vec<Event> {
        let mut out = vec![0.0; model.num_outcomes()];
        sequence
            .events
            .iter()
            .map(|e| {
                model.eval(e.context(), e.values(), &mut out);
                Event::with_values(
                    model.outcome_labels()[argmax(&out)].clone(),
                    e.context().to_vec(),
                    e.values().map(<[f32]>::to_vec),
                )
                .unwrap()
            })
            .collect()
    }
}

fn pairs() -> Pairs {
    Pairs(events().chunks(2).map(|c| Sequence { events: c.to_vec() }).collect())
}

#[test]
fn the_sequence_trainer_fits_the_sequences() {
    let mut p = TrainingParameters::new();
    p.put(Key::CUTOFF, 0);
    p.put(Key::ITERATIONS, 10);
    let (model, report) = train_sequences(&pairs(), &p).unwrap();
    assert_eq!(report.trainer_type, TrainerType::EventModelSequence);
    assert_eq!(report.event_hash, None);
    assert_eq!(model.num_predicates(), 8);
    let source = pairs();
    for s in source.sequences() {
        let tagged = source.update_context(s, &model);
        for (t, e) in tagged.iter().zip(&s.events) {
            assert_eq!(t.outcome(), e.outcome());
        }
    }
    // Without averaging the parameters are whole counts.
    p.put(Key::USE_AVERAGE, false);
    let (raw, _) = train_sequences(&pairs(), &p).unwrap();
    assert!(raw.all_parameters().iter().all(|x| x.fract() == 0.0));
    assert_ne!(raw, model);
}

#[test]
fn iterations_must_parse_before_the_events_are_looked_at() {
    // AbstractTrainer.validate runs before the indexer: with no events
    // and Iterations that does not parse, the JVM reports Iterations.
    let mut p = TrainingParameters::new();
    p.put(Key::ITERATIONS, "many");
    let mut config = TrainingConfiguration::default();
    assert!(matches!(train_events(&[], &p, &mut config), Err(Error::ParameterFormat { .. })));
    // The tolerance is checked in doTrain, after the outcome check.
    let events = [event("s", "a"), event("s", "b")];
    let mut p = TrainingParameters::new();
    p.put(Key::TOLERANCE, -1.0);
    p.put(Key::CUTOFF, 0);
    assert_eq!(train_events(&events, &p, &mut config).err(), Some(Error::InsufficientTrainingData(ONE_OUTCOME)));
}

#[test]
fn an_event_without_values_in_a_real_value_index_is_refused() {
    // PerceptronTrainer reads values[ei][ci] in its first iteration and
    // throws NullPointerException on the event without values.
    let events = [
        Event::with_values("yes", vec![JavaString::new("a")], Some(vec![2.0])).unwrap(),
        Event::with_values("no", vec![JavaString::new("b")], None).unwrap(),
    ];
    let index = DataIndex::index(&events, &IndexParams::new(IndexerKind::OnePassRealValue, 0, false)).unwrap();
    let mut config = TrainingConfiguration::default();
    assert_eq!(train(&index, &TrainingParameters::new(), &mut config).err(), Some(Error::MissingValues { event: 1 }));
    // The tolerance is checked first, as the JVM checks it in doTrain.
    let mut p = TrainingParameters::new();
    p.put(Key::TOLERANCE, -1.0);
    assert_eq!(train(&index, &p, &mut config).err(), Some(Error::Tolerance(-1.0)));
}

#[test]
fn no_iteration_with_averaging_gives_nan_parameters_as_the_jvm_does() {
    // PerceptronTrainer divides the summed parameters by the number of
    // times they were summed, 0.0 / 0, so Iterations 0 (or less) with
    // UseAverage gives a model of NaN parameters; without averaging the
    // zero parameters stay.
    let stop = IterDeltaAccuracyUnderTolerance::default();
    let averaged = PerceptronParams { iterations: 0, ..PerceptronParams::default() };
    let model = train_model(&index(), &averaged, &mut TrainingConfiguration::default(), &stop, None).unwrap();
    assert!(model.all_parameters().iter().all(|p| p.is_nan()));
    let raw = PerceptronParams { iterations: -3, use_average: false, ..PerceptronParams::default() };
    let model = train_model(&index(), &raw, &mut TrainingConfiguration::default(), &stop, None).unwrap();
    assert!(model.all_parameters().iter().all(|&p| p == 0.0));
    // The writer then has nothing to write.
    assert_eq!(turbo_train::writer::write_perceptron(&model).err(), Some(Error::NoParameters));
}

/// A source whose tagging misbehaves.
struct Bad(Vec<Sequence>, bool);

impl SequenceSource for Bad {
    fn sequences(&self) -> &[Sequence] {
        &self.0
    }

    fn update_context(&self, sequence: &Sequence, _: &PerceptronModel) -> Vec<Event> {
        if self.1 {
            sequence.events.iter().map(|e| Event::new("elsewhere", e.context().to_vec())).collect()
        } else {
            Vec::new()
        }
    }
}

#[test]
fn a_source_that_tags_wrongly_is_refused() {
    let sequences: Vec<Sequence> = events().chunks(2).map(|c| Sequence { events: c.to_vec() }).collect();
    assert_eq!(
        train_sequence_model(&Bad(sequences.clone(), false), 1, 0, true, None).err(),
        Some(Error::SequenceLength { expected: 2, got: 0 })
    );
    assert_eq!(
        train_sequence_model(&Bad(sequences, true), 1, 0, true, None).err(),
        Some(Error::UnknownOutcome("elsewhere".to_string()))
    );
}

#[test]
fn the_sequence_trainer_fails_as_the_jvm_does_when_the_cutoff_drops_an_event() {
    // Only "cap" is seen three times, so cutoff 3 drops every event
    // without it: five of the eight.
    let mut p = TrainingParameters::new();
    p.put(Key::CUTOFF, 3);
    p.put(Key::ITERATIONS, 2);
    let index = DataIndex::index(&events(), &IndexParams::new(IndexerKind::OnePass, 3, false)).unwrap();
    assert_eq!(index.num_events(), 3);
    assert_eq!(train_sequences(&pairs(), &p).err(), Some(Error::SequenceEventsDropped { indexed: 3, streamed: 8 }));
}

#[test]
fn a_shorter_sequence_run_is_a_prefix() {
    let source = pairs();
    let mut seen = Vec::new();
    let model =
        train_sequence_model(&source, 5, 0, true, Some(&mut |it, p| seen.push((it.iteration, p.to_vec())))).unwrap();
    assert_eq!(seen.len(), 5);
    assert_eq!(model.all_parameters(), &seen[4].1[..]);
    let three = train_sequence_model(&source, 3, 0, true, None).unwrap();
    assert_eq!(three.all_parameters(), &seen[2].1[..]);
}
