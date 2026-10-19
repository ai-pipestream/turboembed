use super::minimizer::{Search, UpdateInfo};
use super::*;
use crate::index::IndexerKind;

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
    DataIndex::index(&events(), &IndexParams::new(IndexerKind::OnePass, 0, true)).unwrap()
}

#[test]
fn params_come_from_the_training_parameters_and_are_validated() {
    let mut p = TrainingParameters::new();
    assert_eq!(QnParams::from_training_parameters(&p).unwrap(), QnParams::default());
    p.put(Key::ITERATIONS, 7);
    p.put(Key::THREADS, "3");
    p.put(Key::L1_COST, 0.0);
    p.put(Key::L2_COST, "0.5");
    p.put(Key::NUM_OF_UPDATES, 4);
    p.put(Key::MAX_FCT_EVAL, 50);
    assert_eq!(
        QnParams::from_training_parameters(&p).unwrap(),
        QnParams { iterations: 7, threads: 3, l1_cost: 0.0, l2_cost: 0.5, updates: 4, max_fct_eval: 50 }
    );
    p.put(Key::L1_COST, -0.1);
    assert_eq!(QnParams::from_training_parameters(&p), Err(Error::NegativeCost { key: Key::L1_COST, value: -0.1 }));
    p.put(Key::L1_COST, 0.1);
    p.put(Key::THREADS, 0);
    assert_eq!(QnParams::from_training_parameters(&p), Err(Error::NotPositive { key: Key::THREADS, value: 0 }));
    p.put(Key::THREADS, 1);
    p.put(Key::NUM_OF_UPDATES, 0);
    assert_eq!(QnParams::from_training_parameters(&p), Err(Error::NotPositive { key: Key::NUM_OF_UPDATES, value: 0 }));
    p.put(Key::NUM_OF_UPDATES, 1);
    p.put(Key::MAX_FCT_EVAL, -1);
    assert_eq!(QnParams::from_training_parameters(&p), Err(Error::NotPositive { key: Key::MAX_FCT_EVAL, value: -1 }));
    // Iterations are the minimizer's to refuse.
    p.put(Key::MAX_FCT_EVAL, 1);
    p.put(Key::ITERATIONS, 0);
    assert!(QnParams::from_training_parameters(&p).is_ok());
    assert_eq!(train(&index(), &p).err(), Some(Error::NotPositive { key: Key::ITERATIONS, value: 0 }));
}

#[test]
fn one_outcome_is_refused_before_training() {
    let events = [event("s", "a"), event("s", "b")];
    let index = DataIndex::index(&events, &IndexParams::new(IndexerKind::OnePass, 0, true)).unwrap();
    assert_eq!(train(&index, &TrainingParameters::new()).err(), Some(Error::InsufficientTrainingData(ONE_OUTCOME)));
}

#[test]
fn a_bad_parameter_is_refused_before_the_events_are_looked_at() {
    // QNTrainer.init and validate run before the indexer: with one
    // outcome and Threads 0 the JVM reports the threads.
    let events = [event("s", "a"), event("s", "b")];
    let mut p = TrainingParameters::new();
    p.put(Key::THREADS, 0);
    assert_eq!(train_events(&events, &p).err(), Some(Error::NotPositive { key: Key::THREADS, value: 0 }));
    // So does Iterations that does not parse, with no events at all.
    let mut p = TrainingParameters::new();
    p.put(Key::ITERATIONS, "many");
    assert!(matches!(train_events(&[], &p), Err(Error::ParameterFormat { .. })));
}

#[test]
fn an_event_without_values_in_a_real_value_index_is_refused() {
    // The JVM's NegLogLikelihood reads values[ei][ci] and throws
    // NullPointerException on the event without values.
    let events = [
        Event::with_values("yes", vec![JavaString::new("a")], Some(vec![2.0])).unwrap(),
        Event::with_values("no", vec![JavaString::new("b")], None).unwrap(),
    ];
    let index = DataIndex::index(&events, &IndexParams::new(IndexerKind::OnePassRealValue, 0, true)).unwrap();
    assert_eq!(train(&index, &TrainingParameters::new()).err(), Some(Error::MissingValues { event: 1 }));
    // One outcome is reported first, as the JVM checks it first.
    let one = [
        Event::with_values("s", vec![JavaString::new("b")], None).unwrap(),
        Event::with_values("s", vec![JavaString::new("a")], Some(vec![2.0])).unwrap(),
    ];
    let index = DataIndex::index(&one, &IndexParams::new(IndexerKind::OnePassRealValue, 0, true)).unwrap();
    assert_eq!(train(&index, &TrainingParameters::new()).err(), Some(Error::InsufficientTrainingData(ONE_OUTCOME)));
}

#[test]
fn training_fits_the_events() {
    let mut p = TrainingParameters::defaults();
    p.put(Key::CUTOFF, 0);
    let (model, report) = train_events(&events(), &p).unwrap();
    assert_eq!(model.outcome_labels(), ["yes", "no", "maybe"]);
    assert_eq!(model.num_predicates(), 8);
    assert_eq!(model.pattern(3), [0, 1, 2]);
    assert_eq!(model.all_parameters().len(), 24);
    assert_eq!(report.event_hash, Some(crate::event::checksum(&events())));
    let index = index();
    // The elastic net keeps the weights small, but the fit is right.
    let point = {
        // Undo the layout to evaluate the accuracy at the model's point.
        let (n, f) = (model.num_outcomes(), model.num_predicates());
        let mut x = vec![0.0; n * f];
        for pr in 0..f {
            for o in 0..n {
                x[o * f + pr] = model.parameters(pr)[o];
            }
        }
        x
    };
    assert_eq!(training_accuracy(&index, &point), 1.0);
}

#[test]
fn the_observer_sees_every_iteration_and_a_shorter_run_is_a_prefix() {
    let index = index();
    let params = QnParams { l1_cost: 0.0, ..QnParams::default() };
    let mut seen = Vec::new();
    let (model, stop) = train_model(&index, &params, &mut |it, point| {
        seen.push((*it, point.to_vec()));
    })
    .unwrap();
    let ran = seen.len();
    assert!(ran > 3 && ran < 100, "{ran}");
    assert_eq!(stop, QnStop::ChangeRate);
    assert_eq!(seen[ran - 1].0.stop, Some(stop));
    assert!(seen[..ran - 1].iter().all(|(it, _)| it.stop.is_none()));
    assert_eq!(seen.iter().map(|(it, _)| it.iteration).collect::<Vec<_>>(), (1..=ran as i32).collect::<Vec<_>>());
    assert!(seen.windows(2).all(|w| w[1].0.value <= w[0].0.value));
    assert!(seen.windows(2).all(|w| w[1].0.fct_evals > w[0].0.fct_evals));
    assert_eq!(model_parameters(&seen[ran - 1].1, 3, 8, &params), model.all_parameters());
    // A run of two iterations is the first two of the full run.
    let short = QnParams { iterations: 2, ..params };
    let (two, stop) = train_model(&index, &short, &mut |_, _| {}).unwrap();
    assert_eq!(stop, QnStop::Iterations);
    assert_eq!(two.all_parameters(), model_parameters(&seen[1].1, 3, 8, &short));
}

#[test]
fn the_elastic_net_rescales_the_point_at_the_end() {
    let params = QnParams::default();
    let point: Vec<f64> = (0..6).map(|i| i as f64 * 0.5).collect();
    let scaled = model_parameters(&point, 2, 3, &params);
    // Predicate major: x[o * 3 + p].
    assert_eq!(scaled[0], 0.0);
    assert_eq!(scaled[1], 1.1f64.sqrt() * 1.5);
    assert_eq!(scaled[2], 1.1f64.sqrt() * 0.5);
    let l2 = QnParams { l1_cost: 0.0, ..params };
    assert_eq!(model_parameters(&point, 2, 3, &l2), [0.0, 1.5, 0.5, 2.0, 1.0, 2.5]);
}

#[test]
fn chunking_by_threads_changes_only_the_order_of_sums() {
    let index = index();
    let one = train_model(&index, &QnParams::default(), &mut |_, _| {}).unwrap().0;
    let params = QnParams { threads: 3, ..QnParams::default() };
    let three = train_model(&index, &params, &mut |_, _| {}).unwrap().0;
    assert_eq!(one.all_parameters().len(), three.all_parameters().len());
    for (a, b) in one.all_parameters().iter().zip(three.all_parameters()) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
}

#[test]
fn l1_leaves_parameters_at_zero_and_a_low_evaluation_limit_stops_early() {
    let index = index();
    let params = QnParams { l1_cost: 1.0, l2_cost: 0.0, ..QnParams::default() };
    let (model, _) = train_model(&index, &params, &mut |_, _| {}).unwrap();
    assert!(model.all_parameters().contains(&0.0));
    let params = QnParams { max_fct_eval: 2, ..QnParams::default() };
    let (_, stop) = train_model(&index, &params, &mut |_, _| {}).unwrap();
    assert_eq!(stop, QnStop::FunctionEvaluations);
}

#[test]
fn discarded_corrections_alias_as_the_jvm_arrays_do() {
    let mut u = UpdateInfo::new(2, 1);
    let step = |a: f64, b: f64| Search {
        step_size: 1.0,
        value_at_curr: 0.0,
        value_at_next: 0.0,
        grad_at_curr: vec![0.0],
        grad_at_next: vec![b],
        pseudo_grad: vec![],
        curr_point: vec![0.0],
        next_point: vec![a],
        sign: vec![],
        fct_eval_count: 0,
    };
    u.update(&step(1.0, 1.0));
    u.update(&step(2.0, 1.0));
    assert_eq!((u.s[0][0], u.s[1][0]), (1.0, 2.0));
    // The third correction discards the first: slot 0 now names the
    // array of slot 1, and the write into slot 1 shows in both.
    u.update(&step(3.0, 1.0));
    assert_eq!(u.slot, [1, 1]);
    assert_eq!((u.s[u.slot[0]][0], u.s[u.slot[1]][0]), (3.0, 3.0));
    assert_eq!(u.rho, [0.5, 1.0 / 3.0]);
}

#[test]
fn log_sum_of_exps_skips_negative_infinity_and_argmax_takes_the_first() {
    assert_eq!(log_sum_of_exps(&[f64::NEG_INFINITY, 0.0]), 0.0);
    assert_eq!(log_sum_of_exps(&[1.0, 1.0]), 1.0 + strict_math::log(2.0));
    assert_eq!(argmax(&[1.0, 3.0, 3.0]), 1);
    assert!(max_one(f64::NAN).is_nan());
    assert_eq!(max_one(0.5), 1.0);
    assert_eq!(max_one(2.0), 2.0);
}

#[test]
fn non_finite_initial_step_returns_line_search_error_instead_of_looping() {
    let idx = index();
    // When `l1_cost` exceeds every gradient coordinate at the origin, the
    // initial pseudo-gradient is all zero, making `initial_step = 1.0 / 0.0 = +inf`
    // and `0.0 * +inf = NaN`. The constrained line search must return `Error::LineSearch`.
    let big_l1 = QnParams { l1_cost: 100.0, ..QnParams::default() };
    assert_eq!(train_model(&idx, &big_l1, &mut |_, _| {}), Err(Error::LineSearch));

    // When events are balanced across outcomes so the gradient at the origin
    // is zero, `initial_step = +inf` in unconstrained line search too.
    let balanced = [event("a", "f"), event("b", "f")];
    let balanced_idx = DataIndex::index(&balanced, &IndexParams::new(IndexerKind::OnePass, 0, true)).unwrap();
    let no_l1 = QnParams { l1_cost: 0.0, ..QnParams::default() };
    assert_eq!(train_model(&balanced_idx, &no_l1, &mut |_, _| {}), Err(Error::LineSearch));

    // Infinite L2 cost produces a NaN initial gradient and `initial_step = NaN`.
    let inf_l2 = QnParams { l1_cost: 0.0, l2_cost: f64::INFINITY, ..QnParams::default() };
    assert_eq!(train_model(&idx, &inf_l2, &mut |_, _| {}), Err(Error::LineSearch));
}

#[test]
fn huge_thread_count_above_event_count_does_not_allocate_per_thread_empty_chunks() {
    let idx = index();
    let params = QnParams { threads: i32::MAX, iterations: 2, ..QnParams::default() };
    let (model, stop) = train_model(&idx, &params, &mut |_, _| {}).unwrap();
    assert_eq!(stop, QnStop::Iterations);
    assert_eq!(model.all_parameters().len(), 24);
}
