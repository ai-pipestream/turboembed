//! The trainer end to end on small event sets: the solvers, the refusals,
//! and the model bytes written and read back.

use super::*;
use crate::JavaString;

/// Three outcomes, each told by its own predicate, with a shared one and
/// some noise; every event's predicates are in string order.
fn events() -> Vec<Event> {
    let mut out = Vec::new();
    for i in 0..30 {
        let (outcome, cue) = match i % 3 {
            0 => ("a", "cue=a"),
            1 => ("b", "cue=b"),
            _ => ("c", "cue=c"),
        };
        let noise = format!("n={}", i % 4);
        let mut context = vec![cue, "bias", noise.as_str()];
        context.sort_unstable();
        out.push(Event::from_strs(outcome, &context));
    }
    out
}

fn params(solver: &str) -> TrainingParameters {
    let mut p = TrainingParameters::new();
    p.put(LiblinearKey::SOLVER_TYPE, solver);
    p.put(LiblinearKey::C, "1");
    p.put(LiblinearKey::EPS, "0.01");
    p.put(LiblinearKey::BIAS, "1");
    p.put(Key::CUTOFF, 0);
    p
}

fn best(model: &LiblinearModel, context: &[&str]) -> String {
    let mut probs = vec![0.0; model.num_outcomes()];
    model.eval(context, &mut probs);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12, "{probs:?}");
    model.outcome(LiblinearModel::best_outcome(&probs)).to_string_lossy()
}

#[test]
fn each_solver_learns_the_cues() {
    for solver in ["L2R_LR", "L2R_LR_DUAL", "L1R_LR"] {
        let (model, report) = train_events(&events(), &params(solver)).unwrap();
        assert!(report.event_hash.is_some());
        assert_eq!(model.linear().solver().name(), solver);
        assert_eq!(model.linear().nr_class(), 3);
        assert_eq!(model.linear().nr_w(), 3);
        assert_eq!(model.linear().bias(), 1.0);
        for o in ["a", "b", "c"] {
            let cue = format!("cue={o}");
            assert_eq!(best(&model, &["bias", &cue]), o, "{solver}");
            assert_eq!(best(&model, &[&cue, "bias", "unknown"]), o, "{solver}");
        }
    }
}

#[test]
fn two_classes_have_one_weight_vector() {
    let events: Vec<Event> = events().into_iter().filter(|e| !e.outcome().eq_str("c")).collect();
    let (model, _) = train_events(&events, &params("L2R_LR")).unwrap();
    assert_eq!(model.linear().nr_class(), 2);
    assert_eq!(model.linear().nr_w(), 1);
    assert_eq!(model.linear().weights().len(), model.linear().w_size());
    assert_eq!(best(&model, &["cue=b"]), "b");
}

#[test]
fn the_model_bytes_read_back() {
    for solver in ["L2R_LR", "L2R_LR_DUAL", "L1R_LR"] {
        let (model, _) = train_events(&events(), &params(solver)).unwrap();
        let bytes = format::write(&model).unwrap();
        let read = format::read(&bytes).unwrap();
        assert_eq!(read, model);
        assert_eq!(format::write(&read).unwrap(), bytes);
        let text = format::save_model(model.linear());
        assert_eq!(&format::load_model(&text).unwrap(), model.linear());
        assert!(text.starts_with(&format!("solver_type {solver}\nnr_class 3\nlabel 0 1 2\n")), "{text}");
    }
}

#[test]
fn the_parameters_are_refused_as_the_add_on_refuses_them() {
    let e = events();
    let refused = |p: &TrainingParameters| train_events(&e, p).unwrap_err().to_string();
    let mut p = params("L2R_LR");
    p.put(LiblinearKey::SOLVER_TYPE, "NOPE");
    assert_eq!(refused(&p), "solverType [NOPE] is not available!");
    let mut none = TrainingParameters::new();
    none.put(LiblinearKey::C, "1");
    assert_eq!(refused(&none), "solverType [] is not available!");
    assert!(matches!(train_events(&e, &params("L2R_L2LOSS_SVC")), Err(Error::NotScorable(SolverType::L2rL2lossSvc))));
    let mut p = params("L2R_LR");
    p.put(LiblinearKey::C, "0");
    assert_eq!(refused(&p), "C must not be <= 0");
    let mut p = params("L2R_LR");
    p.put(LiblinearKey::EPS, "-1");
    assert_eq!(refused(&p), "eps must not be <= 0");
    let mut p = params("L2R_LR");
    p.put(LiblinearKey::P, "-0.5");
    assert_eq!(refused(&p), "p must not be less than 0");
}

#[test]
fn one_outcome_and_unsorted_events_are_refused() {
    let one: Vec<Event> = events().into_iter().filter(|e| e.outcome().eq_str("a")).collect();
    assert_eq!(train_events(&one, &params("L2R_LR")).unwrap_err().to_string(), crate::error::ONE_OUTCOME);

    let mut e = events();
    e.push(Event::from_strs("a", &["n=1", "cue=a"]));
    assert!(matches!(train_events(&e, &params("L2R_LR")), Err(Error::Unsorted { .. })));
    let mut e = events();
    e.push(Event::from_strs("a", &["cue=a", "cue=a"]));
    assert!(matches!(train_events(&e, &params("L2R_LR")), Err(Error::Unsorted { .. })));
}

#[test]
fn no_bias_feature_when_bias_is_negative() {
    let mut p = params("L2R_LR");
    p.put(LiblinearKey::BIAS, "-1");
    let (model, _) = train_events(&events(), &p).unwrap();
    assert_eq!(model.linear().w_size(), model.linear().nr_feature());
    assert!(format::save_model(model.linear()).contains("\nbias -1.0000000000000000\n"));
}

fn sample() -> Vec<u8> {
    let (model, _) = train_events(&events(), &params("L2R_LR")).unwrap();
    format::write(&model).unwrap()
}

/// The sample's bytes with the text model replaced.
fn with_text(text: &str) -> Vec<u8> {
    let bytes = sample();
    let len = i32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
    let mut out = (text.len() as i32).to_be_bytes().to_vec();
    out.extend_from_slice(text.as_bytes());
    out.extend_from_slice(&bytes[4 + len..]);
    out
}

#[test]
fn the_reader_refuses_what_serialize_does_not_write() {
    let bytes = sample();
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert_eq!(format::read(&trailing), Err(ModelError::Trailing(1)));
    assert!(matches!(format::read(&bytes[..bytes.len() - 1]), Err(ModelError::Data(_))));

    let len = i32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
    let text = std::str::from_utf8(&bytes[4..4 + len]).unwrap();
    let svc = text.replacen("L2R_LR", "L2R_L2LOSS_SVC", 1);
    assert!(matches!(format::read(&with_text(&svc)), Err(ModelError::Invalid(_))));
    let swapped = text.replacen("nr_class 3\nlabel 0 1 2", "label 0 1 2\nnr_class 3", 1);
    assert!(matches!(format::read(&with_text(&swapped)), Err(ModelError::Line { line: 2, .. })));
    let two_spaces = text.replacen("bias ", "bias  ", 1);
    assert!(matches!(format::read(&with_text(&two_spaces)), Err(ModelError::Line { .. })));
    let unended = text.strip_suffix('\n').unwrap();
    assert!(matches!(format::read(&with_text(unended)), Err(ModelError::Invalid(_))));
    let no_space = text.strip_suffix(" \n").unwrap().to_owned() + "\n";
    assert!(matches!(format::read(&with_text(&no_space)), Err(ModelError::Line { .. })));
    let short_row = text.replacen(" \n", "\n", 1);
    assert!(matches!(format::read(&with_text(&short_row)), Err(ModelError::Line { .. })));
    let extra_row = text.to_owned() + "0 0 0 \n";
    assert!(matches!(format::read(&with_text(&extra_row)), Err(ModelError::Line { .. })));
    let nan = text.replacen("bias 1.0000000000000000", "bias NaN", 1);
    assert!(matches!(format::read(&with_text(&nan)), Err(ModelError::Line { line: 5, .. })));
}

#[test]
fn the_parts_are_checked() {
    let (model, _) = train_events(&events(), &params("L2R_LR")).unwrap();
    let linear = model.linear().clone();
    let outcomes = model.outcomes().to_vec();
    let p = |s: &str, id| (JavaString::new(s), id);
    assert!(matches!(
        LiblinearModel::from_parts(linear.clone(), outcomes.clone(), vec![p("x", 0), p("x", 1)]),
        Err(ModelError::DuplicatePredicate(_))
    ));
    assert!(matches!(
        LiblinearModel::from_parts(linear.clone(), outcomes.clone(), vec![p("x", -1)]),
        Err(ModelError::PredicateId { id: -1, .. })
    ));
    assert!(matches!(
        LiblinearModel::from_parts(linear.clone(), outcomes[..2].to_vec(), vec![]),
        Err(ModelError::Invalid(_))
    ));
    let rebuilt = LiblinearModel::from_parts(linear, outcomes, model.predicates().to_vec()).unwrap();
    assert_eq!(rebuilt, model);
}

#[test]
fn eval_skips_unknown_features_and_counts_repeats() {
    let (model, _) = train_events(&events(), &params("L2R_LR")).unwrap();
    let mut once = vec![0.0; 3];
    let mut unknown = vec![0.0; 3];
    let mut twice = vec![0.0; 3];
    model.eval(&["cue=a"], &mut once);
    model.eval(&["cue=a", "nothing", "\u{1F600}"], &mut unknown);
    model.eval(&["cue=a", "cue=a"], &mut twice);
    assert_eq!(once, unknown);
    assert!(twice[0] > once[0]);
    let mut by_bytes = vec![0.0; 3];
    model.eval_modified_utf8(&[b"cue=a"], &mut by_bytes);
    assert_eq!(once, by_bytes);
}
