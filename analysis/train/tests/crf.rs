use turbo_sequence::crf::CrfConstraints;
use turbo_sequence::namefind::NameCodec;
use turbo_train::JavaString;
use turbo_train::crf::{CrfObjective, CrfTrainer, CrfTrainingSequence, IndexedSequence};
#[test]
fn complete_gradient_matches_finite_differences_bio_and_bilou() {
    for (codec, labels, gold) in [
        (NameCodec::Bio, vec!["other", "person-start", "person-cont"], vec![1, 2, 0]),
        (NameCodec::Bilou, vec!["other", "person-start", "person-cont", "person-last", "person-unit"], vec![1, 3, 4]),
    ] {
        let labels = labels.into_iter().map(str::to_owned).collect::<Vec<_>>();
        let mask = CrfConstraints::new(&labels, codec).unwrap();
        let sample = IndexedSequence {
            features: vec![vec![(0, 1.0), (1, -0.4)], vec![(1, 0.7)], vec![(0, 1.0), (1, 0.2)]],
            outcomes: gold,
        };
        let objective = CrfObjective::new(2, mask, vec![sample], 0.17).unwrap();
        let mut x: Vec<_> = (0..objective.dimension()).map(|i| (i as f64 * 0.31).sin() * 0.3).collect();
        let mut gradient = vec![0.0; x.len()];
        objective.value_gradient(&x, &mut gradient).unwrap();
        let mut scratch = vec![0.0; x.len()];
        let h = 1e-5;
        for i in 0..x.len() {
            let old = x[i];
            x[i] = old + h;
            let hi = objective.value_gradient(&x, &mut scratch).unwrap();
            x[i] = old - h;
            let lo = objective.value_gradient(&x, &mut scratch).unwrap();
            x[i] = old;
            assert!((gradient[i] - (hi - lo) / (2.0 * h)).abs() < 1e-5, "component {i}");
        }
    }
}
#[test]
fn transition_training_solves_sequence_ambiguous_at_each_position() {
    // Every token has identical emissions; only sequence transitions can
    // distinguish the entity start, continuation, and outside labels.
    let features = vec![vec![(JavaString::new("bias"), 1.0)]; 3];
    let s = CrfTrainingSequence {
        features: features.clone(),
        outcomes: vec!["p-start".into(), "p-cont".into(), "other".into()],
    };
    let trainer = CrfTrainer { l2: 0.001, iterations: 80, ..CrfTrainer::default() };
    let model = trainer.train(&vec![s; 8]).unwrap();
    let path = model.best_sequences(&features, 1).unwrap().remove(0);
    let labels: Vec<_> = path.labels.iter().map(|&i| model.labels()[i].as_str()).collect();
    assert_eq!(labels, ["p-start", "p-cont", "other"]);
    assert!(path.log_probability.exp() > 0.95);
}
#[test]
fn invalid_sequences_and_numeric_features_are_errors() {
    let mut s =
        CrfTrainingSequence { features: vec![vec![(JavaString::new("x"), 1.0)]], outcomes: vec!["p-cont".into()] };
    assert!(CrfTrainer::default().train(&[s.clone()]).is_err());
    s.outcomes[0] = "other".into();
    s.features[0][0].1 = f64::NAN;
    assert!(CrfTrainer::default().train(&[s]).is_err());
}
