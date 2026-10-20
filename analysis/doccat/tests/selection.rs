use turbo_doccat::{DoccatFactory, DoccatTrainer, FeatureSelection, SelectionMetric};
use turbo_formats::native::DocumentReader;
use turbo_train::{Event, JavaString};
#[test]
fn informative_features_beat_noise_and_ties_are_deterministic() {
    let events = vec![
        Event::from_strs("a", &["signal_a", "all", "noise"]),
        Event::from_strs("a", &["signal_a", "all"]),
        Event::from_strs("b", &["signal_b", "all", "noise"]),
        Event::from_strs("b", &["signal_b", "all"]),
    ];
    for metric in [SelectionMetric::MutualInformation, SelectionMetric::Chi2] {
        let selected = FeatureSelection { metric, top_k: 2 }.fit(&events).unwrap();
        assert_eq!(
            selected.kept().iter().cloned().collect::<Vec<_>>(),
            vec![JavaString::new("signal_a"), JavaString::new("signal_b")]
        );
        let expected = if metric == SelectionMetric::MutualInformation { 2f64.ln() } else { 4. };
        assert!((selected.ranked()[0].1 - expected).abs() < 1e-12);
        assert_eq!(selected.ranked()[2].1, 0.);
        let mut reversed = events.clone();
        reversed.reverse();
        assert_eq!(selected.ranked(), FeatureSelection { metric, top_k: 2 }.fit(&reversed).unwrap().ranked());
    }
}
#[test]
fn duplicates_affect_training_values_but_not_presence_ranking() {
    let plain = vec![Event::from_strs("a", &["x"]), Event::from_strs("b", &["y"])];
    let dup = vec![
        Event::with_values(
            "a",
            vec![JavaString::new("x"), JavaString::new("x"), JavaString::new("noise")],
            Some(vec![2., 3., 0.]),
        )
        .unwrap(),
        plain[1].clone(),
    ];
    let selector = FeatureSelection { metric: SelectionMetric::MutualInformation, top_k: 1 };
    let selected = selector.fit(&dup).unwrap();
    assert_eq!(selected.ranked(), selector.fit(&plain).unwrap().ranked());
    let filtered = selected.filter(&dup).unwrap();
    assert_eq!(filtered[0].context().len(), 2);
    assert_eq!(filtered[0].values(), Some([2., 3.].as_slice()));
    assert!(filtered[1].context().is_empty());
    assert_eq!(filtered[1].outcome(), plain[1].outcome());
}
#[test]
fn trainer_prepares_selected_bow_and_ngram_events() {
    use turbo_doccat::{BuiltinFeatureGenerator, NGramFeatureGenerator};
    let samples: Vec<_> =
        DocumentReader::new("a\tred fox\na\tred fox\nb\tblue bird\nb\tblue bird\n", Default::default())
            .unwrap()
            .map(Result::unwrap)
            .collect();
    let factory =
        DoccatFactory::with_feature_generators(vec![BuiltinFeatureGenerator::NGram(NGramFeatureGenerator::default())]);
    let trainer = DoccatTrainer::new(factory)
        .with_feature_selection(FeatureSelection { metric: SelectionMetric::Chi2, top_k: 1 });
    let events = trainer.events(&samples).unwrap();
    let vocabulary: std::collections::BTreeSet<_> = events.iter().flat_map(Event::context).collect();
    assert_eq!(vocabulary.len(), 1);
}
#[test]
fn invalid_inputs_return_errors() {
    let s = FeatureSelection { metric: SelectionMetric::MutualInformation, top_k: 1 };
    assert!(s.fit(&[]).is_err());
    assert!(s.fit(&[Event::from_strs("a", &["x"])]).is_err());
    let mut events = vec![Event::from_strs("a", &["x"]), Event::from_strs("b", &["y"])];
    assert!(FeatureSelection { top_k: 0, ..s }.fit(&events).is_err());
    for value in [f32::NAN, f32::INFINITY, -1.] {
        events[0] = Event::with_values("a", vec![JavaString::new("x")], Some(vec![value])).unwrap();
        assert!(s.fit(&events).is_err());
    }
}

#[test]
fn selected_features_feed_liblinear_without_noise_weights() {
    let events = vec![
        Event::from_strs("a", &["all", "cue_a"]),
        Event::from_strs("a", &["all", "cue_a", "noise"]),
        Event::from_strs("b", &["all", "cue_b"]),
        Event::from_strs("b", &["all", "cue_b", "noise"]),
    ];
    for metric in [SelectionMetric::MutualInformation, SelectionMetric::Chi2] {
        let s = FeatureSelection { metric, top_k: 2 }.fit(&events).unwrap();
        let filtered = s.filter(&events).unwrap();
        let mut p = turbo_train::TrainingParameters::new();
        for (k, v) in [("solverType", "L2R_LR"), ("c", "1"), ("eps", "0.01"), ("bias", "-1"), ("Cutoff", "0")] {
            p.put(k, v);
        }
        let (m, _) = turbo_train::liblinear::train_events(&filtered, &p).unwrap();
        assert_eq!(m.predicates().len(), 2);
        for (cue, want) in [("cue_a", "a"), ("cue_b", "b")] {
            let mut probs = vec![0.; m.num_outcomes()];
            m.eval(&[cue], &mut probs);
            let best = turbo_train::liblinear::LiblinearModel::best_outcome(&probs);
            assert!(m.outcome(best).eq_str(want));
        }
    }
}
