use turbo_doccat::multilabel::{MultiLabelSample, MultiLabelTrainingParameters, train_multilabel};
use turbo_doccat::{DoccatFactory, FeatureDocument, FeatureSelection, SelectionMetric};

#[test]
fn informative_features_beat_noise_and_ties_are_deterministic() {
    let docs = vec![
        FeatureDocument::from_strs("a", &["signal_a", "all", "noise"]),
        FeatureDocument::from_strs("a", &["signal_a", "all"]),
        FeatureDocument::from_strs("b", &["signal_b", "all", "noise"]),
        FeatureDocument::from_strs("b", &["signal_b", "all"]),
    ];
    for metric in [SelectionMetric::MutualInformation, SelectionMetric::Chi2] {
        let selected = FeatureSelection { metric, top_k: 2 }.fit(&docs).unwrap();
        assert_eq!(
            selected.kept().iter().cloned().collect::<Vec<_>>(),
            vec!["signal_a".to_owned(), "signal_b".to_owned()]
        );
        let expected = if metric == SelectionMetric::MutualInformation { 2f64.ln() } else { 4.0 };
        assert!((selected.ranked()[0].1 - expected).abs() < 1e-12);
        assert_eq!(selected.ranked()[2].1, 0.0);
        let mut reversed = docs.clone();
        reversed.reverse();
        assert_eq!(selected.ranked(), FeatureSelection { metric, top_k: 2 }.fit(&reversed).unwrap().ranked());
    }
}

#[test]
fn duplicates_affect_training_values_but_not_presence_ranking() {
    let plain = vec![FeatureDocument::from_strs("a", &["x"]), FeatureDocument::from_strs("b", &["y"])];
    let dup = vec![
        FeatureDocument::with_values("a", vec!["x".into(), "x".into(), "noise".into()], Some(vec![2.0, 3.0, 0.0]))
            .unwrap(),
        plain[1].clone(),
    ];
    let selector = FeatureSelection { metric: SelectionMetric::MutualInformation, top_k: 1 };
    let selected = selector.fit(&dup).unwrap();
    assert_eq!(selected.ranked(), selector.fit(&plain).unwrap().ranked());
    let filtered = selected.filter(&dup).unwrap();
    assert_eq!(filtered[0].context().len(), 2);
    assert_eq!(filtered[0].values(), Some([2.0, 3.0].as_slice()));
    assert!(filtered[1].context().is_empty());
    assert_eq!(filtered[1].outcome(), plain[1].outcome());
}

#[test]
fn invalid_inputs_return_errors() {
    let s = FeatureSelection { metric: SelectionMetric::MutualInformation, top_k: 1 };
    assert!(s.fit(&[]).is_err());
    assert!(s.fit(&[FeatureDocument::from_strs("a", &["x"])]).is_err());
    let mut docs = vec![FeatureDocument::from_strs("a", &["x"]), FeatureDocument::from_strs("b", &["y"])];
    assert!(FeatureSelection { top_k: 0, ..s }.fit(&docs).is_err());
    for value in [f32::NAN, f32::INFINITY, -1.0] {
        docs[0] = FeatureDocument::with_values("a", vec!["x".into()], Some(vec![value])).unwrap();
        assert!(s.fit(&docs).is_err());
    }
}

#[test]
fn selected_features_feed_multilabel_doccat_without_noise_weights() {
    let docs = vec![
        FeatureDocument::from_strs("a", &["all", "cue_a"]),
        FeatureDocument::from_strs("a", &["all", "cue_a", "noise"]),
        FeatureDocument::from_strs("b", &["all", "cue_b"]),
        FeatureDocument::from_strs("b", &["all", "cue_b", "noise"]),
    ];
    for metric in [SelectionMetric::MutualInformation, SelectionMetric::Chi2] {
        let s = FeatureSelection { metric, top_k: 2 }.fit(&docs).unwrap();
        let filtered = s.filter(&docs).unwrap();
        let samples: Vec<MultiLabelSample> = filtered
            .iter()
            .map(|d| MultiLabelSample::new([d.outcome()], d.context().iter().map(String::as_str)))
            .collect();
        let m =
            train_multilabel("en", &samples, &DoccatFactory::new(), &MultiLabelTrainingParameters::default()).unwrap();
        let idx_a = m.index("a").unwrap();
        let idx_b = m.index("b").unwrap();
        assert!(m.scores(&["cue_a"])[idx_a] > m.scores(&["cue_a"])[idx_b]);
        assert!(m.scores(&["cue_b"])[idx_b] > m.scores(&["cue_b"])[idx_a]);
    }
}

#[test]
fn equal_scores_break_ties_by_native_rust_utf8_order() {
    // U+FB00 ('ﬀ') is > U+1F600 ('😀') in UTF-16 code units (0xFB00 > 0xD83D),
    // whereas in native Rust Unicode scalar order '\u{FB00}' < '\u{1F600}'.
    let docs = vec![FeatureDocument::from_strs("a", &["\u{1F600}"]), FeatureDocument::from_strs("b", &["\u{FB00}"])];
    let selected = FeatureSelection { metric: SelectionMetric::MutualInformation, top_k: 2 }.fit(&docs).unwrap();
    assert_eq!(selected.ranked()[0].0, "\u{FB00}");
    assert_eq!(selected.ranked()[1].0, "\u{1F600}");
}
