mod support;
use turbo_analysis::Ix;
use turbo_analysis_layers::Tokens;
use turbo_depparse::*;
use turbo_train::TrainingParameters;

fn sample() -> DependencySample<Tokens> {
    DependencySample::new(
        vec!["I".into(), "see".into(), "birds".into()],
        vec!["PRON".into(), "VERB".into(), "NOUN".into()],
        DependencyGraph::new(
            vec![Some(Ix::new(1)), None, Some(Ix::new(1))],
            vec!["nsubj".into(), "root".into(), "obj".into()],
        )
        .unwrap(),
    )
    .unwrap()
}
#[test]
fn model_inventory_and_scores_are_checked() {
    for labels in
        [vec!["SHIFT"], vec!["RIGHT_ARC:dep"], vec!["SHIFT", "SHIFT", "RIGHT_ARC:dep"], vec!["SHIFT", "RIGHT_ARC: "]]
    {
        assert!(DependencyModel::from_raw(&support::raw(&labels, "s0t=NOUN", &vec![0.0; labels.len()])).is_err());
    }
    let model =
        DependencyModel::from_raw(&support::raw(&["SHIFT", "RIGHT_ARC:dep"], "s0t=NOUN", &[f64::NAN, 0.0])).unwrap();
    assert!(DependencyParserME::new(model).parse::<Tokens>(&["a"], &["NOUN"]).is_err());
    let p = support::parser();
    assert!(p.parse::<Tokens>(&[], &[]).is_err());
    assert!(p.parse::<Tokens>(&["a"], &[]).is_err());
    let p = DependencyParserME::new(
        DependencyModel::from_raw(&support::raw(&["RIGHT_ARC:dep", "SHIFT"], "never", &[0.0, 0.0])).unwrap(),
    );
    let g = p.parse::<Tokens>(&["a", "b", "c"], &["X", "X", "X"]).unwrap();
    assert_eq!(g.to_layer().head, vec![None, Some(Ix::new(0)), Some(Ix::new(0))]);
}
#[test]
fn events_training_serialization_and_evaluation() {
    let samples = vec![sample(), sample(), sample()];
    let events = DependencyEventStream::generate(&samples).unwrap();
    assert_eq!(events.events.len(), 18);
    assert_eq!(events.events[0].context().len(), 37);
    let mut params = TrainingParameters::new();
    params.put_str("Algorithm", "MAXENT");
    params.put_int("Cutoff", 0);
    params.put_int("Iterations", 60);
    let model = DependencyModel::train("en", &samples, &params).unwrap();
    let reloaded = DependencyModel::open(model.bytes().unwrap().to_vec()).unwrap();
    assert_eq!(model.model_ref().sha256, reloaded.model_ref().sha256);
    let parser = DependencyParserME::new(reloaded);
    let mut scores = DependencyEvaluator::default();
    for s in &samples {
        scores.evaluate(&parser, s).unwrap();
    }
    assert_eq!(scores.all.words, 9);
    assert_eq!(scores.all.uas(), 1.0);
    assert_eq!(scores.all.las(), 1.0);
    let mut cv = DependencyCrossValidator::default();
    let mut sizes = Vec::new();
    cv.evaluate(&samples, 3, |train| {
        sizes.push(train.len());
        Ok(DependencyParserME::new(DependencyModel::train(
            "en",
            &train.iter().map(|s| (*s).clone()).collect::<Vec<_>>(),
            &params,
        )?))
    })
    .unwrap();
    assert_eq!(sizes, [2, 2, 2]);
    assert_eq!(cv.scores.all, scores.all);
    assert!(cv.evaluate(&samples, 1, |_| Ok(parser.clone())).is_err());
}
#[test]
fn nonprojective_training_samples_are_skipped_atomically() {
    let nonprojective = DependencySample::<Tokens>::new(
        vec!["a".into(), "b".into(), "c".into()],
        vec!["X".into(); 3],
        DependencyGraph::new(vec![Some(Ix::new(2)), None, Some(Ix::new(1))], vec!["dep".into(); 3]).unwrap(),
    )
    .unwrap();
    let stream = DependencyEventStream::generate(&[nonprojective, sample()]).unwrap();
    assert_eq!(stream.skipped_nonprojective, 1);
    assert_eq!(stream.events.len(), 6);
}
#[test]
fn scoring_excludes_punctuation_and_does_not_credit_labels_with_wrong_heads() {
    let gold = DependencySample::<Tokens>::new(
        vec!["a".into(), "b".into(), "!".into()],
        vec!["NOUN".into(), "NOUN".into(), "PUNCT".into()],
        DependencyGraph::new(
            vec![None, Some(Ix::new(0)), Some(Ix::new(0))],
            vec!["root".into(), "dep".into(), "punct".into()],
        )
        .unwrap(),
    )
    .unwrap();
    let pred = DependencyGraph::new(
        vec![None, Some(Ix::new(0)), Some(Ix::new(1))],
        vec!["root".into(), "wrong".into(), "punct".into()],
    )
    .unwrap();
    let mut scores = DependencyEvaluator::default();
    scores.score(&gold, &pred).unwrap();
    assert_eq!(scores.all, AttachmentScores { words: 3, heads: 2, labels: 1 });
    assert_eq!(scores.without_punctuation, AttachmentScores { words: 2, heads: 2, labels: 1 });
    let before = scores;
    let short = DependencyGraph::new(vec![None], vec!["root".into()]).unwrap();
    assert!(scores.score(&gold, &short).is_err());
    assert_eq!(scores, before);
}

#[test]
fn counter_overflow_returns_error_without_changing_totals() {
    let gold = sample();
    let mut evaluator = DependencyEvaluator::default();
    evaluator.all.heads = u64::MAX;
    let before = evaluator;
    assert!(evaluator.score(&gold, gold.graph()).is_err());
    assert_eq!(evaluator, before);
}
