use turbo_analysis::ViewId;
use turbo_analysis_layers::Words;
use turbo_depparse::{DependencyGraph, DependencySample, HeldOutEvaluation};
use turbo_formats::conllu::{ConlluDependencySample, Tagset};
use turbo_unicode::WhitespaceMode;
fn sample(text: &str) -> DependencySample<Words> {
    let sentence = turbo_formats::conllu::words::ConlluWordReader::new(text).unwrap().next().unwrap().unwrap();
    let text = &text[sentence.source.start as usize..sentence.source.end as usize];
    DependencySample::from_conllu(
        ConlluDependencySample::read_sentence(text, Tagset::U, WhitespaceMode::Unicode, ViewId::SOURCE).unwrap(),
    )
    .unwrap()
}
#[test]
fn missing_and_invalid_predictions_stay_in_crossing_denominators() {
    let gold = sample(include_str!("data/ud-fi-ftb-crossing.conllu"));
    let mut report = HeldOutEvaluation::default();
    assert!(report.record(&gold, Some(gold.graph())).unwrap());
    assert!(!report.record(&gold, None).unwrap());
    let wrong_length = DependencyGraph::<Words>::new(vec![None], vec!["root".into()]).unwrap();
    assert!(!report.record(&gold, Some(&wrong_length)).unwrap());
    assert_eq!(report.all.sentences, 3);
    assert_eq!(report.nonprojective.sentences, 3);
    assert_eq!(report.projective.sentences, 0);
    assert_eq!(report.all.failed, 2);
    assert_eq!(report.all.scores.all.words, 33);
    assert_eq!(report.all.scores.all.heads, 11);
    assert_eq!(report.all.scores.without_punctuation.words, 30);
    assert!(!report.passes(0.0, 0.0).unwrap());
    assert!(report.passes(f64::NAN, 0.0).is_err());
    assert!(report.passes(0.0, 1.01).is_err());
    assert!(!HeldOutEvaluation::default().passes(0.0, 0.0).unwrap());
}
#[test]
fn projective_and_nonprojective_subsets_add_to_the_complete_population() {
    let projective = sample(include_str!("data/ud-es.conllu"));
    let crossing = sample(include_str!("data/ud-fi-ftb-crossing.conllu"));
    let mut report = HeldOutEvaluation::default();
    for gold in [&projective, &crossing] {
        report.record(gold, Some(gold.graph())).unwrap();
    }
    assert_eq!(report.projective.sentences, 1);
    assert_eq!(report.nonprojective.sentences, 1);
    assert_eq!(report.all.scores.all.words, report.projective.scores.all.words + report.nonprojective.scores.all.words);
    assert!(report.passes(1.0, 1.0).unwrap());
    assert_eq!(report.write_tsv().lines().count(), 7);
}
