use turbo_analysis::ViewId;
use turbo_analysis_layers::Words;
use turbo_depparse::*;
use turbo_formats::conllu::{ConlluDependencySample, Tagset};
use turbo_train::TrainingParameters;
use turbo_unicode::WhitespaceMode;

#[test]
fn plain_training_rejects_crossing_trees_and_default_training_keeps_them() {
    let sample = DependencySample::<Words>::from_conllu(
        ConlluDependencySample::read_sentence(
            include_str!("data/ud-en-nonprojective.conllu"),
            Tagset::U,
            WhitespaceMode::Unicode,
            ViewId::SOURCE,
        )
        .unwrap(),
    )
    .unwrap();
    assert!(!sample.graph().is_projective());
    let samples = [sample];
    let mut params = TrainingParameters::new();
    params.put_str("Algorithm", "MAXENT");
    params.put_int("Iterations", 2);
    params.put_int("Cutoff", 0);
    let settings = FeedforwardSettings {
        epochs: 1,
        embedding_size: 3,
        hidden_size: 5,
        batch_size: 4,
        dropout: 0.0,
        word_cutoff: 1,
        ..Default::default()
    };
    assert!(DependencyModel::train_projective("en", &samples, &params).is_err());
    assert!(FeedforwardDependencyTrainer::train_projective(&samples, &settings).is_err());
    let linear = DependencyModel::train("en", &samples, &params).unwrap();
    let neural = FeedforwardDependencyTrainer::train(&samples, &settings).unwrap();
    assert_eq!(linear.label_encoding(), LabelEncoding::EscapedV1);
    assert_eq!(neural.label_encoding(), LabelEncoding::EscapedV1);
    assert_eq!(DependencyEventStream::generate(&samples).unwrap().skipped_nonprojective, 0);
}
