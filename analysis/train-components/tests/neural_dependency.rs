use turbo_analysis_layers::Words;
use turbo_depparse::{FeatureLayout, FeedforwardDependencyParser, FeedforwardSettings};
use turbo_formats::conllu::Tagset;
use turbo_train_components::depparse::train_feedforward_conllu;

#[test]
fn real_ud_corpus_trains_a_word_parser() {
    let source = include_str!("../../depparse/tests/data/ud-es.conllu");
    let sentences: Vec<_> = source.trim().split("\n\n").collect();
    let settings = FeedforwardSettings {
        embedding_size: 3,
        hidden_size: 5,
        epochs: 2,
        batch_size: 16,
        layout: FeatureLayout::ChenManning18,
        ..Default::default()
    };
    let model = train_feedforward_conllu(&sentences, Tagset::U, &settings).unwrap();
    let parser = FeedforwardDependencyParser::new(model, 4).unwrap();
    parser.parse::<Words>(&["Voy", "a", "el", "mar"], &["VERB", "ADP", "DET", "NOUN"]).unwrap().check_tree().unwrap();
    assert!(train_feedforward_conllu(&["bad input"], Tagset::U, &settings).err().unwrap().contains("sentence 1"));
}

#[test]
fn an_observer_sees_each_epoch_stops_training_and_leaves_the_model_unchanged() {
    use turbo_train_components::depparse::train_feedforward_conllu_observed;
    let source = include_str!("../../depparse/tests/data/ud-es.conllu");
    let sentences: Vec<_> = source.trim().split("\n\n").collect();
    let settings = FeedforwardSettings {
        embedding_size: 3,
        hidden_size: 5,
        epochs: 3,
        batch_size: 16,
        layout: FeatureLayout::ChenManning18,
        ..Default::default()
    };
    let mut seen = Vec::new();
    let observed = train_feedforward_conllu_observed(&sentences, Tagset::U, &settings, &mut |e| {
        seen.push(*e);
        true
    })
    .unwrap();
    assert_eq!(seen.iter().map(|e| (e.epoch, e.epochs)).collect::<Vec<_>>(), [(1, 3), (2, 3), (3, 3)]);
    assert!(seen.iter().all(|e| e.loss.is_finite() && e.loss > 0.0), "{seen:?}");
    assert!(seen[2].loss < seen[0].loss, "the loss falls over the epochs: {seen:?}");
    // Watching changes nothing: the bytes are the unobserved training's.
    let plain = train_feedforward_conllu(&sentences, Tagset::U, &settings).unwrap();
    assert_eq!(observed.to_bytes().unwrap(), plain.to_bytes().unwrap());
    let mut epochs = 0;
    let stopped = train_feedforward_conllu_observed(&sentences, Tagset::U, &settings, &mut |_| {
        epochs += 1;
        false
    });
    assert!(stopped.err().unwrap().contains("stopped"));
    assert_eq!(epochs, 1);
}
