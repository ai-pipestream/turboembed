use turbo_analysis_layers::Words;
use turbo_depparse::{FeatureLayout, FeedforwardDependencyParser, FeedforwardSettings};
use turbo_formats::conllu::Tagset;
use turbo_train_components::depparse::train_feedforward_conllu;
#[test]
fn real_ud_corpus_trains_and_roundtrips_a_word_parser() {
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
    let loaded = turbo_depparse::FeedforwardDependencyModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
    let parser = FeedforwardDependencyParser::new(loaded, 4).unwrap();
    parser.parse::<Words>(&["Voy", "a", "el", "mar"], &["VERB", "ADP", "DET", "NOUN"]).unwrap().check_tree().unwrap();
    assert!(train_feedforward_conllu(&["bad input"], Tagset::U, &settings).err().unwrap().contains("sentence 1"));
}
