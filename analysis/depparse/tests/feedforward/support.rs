use turbo_analysis::Ix;
use turbo_analysis_layers::Tokens;
use turbo_depparse::*;
pub fn samples() -> Vec<DependencySample> {
    vec![
        sample(&["I", "see", "birds"], &["PRON", "VERB", "NOUN"], &[Some(1), None, Some(1)], &["nsubj", "root", "obj"]),
        sample(&["birds", "fly"], &["NOUN", "VERB"], &[Some(1), None], &["nsubj", "root"]),
    ]
}
pub fn sample(words: &[&str], tags: &[&str], heads: &[Option<u32>], labels: &[&str]) -> DependencySample {
    DependencySample::new(
        words.iter().map(|s| s.to_string()).collect(),
        tags.iter().map(|s| s.to_string()).collect(),
        DependencyGraph::<Tokens>::new(
            heads.iter().map(|h| h.map(Ix::new)).collect(),
            labels.iter().map(|s| s.to_string()).collect(),
        )
        .unwrap(),
    )
    .unwrap()
}
pub fn settings(dropout: bool) -> FeedforwardSettings {
    FeedforwardSettings {
        embedding_size: 3,
        hidden_size: 5,
        epochs: 3,
        batch_size: 4,
        learning_rate: 0.02,
        l2: if dropout { 0.001 } else { 0.0 },
        dropout: if dropout { 0.3 } else { 0.0 },
        word_cutoff: 1,
        seed: 17,
        layout: FeatureLayout::OpenNlp14,
        activation: Activation::Cube,
    }
}
pub fn minimal(layout: FeatureLayout) -> FeedforwardDependencyModel {
    let map = |symbols: &[&str], offset| symbols.iter().enumerate().map(|(i, s)| (s.to_string(), offset + i)).collect();
    FeedforwardDependencyModel::new(
        Vocabulary {
            words: map(&["*UNK*", "*NULL*", "*ROOT*"], 0),
            pos: map(&["*UNK*", "*NULL*", "*ROOT*"], 3),
            labels: map(&["*UNK*", "*NULL*"], 6),
        },
        vec![Transition::Shift, Transition::LeftArc("dep".into()), Transition::RightArc("root".into())],
        layout,
        Activation::Cube,
        NeuralWeights {
            embedding_size: 1,
            embeddings: vec![0.0; 8],
            hidden_bias: vec![0.0],
            hidden_weights: vec![0.0; layout.feature_count()],
            output_weights: vec![0.0; 3],
            output_bias: vec![0.0; 3],
        },
    )
    .unwrap()
}
