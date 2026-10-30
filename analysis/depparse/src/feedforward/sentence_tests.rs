use super::*;
use crate::feedforward::{Activation, FeatureLayout, FeedforwardDependencyTrainer, FeedforwardSettings};
use crate::{ArcStandardOracle, DependencyGraph, DependencySample};
use turbo_analysis::Ix;
use turbo_analysis_layers::Tokens;

fn sample(words: &[&str], tags: &[&str], heads: &[Option<u32>], labels: &[&str]) -> DependencySample {
    let graph = DependencyGraph::<Tokens>::new(
        heads.iter().map(|h| h.map(Ix::new)).collect(),
        labels.iter().map(|s| s.to_string()).collect(),
    )
    .unwrap();
    DependencySample::new(
        words.iter().map(|s| s.to_string()).collect(),
        tags.iter().map(|s| s.to_string()).collect(),
        graph,
    )
    .unwrap()
}

#[test]
fn resolved_rows_give_the_ids_of_the_string_context_at_every_state() {
    let train = vec![
        sample(&["I", "see", "Birds"], &["PRON", "VERB", "NOUN"], &[Some(1), None, Some(1)], &["nsubj", "root", "obj"]),
        sample(&["birds", "fly"], &["NOUN", "VERB"], &[Some(1), None], &["nsubj", "root"]),
    ];
    // Unknown words and tags, a word that is a special symbol, and case the
    // normalizer folds.
    let probe = sample(
        &["ΣΟΦΟΣ", "*ROOT*", "BIRDS", "fly", "zz"],
        &["X", "NOUN", "NOUN", "VERB", "ADV"],
        &[Some(3), Some(3), Some(3), None, Some(3)],
        &["dep", "nsubj", "nsubj", "root", "advmod"],
    );
    for layout in [FeatureLayout::OpenNlp14, FeatureLayout::ChenManning18] {
        let settings = FeedforwardSettings {
            embedding_size: 3,
            hidden_size: 5,
            epochs: 1,
            batch_size: 4,
            learning_rate: 0.02,
            l2: 0.0,
            dropout: 0.0,
            word_cutoff: 1,
            seed: 17,
            layout,
            activation: Activation::Cube,
        };
        let model = FeedforwardDependencyTrainer::train(&train, &settings).unwrap();
        for s in train.iter().chain([&probe]) {
            let words = s.tokens().iter().map(String::as_str).collect::<Vec<_>>();
            let tags = s.tags().iter().map(String::as_str).collect::<Vec<_>>();
            let sentence = SentenceIds::new(&model, &words, &tags);
            let (mut positions, mut ids) = (Vec::new(), Vec::new());
            let mut state = ArcStandardState::<Tokens>::new(words.len()).unwrap();
            for step in ArcStandardOracle::transitions(s.graph()).unwrap() {
                let context = FeedforwardContext::extract(&state, &words, &tags, layout).unwrap();
                sentence.fill(&model, &state, &mut positions, &mut ids);
                assert_eq!(ids, model.feature_ids(&context).unwrap());
                state.apply(&step).unwrap();
            }
        }
    }
}
