use super::*;
use crate::feedforward::{NeuralWeights, Vocabulary, normalize::normalize};
use std::collections::BTreeMap;
use turbo_jvm::{hash_map::JavaHashMap, random::JavaRandom};

pub(super) fn initialize<A: WordOrTokenAnchor>(
    samples: &[DependencySample<A>],
    settings: &FeedforwardSettings,
) -> Result<FeedforwardDependencyModel> {
    let mut counts = JavaHashMap::<String, usize>::new();
    let mut tags = JavaHashMap::new();
    let mut labels = JavaHashMap::new();
    let mut transitions = BTreeMap::new();
    for sample in samples {
        if !sample.graph().is_projective() {
            continue;
        }
        for word in sample.tokens() {
            counts.merge(normalize(word), 1, |n, v| *n += v);
        }
        for tag in sample.tags() {
            tags.put(tag.clone(), ());
        }
        for i in 0..sample.graph().len() {
            labels.put(sample.graph().arc(turbo_analysis::Ix::new(i as u32))?.relation.to_owned(), ());
        }
        for t in ArcStandardOracle::transitions(sample.graph())? {
            transitions.insert(t.encode(), t);
        }
    }
    let mut row = 0;
    let mut vocab = |symbols: &[&str], values: Vec<&String>| {
        let mut out = BTreeMap::new();
        for word in symbols.iter().copied().chain(values.into_iter().map(String::as_str)) {
            if !out.contains_key(word) {
                out.insert(word.to_owned(), row);
                row += 1;
            }
        }
        out
    };
    let words = vocab(
        &["*UNK*", "*NULL*", "*ROOT*"],
        counts.iter().filter(|(_, n)| **n >= settings.word_cutoff).map(|(s, _)| s).collect(),
    );
    let pos = vocab(&["*UNK*", "*NULL*", "*ROOT*"], tags.iter().map(|(s, _)| s).collect());
    let labels = vocab(&["*UNK*", "*NULL*"], labels.iter().map(|(s, _)| s).collect());
    let vocabulary = Vocabulary { words, pos, labels };
    let mut transitions: Vec<_> = transitions.into_values().collect();
    transitions.sort_by(|a, b| turbo_jvm::chars::compare(&a.encode(), &b.encode()));
    settings.check_shape(row, transitions.len())?;
    let mut random = JavaRandom::new(settings.seed);
    let mut uniform = |count, scale| (0..count).map(|_| ((random.next_double() * 2.0 - 1.0) * scale) as f32).collect();
    let (e, h, o) = (settings.embedding_size, settings.hidden_size, transitions.len());
    let input = settings.layout.feature_count() * e;
    let embeddings = uniform(row * e, 0.01);
    let hidden_weights = uniform(h * input, (6.0 / (input + h) as f64).sqrt());
    let output_weights = uniform(o * h, (6.0 / (h + o) as f64).sqrt());
    let weights = NeuralWeights {
        embedding_size: e,
        embeddings,
        hidden_weights,
        output_weights,
        hidden_bias: vec![0.0; h],
        output_bias: vec![0.0; o],
    };
    FeedforwardDependencyModel::new(vocabulary, transitions, settings.layout, settings.activation, weights)
}
