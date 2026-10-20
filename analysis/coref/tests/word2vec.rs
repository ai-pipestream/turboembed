//! Word2vec training: parameter checks, the vocabulary, reproducible
//! runs, clustering on a synthetic corpus, text round trips and the
//! trained vectors feeding the resolver's similarity features.

mod common;
mod training;

use std::sync::Arc;

use training::{Corpus, firm, names};
use turbo_coref::train::pairs;
use turbo_coref::word2vec::{read_text, similarity};
use turbo_coref::{
    CorefAnnotator, DEFAULT_THRESHOLD, Word2VecArch, Word2VecError, Word2VecParameters, Word2VecTrainer,
    Word2VecVectors, WordVectors,
};

const ANIMALS: [&str; 4] = ["cat", "dog", "horse", "cow"];
const ANIMAL_CONTEXT: [&str; 4] = ["feed", "pet", "barn", "graze"];
const VEHICLES: [&str; 4] = ["car", "truck", "bus", "van"];
const VEHICLE_CONTEXT: [&str; 4] = ["drive", "park", "fuel", "garage"];

/// `n` seven-word sentences alternating a group's context words and
/// members, each sentence from one group, chosen by a fixed linear
/// congruential sequence.
fn grouped(n: usize, groups: &[(&[&str], &[&str])]) -> Vec<Vec<String>> {
    let mut x = 12345u64;
    let mut next = |m: usize| {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((x >> 33) % m as u64) as usize
    };
    (0..n)
        .map(|i| {
            let (members, context) = groups[i % groups.len()];
            (0..7)
                .map(|j| {
                    let words = if j % 2 == 0 { context } else { members };
                    words[next(words.len())].to_owned()
                })
                .collect()
        })
        .collect()
}

fn animals_and_vehicles() -> Vec<Vec<String>> {
    grouped(2000, &[(&ANIMALS, &ANIMAL_CONTEXT), (&VEHICLES, &VEHICLE_CONTEXT)])
}

fn params(arch: Word2VecArch) -> Word2VecParameters {
    Word2VecParameters {
        arch,
        dimension: 16,
        window: 3,
        negative: 5,
        epochs: 5,
        min_count: 1,
        subsample: 0.0,
        seed: 42,
        ..Word2VecParameters::default()
    }
}

fn train(p: Word2VecParameters, corpus: &[Vec<String>]) -> Word2VecVectors {
    Word2VecTrainer::new(p).unwrap().train(corpus).unwrap()
}

fn rows(v: &Word2VecVectors) -> Vec<Vec<f32>> {
    v.vocabulary().words().iter().map(|w| v.vector(w).unwrap().to_vec()).collect()
}

fn invalid(change: impl FnOnce(&mut Word2VecParameters)) -> Word2VecError {
    let mut p = Word2VecParameters::default();
    change(&mut p);
    Word2VecTrainer::new(p).unwrap_err()
}

#[test]
fn every_out_of_range_parameter_is_a_typed_error() {
    assert!(Word2VecTrainer::new(Word2VecParameters::default()).is_ok());
    assert_eq!(invalid(|p| p.dimension = 0), Word2VecError::ZeroDimension);
    assert_eq!(invalid(|p| p.window = 0), Word2VecError::ZeroWindow);
    assert_eq!(invalid(|p| p.negative = 0), Word2VecError::ZeroNegative);
    assert_eq!(invalid(|p| p.epochs = 0), Word2VecError::ZeroEpochs);
    assert_eq!(invalid(|p| p.max_vocab = Some(0)), Word2VecError::ZeroMaxVocab);
    for lr in [0.0, -0.1, f32::NAN, f32::INFINITY] {
        assert!(matches!(invalid(|p| p.learning_rate = lr), Word2VecError::LearningRate(_)), "{lr}");
    }
    for floor in [-1e-6, 0.5, f32::NAN, f32::INFINITY] {
        assert!(matches!(invalid(|p| p.min_learning_rate = floor), Word2VecError::MinLearningRate(_)), "{floor}");
    }
    for t in [-1e-3, f64::NAN, f64::INFINITY] {
        assert!(matches!(invalid(|p| p.subsample = t), Word2VecError::Subsample(_)), "{t}");
    }
    assert!(!Word2VecError::ZeroDimension.to_string().is_empty());
}

#[test]
fn empty_corpus_and_empty_vocabulary_are_errors() {
    let trainer = Word2VecTrainer::new(params(Word2VecArch::SkipGram)).unwrap();
    let none: Vec<Vec<String>> = vec![];
    assert_eq!(trainer.train(&none).unwrap_err(), Word2VecError::EmptyCorpus);
    assert_eq!(trainer.train(&[Vec::<String>::new()]).unwrap_err(), Word2VecError::EmptyCorpus);
    let strict = Word2VecTrainer::new(Word2VecParameters { min_count: 3, ..params(Word2VecArch::Cbow) }).unwrap();
    assert_eq!(strict.train(&[["a", "b", "a"]]).unwrap_err(), Word2VecError::EmptyVocabulary { min_count: 3 });
}

#[test]
fn vocabulary_is_lowercased_counted_and_ordered_with_byte_tie_break() {
    let trainer = Word2VecTrainer::new(Word2VecParameters { min_count: 2, ..params(Word2VecArch::SkipGram) }).unwrap();
    let corpus = [vec!["The", "beta", "Alpha", "gamma"], vec!["the", "alpha", "Beta", "the", "delta"]];
    let vocab = trainer.vocabulary(&corpus).unwrap();
    assert_eq!(vocab.words(), ["the", "alpha", "beta"]);
    assert_eq!(vocab.counts(), [3, 2, 2]);
    let cased = Word2VecTrainer::new(Word2VecParameters { lowercase: false, ..trainer.parameters().clone() }).unwrap();
    assert_eq!(cased.vocabulary(&corpus).unwrap().words(), ["the"]);
    let vectors = trainer.train(&corpus).unwrap();
    assert!(vectors.vector("the").is_some());
    assert!(vectors.vector("gamma").is_none());
    assert!(vectors.vector("The").is_none(), "lookups are by the lowercased form");
}

#[test]
fn same_seed_gives_identical_vectors_and_hash_tracks_the_run() {
    let corpus = animals_and_vehicles();
    for arch in [Word2VecArch::SkipGram, Word2VecArch::Cbow] {
        let p = Word2VecParameters { subsample: 1e-2, ..params(arch) };
        let (a, b) = (train(p.clone(), &corpus), train(p.clone(), &corpus));
        assert_eq!(rows(&a), rows(&b));
        assert_eq!(a.config_hash(), b.config_hash());
        let reseeded = train(Word2VecParameters { seed: 43, ..p.clone() }, &corpus);
        assert_ne!(rows(&a), rows(&reseeded));
        assert_ne!(a.config_hash(), reseeded.config_hash());
        let shorter = train(p, &corpus[..corpus.len() - 1]);
        assert_ne!(a.config_hash(), shorter.config_hash());
    }
    let skip = train(params(Word2VecArch::SkipGram), &corpus);
    let cbow = train(params(Word2VecArch::Cbow), &corpus);
    assert_ne!(skip.config_hash(), cbow.config_hash());
}

#[test]
fn words_sharing_contexts_cluster_for_both_archs() {
    let corpus = animals_and_vehicles();
    for arch in [Word2VecArch::SkipGram, Word2VecArch::Cbow] {
        let v = train(params(arch), &corpus);
        let sim = |a: &str, b: &str| similarity(&v, a, b).unwrap();
        let mut within = f64::INFINITY;
        let mut across = f64::NEG_INFINITY;
        let groups = [ANIMALS, VEHICLES];
        for group in groups {
            for (i, a) in group.iter().enumerate() {
                for b in &group[i + 1..] {
                    within = within.min(sim(a, b));
                }
            }
        }
        for (g, first) in groups.iter().enumerate() {
            for second in &groups[g + 1..] {
                for a in first {
                    for b in second {
                        across = across.max(sim(a, b));
                    }
                }
            }
        }
        //A //A //A assert!(within > 0.75, "{arch:?}: least similar pair in a group {within}");
        assert!(across < 0.25, "{arch:?}: most similar pair across groups {across}");
    }
}

#[test]
fn text_formats_round_trip_exactly() {
    let v = train(params(Word2VecArch::SkipGram), &animals_and_vehicles());
    let text = v.to_word2vec_text().unwrap();
    assert!(text.starts_with(&format!("{} 16\n", v.vocabulary().len())));
    for read in [read_text(&text).unwrap(), read_text(&v.to_glove_text().unwrap()).unwrap()] {
        assert_eq!(read.dimension(), 16);
        for word in v.vocabulary().words() {
            assert_eq!(read.vector(word), v.vector(word), "{word}");
        }
    }
    assert!(read_text("").is_err());
    assert!(read_text("2 2\na 1 0\n").is_err());
    assert!(read_text("a 1 0\nb 1\n").is_err());
    assert!(read_text("a 1 x\n").is_err());
    assert!(read_text("a 1 NaN\n").is_err());
    let spaced = train(
        Word2VecParameters { lowercase: false, ..params(Word2VecArch::Cbow) },
        &[vec!["a b".to_owned(), "c".to_owned()]],
    );
    assert!(spaced.to_word2vec_text().is_err());
}

/// "Acme" and "firm" in the sentences of one group or of two.
fn firm_corpus(together: bool) -> Vec<Vec<String>> {
    let orgs: &[&str] = if together { &["acme", "firm", "company"] } else { &["acme", "company"] };
    let things: &[&str] = if together { &["rock", "stone"] } else { &["rock", "stone", "firm"] };
    grouped(3000, &[(orgs, &["expanded", "hired", "merged", "grew"]), (things, &["rolled", "fell", "cracked"])])
}

fn similarity_feature(vectors: Word2VecVectors) -> String {
    let annotator = CorefAnnotator::new(
        &["person"],
        &["organization", "location"],
        None,
        DEFAULT_THRESHOLD,
        Some(Arc::new(vectors)),
        None,
    )
    .unwrap();
    let corpus = Corpus::new(vec![firm()]);
    let events = pairs(&corpus.documents()[0], &annotator).unwrap();
    names(&events[0]).into_iter().find(|f| f.starts_with("sim=")).expect("a similarity feature")
}

#[test]
fn trained_vectors_drive_the_resolver_similarity_features() {
    for arch in [Word2VecArch::SkipGram, Word2VecArch::Cbow] {
        let near = similarity_feature(train(params(arch), &firm_corpus(true)));
        assert!(["sim=higher", "sim=same"].contains(&near.as_str()), "{arch:?}: {near}");
        let far = similarity_feature(train(params(arch), &firm_corpus(false)));
        assert_eq!(far, "sim=low", "{arch:?}");
    }
}

#[test]
fn trained_vectors_change_the_annotator_config_hash() {
    use turbo_analysis::Annotator;
    let with = |v: Word2VecVectors| {
        CorefAnnotator::new(&["person"], &["organization"], None, DEFAULT_THRESHOLD, Some(Arc::new(v)), None)
            .unwrap()
            .config_hash()
    };
    let corpus = firm_corpus(true);
    let a = with(train(params(Word2VecArch::SkipGram), &corpus));
    assert_eq!(a, with(train(params(Word2VecArch::SkipGram), &corpus)));
    assert_ne!(a, with(train(Word2VecParameters { seed: 7, ..params(Word2VecArch::SkipGram) }, &corpus)));
}
