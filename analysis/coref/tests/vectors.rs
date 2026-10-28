//! Port of the vector parts of OpenNLP's `CorefTrainerTest`: word
//! vectors for head similarity, token vectors for span features, and
//! the validation of both providers.

mod common;
mod training;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::{Fixture, chain_ids};
use training::{Corpus, Encoder, Gold, corpus, document, encoder, firm, names};
use turbo_coref::train::{pairs, train_ranking};
use turbo_coref::{CorefAnnotator, CorefModel, DEFAULT_THRESHOLD, MapWordVectors, TokenVectors, WordVectors};

const PERSON: &[&str] = &["person"];
const NEUTRAL: &[&str] = &["organization", "location"];

fn with_words(vectors: Arc<dyn WordVectors>) -> CorefAnnotator {
    CorefAnnotator::new(PERSON, NEUTRAL, None, DEFAULT_THRESHOLD, Some(vectors), None).unwrap()
}

fn with_encoder(encoder: Arc<dyn TokenVectors>) -> CorefAnnotator {
    CorefAnnotator::new(PERSON, NEUTRAL, None, DEFAULT_THRESHOLD, None, Some(encoder)).unwrap()
}

/// An encoder's answer for one sentence.
type Answer = fn(&[&str]) -> Vec<Vec<f32>>;

fn encoder_of(dimension: usize, f: Answer) -> Arc<dyn TokenVectors> {
    Arc::new(Encoder { dimension, f })
}

fn firm_pairs(annotator: &CorefAnnotator) -> Result<Vec<turbo_train::Event>, String> {
    let corpus = Corpus::new(vec![firm()]);
    pairs(&corpus.documents()[0], annotator)
}

/// Word vectors answering from a function, counting lookups of "firm".
struct FnVectors {
    dimension: usize,
    table: Vec<(&'static str, Vec<f32>)>,
    fallback: Option<Vec<f32>>,
    firm_lookups: AtomicUsize,
}

impl FnVectors {
    fn new(dimension: usize, table: Vec<(&'static str, Vec<f32>)>, fallback: Option<Vec<f32>>) -> FnVectors {
        FnVectors { dimension, table, fallback, firm_lookups: AtomicUsize::new(0) }
    }
}

impl WordVectors for FnVectors {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn vector(&self, word: &str) -> Option<&[f32]> {
        if word == "firm" {
            self.firm_lookups.fetch_add(1, Ordering::Relaxed);
        }
        self.table.iter().find(|(w, _)| *w == word).map(|(_, v)| v.as_slice()).or(self.fallback.as_deref())
    }

    fn config_hash(&self) -> u64 {
        self.dimension as u64
    }
}

#[test]
fn word_vectors_feed_similarity_features_for_nominal_anaphors() {
    let mut vectors = MapWordVectors::new(2);
    vectors.insert("acme", vec![1.0, 0.1]).unwrap();
    vectors.insert("firm", vec![1.0, 0.1]).unwrap();
    let events = firm_pairs(&with_words(Arc::new(vectors))).unwrap();
    assert_eq!(events.len(), 1);
    let features = names(&events[0]);
    assert!(features.contains(&"sim=same".to_owned()), "{features:?}");
    assert!(features.contains(&"clusterSim=same".to_owned()), "{features:?}");
    let plain = names(&firm_pairs(&CorefAnnotator::default()).unwrap()[0]);
    assert!(!plain.iter().any(|f| f.starts_with("sim=")), "{plain:?}");
}

#[test]
fn word_vectors_must_be_valid_and_consistent() {
    assert!(MapWordVectors::new(2).insert("firm", vec![1.0]).is_err());
    assert!(MapWordVectors::new(1).insert("firm", vec![f32::NAN]).is_err());
    let empty = Arc::new(FnVectors::new(1, vec![], Some(vec![])));
    assert!(firm_pairs(&with_words(empty)).is_err());
    let nonfinite = Arc::new(FnVectors::new(1, vec![], Some(vec![f32::NAN])));
    assert!(firm_pairs(&with_words(nonfinite)).is_err());
    let inconsistent = Arc::new(FnVectors::new(2, vec![("firm", vec![1.0, 0.0])], Some(vec![1.0])));
    assert!(firm_pairs(&with_words(inconsistent)).is_err());
}

#[test]
fn missing_word_vector_is_looked_up_once_per_document() {
    let vectors = Arc::new(FnVectors::new(1, vec![], None));
    firm_pairs(&with_words(vectors.clone())).unwrap();
    assert_eq!(vectors.firm_lookups.load(Ordering::Relaxed), 1);
}

#[test]
fn token_vectors_add_span_features_with_values() {
    let events = firm_pairs(&with_encoder(encoder())).unwrap();
    assert_eq!(events.len(), 1);
    let names = names(&events[0]);
    let values = events[0].values().expect("values");
    assert_eq!(names.len(), values.len());
    let value = |n: &str| values[names.iter().position(|x| x == n).unwrap_or_else(|| panic!("{n} in {names:?}"))];
    // "The firm" averages {0, 1} and {1, 0}; "Acme" is {1, 0}: cosine 0.707.
    assert!(names.contains(&"ctx=7".to_owned()), "{names:?}");
    assert!(names.contains(&"ctxCluster=7".to_owned()), "{names:?}");
    assert!((value("vp0") - 0.5).abs() < 1e-6);
    assert!(value("vp1").abs() < 1e-6);
    assert!((value("vd0") - 0.5).abs() < 1e-6);
    assert!((value("vd1") - 0.5).abs() < 1e-6);
    assert!((value("kinds=nominal>entity") - 1.0).abs() < 1e-6);
    assert!(firm_pairs(&CorefAnnotator::default()).unwrap()[0].values().is_none());
}

#[test]
fn ranking_model_trains_and_decodes_with_token_vectors() {
    let rules = with_encoder(encoder());
    let trained = train_ranking("eng", &corpus().documents(), 10, 0.05, 0.001, &rules).unwrap();
    assert_eq!(trained.token_vector_dimension(), 2);
    let model = CorefModel::read(trained.to_bytes().to_vec()).unwrap();
    assert!(model.is_ranking());
    assert_eq!(model.token_vector_dimension(), 2);
    let p = model.link_probability(&["vp0".to_owned()], Some(&[1.0])).unwrap();
    assert!(p > 0.5, "a shared direction should favor linking: {p}");
    let model = Arc::new(model);
    let annotator =
        CorefAnnotator::new(PERSON, NEUTRAL, Some(model.clone()), DEFAULT_THRESHOLD, None, Some(encoder())).unwrap();
    assert_eq!(annotator.token_vector_dimension(), 2);
    let linked = document("Cyberdyne", "organization", "expanded", "It", 0);
    assert_eq!(chain_ids(&linked.fixture, &annotator), [0, 0]);
    let apart = document("Cyberdyne", "organization", "expanded", "He", 1);
    assert_eq!(chain_ids(&apart.fixture, &annotator), [0, 1]);
    // The model requires its encoder, of its dimension.
    assert!(CorefAnnotator::with_model(model.clone()).is_err());
    let one = encoder_of(1, |t| vec![vec![1.0]; t.len()]);
    assert!(CorefAnnotator::new(PERSON, NEUTRAL, Some(model), DEFAULT_THRESHOLD, None, Some(one)).is_err());
}

#[test]
fn vector_dimensions_are_validated_at_construction() {
    let plain = train_ranking("eng", &corpus().documents(), 1, 0.05, 0.0, &CorefAnnotator::default()).unwrap();
    assert_eq!(plain.token_vector_dimension(), 0);
    let plain = Arc::new(plain);
    assert!(CorefAnnotator::new(PERSON, NEUTRAL, Some(plain), DEFAULT_THRESHOLD, None, Some(encoder())).is_err());
    let zero_words = MapWordVectors::new(0);
    assert!(CorefAnnotator::new(PERSON, NEUTRAL, None, DEFAULT_THRESHOLD, Some(Arc::new(zero_words)), None).is_err());
    let zero_encoder = encoder_of(0, |t| vec![vec![]; t.len()]);
    assert!(CorefAnnotator::new(PERSON, NEUTRAL, None, DEFAULT_THRESHOLD, None, Some(zero_encoder)).is_err());
}

#[test]
fn encoder_answers_are_validated() {
    let bad: [Answer; 5] = [
        |_| vec![vec![1.0, 0.0]],
        |t| (0..t.len()).map(|i| if i == 1 { vec![] } else { vec![1.0, 0.0] }).collect(),
        |t| vec![vec![]; t.len()],
        |t| (0..t.len()).map(|i| if i == 1 { vec![1.0] } else { vec![1.0, 0.0] }).collect(),
        |t| (0..t.len()).map(|i| if i == 0 { vec![f32::INFINITY, 0.0] } else { vec![0.0, 0.0] }).collect(),
    ];
    for (n, f) in bad.into_iter().enumerate() {
        let annotator = with_encoder(encoder_of(2, f));
        let built = firm().fixture.build();
        assert!(annotator.resolve(&built.input()).is_err(), "case {n}");
        assert!(firm_pairs(&annotator).is_err(), "case {n}");
    }
}

#[test]
fn ranking_rejects_an_encoder_that_changes_dimension_between_documents() {
    let changing = encoder_of(2, |t| {
        let d = if t[0] == "Acme" || t[0] == "It" { 2 } else { 3 };
        vec![(0..d).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect(); t.len()]
    });
    let docs = Corpus::new(vec![
        document("Acme", "organization", "expanded", "It", 0),
        document("Kowalczyk", "person", "arrived", "He", 0),
    ]);
    assert!(train_ranking("eng", &docs.documents(), 1, 0.05, 0.0, &with_encoder(changing)).is_err());
}

#[test]
fn token_vectors_tolerate_an_entity_without_an_aligned_token() {
    let mut f = Fixture::new()
        .sentence(&["Ada/NNP", "spoke/VBD", "./."])
        .sentence(&["She/PRP", "left/VBD", "./."])
        .entity("Ada", "person");
    // Drop the token under the entity: the entity aligns to no token and
    // is not a mention (a documented deviation from OpenNLP, which keeps it).
    f.tokens.remove(0);
    f.tags.remove(0);
    let gold = Gold::new(f, &[("Ada", 0), ("She", 0)]);
    let events = pairs(&Corpus::new(vec![gold]).documents()[0], &with_encoder(encoder())).unwrap();
    assert!(events.is_empty());
}
