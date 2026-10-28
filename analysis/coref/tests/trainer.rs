//! Port of OpenNLP's `CorefTrainerTest`: pair events, gold validation,
//! the pairwise and ranking trainers, and model round trips.

mod common;
mod training;

use std::sync::Arc;

use common::chain_ids;
use training::{Corpus, Gold, corpus, document, names, parameters};
use turbo_coref::train::{DEFAULT_EPOCHS, DEFAULT_L2, DEFAULT_LEARNING_RATE, GoldMention, pairs, train, train_ranking};
use turbo_coref::{APART, CorefAnnotator, CorefModel, DEFAULT_THRESHOLD, LINK};

fn one(gold: Gold) -> Corpus {
    Corpus::new(vec![gold])
}

fn rules() -> CorefAnnotator {
    CorefAnnotator::default()
}

fn ranking(corpus: &Corpus) -> CorefModel {
    train_ranking("eng", &corpus.documents(), DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_L2, &rules()).unwrap()
}

#[test]
fn pairs_label_the_gold_antecedent_and_the_rest() {
    let linked = one(document("Acme", "organization", "expanded", "It", 0));
    let events = pairs(&linked.documents()[0], &rules()).unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].outcome(), LINK);
    assert!(names(&events[0]).contains(&"kinds=pronoun>entity".to_owned()));
    let apart = one(document("Acme", "organization", "expanded", "He", 1));
    let events = pairs(&apart.documents()[0], &rules()).unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].outcome(), APART);
}

#[test]
fn pairs_reject_duplicate_spans_and_noncanonical_chain_ids() {
    let base = document("Acme", "organization", "expanded", "It", 0);
    let first = base.gold[0].span;
    let second = base.gold[1].span;
    let duplicate = Gold {
        fixture: base.fixture.clone(),
        gold: vec![GoldMention { span: first, chain: 0 }, GoldMention { span: first, chain: 1 }],
    };
    let err = pairs(&one(duplicate).documents()[0], &rules()).unwrap_err();
    assert!(err.contains("duplicate gold mention span"), "{err}");
    let sparse = Gold {
        fixture: base.fixture,
        gold: vec![GoldMention { span: first, chain: 7 }, GoldMention { span: second, chain: 7 }],
    };
    let err = pairs(&one(sparse).documents()[0], &rules()).unwrap_err();
    assert!(err.contains("first-mention order"), "{err}");
}

#[test]
fn trained_pair_model_round_trips_and_links() {
    let corpus = corpus();
    let trained = train("eng", &corpus.documents(), &parameters(), &rules()).unwrap();
    assert_eq!(trained.language(), "eng");
    assert_eq!(trained.pair_model().num_outcomes(), 2);
    assert!(!trained.is_ranking());
    let model = CorefModel::read(trained.to_bytes().to_vec()).unwrap();
    assert_eq!(model.to_bytes(), trained.to_bytes());
    assert_eq!(model.model_ref().sha256, trained.model_ref().sha256);
    // The synthetic corpus is balanced, so the pairs are calibrated around one half.
    let annotator =
        CorefAnnotator::new(&["person"], &["organization", "location"], Some(Arc::new(model)), 0.5, None, None)
            .unwrap();
    let linked = document("Cyberdyne", "organization", "expanded", "It", 0);
    assert_eq!(chain_ids(&linked.fixture, &annotator), [0, 0]);
    let apart = document("Cyberdyne", "organization", "expanded", "He", 1);
    assert_eq!(chain_ids(&apart.fixture, &annotator), [0, 1]);
}

#[test]
fn ranking_model_round_trips_and_links_against_the_new_chain_option() {
    let trained = ranking(&corpus());
    assert!(trained.is_ranking());
    let model = CorefModel::read(trained.to_bytes().to_vec()).unwrap();
    assert!(model.is_ranking());
    let annotator = CorefAnnotator::with_model(model).unwrap();
    let linked = document("Cyberdyne", "organization", "expanded", "It", 0);
    assert_eq!(chain_ids(&linked.fixture, &annotator), [0, 0]);
    let apart = document("Cyberdyne", "organization", "expanded", "He", 1);
    assert_eq!(chain_ids(&apart.fixture, &annotator), [0, 1]);
    // The model's edges carry its probability, not the sieve score of 1.
    let built = linked.fixture.build();
    let (_, _, edges) = annotator.resolve(&built.input()).unwrap();
    assert_eq!(edges.from.len(), 1);
    assert!(edges.score[0] > 0.0 && edges.score[0] <= 1.0);
}

#[test]
fn model_annotator_reports_its_model_ref() {
    use turbo_analysis::Annotator;
    let trained = ranking(&corpus());
    let sha = trained.model_ref().sha256.clone();
    let annotator = CorefAnnotator::with_model(trained).unwrap();
    assert_eq!(annotator.model().map(|m| m.sha256), Some(sha));
}

#[test]
fn gold_singletons_mark_partial_annotation() {
    let three = || {
        common::Fixture::new()
            .sentence(&["Acme/NNP", "expanded/VBD", "./."])
            .sentence(&["The/DT", "firm/NN", "grew/VBD", "./."])
            .sentence(&["The/DT", "company/NN", "hired/VBD", "./."])
            .entity("Acme", "organization")
            .chunk("Acme", "NP")
            .chunk("The firm", "NP")
            .chunk("The company", "NP")
    };
    // No singletons: the unannotated "The company" is taught to start a chain.
    let complete = one(Gold::new(three(), &[("Acme", 0), ("The firm", 0)]));
    assert_eq!(pairs(&complete.documents()[0], &rules()).unwrap().len(), 3);
    // A gold with singletons annotates every mention of its scheme.
    let partial = one(Gold::new(three(), &[("Acme", 0), ("The firm", 1)]));
    assert_eq!(pairs(&partial.documents()[0], &rules()).unwrap().len(), 1);
}

#[test]
fn ranking_rejects_bad_settings_and_empty_corpora() {
    let corpus = corpus();
    let docs = corpus.documents();
    for (epochs, rate, l2) in
        [(0, 0.1, 0.0), (1, 0.0, 0.0), (1, 0.1, -1.0), (1, f64::INFINITY, 0.0), (1, 0.1, f64::INFINITY)]
    {
        assert!(train_ranking("eng", &docs, epochs, rate, l2, &rules()).is_err(), "{epochs} {rate} {l2}");
    }
    assert!(train_ranking("eng", &[], 1, 0.1, 0.0, &rules()).is_err());
    assert!(train("eng", &[], &parameters(), &rules()).is_err());
}

#[test]
fn model_and_training_reject_missing_language_codes() {
    let corpus = corpus();
    let docs = corpus.documents();
    for language in ["", " "] {
        assert!(train(language, &docs, &parameters(), &rules()).unwrap_err().contains("languageCode"));
        let err = train_ranking(language, &docs, 1, 0.1, 0.0, &rules()).unwrap_err();
        assert!(err.contains("languageCode"));
    }
}

#[test]
fn annotator_rejects_bad_threshold_with_model_and_model_reads_reject_garbage() {
    let trained = train("eng", &corpus().documents(), &parameters(), &rules()).unwrap();
    let model = Some(Arc::new(trained));
    assert!(CorefAnnotator::new(&["person"], &["organization"], model.clone(), 1.5, None, None).is_err());
    assert!(CorefAnnotator::new(&["person"], &["organization"], model, DEFAULT_THRESHOLD, None, None).is_ok());
    assert!(CorefModel::read(b"not a model".to_vec()).is_err());
}

#[test]
fn pair_model_accepts_link_at_the_threshold() {
    // Any probability passes a threshold of zero.
    let trained = train("eng", &corpus().documents(), &parameters(), &rules()).unwrap();
    let annotator =
        CorefAnnotator::new(&["person"], &["organization"], Some(Arc::new(trained)), 0.0, None, None).unwrap();
    let doc = document("Cyberdyne", "organization", "expanded", "It", 0);
    assert_eq!(chain_ids(&doc.fixture, &annotator), [0, 0]);
}
