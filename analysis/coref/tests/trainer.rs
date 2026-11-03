//! Pair events, gold validation and settings checks without archive models.

mod common;
mod training;

use training::{Corpus, Gold, corpus, document, names, parameters};
use turbo_coref::train::{GoldMention, pairs, train, train_ranking};
use turbo_coref::{APART, CorefAnnotator, CorefModel, DEFAULT_THRESHOLD, LINK};

fn one(gold: Gold) -> Corpus {
    Corpus::new(vec![gold])
}

fn rules() -> CorefAnnotator {
    CorefAnnotator::default()
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
    for (epochs, rate, l2, message) in [
        (0, 0.1, 0.0, "epochs must be positive"),
        (1, 0.0, 0.0, "learningRate must be positive and finite"),
        (1, 0.1, -1.0, "l2 must be finite and not negative"),
        (1, f64::INFINITY, 0.0, "learningRate must be positive and finite"),
        (1, 0.1, f64::INFINITY, "l2 must be finite and not negative"),
    ] {
        let error = train_ranking("eng", &docs, epochs, rate, l2, &rules()).unwrap_err();
        assert!(error.contains(message), "{error}");
    }
    assert_eq!(
        train_ranking("eng", &[], 1, 0.1, 0.0, &rules()).unwrap_err(),
        "training data produced no ranking instances"
    );
    let empty = train("eng", &[], &parameters(), &rules()).unwrap_err();
    assert_eq!(empty, "Insufficient training data to create model.");
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
fn rules_reject_bad_threshold_and_model_reads_reject_garbage() {
    for threshold in [1.5, -0.1, f64::NAN, f64::INFINITY] {
        assert!(CorefAnnotator::new(&["person"], &["organization"], None, threshold, None, None).is_err());
    }
    assert!(CorefAnnotator::new(&["person"], &["organization"], None, DEFAULT_THRESHOLD, None, None).is_ok());
    assert!(CorefModel::read(b"not a model".to_vec()).is_err());
}
