//! Integration tests for `NGramLanguageModel` (`opennlp.tools.languagemodel.NgramLanguageModelTest`).

use turbo_ngram::{DEFAULT_N, Error, LanguageModel, NGramLanguageModel};

const NGRAM_MODEL_XML: &[u8] = br#"<?xml version="1.0" encoding="UTF-8"?>
<dictionary case_sensitive="false">
    <entry count="1">
        <token>brown</token>
        <token>fox</token>
    </entry>
    <entry count="1">
        <token>fox</token>
    </entry>
    <entry count="1">
        <token>brown</token>
        <token>fox</token>
        <token>jumped</token>
    </entry>
    <entry count="2">
        <token>the</token>
    </entry>
    <entry count="1">
        <token>the</token>
        <token>brown</token>
        <token>fox</token>
    </entry>
    <entry count="1">
        <token>the</token>
        <token>brown</token>
        <token>Fox</token>
    </entry>
    <entry count="2">
        <token>jumped</token>
    </entry>
    <entry count="2">
        <token>brown</token>
    </entry>
    <entry count="1">
        <token>brown</token>
        <token>Fox</token>
        <token>jumped</token>
    </entry>
    <entry count="1">
        <token>Fox</token>
    </entry>
    <entry count="1">
        <token>fox</token>
        <token>jumped</token>
    </entry>
    <entry count="2">
        <token>the</token>
        <token>brown</token>
    </entry>
    <entry count="1">
        <token>brown</token>
        <token>Fox</token>
    </entry>
    <entry count="1">
        <token>Fox</token>
        <token>jumped</token>
    </entry>
</dictionary>
"#;

const SENTENCES_TXT: &str = include_str!("fixtures/sentences.txt");

#[test]
fn language_model_empty_vocabulary_probability() {
    let model = NGramLanguageModel::default();
    assert_eq!(model.order(), DEFAULT_N);
    assert_eq!(model.calculate_probability::<&str>(&[]), 0.0, "probability with an empty vocabulary is always 0");
    assert_eq!(model.calculate_probability(&["1", "2", "3"]), 0.0, "probability with an empty vocabulary is always 0");
    assert_eq!(model.predict_next_tokens(&["1"]), None);
}

#[test]
fn language_model_invalid_order_rejected() {
    assert_eq!(NGramLanguageModel::new(0), Err(Error::InvalidOrder(0)));
    assert_eq!(NGramLanguageModel::from_xml_with_order(NGRAM_MODEL_XML, 0), Err(Error::InvalidOrder(0)));
}

#[test]
fn language_model_trigram_and_bigram_predictions() {
    let mut model = NGramLanguageModel::default();
    model.add(&["the", "green", "house"]).unwrap();
    model.add(&["my", "red", "house"]).unwrap();
    model.add(&["his", "green", "shirt"]).unwrap();

    let prob = model.calculate_probability(&["the", "green", "house"]);
    assert_eq!(prob.to_bits(), 0x3FF0_0000_0000_0000);
    let pred = model.predict_next_tokens(&["the", "green"]);
    assert_eq!(pred, Some(vec!["house".to_string()]));
}

#[test]
fn language_model_from_sentences_corpus() {
    let mut model = NGramLanguageModel::default();
    for line in SENTENCES_TXT.lines() {
        let array: Vec<&str> = line.split(' ').collect();
        model.add(&array).unwrap();
    }
    // "the afternoon" is followed only by "boat" in the corpus, so the
    // trigram score for that continuation is 1.
    let tokens = model.predict_next_tokens(&["take", "the", "afternoon"]);
    assert_eq!(tokens, Some(vec!["boat".to_string()]));

    // Unseen words have a unigram count of 0, so the log score is -inf.
    let p1 = model
        .calculate_probability(&["keeper", "of", "the", "station", "checks", "the", "zeppelin", "quartz", "vortex"]);
    // Every trigram here is in the corpus. The exact product of the trigram
    // ratios is 1/280; summing logs and exponentiating with StrictMath lands
    // four ulps below the nearest double to 1/280 (0x3F6D_41D4_1D41_D41D).
    let p2 = model.calculate_probability(&[
        "keeper",
        "of",
        "the",
        "station",
        "checks",
        "the",
        "instruments",
        "every",
        "morning",
        "and",
        "writes",
        "a",
        "short",
        "note",
    ]);
    assert_eq!(p1, 0.0);
    assert_eq!(p2.to_bits(), 0x3F6D_41D4_1D41_D419);
}

#[test]
fn language_model_from_ngram_model_xml() {
    let model = NGramLanguageModel::from_xml(NGRAM_MODEL_XML).unwrap();
    let prob = model.calculate_probability(&["the", "brown", "fox"]);
    assert_eq!(prob.to_bits(), 0x3FE0_0000_0000_0000);
    let tokens = model.predict_next_tokens(&["the", "brown"]);
    assert_eq!(tokens, Some(vec!["fox".to_string()]));
}

#[test]
fn language_model_negative_counts_do_not_yield_certainty() {
    let mut model = NGramLanguageModel::default();
    model.add(&["the", "green", "house"]).unwrap();
    model.set_count(&["house"], -5).unwrap();
    assert_eq!(model.calculate_probability(&["house"]), 0.0);
    assert_eq!(model.calculate_probability(&["unknown", "house"]), 0.0);

    // Negative prefix count falls back to vocabulary size instead of a negative denominator.
    model.set_count(&["the", "green"], -2).unwrap();
    let p = model.calculate_probability(&["the", "green", "house"]);
    assert!(p > 0.0 && p < 1.0, "expected probability in (0, 1), got {p}");
}
