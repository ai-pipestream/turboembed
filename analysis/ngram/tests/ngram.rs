//! Comprehensive tests for `turbo-ngram`, porting OpenNLP's `NGramModelTest`,
//! `NGramCharModelTest`, `NGramUtilsTest`, `NGramGeneratorTest`, and
//! `NgramLanguageModelTest`.

use turbo_ngram::{
    DEFAULT_N, Error, LanguageModel, NGramCharModel, NGramGenerator, NGramLanguageModel, NGramModel, NGramUtils,
};

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
fn ngram_model_zero_get_count() {
    let model = NGramModel::new();
    let count = model.get_count(&[""]);
    assert_eq!(count, 0);
    assert_eq!(model.size(), 0);
}

#[test]
fn ngram_model_zero_get_count_after_add() {
    let mut model = NGramModel::new();
    model.add(&["the", "brown", "fox", "jumped"]).unwrap();
    let count = model.get_count(&[""]);
    assert_eq!(count, 0);
    assert_eq!(model.size(), 1);
}

#[test]
fn ngram_model_add() {
    let mut model = NGramModel::new();
    model.add(&["the", "brown", "fox", "jumped"]).unwrap();
    let count = model.get_count(&["the", "brown", "fox", "jumped"]);
    assert_eq!(count, 1);
    assert_eq!(model.size(), 1);
}

#[test]
fn ngram_model_same_add() {
    let mut model = NGramModel::new();
    model.add(&["the", "brown", "fox", "jumped"]).unwrap();
    model.add(&["the", "brown", "fox", "jumped"]).unwrap();
    let count = model.get_count(&["the", "brown", "fox", "jumped"]);
    assert_eq!(count, 2);
    assert_eq!(model.size(), 1);
    assert_eq!(model.number_of_grams(), 2);
}

#[test]
fn ngram_model_multi_add_range() {
    let mut model = NGramModel::new();
    model.add_range(&["the", "brown", "fox", "jumped"], 1, 3).unwrap();
    for i in 1..=3 {
        let tokens = &["the", "brown", "fox", "jumped"][..i];
        assert_eq!(model.get_count(tokens), 1);
    }
    assert_eq!(model.size(), 9);
}

#[test]
fn ngram_model_remove() {
    let mut model = NGramModel::new();
    let tokens = ["the", "brown", "fox", "jumped"];
    model.add(&tokens).unwrap();
    assert!(model.remove(&tokens));
    assert_eq!(model.size(), 0);
    assert!(!model.remove(&tokens));
}

#[test]
fn ngram_model_contains() {
    let mut model = NGramModel::new();
    let tokens = ["the", "brown", "fox", "jumped"];
    model.add(&tokens).unwrap();
    assert!(!model.contains(&["the"]));
    assert!(model.contains(&["the", "brown", "fox", "jumped"]));
}

#[test]
fn ngram_model_number_of_grams() {
    let mut model = NGramModel::new();
    let tokens = ["the", "brown", "fox", "jumped"];
    model.add_range(&tokens, 1, 3).unwrap();
    assert_eq!(model.number_of_grams(), 9);
}

#[test]
fn ngram_model_cutoff_1() {
    let mut model = NGramModel::new();
    let tokens = ["the", "brown", "fox", "jumped"];
    model.add_range(&tokens, 1, 3).unwrap();
    model.cutoff(2, 4);
    assert_eq!(model.size(), 0);
}

#[test]
fn ngram_model_cutoff_2() {
    let mut model = NGramModel::new();
    let tokens = ["the", "brown", "fox", "jumped"];
    model.add_range(&tokens, 1, 3).unwrap();
    model.cutoff(1, 3);
    assert_eq!(model.size(), 9);
}

#[test]
fn ngram_model_to_dictionary() {
    let mut model = NGramModel::new();
    model.add_range(&["the", "brown", "fox", "jumped"], 1, 3).unwrap();
    model.add_range(&["the", "brown", "Fox", "jumped"], 1, 3).unwrap();
    let dict = model.to_dictionary(false);
    assert!(!dict.is_case_sensitive());
    assert_eq!(dict.len(), 9);
    assert_eq!(dict.min_tokens(), 1);
    assert_eq!(dict.max_tokens(), 3);
}

#[test]
fn ngram_model_to_dictionary_case_sensitive() {
    let mut model = NGramModel::new();
    model.add_range(&["the", "brown", "fox", "jumped"], 1, 3).unwrap();
    model.add_range(&["the", "brown", "Fox", "jumped"], 1, 3).unwrap();
    let dict = model.to_dictionary(true);
    assert!(dict.is_case_sensitive());
    assert_eq!(dict.len(), 14);
    assert_eq!(dict.min_tokens(), 1);
    assert_eq!(dict.max_tokens(), 3);
}

#[test]
fn ngram_model_invalid_format_missing_count() {
    let xml = b"<dictionary case_sensitive=\"false\"><entry><token>the</token></entry></dictionary>";
    assert_eq!(NGramModel::read(xml), Err(Error::MissingCountAttribute { entry_index: 0 }));
}

#[test]
fn ngram_model_invalid_format_not_a_number() {
    let xml = b"<dictionary case_sensitive=\"false\"><entry count=\"two\"><token>the</token></entry></dictionary>";
    assert_eq!(NGramModel::read(xml), Err(Error::InvalidCountAttribute { entry_index: 0, value: "two".to_string() }));
}

#[test]
fn ngram_model_from_chars() {
    let mut model = NGramModel::new();
    model.add_chars("the", 1, 3).unwrap();
    assert_eq!(model.size(), 6);
    assert_eq!(model.get_count(&["th"]), 1);
    assert_eq!(model.to_string(), "Size: 6");

    let mut full = NGramModel::new();
    full.add_chars("the brown fox jumped", 1, 3).unwrap();
    assert_eq!(full.size(), 53);
    assert_eq!(full.number_of_grams(), 57);
    assert_eq!(full.to_string(), "Size: 53");
}

#[test]
fn ngram_model_serialize_and_read_roundtrip() {
    let model1 = NGramModel::read(NGRAM_MODEL_XML).unwrap();
    let mut model2 = NGramModel::new();
    model2.add_range(&["the", "brown", "fox", "jumped"], 1, 3).unwrap();
    model2.add_range(&["the", "brown", "Fox", "jumped"], 1, 3).unwrap();
    assert_eq!(model1, model2);
    assert_eq!(model1.size(), 14);
    assert_eq!(model1.number_of_grams(), 18);

    let bytes = model1.serialize().unwrap();
    let model3 = NGramModel::read(&bytes).unwrap();
    assert_eq!(model3, model2);
    let ordered1: Vec<(&[String], i32)> = model1.iter_with_counts().collect();
    let ordered3: Vec<(&[String], i32)> = model3.iter_with_counts().collect();
    assert_eq!(ordered1, ordered3);
}

#[test]
fn ngram_char_model_tests() {
    let mut model = NGramCharModel::new();
    assert_eq!(model.get_count(""), 0);
    assert_eq!(model.size(), 0);

    model.add("the");
    assert_eq!(model.get_count("fox"), 0);
    assert_eq!(model.get_count("the"), 1);
    assert_eq!(model.size(), 1);

    model.add("the");
    assert_eq!(model.get_count("the"), 2);
    assert_eq!(model.size(), 1);

    let mut short_model = NGramCharModel::new();
    short_model.add_chars("the", 1, 3).unwrap();
    assert_eq!(short_model.get_count("th"), 1);
    assert_eq!(short_model.size(), 6);
    short_model.cutoff(2, 4);
    assert_eq!(short_model.size(), 0);

    let mut range_model = NGramCharModel::new();
    range_model.add_chars("the brown fox jumped", 1, 3).unwrap();
    for i in 1..=3 {
        assert!(range_model.get_count(&"the brown fox jumped"[..i]) >= 1);
    }
    assert_eq!(range_model.size(), 53);
    assert_eq!(range_model.number_of_grams(), 57);

    let mut rem_model = NGramCharModel::new();
    rem_model.add("the");
    assert!(rem_model.remove("the"));
    assert_eq!(rem_model.size(), 0);

    let mut contains_model = NGramCharModel::new();
    contains_model.add("the");
    assert!(!contains_model.contains("fox"));
    assert!(contains_model.contains("the"));
    assert!(contains_model.contains("THE"));

    let mut cutoff_model = NGramCharModel::new();
    cutoff_model.add_chars("the brown fox jumped", 1, 3).unwrap();
    cutoff_model.cutoff(2, 4);
    assert_eq!(cutoff_model.size(), 3);

    let mut cutoff_model2 = NGramCharModel::new();
    cutoff_model2.add_chars("the brown fox jumped", 1, 3).unwrap();
    cutoff_model2.cutoff(1, 2);
    assert_eq!(cutoff_model2.size(), 52);
}

#[test]
fn ngram_utils_probabilities_and_slices() {
    let set = [
        vec!["<s>", "I", "am", "Sam", "</s>"],
        vec!["<s>", "Sam", "I", "am", "</s>"],
        vec!["<s>", "I", "do", "not", "like", "green", "eggs", "and", "ham", "</s>"],
        vec![""],
    ];
    let d = NGramUtils::calculate_bigram_ml_probability(&"<s>", &"I", &set);
    assert_eq!(d, 0.6666666666666666);
    let d = NGramUtils::calculate_bigram_ml_probability(&"Sam", &"</s>", &set);
    assert_eq!(d, 0.5);
    let d = NGramUtils::calculate_bigram_ml_probability(&"<s>", &"Sam", &set);
    assert_eq!(d, 0.3333333333333333);

    let d = NGramUtils::calculate_trigram_ml_probability(&"I", &"am", &"Sam", &set);
    assert_eq!(d, 0.5);
    let d = NGramUtils::calculate_trigram_ml_probability(&"Sam", &"I", &"am", &set);
    assert_eq!(d, 1.0);

    let d = NGramUtils::calculate_ngram_ml_probability(&["I", "am", "Sam"], &set);
    assert_eq!(d, 0.5);
    let d = NGramUtils::calculate_ngram_ml_probability(&["Sam", "I", "am"], &set);
    assert_eq!(d, 1.0);

    let p_laplace = NGramUtils::calculate_laplace_smoothing_probability(&["<s>", "I"], &set, 1.0);
    assert_eq!(p_laplace.to_bits(), 0x3FE8_0000_0000_0000);
    let p_prior = NGramUtils::calculate_bigram_prior_smoothing_probability(&"<s>", &"I", &set, 1.0);
    assert_eq!(p_prior.to_bits(), 0x3FD3_9782_9CBC_14E6);
    let p_mass = NGramUtils::calculate_missing_ngram_probability_mass(&["<s>"], 0.5, &set);
    assert_eq!(p_mass.to_bits(), 0x4001_5555_5555_5555);

    // The distinct words are summed in `java.util.HashSet` order (zz, q, b, c,
    // d, x, I for this set), not first occurrence order, which would give
    // 0x3FE6_6666_6666_6665.
    let order_set = [vec!["c", "b"], vec!["x", "d", "q", "I", "zz"]];
    let p_mass = NGramUtils::calculate_missing_ngram_probability_mass(&["c"], 0.1, &order_set);
    assert_eq!(p_mass.to_bits(), 0x3FE6_6666_6666_6666);

    let interp_set = [
        vec!["the", "green", "book", "STOP"],
        vec!["my", "blue", "book", "STOP"],
        vec!["his", "green", "house", "STOP"],
        vec!["book", "STOP"],
    ];
    let lambda = 1.0 / 3.0;
    let d = NGramUtils::calculate_trigram_linear_interpolation_probability(
        &"the",
        &"green",
        &"book",
        &interp_set,
        lambda,
        lambda,
        lambda,
    )
    .unwrap();
    assert_eq!(d, 0.5714285714285714);

    let interp_set2 = [vec!["D", "N", "V", "STOP"], vec!["D", "N", "V", "STOP"]];
    let d2 = NGramUtils::calculate_trigram_linear_interpolation_probability(
        &"N",
        &"V",
        &"STOP",
        &interp_set2,
        lambda,
        lambda,
        lambda,
    )
    .unwrap();
    assert_eq!(d2, 0.75);

    assert!(matches!(
        NGramUtils::calculate_trigram_linear_interpolation_probability(
            &"N",
            &"V",
            &"STOP",
            &interp_set2,
            0.5,
            0.5,
            0.5,
        ),
        Err(Error::InvalidInterpolationWeights { .. })
    ));

    let tokens = ["I", "saw", "brown", "fox"];
    let ngrams = NGramUtils::get_ngrams(&tokens, 2);
    assert_eq!(ngrams, vec![&["I", "saw"][..], &["saw", "brown"][..], &["brown", "fox"][..]]);
    let ngrams3 = NGramUtils::get_ngrams(&tokens, 3);
    assert_eq!(ngrams3, vec![&["I", "saw", "brown"][..], &["saw", "brown", "fox"][..]]);
}

#[test]
fn ngram_generator_token_and_char_slices() {
    let list = ["This", "is", "a", "sentence", "for", "testing", "the", "NGram", "generator"];
    let unigrams = NGramGenerator::generate(&list, 1, "-");
    assert_eq!(unigrams, list);

    let bigrams = NGramGenerator::generate(&list, 2, "-");
    assert_eq!(
        bigrams,
        vec![
            "This-is",
            "is-a",
            "a-sentence",
            "sentence-for",
            "for-testing",
            "testing-the",
            "the-NGram",
            "NGram-generator",
        ]
    );

    let trigrams = NGramGenerator::generate(&list, 3, "-");
    assert_eq!(
        trigrams,
        vec![
            "This-is-a",
            "is-a-sentence",
            "a-sentence-for",
            "sentence-for-testing",
            "for-testing-the",
            "testing-the-NGram",
            "the-NGram-generator",
        ]
    );

    let fourgrams = NGramGenerator::generate(&list, 4, "-");
    assert_eq!(
        fourgrams,
        vec![
            "This-is-a-sentence",
            "is-a-sentence-for",
            "a-sentence-for-testing",
            "sentence-for-testing-the",
            "for-testing-the-NGram",
            "testing-the-NGram-generator",
        ]
    );

    assert!(NGramGenerator::generate(&list, 11, "-").is_empty());
    assert!(NGramGenerator::generate::<&str>(&[], 1, "-").is_empty());

    let chars = ['T', 'e', 's', 't', 'i', 'n', 'g'];
    assert_eq!(NGramGenerator::generate_chars(&chars, 1, "-"), vec!["T", "e", "s", "t", "i", "n", "g"]);
    assert_eq!(NGramGenerator::generate_chars(&chars, 2, "-"), vec!["T-e", "e-s", "s-t", "t-i", "i-n", "n-g"]);
    assert_eq!(NGramGenerator::generate_chars(&chars, 3, "-"), vec!["T-e-s", "e-s-t", "s-t-i", "t-i-n", "i-n-g"]);
    assert_eq!(NGramGenerator::generate_chars(&chars, 4, "-"), vec!["T-e-s-t", "e-s-t-i", "s-t-i-n", "t-i-n-g"]);
    assert!(NGramGenerator::generate_chars(&chars, 8, "-").is_empty());
    assert!(NGramGenerator::generate_chars(&[], 1, "-").is_empty());
}

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
    let tokens = model.predict_next_tokens(&["neural", "network", "language"]);
    assert_eq!(tokens, Some(vec!["models".to_string()]));

    let p1 = model.calculate_probability(&[
        "use",
        "the",
        "word2vec",
        "to",
        "train",
        "larry",
        "crust",
        "apple",
        "uncharted",
        "once",
        "upon",
        "a",
        "time",
    ]);
    let p2 = model.calculate_probability(&[
        "use",
        "the",
        "word2vec",
        "to",
        "train",
        "wikitravel",
        "corpus",
        "and",
        "got",
        "the",
        "word",
        "vector",
    ]);
    assert_eq!(p1, 0.0);
    assert_eq!(p2.to_bits(), 0x3FC0_0000_0000_0001);
}

#[test]
fn language_model_from_ngram_model_xml() {
    let model = NGramLanguageModel::from_xml(NGRAM_MODEL_XML).unwrap();
    let prob = model.calculate_probability(&["the", "brown", "fox"]);
    assert_eq!(prob.to_bits(), 0x3FE0_0000_0000_0000);
    let tokens = model.predict_next_tokens(&["the", "brown"]);
    assert_eq!(tokens, Some(vec!["fox".to_string()]));
}
