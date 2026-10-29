//! Crate-local snapshot of DictionaryLemmatizer over gold UD tags.

use turbo_lemmatizer::DictionaryLemmatizer;

const DICTIONARY: &[u8] = include_bytes!("../../goldens/data/fixtures/lemmatizer-dictionary.txt");

#[test]
fn dictionary_fixture_snapshot() {
    let dict = DictionaryLemmatizer::parse(DICTIONARY, "fixture").expect("dictionary");
    let cases = [
        ("the", "DET", "the"),
        ("went", "VERB", "go"),
        ("leaves", "VERB", "leave"),
        ("leaves", "NOUN", "leaf"),
        ("silt", "NOUN", "O"),
        ("line", "NOUN", "line"),
        ("The", "DET", "the"),
        ("U.S.", "PROPN", "United States"),
        ("café", "NOUN", "café"),
        ("東京", "PROPN", "Tokyo"),
        ("münchen", "PROPN", "München"),
        ("unknown", "NN", "O"),
    ];
    let actual: Vec<_> = cases.iter().map(|(word, tag, _)| dict.lemma(word, tag)).collect();
    let expected: Vec<_> = cases.iter().map(|(_, _, lemma)| *lemma).collect();
    assert_eq!(actual, expected);
    assert_eq!(dict.lemmas("went", "VERB").unwrap(), ["go"]);
    assert_eq!(dict.lemmas("leaves", "VERB").unwrap(), ["leave", "leaf"]);
    assert!(dict.lemmas("silt", "NOUN").unwrap().is_empty());
}

#[test]
fn dictionary_fixture_lookup_is_case_insensitive() {
    let dict = DictionaryLemmatizer::parse(DICTIONARY, "fixture").expect("dictionary");
    assert_eq!(dict.lemma("WENT", "VERB"), "go");
    assert_eq!(dict.lemma("MÜNCHEN", "PROPN"), "München");
}
