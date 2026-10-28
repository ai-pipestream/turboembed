//! Crate-local snapshot of DictionaryLemmatizer over gold UD tags.

use turbo_lemmatizer::DictionaryLemmatizer;

#[test]
fn dictionary_lemmatizer_reads_gold_ud_tags() {
    let dict = DictionaryLemmatizer::parse(b"running\tVBG\trun\nwent\tVBD\tgo\nbetter\tJJR\tgood\n", "model")
        .expect("dictionary");
    assert_eq!(dict.lemma("running", "VBG"), "run");
    assert_eq!(dict.lemma("went", "VBD"), "go");
    assert_eq!(dict.lemma("better", "JJR"), "good");
    assert_eq!(dict.lemma("unknown", "NN"), "O");
}

#[test]
fn dictionary_lemmatizer_is_case_insensitive() {
    let dict = DictionaryLemmatizer::parse(b"running\tVBG\trun\n", "model").expect("dictionary");
    assert_eq!(dict.lemma("running", "VBG"), "run");
    assert_eq!(dict.lemma("RUNNING", "VBG"), "run");
}
