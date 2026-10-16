//! spaCy's `PhraseMatcher` tests (spaCy, MIT: `spacy/tests/matcher/
//! test_phrase_matcher.py`), ported with the same expected matches. A
//! pattern document `Doc(vocab, words=[...])` is a phrase's tokens here,
//! and a pattern's `pos` or `norm` values are the phrase's tokens over that
//! attribute.

mod common;

use common::Doc;
use turbo_analysis::IxRange;
use turbo_matcher::{Attr, Field, Flag, MatchError, PatternError, PhraseMatcher};

fn text() -> PhraseMatcher {
    PhraseMatcher::new(Field::Attr(Attr::Text))
}

fn owned(v: &[(&str, usize, usize)]) -> Vec<(String, usize, usize)> {
    v.iter().map(|&(l, s, e)| (l.to_string(), s, e)).collect()
}

#[test]
fn issue3248_1() {
    let mut m = text();
    for p in ["a", "b", "c"] {
        m.add("TEST1", &[p]).unwrap();
    }
    m.add("TEST2", &["d"]).unwrap();
    assert_eq!(m.len(), 2);
}

#[test]
fn issue3331() {
    let mut m = text();
    m.add("A", &["Barack", "Obama"]).unwrap();
    m.add("B", &["Barack", "Obama"]).unwrap();
    let found = Doc::split("Barack Obama lifts America").run_phrases(&m);
    let mut labels: Vec<String> = found.iter().map(|f| f.0.clone()).collect();
    labels.sort();
    assert_eq!(labels, ["A", "B"]);
}

#[test]
fn issue3972() {
    let mut m = text();
    m.add("A", &["New", "York"]).unwrap();
    m.add("B", &["New", "York"]).unwrap();
    assert_eq!(Doc::split("I live in New York").run_phrases(&m), owned(&[("A", 3, 5), ("B", 3, 5)]));
}

#[test]
fn issue4002() {
    let doc = Doc::split("a b c d").with_norms(&["a", "b", "c", "d"]);
    let mut m = PhraseMatcher::new(Field::Attr(Attr::Norm));
    m.add("TEST", &["c", "d"]).unwrap();
    assert_eq!(doc.run_phrases(&m).len(), 1);
    // A pattern whose words are "1" and "2" with their norms set to "c"
    // and "d": over NORM, the phrase is its norms.
    let mut m = PhraseMatcher::new(Field::Attr(Attr::Norm));
    m.add("TEST", &["c", "d"]).unwrap();
    assert_eq!(doc.run_phrases(&m).len(), 1);
}

#[test]
fn issue6839_and_span_in_phrasematcher() {
    let doc = Doc::split("I like Spans and Docs in my input , and nothing else .");
    let mut m = text();
    m.add("SPACY", &["Spans", "and", "Docs"]).unwrap();
    assert_eq!(doc.run_phrases(&m).len(), 1);
    assert_eq!(m.find_in(&doc.input(), IxRange::new(0, 8)).unwrap().len(), 1);
}

#[test]
fn span_v_doc_in_phrasematcher() {
    // spaCy's word list joins "," and "and" into one word ",and" by a
    // missing comma; kept as it is.
    let doc = Doc::words(&[
        "I",
        "like",
        "Spans",
        "and",
        "Docs",
        "in",
        "my",
        "input",
        ",",
        "Spans",
        "and",
        "Docs",
        "in",
        "my",
        "matchers",
        ",and",
        "Spans",
        "and",
        "Docs",
        "everywhere",
        ".",
    ]);
    let mut m = text();
    m.add("SPACY", &["Spans", "and", "Docs"]).unwrap();
    assert_eq!(doc.run_phrases(&m).len(), 3);
    assert_eq!(m.find_in(&doc.input(), IxRange::new(9, 15)).unwrap().len(), 1);
}

#[test]
fn issue10643() {
    let doc = Doc::split("Only save out the binary data for the individual components .");
    let mut m = text();
    m.add("0", &["binary"]).unwrap();
    m.add("1", &["binary", "data"]).unwrap();
    assert_eq!(doc.run_phrases(&m), owned(&[("0", 4, 5), ("1", 4, 6)]));
    m.remove("0").unwrap();
    assert_eq!(m.len(), 1);
    assert_eq!(doc.run_phrases(&m), owned(&[("1", 4, 6)]));
    m.remove("1").unwrap();
    assert_eq!(m.len(), 0);
    assert!(doc.run_phrases(&m).is_empty());
}

#[test]
fn matcher_phrase_matcher() {
    let doc = Doc::split("I like Google Now best");
    for phrase in [&["Google", "Now"][..], &["I"], &["I", "like"], &["best"], &["Now", "best"]] {
        let mut m = text();
        m.add("P", phrase).unwrap();
        assert_eq!(doc.run_phrases(&m).len(), 1, "{phrase:?}");
    }
}

#[test]
fn phrase_matcher_length_and_contains() {
    let mut m = text();
    assert_eq!(m.len(), 0);
    m.add("TEST", &["test"]).unwrap();
    assert_eq!(m.len(), 1);
    m.add("TEST2", &["test2"]).unwrap();
    assert_eq!(m.len(), 2);
    assert!(m.contains("TEST") && !m.contains("TEST3"));
}

#[test]
fn phrase_matcher_add_new_api() {
    let doc = Doc::split("a b");
    let mut m = text();
    m.add("NEW_API", &["a"]).unwrap();
    m.add("NEW_API", &["a", "b"]).unwrap();
    assert_eq!(doc.run_phrases(&m).len(), 2);
}

#[test]
fn phrase_matcher_repeated_add() {
    let mut m = text();
    let first = m.add("TEST", &["like"]).unwrap();
    for _ in 0..3 {
        assert_eq!(m.add("TEST", &["like"]).unwrap(), first);
    }
    assert!(m.contains("TEST") && !m.contains("TEST2"));
    assert_eq!(Doc::split("I like Google Now best").run_phrases(&m).len(), 1);
}

#[test]
fn phrase_matcher_remove() {
    let doc = Doc::split("I like Google Now best");
    let mut m = text();
    m.add("TEST1", &["like"]).unwrap();
    m.add("TEST2", &["best"]).unwrap();
    assert_eq!(doc.run_phrases(&m).len(), 2);
    m.remove("TEST1").unwrap();
    assert!(!m.contains("TEST1") && m.contains("TEST2"));
    assert_eq!(doc.run_phrases(&m).len(), 1);
    m.remove("TEST2").unwrap();
    assert_eq!(doc.run_phrases(&m).len(), 0);
    assert_eq!(m.remove("TEST3"), Err(MatchError::NoLabel("TEST3".into())));
    assert_eq!(doc.run_phrases(&m).len(), 0);
}

#[test]
fn phrase_matcher_overlapping_with_remove() {
    let doc = Doc::split("I like Google Now best");
    let mut m = text();
    m.add("TEST", &["like"]).unwrap();
    m.add("TEST2", &["like"]).unwrap();
    assert_eq!((m.len(), doc.run_phrases(&m).len()), (2, 2));
    m.remove("TEST").unwrap();
    assert_eq!(m.len(), 1);
    assert_eq!(doc.run_phrases(&m), owned(&[("TEST2", 1, 2)]));
    m.remove("TEST2").unwrap();
    assert_eq!((m.len(), doc.run_phrases(&m).len()), (0, 0));
}

#[test]
fn phrase_matcher_remove_overlapping_patterns() {
    let mut m = text();
    for p in [&["this"][..], &["this", "is"], &["this", "is", "a"], &["this", "is", "a", "word"]] {
        m.add("THIS", p).unwrap();
    }
    m.remove("THIS").unwrap();
    assert!(Doc::split("this is a word").run_phrases(&m).is_empty());
}

#[test]
fn phrase_matcher_string_attrs() {
    let mut m = PhraseMatcher::new(Field::Attr(Attr::Pos));
    m.add("TEST", &["PRON", "VERB", "NOUN"]).unwrap();
    let doc =
        Doc::split("Yes , you hate dogs very much").with_tags(&["INTJ", "PUNCT", "PRON", "VERB", "NOUN", "ADV", "ADV"]);
    assert_eq!(doc.run_phrases(&m), owned(&[("TEST", 2, 5)]));
}

#[test]
fn phrase_matcher_string_attrs_negative() {
    let mut m = PhraseMatcher::new(Field::Attr(Attr::Pos));
    m.add("TEST", &["PRON", "VERB", "NOUN"]).unwrap();
    let doc = Doc::split("matcher:POS-PRON matcher:POS-VERB matcher:POS-NOUN").with_tags(&["X", "X", "X"]);
    assert!(doc.run_phrases(&m).is_empty());
}

#[test]
fn phrase_matcher_bool_attrs() {
    let mut m = PhraseMatcher::new(Field::Flag(Flag::Punct));
    m.add("TEST", &["Hello", "world", "!"]).unwrap();
    let doc = Doc::split("No problem , he said .");
    assert_eq!(doc.run_phrases(&m), owned(&[("TEST", 0, 3), ("TEST", 3, 6)]));
}

#[test]
fn attr_pipeline_checks() {
    // The document must have the layer the field reads; text needs only
    // tokens.
    for (attr, layer) in [(Attr::Pos, "opennlp:pos"), (Attr::Lemma, "opennlp:lemmas")] {
        let mut m = PhraseMatcher::new(Field::Attr(attr));
        m.add("TEST", &["X"]).unwrap();
        assert_eq!(m.find(&Doc::split("Test").input()), Err(MatchError::Missing(layer)));
    }
    for attr in [Attr::Text, Attr::Lower] {
        let mut m = PhraseMatcher::new(Field::Attr(attr));
        m.add("TEST", &["Test"]).unwrap();
        assert_eq!(Doc::split("Test").run_phrases(&m).len(), 1);
    }
}

#[test]
fn phrase_matcher_as_spans() {
    let mut m = text();
    m.add("A", &["hello", "world"]).unwrap();
    m.add("B", &["test"]).unwrap();
    let doc = Doc::split("... hello world this is a test");
    let found = m.find(&doc.input()).unwrap();
    assert_eq!(found.len(), 2);
    assert_eq!((doc.text_of(found[0].tokens), m.label_name(found[0].label)), ("hello world", "A"));
    assert_eq!((doc.text_of(found[1].tokens), m.label_name(found[1].label)), ("test", "B"));
}

#[test]
fn phrase_matcher_sent_start() {
    // spaCy only checks that the matcher can be made over SENT_START. Its
    // value comes from the document's sentences, not a phrase's text.
    let mut m = PhraseMatcher::new(Field::Flag(Flag::SentStart));
    assert_eq!(m.add("S", &["a"]), Err(PatternError::NotFromText(Field::Flag(Flag::SentStart))));
}

#[test]
fn lower_phrases_lowercase_their_tokens() {
    let mut m = PhraseMatcher::new(Field::Attr(Attr::Lower));
    m.add("CITY", &["New", "YORK"]).unwrap();
    assert_eq!(Doc::split("in new York today").run_phrases(&m), owned(&[("CITY", 1, 3)]));
    assert_eq!(m.add("CITY", &[]), Err(PatternError::Empty));
}
