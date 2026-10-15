//! spaCy's `PhraseMatcher` tests, ported as fixtures, from
//! `spacy/tests/matcher/test_phrase_matcher.py` of spaCy 3.8.7 (MIT
//! License, Copyright (C) 2016-2024 ExplosionAI GmbH, 2016 spaCy GmbH,
//! 2015 Matthew Honnibal). A spaCy phrase is a `Doc`; here it is the
//! token values, or a `Sequence` run through `add_sequence`.
//!
//! Not ported: callbacks, pickling, deprecation warnings, `Span` inputs,
//! and validation of spaCy's `Doc` patterns.

mod common;

use common::{doc, phrase_found};
use turbo_matcher::{Attr, Flag, PatternError, PhraseMatcher, PhraseOn};

fn labelled(rows: &[(&str, u32, u32)]) -> Vec<(String, u32, u32)> {
    rows.iter().map(|&(l, s, e)| (l.into(), s, e)).collect()
}

fn text_matcher(phrases: &[(&str, &[&str])]) -> PhraseMatcher {
    let mut b = PhraseMatcher::builder(PhraseOn::Attr(Attr::Text));
    for (label, words) in phrases {
        b.add(label, words).unwrap();
    }
    b.build().unwrap()
}

#[test]
fn issue3248_labels_counted_once() {
    let mut b = PhraseMatcher::builder(PhraseOn::Attr(Attr::Lower));
    b.add("TEST1", &["a"]).unwrap();
    b.add("TEST1", &["b"]).unwrap();
    b.add("TEST1", &["c"]).unwrap();
    b.add("TEST2", &["d"]).unwrap();
    assert_eq!(b.len(), 2);
}

#[test]
fn issue3331_and_3972_one_phrase_two_labels() {
    let m = text_matcher(&[("A", &["Barack", "Obama"]), ("B", &["Barack", "Obama"])]);
    assert_eq!(phrase_found(&m, &doc(&["Barack", "Obama", "lifts", "America"])), labelled(&[("A", 0, 2), ("B", 0, 2)]));
    let m = text_matcher(&[("A", &["New", "York"]), ("B", &["New", "York"])]);
    assert_eq!(phrase_found(&m, &doc(&["I", "live", "in", "New", "York"])), labelled(&[("A", 3, 5), ("B", 3, 5)]));
}

#[test]
fn issue10643_nested_phrases() {
    let words = ["Only", "save", "out", "the", "binary", "data", "for", "the", "individual", "components", "."];
    let mut b = PhraseMatcher::builder(PhraseOn::Attr(Attr::Text));
    b.add("0", &["binary"]).unwrap();
    b.add("1", &["binary", "data"]).unwrap();
    let d = doc(&words);
    assert_eq!(phrase_found(&b.build().unwrap(), &d), labelled(&[("0", 4, 5), ("1", 4, 6)]));
    let mut b = PhraseMatcher::builder(PhraseOn::Attr(Attr::Text));
    b.add("0", &["binary"]).unwrap();
    b.add("1", &["binary", "data"]).unwrap();
    assert!(b.remove("0"));
    assert_eq!(b.len(), 1);
    assert_eq!(phrase_found(&b.build().unwrap(), &d), labelled(&[("1", 4, 6)]));
}

#[test]
fn phrases_anywhere() {
    let d = doc(&["I", "like", "Google", "Now", "best"]);
    for (label, words) in [
        ("COMPANY", &["Google", "Now"][..]),
        ("I", &["I"][..]),
        ("ILIKE", &["I", "like"][..]),
        ("BEST", &["best"][..]),
        ("NOWBEST", &["Now", "best"][..]),
    ] {
        assert_eq!(phrase_found(&text_matcher(&[(label, words)]), &d).len(), 1, "{label}");
    }
}

#[test]
fn length_contains_and_repeated_add() {
    let mut b = PhraseMatcher::builder(PhraseOn::Attr(Attr::Text));
    assert_eq!(b.len(), 0);
    b.add("TEST", &["test"]).unwrap();
    assert_eq!(b.len(), 1);
    b.add("TEST2", &["test2"]).unwrap();
    assert_eq!(b.len(), 2);
    assert!(b.contains("TEST") && !b.contains("TEST3"));
    let m = text_matcher(&[("TEST", &["like"]), ("TEST", &["like"]), ("TEST", &["like"]), ("TEST", &["like"])]);
    assert_eq!(phrase_found(&m, &doc(&["I", "like", "Google", "Now", "best"])).len(), 1);
}

#[test]
fn two_phrases_one_label() {
    let m = text_matcher(&[("OLD_API", &["a"]), ("OLD_API", &["a", "b"])]);
    assert_eq!(phrase_found(&m, &doc(&["a", "b"])).len(), 2);
}

#[test]
fn remove_labels() {
    let d = doc(&["I", "like", "Google", "Now", "best"]);
    let mut b = PhraseMatcher::builder(PhraseOn::Attr(Attr::Text));
    b.add("TEST1", &["like"]).unwrap();
    b.add("TEST2", &["best"]).unwrap();
    assert!(b.remove("TEST1"));
    assert!(!b.contains("TEST1") && b.contains("TEST2"));
    assert!(!b.remove("TEST1"));
    assert!(!b.remove("TEST3"));
    assert_eq!(phrase_found(&b.build().unwrap(), &d), labelled(&[("TEST2", 4, 5)]));
    let mut b = PhraseMatcher::builder(PhraseOn::Attr(Attr::Text));
    b.add("TEST", &["like"]).unwrap();
    b.add("TEST2", &["like"]).unwrap();
    assert!(b.remove("TEST"));
    assert_eq!(phrase_found(&b.build().unwrap(), &d), labelled(&[("TEST2", 1, 2)]));
}

#[test]
fn string_attributes() {
    // A phrase of tags from a tagged sequence.
    let pattern = doc(&["I", "like", "cats"]).tags(&["PRON", "VERB", "NOUN"]);
    let mut b = PhraseMatcher::builder(PhraseOn::Attr(Attr::Pos));
    b.add_sequence("TEST", &pattern.seq()).unwrap();
    let m = b.build().unwrap();
    let d = doc(&["Yes", ",", "you", "hate", "dogs", "very", "much"])
        .tags(&["INTJ", "PUNCT", "PRON", "VERB", "NOUN", "ADV", "ADV"]);
    assert_eq!(phrase_found(&m, &d), labelled(&[("TEST", 2, 5)]));
    // Tokens whose text looks like the matcher's own keys are not tags.
    let d = doc(&["matcher:POS-PRON", "matcher:POS-VERB", "matcher:POS-NOUN"]).tags(&["X", "X", "X"]);
    assert!(phrase_found(&m, &d).is_empty());
}

#[test]
fn flag_attributes() {
    let mut b = PhraseMatcher::builder(PhraseOn::Flag(Flag::Punct));
    b.add_sequence("TEST", &doc(&["Hello", "world", "!"]).seq()).unwrap();
    let m = b.build().unwrap();
    let d = doc(&["No", "problem", ",", "he", "said", "."]);
    assert_eq!(phrase_found(&m, &d), labelled(&[("TEST", 0, 3), ("TEST", 3, 6)]));
}

#[test]
fn removing_overlapping_phrases() {
    let mut b = PhraseMatcher::builder(PhraseOn::Attr(Attr::Text));
    for p in [&["this"][..], &["this", "is"], &["this", "is", "a"], &["this", "is", "a", "word"]] {
        b.add("THIS", p).unwrap();
    }
    assert!(b.remove("THIS"));
    assert!(!b.contains("THIS"));
}

#[test]
fn matches_in_text_order_with_labels() {
    let m = text_matcher(&[("A", &["hello", "world"]), ("B", &["test"])]);
    let d = doc(&["...", "hello", "world", "this", "is", "a", "test"]);
    assert_eq!(phrase_found(&m, &d), labelled(&[("A", 1, 3), ("B", 6, 7)]));
}

#[test]
fn only_the_phrase_matches() {
    let m = text_matcher(&[("TEST", &["helllo"])]);
    assert!(phrase_found(&m, &doc(&["hello", "world"])).is_empty());
    let words = [
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
        ",",
        "and",
        "Spans",
        "and",
        "Docs",
        "everywhere",
        ".",
    ];
    let m = text_matcher(&[("SPACY", &["Spans", "and", "Docs"])]);
    assert_eq!(phrase_found(&m, &doc(&words)).len(), 3);
}

#[test]
fn bad_phrases_are_refused() {
    let mut b = PhraseMatcher::builder(PhraseOn::Attr(Attr::Text));
    assert_eq!(b.add("TEST", &[]), Err(PatternError::EmptyPhrase));
    assert_eq!(b.add_flags("TEST", &[true]), Err(PatternError::PhraseValue));
    assert_eq!(b.add("", &["x"]), Err(PatternError::EmptyLabel));
    let mut b = PhraseMatcher::builder(PhraseOn::Flag(Flag::Punct));
    assert_eq!(b.add("TEST", &["x"]), Err(PatternError::PhraseValue));
}

#[test]
fn a_label_added_again_after_remove_has_only_its_new_phrases() {
    let mut b = PhraseMatcher::builder(PhraseOn::Attr(Attr::Text));
    b.add("A", &["x"]).unwrap();
    assert!(b.remove("A"));
    b.add("A", &["y"]).unwrap();
    assert_eq!(phrase_found(&b.build().unwrap(), &doc(&["x", "y"])), labelled(&[("A", 1, 2)]));
    // A refused phrase does not bring a removed label back.
    let mut b = PhraseMatcher::builder(PhraseOn::Attr(Attr::Text));
    b.add("A", &["x"]).unwrap();
    assert!(b.remove("A"));
    assert_eq!(b.add("A", &[]), Err(PatternError::EmptyPhrase));
    assert!(!b.contains("A"));
    assert_eq!(b.len(), 0);
}
