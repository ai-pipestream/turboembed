//! A synonym's replacement terms are made with the field's own tokenizer
//! (`#2339`): a whitespace field indexes `foo-bar` as one term, so the
//! query plan must ask for that one term, not the word tokenizer's two.

use std::sync::Arc;

use turbo_chunker::terms::TermForm;
use turbo_matcher::PhraseSynonymExpander;
use turbo_query::{FieldQuerySpec, FieldTokenizerKind, OnePassQueryAnalyzer};

fn analyzer() -> OnePassQueryAnalyzer {
    let mut syns = PhraseSynonymExpander::lower();
    syns.add_equivalent_group(&[&["x"], &["foo-bar"]]).unwrap();
    syns.add_equivalent_group(&[&["y"], &["up-to-date", "list"]]).unwrap();
    OnePassQueryAnalyzer::builder()
        .with_synonyms(Arc::new(syns))
        .add_field(
            FieldQuerySpec::new("ws").with_tokenizer(FieldTokenizerKind::Whitespace).with_term_form(TermForm::Folded),
        )
        .add_field(
            FieldQuerySpec::new("word").with_tokenizer(FieldTokenizerKind::Word).with_term_form(TermForm::Folded),
        )
        .build()
        .unwrap()
}

#[test]
fn a_whitespace_field_keeps_a_hyphenated_replacement_whole() {
    let plan = analyzer().analyze("find x here").unwrap();
    let ws = &plan.fields[0];
    assert_eq!(ws.synonyms.len(), 1, "{:?}", ws.synonyms);
    assert_eq!(ws.synonyms[0].surface, "x");
    assert_eq!(ws.synonyms[0].terms, vec!["foo-bar"]);
    // The replacement is the term the field's own analysis gives the text.
    let direct = analyzer().analyze("foo-bar").unwrap();
    let indexed: Vec<&str> = direct.fields[0].bm25_terms.iter().map(|t| t.term.as_str()).collect();
    assert_eq!(indexed, vec!["foo-bar"]);
}

#[test]
fn a_word_field_splits_the_same_replacement() {
    let plan = analyzer().analyze("find x here").unwrap();
    let word = &plan.fields[1];
    assert_eq!(word.synonyms.len(), 1, "{:?}", word.synonyms);
    assert_eq!(word.synonyms[0].terms, vec!["foo", "bar"]);
}

#[test]
fn a_multi_word_replacement_follows_each_field() {
    let plan = analyzer().analyze("the y").unwrap();
    assert_eq!(plan.fields[0].synonyms[0].terms, vec!["up-to-date", "list"]);
    assert_eq!(plan.fields[1].synonyms[0].terms, vec!["up", "to", "date", "list"]);
}
