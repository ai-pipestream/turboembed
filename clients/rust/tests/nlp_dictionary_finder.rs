//! `OnePassDocumentNlp` with a dictionary name finder (`#2341`): the
//! finder reads the same cleaned view the tokens were made on, and its
//! entities come back on the caller's source text.

use turbo_client::nlp::OnePassDocumentNlp;
use turbo_namefind::{Dictionary, DictionaryNameFinder};

fn nlp() -> OnePassDocumentNlp {
    let dict = Dictionary::read("case-insensitive\nORG\tAlphabet\n").unwrap();
    OnePassDocumentNlp::builder().with_dictionary_finder(DictionaryNameFinder::new(dict).unwrap()).build().unwrap()
}

#[test]
fn a_dictionary_name_is_found_in_plain_text() {
    let source = "Alphabet builds tools.";
    let out = nlp().analyze(source).expect("the finder and the tokens share one view");
    assert_eq!(out.affirmed_entities.len(), 1, "{:?}", out.affirmed_entities);
    let hit = &out.affirmed_entities[0];
    assert_eq!((hit.surface.as_str(), hit.kind.as_str()), ("Alphabet", "ORG"));
}

#[test]
fn a_dictionary_name_is_found_where_cleaning_changes_the_text() {
    // A soft hyphen inside the name and a zero-width space before it: the
    // cleaned view reads "Alphabet", the source does not.
    let source = "Today \u{200B}Alpha\u{00AD}bet builds tools.";
    let out = nlp().analyze(source).expect("the finder and the tokens share one view");
    assert_eq!(out.affirmed_entities.len(), 1, "{:?}", out.affirmed_entities);
    let hit = &out.affirmed_entities[0];
    assert_eq!(hit.kind, "ORG");
    assert_eq!(hit.surface.replace('\u{00AD}', ""), "Alphabet", "{:?}", hit.surface);
}

#[test]
fn a_batch_finds_the_name_in_every_document() {
    let out = nlp().analyze_batch(&["Alphabet builds tools.", "No name here.", "We met Alphabet today."]);
    let found: Vec<usize> = out.into_iter().map(|r| r.unwrap().affirmed_entities.len()).collect();
    assert_eq!(found, vec![1, 0, 1]);
}
