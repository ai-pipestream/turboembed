//! The builtin index's lookups by language and component.

use turbo_zoo::{Component, Index, LicenceClass};

#[test]
fn the_neural_checkpoints_are_found_by_language_and_component() {
    let index = Index::builtin();
    let ids = |l: &str, c| index.find(l, c).map(|a| a.id.as_str().to_owned()).collect::<Vec<_>>();
    assert_eq!(ids("en", Component::ExtractiveQa), ["deepset-roberta-base-squad2"]);
    assert_eq!(ids("en", Component::SchemaExtractor), ["fastino-gliner2-base-v1"]);
    assert_eq!(ids("en", Component::RelationExtractor), ["jackboyla-glirel-large-v0"]);
    assert_eq!(ids("en", Component::EntityExtractor), ["urchade-gliner-small-v2.1"]);
    assert_eq!(ids("mul", Component::EntityExtractor), ["urchade-gliner-multi-v2.1"]);
    assert_eq!(ids("en", Component::ZeroShotCategorizer), ["cross-encoder-nli-minilm2-l6-h768"]);
    // The tokenizers are fetched by id beside their models: no component,
    // so OpenNLP's lookup never takes one for a tokenizer model.
    assert!(index.list().iter().filter(|a| a.id.as_str().ends_with("-tokenizer")).all(|a| a.component.is_none()));
}

#[test]
fn find_within_keeps_the_classes_asked_for_least_restrictive_first() {
    let index = Index::builtin();
    let ids = |at_most| {
        let found = index.find_within("en", Component::SentenceDetector, at_most);
        found.iter().map(|a| a.id.as_str().to_owned()).collect::<Vec<_>>()
    };
    assert_eq!(ids(LicenceClass::Permissive), ["turbo-en-masc-3.0.0-sentence-1"]);
    let all = ids(LicenceClass::ShareAlike);
    assert_eq!(all, ["turbo-en-masc-3.0.0-sentence-1", "opennlp-en-ud-ewt-sentence-1.3-2.5.4"]);
    assert_eq!(index.find("en", Component::SentenceDetector).count(), 2);
    assert!(index.find_within("en", Component::ExtractiveQa, LicenceClass::Permissive).is_empty());
    // Trained on SNLI and MultiNLI, both share-alike.
    assert!(index.find_within("en", Component::ZeroShotCategorizer, LicenceClass::Permissive).is_empty());
}
