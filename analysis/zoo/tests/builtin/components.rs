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
    for language in ["es", "ca", "fi", "it"] {
        assert!(ids(language, Component::DependencyParser).is_empty());
    }
    assert_eq!(
        ids("en", Component::DocumentCategorizer),
        [
            "turbo-en-goemotions-minilm-l6-v2-head-1",
            "turbo-en-goemotions-bge-small-en-v1.5-head-1",
            "turbo-en-civil-comments-minilm-l6-v2-head-1",
            "turbo-en-civil-comments-minilm-l6-v2-context-head-1",
            "turbo-en-civil-comments-bge-small-en-v1.5-head-1",
            "turbo-en-civil-comments-bge-small-en-v1.5-context-head-1",
        ]
    );
    // The tokenizers are fetched by id beside their models: no component,
    // so component lookup does not confuse a tokenizer asset with a model.
    assert!(index.list().iter().filter(|a| a.id.as_str().ends_with("-tokenizer")).all(|a| a.component.is_none()));
}

#[test]
fn retired_tasks_are_absent_and_neural_licence_filters_remain() {
    let index = Index::builtin();
    assert!(index.find("en", Component::SentenceDetector).next().is_none());
    assert!(index.find_within("en", Component::ExtractiveQa, LicenceClass::Permissive).is_empty());
    // Trained on SNLI and MultiNLI, both share-alike.
    assert!(index.find_within("en", Component::ZeroShotCategorizer, LicenceClass::Permissive).is_empty());
}
