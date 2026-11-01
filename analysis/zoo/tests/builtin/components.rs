//! The builtin index's lookups by language and component.

use turbo_zoo::{Component, Index, Kind, LicenceClass};

#[test]
fn the_neural_checkpoints_are_found_by_language_and_component() {
    let index = Index::builtin();
    let ids = |l: &str, c| index.find(l, c).map(|a| a.id.as_str().to_owned()).collect::<Vec<_>>();
    assert_eq!(ids("en", Component::ExtractiveQa), ["deepset-roberta-base-squad2"]);
    assert_eq!(ids("en", Component::SchemaExtractor), ["fastino-gliner2-base-v1"]);
    assert_eq!(ids("en", Component::RelationExtractor), ["jackboyla-glirel-large-v0"]);
    assert_eq!(
        ids("en", Component::EntityExtractor),
        ["turbo-en-wnut17-modernbert-base-l12-open-span-1", "urchade-gliner-small-v2.1"]
    );
    assert_eq!(ids("mul", Component::EntityExtractor), ["urchade-gliner-multi-v2.1"]);
    assert_eq!(ids("en", Component::ZeroShotCategorizer), ["cross-encoder-nli-minilm2-l6-h768"]);
    assert_eq!(ids("en", Component::SparseEncoder), ["opensearch-neural-sparse-encoding-v1"]);
    assert_eq!(ids("en", Component::LateInteractionEncoder), ["colbert-ir-colbertv2.0"]);
    assert_eq!(ids("en", Component::AspectSentiment), ["turbo-en-masc-3.0.0-aspect-sentiment-1"]);
    assert_eq!(ids("en", Component::Pos), ["turbo-en-masc-pos-modernbert-base-1"]);
    for language in ["es", "ca", "fi", "it"] {
        assert!(ids(language, Component::DependencyParser).is_empty());
    }
    assert_eq!(
        ids("en", Component::DocumentCategorizer),
        [
            "turbo-en-masc-3.0.0-modernbert-base-polarity-head-1",
            "turbo-en-goemotions-modernbert-base-head-1",
            "turbo-en-civil-comments-modernbert-base-head-1",
            "turbo-en-civil-comments-modernbert-base-context-head-1",
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

#[test]
fn every_listed_model_names_a_component_except_the_tokenizers() {
    let index = Index::builtin();
    let missing: Vec<&str> = index
        .list()
        .iter()
        .filter(|a| a.kind == Kind::Model && a.component.is_none() && !a.id.as_str().ends_with("-tokenizer"))
        .map(|a| a.id.as_str())
        .collect();
    assert!(missing.is_empty(), "models with no component: {missing:?}");
}

#[test]
fn the_retrieval_encoders_and_the_aspect_heads_keep_their_classes() {
    let index = Index::builtin();
    // MS MARCO keeps both retrieval encoders and the legacy MiniLM aspect head local-only.
    assert!(index.find_within("en", Component::SparseEncoder, LicenceClass::Permissive).is_empty());
    assert!(index.find_within("en", Component::LateInteractionEncoder, LicenceClass::Permissive).is_empty());
    assert!(index.find_within("en", Component::AspectSentiment, LicenceClass::Permissive).is_empty());
    let doccat = index.find_within("en", Component::DocumentCategorizer, LicenceClass::Permissive);
    assert_eq!(
        doccat.iter().map(|a| a.id.as_str()).collect::<Vec<_>>(),
        [
            "turbo-en-masc-3.0.0-modernbert-base-polarity-head-1",
            "turbo-en-goemotions-modernbert-base-head-1",
            "turbo-en-civil-comments-modernbert-base-head-1",
            "turbo-en-civil-comments-modernbert-base-context-head-1",
        ]
    );
}

#[test]
fn the_component_constants_round_trip() {
    for c in Component::ALL {
        assert_eq!(c.name().parse::<Component>(), Ok(*c));
    }
    assert!("COMPONENT_EMBEDDER".parse::<Component>().is_err());
}
