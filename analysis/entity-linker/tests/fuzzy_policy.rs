use turbo_analysis::Annotator;
use turbo_entity_linker::{CatalogProvider, EntityLinker, KbEntity, KbProvider, KnowledgeBase, LinkerConfig};
use turbo_matcher::FuzzyMatchPolicy;

fn policy(distance: u8, phonetic: bool) -> FuzzyMatchPolicy {
    FuzzyMatchPolicy::new(distance, 4, phonetic, 0.1).unwrap()
}

fn provider(name: &str) -> CatalogProvider {
    let entity = KbEntity::new("E1", name).with_alias(name, 1.0).with_types(["PERSON"]).with_prior(1.0);
    CatalogProvider::new(KnowledgeBase::from_entities("test", vec![entity]).unwrap())
}

#[test]
fn entity_policy_recovers_transposition_and_phonetic_aliases() {
    let california = provider("California");
    assert!(california.generate_candidates("Califronia", None, 8, 1.0).is_empty());
    let hits = california.generate_candidates_with_policy("Califronia", None, 8, 1.0, policy(1, false));
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].kb_id, "E1");
    assert!(hits[0].match_similarity > 0.0 && hits[0].match_similarity < 1.0);

    let catherine = provider("Catherine");
    let hits = catherine.generate_candidates_with_policy("Kathryn", None, 8, 1.0, policy(0, true));
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].matched_alias, "Catherine");
}

#[test]
fn entity_policy_is_validated_and_changes_hash_only_when_enabled() {
    let plain = EntityLinker::new().add_provider(provider("Catherine"));
    let disabled = EntityLinker::new()
        .add_provider(provider("Catherine"))
        .with_config(LinkerConfig { fuzzy_policy: FuzzyMatchPolicy::default(), ..LinkerConfig::default() });
    let enabled = EntityLinker::new()
        .add_provider(provider("Catherine"))
        .with_config(LinkerConfig { fuzzy_policy: policy(1, false), ..LinkerConfig::default() });
    assert_eq!(plain.config_hash(), disabled.config_hash());
    assert_ne!(plain.config_hash(), enabled.config_hash());
    assert!(LinkerConfig::default().validate().is_ok());
    assert!(enabled.config().validate().is_ok());
}
