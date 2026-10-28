use turbo_analysis::{Annotator, Span};
use turbo_geo::{
    EntryParts, FEATURE_CLASS_CITY, GazetteerEntry, GeoEntityLinker, GeoPoint, Geocoder, InMemoryGazetteer,
};
use turbo_matcher::FuzzyMatchPolicy;

fn policy(distance: u8, phonetic: bool) -> FuzzyMatchPolicy {
    FuzzyMatchPolicy::new(distance, 4, phonetic, 0.1).unwrap()
}

fn gazetteer(name: &str) -> InMemoryGazetteer {
    InMemoryGazetteer::from_entries(&[GazetteerEntry::new(EntryParts {
        source: "test".into(),
        record_id: "1".into(),
        name: name.into(),
        alternate_names: Vec::new(),
        location: GeoPoint::new(0.0, 0.0).unwrap(),
        bounding_box: None,
        country_code: None,
        containment: Vec::new(),
        population: 1,
        feature_class: Some(FEATURE_CLASS_CITY.into()),
        attributes: Vec::new(),
    })
    .unwrap()])
    .unwrap()
}

#[test]
fn geo_policy_recovers_transposition_and_phonetic_names_with_source_spans() {
    let text = "Califronia";
    let span = Span::new(0, text.len() as u32);
    // The default Combined scorer already recovers the transposition, so a
    // disabled policy must reproduce that baseline result exactly.
    let baseline = GeoEntityLinker::new(gazetteer("California"));
    let expected = baseline.resolve(text, &[span]).unwrap();
    let disabled = GeoEntityLinker::new(gazetteer("California")).with_fuzzy_match_policy(FuzzyMatchPolicy::default());
    assert_eq!(disabled.resolve(text, &[span]).unwrap(), expected);

    let fuzzy = GeoEntityLinker::new(gazetteer("California")).with_fuzzy_match_policy(policy(1, false));
    let resolved = fuzzy.resolve(text, &[span]).unwrap();
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].mention(), span);
    assert_eq!(resolved[0].entry().name(), "California");
    assert!(resolved[0].confidence() > 0.0 && resolved[0].confidence() <= 1.0);

    // Phonetic recovery is new: the baseline scorer cannot relate "Kathryn"
    // to "Catherine" at its similarity threshold.
    let text = "Kathryn";
    let span = Span::new(0, text.len() as u32);
    let plain = GeoEntityLinker::new(gazetteer("Catherine"));
    assert!(plain.resolve(text, &[span]).unwrap().is_empty());
    let phonetic = plain.with_fuzzy_match_policy(policy(0, true));
    let resolved = phonetic.resolve(text, &[span]).unwrap();
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].mention(), span);
    assert_eq!(resolved[0].entry().name(), "Catherine");
}

#[test]
fn geo_policy_changes_hash_only_when_enabled() {
    let plain = GeoEntityLinker::new(gazetteer("California"));
    let disabled = GeoEntityLinker::new(gazetteer("California")).with_fuzzy_match_policy(FuzzyMatchPolicy::default());
    let enabled = GeoEntityLinker::new(gazetteer("California")).with_fuzzy_match_policy(policy(1, false));
    assert_eq!(plain.config_hash(), disabled.config_hash());
    assert_ne!(plain.config_hash(), enabled.config_hash());
}
