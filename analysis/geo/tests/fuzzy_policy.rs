use turbo_analysis::{Annotator, Span};
use turbo_geo::{
    EntryParts, FEATURE_CLASS_CITY, GazetteerEntry, GeoEntityLinker, GeoPoint, Geocoder, InMemoryGazetteer,
    jaro_winkler,
};
use turbo_matcher::FuzzyMatchPolicy;

fn policy(distance: u8, phonetic: bool) -> FuzzyMatchPolicy {
    FuzzyMatchPolicy::new(distance, 4, phonetic, 0.1).unwrap()
}

fn entry(id: &str, name: &str, population: i64) -> GazetteerEntry {
    GazetteerEntry::new(EntryParts {
        source: "test".into(),
        record_id: id.into(),
        name: name.into(),
        alternate_names: Vec::new(),
        location: GeoPoint::new(0.0, 0.0).unwrap(),
        bounding_box: None,
        country_code: None,
        containment: Vec::new(),
        population,
        feature_class: Some(FEATURE_CLASS_CITY.into()),
        attributes: Vec::new(),
    })
    .unwrap()
}

fn gazetteer(name: &str) -> InMemoryGazetteer {
    InMemoryGazetteer::from_entries(&[entry("1", name, 1)]).unwrap()
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

#[test]
fn geo_policy_surfaces_low_similarity_phonetic_match_past_raw_similarity_decoys() {
    // Mention "Kathryn": raw Jaro-Winkler ranks one-edit names such as
    // "Kathryna" (about 0.975) and the refused decoys "Kathy" (0.943) and
    // "Kath" (0.914) above the phonetic match "Catherine" (0.757). The
    // policy admits every one-edit name and "Catherine" at 0.9 and refuses
    // both decoys. The retrieval-limit cap is applied only after that
    // policy ranking, so "Catherine" (highest population) ranks first and
    // survives the cap while the refused decoys never appear.
    let jw_catherine = jaro_winkler("kathryn", "catherine");
    assert!(jaro_winkler("kathryn", "kathy") > jw_catherine);
    assert!(jaro_winkler("kathryn", "kath") > jw_catherine);

    let mut entries: Vec<GazetteerEntry> = ('a'..='y')
        .enumerate()
        .map(|(i, suffix)| entry(&format!("f-{i}"), &format!("Kathryn{suffix}"), 1_000))
        .collect();
    entries.push(entry("phony", "Catherine", 9_000_000));
    entries.push(entry("decoy-kathy", "Kathy", 8_000_000));
    entries.push(entry("decoy-kath", "Kath", 7_000_000));
    let gaz = InMemoryGazetteer::from_entries(&entries).unwrap();
    let linker = GeoEntityLinker::new(gaz).with_fuzzy_match_policy(policy(0, true)).with_max_candidates(2);

    let text = "Kathryn";
    let linked = linker.link_mentions(text, &[Span::new(0, text.len() as u32)]).unwrap();
    assert_eq!(linked.len(), 1);
    let candidates = &linked[0];
    assert_eq!(candidates.len(), 2, "hard cap stays at max_candidates");
    assert_eq!(candidates[0].entry.record_id(), "phony");
    assert!((candidates[0].fuzzy_score - 0.9).abs() < 1e-5);
    for c in candidates {
        assert_ne!(c.entry.record_id(), "decoy-kathy");
        assert_ne!(c.entry.record_id(), "decoy-kath");
    }
}

#[test]
fn geo_policy_ranks_first_edit_matches_from_policy_order_before_limit() {
    // Twenty-six one-edit "Kathryn*" names share the highest raw similarity
    // (about 0.975) and fill the retrieval order ahead of "Pathryn" and
    // "Oathryn" (about 0.905), which are also one-edit admits but carry the
    // highest populations. Capping before the policy ranking would drop
    // both; ranking first keeps them on top of the output.
    let mut entries: Vec<GazetteerEntry> = ('a'..='z')
        .enumerate()
        .map(|(i, suffix)| entry(&format!("f-{i}"), &format!("Kathryn{suffix}"), 1_000))
        .collect();
    entries.push(entry("p1", "Pathryn", 9_000_000));
    entries.push(entry("p2", "Oathryn", 8_000_000));
    let gaz = InMemoryGazetteer::from_entries(&entries).unwrap();
    let linker = GeoEntityLinker::new(gaz).with_fuzzy_match_policy(policy(1, false)).with_max_candidates(2);

    let text = "Kathryn";
    let linked = linker.link_mentions(text, &[Span::new(0, text.len() as u32)]).unwrap();
    assert_eq!(linked[0].len(), 2);
    assert_eq!(linked[0][0].entry.record_id(), "p1");
    assert_eq!(linked[0][1].entry.record_id(), "p2");
}
