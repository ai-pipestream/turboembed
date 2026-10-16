//! The overlay: additions first, suppressed base records hidden from every
//! query, and the region representative chosen between the two sides.

use std::path::PathBuf;

use turbo_geo::*;

fn fixture(name: &str) -> PathBuf {
    turbo_goldens::data_dir().join("fixtures").join(name)
}

fn overlay() -> OverlayGazetteer<&'static BundledGazetteer, UserGazetteer> {
    let additions = UserGazetteer::load(fixture("gazetteer-user.txt"), "sites").unwrap();
    let rules = UserGazetteer::load_suppressions(fixture("gazetteer-suppressions.txt")).unwrap();
    OverlayGazetteer::new(BundledGazetteer::get(), Some(additions), rules).unwrap()
}

fn sources_and_ids<'a>(found: &[Entry<'a>]) -> Vec<(&'a str, &'a str)> {
    found.iter().map(|e| (e.source(), e.record_id())).collect()
}

#[test]
fn additions_come_first_and_rules_hide_base_records() {
    let o = overlay();
    let base = BundledGazetteer::get();
    // Every bundled Springfield is a US city, so the rule hides them all;
    // the plant's alternate name is an addition.
    assert_eq!(base.lookup("Springfield").len(), 5);
    assert_eq!(sources_and_ids(&o.lookup("springfield")), [("sites", "plant-1")]);
    // "Paris" in FR is hidden, the US Paris records stay, after the lab.
    let paris = o.lookup("Paris");
    assert_eq!(sources_and_ids(&paris[..1]), [("sites", "paris-lab")]);
    assert!(paris[1..].iter().all(|e| e.source() == "naturalearth" && e.country_code() != Some("FR")));
    assert_eq!(paris.len(), base.lookup("Paris").len());
    // The feature class filter ignores case: "city" hides the CITY London records.
    assert!(o.lookup("London").is_empty());
    // "Sao Paulo" with BR hides the bundled São Paulo whatever the spelling.
    assert!(o.lookup("São Paulo").is_empty());
    assert_eq!(sources_and_ids(&o.lookup("são paulo office")), [("sites", "zz")]);
}

#[test]
fn hidden_records_are_hidden_from_ids_and_regions() {
    let o = overlay();
    assert!(o.by_id("naturalearth", "1159151613").is_none(), "Paris, FR");
    assert!(o.by_id("naturalearth", "1159151577").is_none(), "London, GB");
    assert_eq!(o.by_id("sites", "hq").unwrap().name(), "Acme Headquarters");
    assert!(o.by_id("naturalearth", "1159146271").is_none(), "Springfield, Illinois");
    // The base's representative of FR is Paris, which is hidden, so the
    // base offers none; the lab is the additions' representative.
    assert_eq!(o.by_region(RegionCode::parse("FR").unwrap()).unwrap().record_id(), "paris-lab");
    // The most populous US record is New York, more populous than the HQ.
    assert_eq!(o.by_region(RegionCode::parse("US").unwrap()).unwrap().name(), "New York");
    // GB: London is hidden, so the base has no representative.
    assert!(o.by_region(RegionCode::parse("GB").unwrap()).is_none());
    assert_eq!(o.sources(), ["sites", "naturalearth"]);
}

#[test]
fn an_overlay_must_change_something() {
    let none: Option<UserGazetteer> = None;
    assert_eq!(OverlayGazetteer::new(BundledGazetteer::get(), none, vec![]).err(), Some(EmptyOverlay));
    let rules = vec![Suppression::name_only("Paris").unwrap()];
    let only_rules = OverlayGazetteer::<_, UserGazetteer>::new(BundledGazetteer::get(), None, rules).unwrap();
    assert!(only_rules.lookup("paris").is_empty());
    assert_eq!(only_rules.sources(), ["naturalearth"]);
}

#[test]
fn ties_between_the_sides_go_to_the_additions() {
    let entry = |source: &str, id: &str, population: i64| {
        GazetteerEntry::new(EntryParts {
            source: source.into(),
            record_id: id.into(),
            name: "Place".into(),
            alternate_names: vec![],
            location: GeoPoint::new(0.0, 0.0).unwrap(),
            bounding_box: None,
            country_code: Some("NL".into()),
            containment: vec![],
            population,
            feature_class: Some("POI".into()),
            attributes: vec![],
        })
        .unwrap()
    };
    let base = InMemoryGazetteer::from_entries(&[entry("b", "1", 10)]).unwrap();
    let added = InMemoryGazetteer::from_entries(&[entry("a", "1", 10)]).unwrap();
    let o = OverlayGazetteer::new(&base, Some(&added), vec![]).unwrap();
    assert_eq!(o.by_region(RegionCode::parse("NL").unwrap()).unwrap().source(), "a");
    assert_eq!(sources_and_ids(&o.lookup("place")), [("a", "1"), ("b", "1")]);
}
