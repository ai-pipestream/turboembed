//! The bundled Natural Earth gazetteer: its table, its records and the
//! lookups the folding rule promises.

use turbo_geo::{BUNDLED_TABLE, BundledGazetteer, FEATURE_CLASS_CITY, Gazetteer, RegionCode};

#[test]
fn every_data_row_of_the_table_is_a_record() {
    let rows = BUNDLED_TABLE.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).count();
    let g = BundledGazetteer::get();
    assert_eq!(rows, 7342);
    assert_eq!(g.len(), rows);
    assert_eq!(g.sources(), ["naturalearth"]);
    assert!(BUNDLED_TABLE.is_ascii());
}

#[test]
fn a_record_has_every_field_of_its_row() {
    let g = BundledGazetteer::get();
    let paris = g.by_id("naturalearth", "1159151613").expect("Paris is bundled");
    assert_eq!(paris.name(), "Paris");
    assert_eq!(paris.alternate_names().count(), 0);
    assert_eq!(paris.location().latitude(), 48.85809);
    assert_eq!(paris.location().longitude(), 2.35299);
    assert_eq!(paris.country_code(), Some("FR"));
    assert_eq!(paris.containment().collect::<Vec<_>>(), ["Ile-de-France"]);
    assert_eq!(paris.population(), 9_904_000);
    assert_eq!(paris.feature_class(), Some(FEATURE_CLASS_CITY));
    let attributes: Vec<(&str, &str, &str)> = paris.attributes().map(|a| (a.key, a.value, a.source)).collect();
    assert_eq!(
        attributes,
        [
            ("wikidata", "Q90", "naturalearth"),
            ("geonames", "2988507", "naturalearth"),
            ("whosonfirst", "101751119", "naturalearth")
        ]
    );
    assert_eq!(paris.bounding_box(), None);
}

#[test]
fn candidates_are_ranked_by_population() {
    let g = BundledGazetteer::get();
    let found: Vec<(&str, u64)> =
        g.lookup("springfield").iter().map(|e| (e.containment().next().unwrap(), e.population())).collect();
    assert_eq!(found.len(), 5);
    assert_eq!(found[0], ("Massachusetts", 421_780));
    assert!(found.windows(2).all(|w| w[0].1 >= w[1].1), "{found:?}");
    let london: Vec<&str> = g.lookup("London").iter().map(|e| e.country_code().unwrap()).collect();
    assert_eq!(london[..2], ["GB", "CA"]);
}

#[test]
fn queries_fold_like_the_names() {
    let g = BundledGazetteer::get();
    let plain = g.lookup("Sao Paulo");
    assert!(!plain.is_empty());
    for query in ["São Paulo", "SAO PAULO", "  sao\tpaulo ", "Sa\u{303}o-Paulo"] {
        assert_eq!(g.lookup(query), plain, "{query}");
    }
    assert!(g.lookup("").is_empty());
    assert!(g.lookup("--").is_empty());
    assert!(g.lookup("Sao").is_empty());
}

#[test]
fn a_region_is_represented_by_its_most_populous_record() {
    let g = BundledGazetteer::get();
    let fr = g.by_region(RegionCode::parse("fr").unwrap()).unwrap();
    assert_eq!(fr.name(), "Paris");
    let us = g.by_region(RegionCode::parse("US").unwrap()).unwrap();
    assert_eq!(us.name(), "New York");
    assert!(g.by_region(RegionCode::parse("ZZ").unwrap()).is_none());
}

#[test]
fn ids_are_scoped_to_their_source_and_not_folded() {
    let g = BundledGazetteer::get();
    assert!(g.by_id("naturalearth", "1159151613").is_some());
    assert!(g.by_id("geonames", "1159151613").is_none());
    assert!(g.by_id("NaturalEarth", "1159151613").is_none());
    assert!(g.by_id("naturalearth", " 1159151613").is_none());
}

#[test]
fn a_fresh_parse_gives_the_same_index() {
    let fresh = BundledGazetteer::parse().unwrap();
    let shared = BundledGazetteer::get();
    for name in ["Springfield", "Paris", "san jose", "Victoria"] {
        let a: Vec<_> = fresh.lookup(name).iter().map(|e| e.record_id()).collect();
        let b: Vec<_> = shared.lookup(name).iter().map(|e| e.record_id()).collect();
        assert_eq!(a, b);
    }
    assert_eq!(fresh.key_count(), shared.key_count());
}
