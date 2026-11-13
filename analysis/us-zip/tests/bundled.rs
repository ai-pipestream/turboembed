//! The bundled tables, read through the public API.

use turbo_geo::{ATTRIBUTE_KEY_FIPS, ATTRIBUTE_KEY_GEOID, ATTRIBUTE_KEY_ZCTA};
use turbo_us_zip::{COUNTY_TABLE, PLACE_TABLE, PREFIX_TABLE, SOURCE_CENSUS_ZCTA, UsZipTable, ZCTA_TABLE, zcta_entry};

#[test]
fn bundled_tables_parse_whole() {
    let t = UsZipTable::bundled();
    assert_eq!(t.len(), 33_791);
    assert_eq!(t.zctas().count(), 33_791);
    let fresh = UsZipTable::parse(ZCTA_TABLE, PLACE_TABLE, COUNTY_TABLE, PREFIX_TABLE).unwrap();
    assert_eq!(fresh.len(), t.len());
}

#[test]
fn a_zip_code_finds_its_state_county_and_place() {
    let t = UsZipTable::bundled();
    let z = t.zcta("62704").unwrap();
    assert_eq!(z.code(), "62704");
    assert_eq!(z.state(), "IL");
    let (county, share) = z.counties().next().unwrap();
    assert_eq!((county.name(), county.geoid().as_str(), county.state()), ("Sangamon County", "17167", "IL"));
    assert!(share > 900, "{share}");
    let (place, _) = z.places().next().unwrap();
    assert_eq!((place.name(), place.kind(), place.geoid().as_str()), ("Springfield", "city", "1772000"));
    assert!(z.bounding_box().contains(z.point()));
    assert!(z.land_km2() > 1.0);
}

#[test]
fn zip_plus_four_is_the_same_zcta() {
    let t = UsZipTable::bundled();
    assert_eq!(t.zcta("62704-1234").unwrap().code(), "62704");
    for bad in ["62704-123", "62704-12345", "62704 1234", "6270", "627041", "", "abcde", "6270\u{663}"] {
        assert!(t.zcta(bad).is_none(), "{bad:?}");
    }
}

#[test]
fn digits_of_other_scripts_are_no_zip_code() {
    let t = UsZipTable::bundled();
    // Fullwidth and Arabic-Indic digits are not the ASCII digits ZIP Codes are written in.
    assert!(t.zcta("\u{ff16}\u{ff12}\u{ff17}\u{ff10}\u{ff14}").is_none());
    assert!(t.zcta("\u{666}\u{662}\u{667}\u{660}\u{664}").is_none());
}

#[test]
fn a_zip_code_without_a_zcta_finds_nothing() {
    let t = UsZipTable::bundled();
    // A ZIP Code of a single organization and a military ZIP Code.
    assert!(t.zcta("00501").is_none());
    assert!(t.zcta("09012").is_none());
}

#[test]
fn puerto_rico_and_accented_place_names_are_read() {
    let t = UsZipTable::bundled();
    assert_eq!(t.zcta("00601").unwrap().state(), "PR");
    let accented = t.zctas().flat_map(|z| z.places().map(|(p, _)| p.name())).find(|n| !n.is_ascii());
    assert!(accented.is_some(), "some place name has a letter outside ASCII");
}

#[test]
fn every_point_is_inside_its_box() {
    for z in UsZipTable::bundled().zctas() {
        assert!(z.bounding_box().contains(z.point()), "{z:?}");
        assert!(z.counties().next().is_some(), "{z:?}");
        let shares: Vec<u16> = z.counties().map(|(_, s)| s).collect();
        assert!(shares.windows(2).all(|w| w[0] >= w[1]), "{z:?}");
    }
}

#[test]
fn prefixes_list_their_states() {
    let t = UsZipTable::bundled();
    let s: Vec<(&str, u32)> = t.prefix_states("627").collect();
    assert_eq!(s.len(), 1);
    assert_eq!(s[0].0, "IL");
    let border: Vec<&str> = t.prefix_states("739").map(|(s, _)| s).collect();
    assert_eq!(border, ["OK", "TX"]);
    assert_eq!(t.prefix_states("000").count(), 0);
    assert_eq!(t.prefix_states("62").count(), 0);
}

#[test]
fn places_and_counties_are_found_by_geoid() {
    let t = UsZipTable::bundled();
    assert_eq!(t.place("1772000").unwrap().state(), "IL");
    assert_eq!(t.county("17167").unwrap().name(), "Sangamon County");
    assert!(t.place("177200").is_none());
    assert!(t.county("99999").is_none());
}

#[test]
fn a_zcta_becomes_a_gazetteer_record() {
    let t = UsZipTable::bundled();
    let e = zcta_entry(&t.zcta("62704").unwrap());
    assert_eq!((e.source(), e.record_id(), e.name()), (SOURCE_CENSUS_ZCTA, "62704", "62704"));
    assert_eq!(e.country_code(), Some("US"));
    assert_eq!(e.containment(), ["IL", "Sangamon County", "Springfield"]);
    assert_eq!(e.feature_class(), Some("POSTAL"));
    assert_eq!(e.attribute(ATTRIBUTE_KEY_ZCTA).unwrap().value(), "62704");
    assert_eq!(e.attribute(ATTRIBUTE_KEY_FIPS).unwrap().value(), "17167");
    assert_eq!(e.attribute(ATTRIBUTE_KEY_GEOID).unwrap().value(), "1772000");
    assert!(e.bounding_box().unwrap().contains(e.location()));
}

#[test]
fn a_border_zcta_record_names_one_state() {
    let t = UsZipTable::bundled();
    for z in t.zctas() {
        let e = zcta_entry(&z);
        let c = e.containment();
        if let Some(county) = c.get(1) {
            let found = z.counties().find(|(k, _)| k.name() == county).unwrap().0;
            assert_eq!(found.state(), z.state(), "{z:?}");
        }
    }
}

#[test]
fn a_border_zcta_keeps_its_prefix_state() {
    let t = UsZipTable::bundled();
    // Evanston, Wyoming: most of the ZCTA's land is in Utah.
    let z = t.zcta("82930").unwrap();
    assert_eq!(z.state(), "WY");
    assert_eq!(zcta_entry(&z).containment()[0], "WY");
    assert_eq!(zcta_entry(&z).containment()[1], "Uinta County");
    // American Samoa's ZCTA in Hawaii's prefix 967 has no land in Hawaii.
    assert_eq!(t.zcta("96799").unwrap().state(), "AS");
    // Each ZCTA's state is the main state of its prefix, unless it holds
    // no land there.
    for z in t.zctas() {
        let code = z.code();
        let main = t.prefix_states(&code[..3]).max_by_key(|&(_, n)| n).unwrap().0;
        if z.state() != main {
            assert!(z.counties().all(|(c, _)| c.state() != main), "{z:?}");
        }
    }
}
