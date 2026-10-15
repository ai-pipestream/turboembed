//! Snapshots: a loaded gazetteer written and read back answers every query
//! as before, and damaged bytes are refused, never read out of bounds.

use std::path::PathBuf;

use turbo_geo::*;

fn fixture(name: &str) -> PathBuf {
    turbo_goldens::data_dir().join("fixtures").join(name)
}

/// Every record field of every lookup of every query, for comparing two
/// gazetteers.
fn answers(g: &impl Gazetteer, queries: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for q in queries {
        for e in g.lookup(q) {
            out.push(format!(
                "{q} -> {} {} {} {:?} {:?} {:?} {:?} {:?} {} {:?} {:?}",
                e.source(),
                e.record_id(),
                e.name(),
                e.alternate_names().collect::<Vec<_>>(),
                e.location(),
                e.bounding_box(),
                e.country_code(),
                e.containment().collect::<Vec<_>>(),
                e.population(),
                e.feature_class(),
                e.attributes().collect::<Vec<_>>(),
            ));
        }
    }
    for code in ["DE", "US", "JP", "FJ", "ZZ"] {
        out.push(format!("{code} -> {:?}", g.by_region(RegionCode::parse(code).unwrap()).map(|e| e.record_id())));
    }
    out
}

const QUERIES: &[&str] = &["Berlin", "springfield", "東京", "tokio", "Bavaria", "Acme HQ", "dateline depot", "Paris"];

#[test]
fn every_kind_of_gazetteer_reads_back_its_snapshot() {
    let geonames = GeoNamesGazetteer::load(fixture("gazetteer-geonames.txt")).unwrap();
    let mut bytes = Vec::new();
    geonames.write_snapshot(&mut bytes).unwrap();
    let back = GeoNamesGazetteer::from_snapshot(&bytes).unwrap();
    assert_eq!(answers(&geonames, QUERIES), answers(&back, QUERIES));
    assert_eq!(back.by_id("geonames", "2950159").unwrap().name(), "Berlin");

    let overture = OvertureGazetteer::load(fixture("gazetteer-overture.txt")).unwrap();
    let mut bytes = Vec::new();
    overture.write_snapshot(&mut bytes).unwrap();
    assert_eq!(answers(&overture, QUERIES), answers(&OvertureGazetteer::from_snapshot(&bytes).unwrap(), QUERIES));

    let user = UserGazetteer::load(fixture("gazetteer-user.txt"), "sites").unwrap();
    let mut bytes = Vec::new();
    user.write_snapshot(&mut bytes).unwrap();
    let back = UserGazetteer::from_snapshot(&bytes).unwrap();
    assert_eq!(back.source(), "sites");
    assert_eq!(back.sources(), ["sites"]);
    assert_eq!(answers(&user, QUERIES), answers(&back, QUERIES));

    // Not Berlin: its "--" would be refused in records built in code.
    let entries: Vec<GazetteerEntry> = geonames.lookup("tokio").iter().map(|e| e.to_entry()).collect();
    let memory = InMemoryGazetteer::from_entries(&entries).unwrap();
    let mut bytes = Vec::new();
    memory.write_snapshot(&mut bytes).unwrap();
    assert_eq!(answers(&memory, QUERIES), answers(&InMemoryGazetteer::from_snapshot(&bytes).unwrap(), QUERIES));
}

#[test]
fn a_snapshot_is_refused_by_the_wrong_kind_and_when_damaged() {
    let g = UserGazetteer::load(fixture("gazetteer-user.txt"), "sites").unwrap();
    let mut bytes = Vec::new();
    g.write_snapshot(&mut bytes).unwrap();
    assert!(matches!(GeoNamesGazetteer::from_snapshot(&bytes), Err(SnapshotError::Kind { .. })));
    assert!(matches!(UserGazetteer::from_snapshot(b"not a snapshot"), Err(SnapshotError::NotASnapshot)));
    let mut version = bytes.clone();
    version[8] = 9;
    assert!(matches!(UserGazetteer::from_snapshot(&version), Err(SnapshotError::Version(9))));
    for cut in [13, 20, bytes.len() / 2, bytes.len() - 1] {
        assert!(UserGazetteer::from_snapshot(&bytes[..cut]).is_err(), "cut at {cut}");
    }
    let mut longer = bytes.clone();
    longer.push(0);
    assert!(matches!(UserGazetteer::from_snapshot(&longer), Err(SnapshotError::Corrupt(_))));
    // Flip each byte in turn: every result is a refusal or a gazetteer
    // whose queries run without panicking.
    for i in 13..bytes.len() {
        let mut damaged = bytes.clone();
        damaged[i] ^= 0x5A;
        if let Ok(g) = UserGazetteer::from_snapshot(&damaged) {
            let _ = answers(&g, QUERIES);
            let _ = g.by_id("sites", "hq").map(|e| e.to_entry());
        }
    }
}
