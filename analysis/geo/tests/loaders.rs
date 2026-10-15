//! The table loaders: what each reads from a row, what it refuses and on
//! which line, and that a large table read on several threads gives the
//! index one thread gives.

use std::path::PathBuf;

use turbo_geo::*;

fn fixture(name: &str) -> PathBuf {
    turbo_goldens::data_dir().join("fixtures").join(name)
}

fn ids(found: &[Entry<'_>]) -> Vec<String> {
    found.iter().map(|e| e.record_id().to_string()).collect()
}

fn row_problem(e: LoadError) -> (usize, RowProblem) {
    match e {
        LoadError::Row { line, problem } => (line, problem),
        other => panic!("expected a row problem, got {other:?}"),
    }
}

/// A GeoNames row with the given cells in columns 0 to 8 and 14.
fn geonames_row(
    id: &str,
    name: &str,
    ascii: &str,
    alternates: &str,
    lat: &str,
    lon: &str,
    class: &str,
    cc: &str,
    pop: &str,
) -> String {
    [id, name, ascii, alternates, lat, lon, class, "PPL", cc, "", "", "", "", "", pop, "", "0", "", "2026-01-01"]
        .join("\t")
}

#[test]
fn geonames_rows_give_their_names_location_class_and_population() {
    let g = GeoNamesGazetteer::load(fixture("gazetteer-geonames.txt")).unwrap();
    assert_eq!(g.sources(), ["geonames"]);
    let berlin = g.by_id("geonames", "2950159").unwrap();
    // The ASCII name equals the name; blanks, the name itself and repeats
    // are dropped; "--" stays on the record but is not indexed.
    assert_eq!(
        berlin.alternate_names().collect::<Vec<_>>(),
        ["BER", "Berlín", "Berlino", "Berlyn", "Берлин", "ベルリン", "柏林", "--"]
    );
    assert!(g.lookup("--").is_empty());
    assert_eq!(ids(&g.lookup("berlin")), ["2950159", "5083330"]);
    assert_eq!(ids(&g.lookup("БЕРЛИН")), ["2950159"]);
    let zurich = g.by_id("geonames", "2657896").unwrap();
    assert_eq!(zurich.alternate_names().next(), Some("Zurich"));
    assert_eq!(zurich.feature_class(), Some(FEATURE_CLASS_CITY));
    let ocean = g.by_id("geonames", "2363254").unwrap();
    assert_eq!((ocean.country_code(), ocean.population(), ocean.feature_class()), (None, 0, Some(FEATURE_CLASS_POI)));
    assert_eq!(g.by_id("geonames", "2951839").unwrap().feature_class(), Some(FEATURE_CLASS_ADMIN));
    // "11.57549d" is a Java double.
    assert_eq!(g.by_id("geonames", "2867714").unwrap().location().longitude(), 11.57549);
    // Unicode white space is trimmed from every cell.
    let madrid = g.by_id("geonames", "3117735").unwrap();
    assert_eq!(madrid.name(), "Madrid");
    assert_eq!(madrid.alternate_names().collect::<Vec<_>>(), ["Madri", "マドリード"]);
    assert_eq!(g.by_id("geonames", "4951788").unwrap().population(), 155_929);
}

#[test]
fn ties_fall_to_the_feature_class_then_the_record_id_as_java_compares_it() {
    let g = GeoNamesGazetteer::load(fixture("gazetteer-geonames.txt")).unwrap();
    // 155929 first; then the three of population 0: the two cities by id
    // ("10" before "9" as strings), then the POI.
    assert_eq!(ids(&g.lookup("Springfield")), ["4951788", "10", "9", "90"]);
}

#[test]
fn a_geonames_row_needs_nineteen_columns_and_java_numbers() {
    let ok = geonames_row("1", "A", "", "", "1", "2", "P", "", "");
    let wide = format!("{ok}\textra");
    let e = GeoNamesGazetteer::from_bytes(format!("{ok}\n{wide}\n").as_bytes()).err().unwrap();
    assert_eq!(row_problem(e), (2, RowProblem::Columns { found: 20, expected: ColumnCount::Exactly(19) }));
    let bad = geonames_row("2", "B", "", "", "north", "2", "P", "", "");
    let e = GeoNamesGazetteer::from_bytes(format!("{ok}\n\n{bad}").as_bytes()).err().unwrap();
    assert_eq!(row_problem(e), (3, RowProblem::Number(NumberFormat("north".into()))));
    let far = geonames_row("2", "B", "", "", "91", "2", "P", "", "");
    let e = GeoNamesGazetteer::from_bytes(far.as_bytes()).err().unwrap();
    assert_eq!(row_problem(e), (1, RowProblem::Entry(EntryError::Latitude(91.0))));
    let lower = geonames_row("2", "B", "", "", "1", "2", "P", "de", "");
    let e = GeoNamesGazetteer::from_bytes(lower.as_bytes()).err().unwrap();
    assert_eq!(row_problem(e), (1, RowProblem::Entry(EntryError::CountryCode("de".into()))));
    let negative = geonames_row("2", "B", "", "", "1", "2", "P", "", "-5");
    let e = GeoNamesGazetteer::from_bytes(negative.as_bytes()).err().unwrap();
    assert_eq!(row_problem(e), (1, RowProblem::Entry(EntryError::Population(-5))));
    let repeated = format!("{ok}\r\n{ok}\r\n");
    let e = GeoNamesGazetteer::from_bytes(repeated.as_bytes()).err().unwrap();
    assert_eq!(row_problem(e), (2, RowProblem::RepeatedId("1".into())));
    // A comment is no comment in this format.
    let e = GeoNamesGazetteer::from_bytes(format!("# header\n{ok}").as_bytes()).err().unwrap();
    assert_eq!(row_problem(e).0, 1);
}

#[test]
fn a_canonical_name_without_a_word_token_is_refused_and_an_alternate_skipped() {
    let ok = geonames_row("1", "A", "", "...", "1", "2", "P", "", "");
    let g = GeoNamesGazetteer::from_bytes(ok.as_bytes()).unwrap();
    assert_eq!(g.by_id("geonames", "1").unwrap().alternate_names().collect::<Vec<_>>(), ["..."]);
    let bad = geonames_row("2", "...", "", "", "1", "2", "P", "", "");
    let e = GeoNamesGazetteer::from_bytes(format!("{ok}\n{bad}\n").as_bytes()).err().unwrap();
    let (line, problem) = row_problem(e);
    assert_eq!(line, 2);
    assert!(matches!(problem, RowProblem::Unmatchable { name, .. } if name == "..."));
    // The repeated id is reported before the unmatchable name.
    let both = geonames_row("1", "...", "", "", "1", "2", "P", "", "");
    let e = GeoNamesGazetteer::from_bytes(format!("{ok}\n{both}\n").as_bytes()).err().unwrap();
    assert_eq!(row_problem(e), (2, RowProblem::RepeatedId("1".into())));
}

#[test]
fn empty_tables_bad_utf8_and_blank_sources_are_refused() {
    assert!(matches!(GeoNamesGazetteer::from_bytes(b""), Err(LoadError::NoRows)));
    assert!(matches!(GeoNamesGazetteer::from_bytes(b"\xEF\xBB\xBF \n\t\n"), Err(LoadError::NoRows)));
    assert!(matches!(OvertureGazetteer::from_bytes(b"# only a comment\n"), Err(LoadError::NoRows)));
    let e = GeoNamesGazetteer::from_bytes(b"\n\nab\xFFc").err().unwrap();
    assert!(matches!(e, LoadError::Utf8 { line: 3 }), "{e:?}");
    assert!(matches!(UserGazetteer::from_bytes(b"a\tb\t1\t2", " "), Err(LoadError::BlankSource)));
    assert!(matches!(GeoNamesGazetteer::load("/nonexistent/geonames.txt"), Err(LoadError::Io(_))));
}

#[test]
fn overture_rows_map_subtypes_and_skip_comments() {
    let g = OvertureGazetteer::load(fixture("gazetteer-overture.txt")).unwrap();
    assert_eq!(g.sources(), ["overture"]);
    assert_eq!(g.len(), 10);
    let class = |id: &str| g.by_id("overture", id).unwrap().feature_class().unwrap();
    assert_eq!(class("ov-au"), FEATURE_CLASS_ADMIN);
    assert_eq!(class("ov-brooklyn"), FEATURE_CLASS_POI);
    assert_eq!(class("ov-mitte"), FEATURE_CLASS_POI);
    assert_eq!(class("ov-springfield-il"), FEATURE_CLASS_CITY);
    assert_eq!(class("ov-springfield-co"), FEATURE_CLASS_ADMIN);
    assert_eq!(ids(&g.lookup("bavaria")), ["ov-by", "ov-bavaria-county"]);
    assert_eq!(ids(&g.lookup("BAYERN")), ["ov-by"]);
    let sp = g.by_id("overture", "ov-sp").unwrap();
    assert_eq!(sp.alternate_names().collect::<Vec<_>>(), ["Sao Paulo", "Sampa"]);
    assert_eq!(g.by_id("overture", "ov-x").unwrap().population(), 0);
}

#[test]
fn user_rows_take_boxes_attributes_containment_and_optional_columns() {
    let g = UserGazetteer::load(fixture("gazetteer-user.txt"), "sites").unwrap();
    assert_eq!(g.sources(), ["sites"]);
    let hq = g.by_id("sites", "hq").unwrap();
    assert_eq!(hq.alternate_names().collect::<Vec<_>>(), ["Acme HQ", "HQ", "Head Office"]);
    assert_eq!(hq.containment().collect::<Vec<_>>(), ["New York", "Manhattan"]);
    let attributes: Vec<_> = hq.attributes().map(|a| (a.key, a.value, a.source, a.notes)).collect();
    assert_eq!(attributes, [("fips", "36061", "sites", ""), ("note", "main site", "sites", "")]);
    assert_eq!(hq.population(), 0);
    let plant = g.by_id("sites", "plant-1").unwrap();
    // Only a box: the location is its center.
    assert_eq!((plant.location().latitude(), plant.location().longitude()), (39.8, -89.6));
    assert!(plant.bounding_box().is_some());
    let depot = g.by_id("sites", "dateline").unwrap();
    assert_eq!(depot.location().longitude(), 180.0);
    assert_eq!(depot.feature_class(), Some("city"));
    let office = g.by_id("sites", "zz").unwrap();
    assert_eq!((office.country_code(), office.population()), (None, 7));
    let tiny = g.by_id("sites", "tiny").unwrap();
    assert_eq!((tiny.feature_class(), tiny.country_code(), tiny.containment().len()), (None, None, 0));
}

#[test]
fn user_rows_are_refused_with_their_lines() {
    let load = |text: &str| row_problem(UserGazetteer::from_bytes(text.as_bytes(), "s").err().unwrap());
    assert_eq!(load("only-id"), (1, RowProblem::Columns { found: 1, expected: ColumnCount::AtLeast(2) }));
    assert_eq!(load("# c\na\tA\t\t1"), (2, RowProblem::HalfLocation));
    assert_eq!(load("a\tA"), (1, RowProblem::NoLocation));
    assert_eq!(load("a\tA\t\t\t\t\t\t\t1,2,3"), (1, RowProblem::BoxEdges("1,2,3".into())));
    assert_eq!(load("a\tA\t\t1\t2\t\t\t\t\t\tnokey"), (1, RowProblem::AttributePair("nokey".into())));
    assert_eq!(load("a\tA\t\t1\t2\t\t\t\t\t\t=v"), (1, RowProblem::AttributePair("=v".into())));
    assert_eq!(load("a\tA\t\t1\t2\t\t\t\t\t\tk=v\tk=w"), (1, RowProblem::DuplicateAttribute("k".into())));
    assert_eq!(load("a\tA\t\t1\t2\tus"), (1, RowProblem::Entry(EntryError::CountryCode("us".into()))));
    // A hand written table refuses an alternate name with no word token.
    let (line, problem) = load("a\tA\tB|--\t1\t2");
    assert_eq!(line, 1);
    assert!(matches!(problem, RowProblem::Unmatchable { name, .. } if name == "--"));
}

#[test]
fn suppression_files_give_rules_in_order() {
    let rules = UserGazetteer::load_suppressions(fixture("gazetteer-suppressions.txt")).unwrap();
    let summary: Vec<_> =
        rules.iter().map(|r| (r.name(), r.country().map(|c| c.to_string()), r.feature_class())).collect();
    assert_eq!(
        summary,
        [
            ("Springfield", Some("US".to_string()), Some("CITY")),
            ("London", None, Some("city")),
            ("Sao Paulo", Some("BR".to_string()), None),
            ("Paris", Some("FR".to_string()), None),
            ("Tiny Place", None, None),
        ]
    );
    assert!(UserGazetteer::suppressions_from_bytes(b"").unwrap().is_empty());
    let e = UserGazetteer::suppressions_from_bytes(b"a\tb\tc\td").err().unwrap();
    assert_eq!(row_problem(e), (1, RowProblem::Columns { found: 4, expected: ColumnCount::AtMost(3) }));
    let e = UserGazetteer::suppressions_from_bytes(b"ok\n...\n").err().unwrap();
    assert!(matches!(row_problem(e), (2, RowProblem::Suppression(SuppressionError::Unmatchable(_)))));
}

#[test]
fn records_built_in_code_are_checked_and_indexed() {
    let parts = |id: &str, name: &str, alternates: &[&str]| EntryParts {
        source: "db".into(),
        record_id: id.into(),
        name: name.into(),
        alternate_names: alternates.iter().map(|s| s.to_string()).collect(),
        location: GeoPoint::new(1.0, 2.0).unwrap(),
        bounding_box: None,
        country_code: Some("NL".into()),
        containment: vec![],
        population: 5,
        feature_class: None,
        attributes: vec![("k".into(), AttributeValue::new("v", "db", "from a test").unwrap())],
    };
    let a = GazetteerEntry::new(parts("1", "Den Haag", &["The Hague", "'s-Gravenhage"])).unwrap();
    let b = GazetteerEntry::new(parts("2", "Haag", &[])).unwrap();
    let g = InMemoryGazetteer::from_entries(&[a.clone(), b]).unwrap();
    assert_eq!(ids(&g.lookup("the hague")), ["1"]);
    // The apostrophe is no part of a word: the key is "s gravenhage".
    assert_eq!(ids(&g.lookup("S GRAVENHAGE")), ["1"]);
    assert_eq!(ids(&g.lookup("'s-Gravenhage")), ["1"]);
    assert_eq!(g.by_id("db", "1").unwrap().to_entry(), a);
    assert_eq!(g.by_region(RegionCode::parse("nl").unwrap()).unwrap().record_id(), "1");
    assert!(matches!(InMemoryGazetteer::from_entries(&[a.clone(), a.clone()]), Err(IndexError::Duplicate { .. })));
    let unmatchable = GazetteerEntry::new(parts("3", "X", &["?"])).unwrap();
    assert!(matches!(InMemoryGazetteer::from_entries(&[unmatchable]), Err(IndexError::Unmatchable { .. })));
    assert!(GazetteerEntry::new(parts("4", " ", &[])).is_err());
    assert!(GazetteerEntry::new(parts("4", "Y", &[""])).is_err());
    assert!(InMemoryGazetteer::from_entries(&[]).unwrap().is_empty());
}

/// A GeoNames table of several megabytes, so it is read on several threads.
fn large_table(rows: usize) -> String {
    let mut text = String::new();
    for i in 0..rows {
        let name = format!("Place {} Ünïcode {}", i % 9973, i % 7);
        let alternates = format!("Alt {i},Ålt {},Plàce {}", i % 101, i % 13);
        let class = ["P", "A", "S"][i % 3];
        text.push_str(&geonames_row(
            &i.to_string(),
            &name,
            "",
            &alternates,
            "1.5",
            "2.5",
            class,
            "DE",
            &(i % 1000).to_string(),
        ));
        text.push(if i % 2 == 0 { '\n' } else { '\r' });
        if i % 3 == 0 {
            text.push('\n');
        }
    }
    text
}

#[test]
fn a_table_read_on_several_threads_matches_one_read_on_one() {
    let text = large_table(60_000);
    assert!(text.len() > 4 << 20);
    let parallel = GeoNamesGazetteer::from_bytes(text.as_bytes()).unwrap();
    assert_eq!(parallel.len(), 60_000);
    // The same records indexed in one piece on this thread.
    let entries: Vec<GazetteerEntry> =
        (0..60_000).map(|i| parallel.by_id("geonames", &i.to_string()).unwrap().to_entry()).collect();
    let single = InMemoryGazetteer::from_entries(&entries).unwrap();
    assert_eq!(parallel.key_count(), single.key_count());
    for query in ["Place 17 Unicode 3", "alt 59999", "ALT 100", "plàce 12", "place 0 unicode 0"] {
        let found = ids(&parallel.lookup(query));
        assert!(!found.is_empty(), "{query}");
        assert_eq!(found, ids(&single.lookup(query)), "{query}");
    }
    let de = RegionCode::parse("DE").unwrap();
    assert_eq!(parallel.by_region(de).unwrap().record_id(), single.by_region(de).unwrap().record_id());
    // A repeated id far into the table is reported on its line.
    let repeated = format!("{text}{}\n", geonames_row("59990", "X", "", "", "1", "2", "P", "", ""));
    let e = GeoNamesGazetteer::from_bytes(repeated.as_bytes()).err().unwrap();
    let lines = text.replace("\r\n", "\n").replace('\r', "\n").lines().count();
    assert_eq!(row_problem(e), (lines + 1, RowProblem::RepeatedId("59990".into())));
}
