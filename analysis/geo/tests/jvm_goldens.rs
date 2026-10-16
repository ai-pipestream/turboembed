//! The gazetteers against the JVM: every `gazetteer` golden of
//! `analysis/goldens` (written by `turbo-goldens generate`, see
//! analysis/goldens-tool/README.md and `GazetteerGolden.java`).
//!
//! A query golden is what one gazetteer answered for each line of one input
//! of the `gazetteer` set: `lookup` for the `names` inputs, `byId` for `ids`
//! and `byRegion` for `regions`, with every field of every entry found. A
//! load golden is whether one table of the set loaded, with OpenNLP's
//! message when it did not. Each test asks the same here and rebuilds every
//! layer, cell for cell; coordinates are compared bit for bit.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use turbo_geo::{
    BUNDLED_TABLE, BundledGazetteer, Entry, Gazetteer, GeoNamesGazetteer, LoadError, OverlayGazetteer,
    OvertureGazetteer, RegionCode, UserGazetteer,
};
use turbo_goldens::{ColumnData, ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape};

const U32: ColumnType = ColumnType::U32;
const F64: ColumnType = ColumnType::F64;
const STR: ColumnType = ColumnType::Str;

/// The source the goldens load the user table with.
const USER_SOURCE: &str = "sites";

fn shapes() -> Vec<Shape> {
    const SOURCES: &[ColumnSpec] = &[ColumnSpec::new("source", STR)];
    const QUERIES: &[ColumnSpec] =
        &[ColumnSpec::new("found", U32), ColumnSpec::new("error-class", STR), ColumnSpec::new("error-message", STR)];
    const CANDIDATES: &[ColumnSpec] = &[ColumnSpec::new("query", U32), ColumnSpec::new("entry", U32)];
    const ENTRIES: &[ColumnSpec] = &[
        ColumnSpec::new("source", STR),
        ColumnSpec::new("record-id", STR),
        ColumnSpec::new("name", STR),
        ColumnSpec::new("latitude", F64),
        ColumnSpec::new("longitude", F64),
        ColumnSpec::new("has-box", U32),
        ColumnSpec::new("west", F64),
        ColumnSpec::new("south", F64),
        ColumnSpec::new("east", F64),
        ColumnSpec::new("north", F64),
        ColumnSpec::new("has-country", U32),
        ColumnSpec::new("country", STR),
        ColumnSpec::new("population", STR),
        ColumnSpec::new("has-feature-class", U32),
        ColumnSpec::new("feature-class", STR),
    ];
    const NAMES: &[ColumnSpec] = &[ColumnSpec::new("entry", U32), ColumnSpec::new("name", STR)];
    const ATTRIBUTES: &[ColumnSpec] = &[
        ColumnSpec::new("entry", U32),
        ColumnSpec::new("key", STR),
        ColumnSpec::new("value", STR),
        ColumnSpec::new("source", STR),
        ColumnSpec::new("notes", STR),
    ];
    const LOAD: &[ColumnSpec] =
        &[ColumnSpec::new("loaded", U32), ColumnSpec::new("error-class", STR), ColumnSpec::new("error-message", STR)];
    const RULES: &[ColumnSpec] = &[
        ColumnSpec::new("name", STR),
        ColumnSpec::new("has-country", U32),
        ColumnSpec::new("country", STR),
        ColumnSpec::new("has-feature-class", U32),
        ColumnSpec::new("feature-class", STR),
    ];
    let doc = |id: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Document, columns };
    vec![
        doc("turbo:gazetteer-sources", SOURCES),
        doc("turbo:gazetteer-queries", QUERIES),
        doc("turbo:gazetteer-candidates", CANDIDATES),
        doc("turbo:gazetteer-entries", ENTRIES),
        doc("turbo:gazetteer-alternates", NAMES),
        doc("turbo:gazetteer-containment", NAMES),
        doc("turbo:gazetteer-attributes", ATTRIBUTES),
        doc("turbo:gazetteer-load", LOAD),
        doc("turbo:gazetteer-suppressions", RULES),
    ]
}

/// The layers rebuilt here, by id, each as rows of cells written as text.
/// A cell of `None` is not compared.
type Rebuilt = BTreeMap<&'static str, Vec<Vec<Option<String>>>>;

fn s(v: impl ToString) -> Option<String> {
    Some(v.to_string())
}

/// A double as the bits the comparison uses.
fn bits(v: f64) -> Option<String> {
    Some(format!("{:016x}", v.to_bits()))
}

fn cells(layer: &GoldenLayer) -> Vec<Vec<String>> {
    (0..layer.len())
        .map(|i| {
            layer
                .columns
                .iter()
                .map(|c| match &c.data {
                    ColumnData::U32(v) => v[i].to_string(),
                    ColumnData::Str(v) => v.get(i).to_owned(),
                    ColumnData::F64(v) => format!("{:016x}", v[i].to_bits()),
                })
                .collect()
        })
        .collect()
}

fn fixture(name: &str) -> PathBuf {
    turbo_goldens::data_dir().join("fixtures").join(name)
}

/// The lines of an input as the generator splits it: at line feeds, without
/// the empty line after a last one.
fn lines(text: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = text.split('\n').collect();
    if lines.last() == Some(&"") {
        lines.pop();
    }
    lines
}

/// What one question returned: the entries found, or OpenNLP's exception
/// class and message.
type Answer<'g> = Result<Vec<Entry<'g>>, (&'static str, String)>;

fn ask<'g>(g: &'g dyn Gazetteer, id: &str, line: &str) -> Answer<'g> {
    if id.starts_with("names") {
        Ok(g.lookup(line))
    } else if id.starts_with("ids") {
        let (source, record_id) = line.split_once('\t').expect("an ids line has a tab");
        Ok(g.by_id(source, record_id).into_iter().collect())
    } else if id.starts_with("regions") {
        match RegionCode::parse(line) {
            Ok(code) => Ok(g.by_region(code).into_iter().collect()),
            Err(e) => Err(("java.lang.IllegalArgumentException", e.to_string())),
        }
    } else {
        panic!("the gazetteer input {id} is not names, ids or regions")
    }
}

fn rebuild_queries(g: &dyn Gazetteer, id: &str, text: &str) -> Rebuilt {
    let mut out = Rebuilt::new();
    let mut sources = g.sources();
    sources.sort_unstable();
    out.insert("turbo:gazetteer-sources", sources.iter().map(|x| vec![s(x)]).collect());
    let mut queries = Vec::new();
    let mut candidates = Vec::new();
    let mut entries: Vec<Entry<'_>> = Vec::new();
    for (q, line) in lines(text).into_iter().enumerate() {
        match ask(g, id, line) {
            Ok(found) => {
                queries.push(vec![s(found.len()), s(""), s("")]);
                for e in found {
                    let i = entries.iter().position(|x| *x == e).unwrap_or_else(|| {
                        entries.push(e);
                        entries.len() - 1
                    });
                    candidates.push(vec![s(q), s(i)]);
                }
            }
            Err((class, message)) => queries.push(vec![s(0), s(class), s(message)]),
        }
    }
    out.insert("turbo:gazetteer-queries", queries);
    out.insert("turbo:gazetteer-candidates", candidates);
    let mut rows = Vec::new();
    let (mut alternates, mut containment, mut attributes) = (Vec::new(), Vec::new(), Vec::new());
    for (i, e) in entries.iter().enumerate() {
        let location = e.location();
        let bbox = e.bounding_box();
        let [west, south, east, north] = bbox.map_or([0.0; 4], |b| [b.west(), b.south(), b.east(), b.north()]);
        rows.push(vec![
            s(e.source()),
            s(e.record_id()),
            s(e.name()),
            bits(location.latitude()),
            bits(location.longitude()),
            s(u32::from(bbox.is_some())),
            bits(west),
            bits(south),
            bits(east),
            bits(north),
            s(u32::from(e.country_code().is_some())),
            s(e.country_code().unwrap_or("")),
            s(e.population()),
            s(u32::from(e.feature_class().is_some())),
            s(e.feature_class().unwrap_or("")),
        ]);
        alternates.extend(e.alternate_names().map(|n| vec![s(i), s(n)]));
        containment.extend(e.containment().map(|n| vec![s(i), s(n)]));
        let mut sorted: Vec<_> = e.attributes().collect();
        // Java sorts the keys by String.compareTo, UTF-16 order; the keys
        // here are ASCII, where that is byte order.
        sorted.sort_by(|a, b| a.key.cmp(b.key));
        attributes.extend(sorted.into_iter().map(|a| vec![s(i), s(a.key), s(a.value), s(a.source), s(a.notes)]));
    }
    out.insert("turbo:gazetteer-entries", rows);
    out.insert("turbo:gazetteer-alternates", alternates);
    out.insert("turbo:gazetteer-containment", containment);
    out.insert("turbo:gazetteer-attributes", attributes);
    out
}

/// The message OpenNLP gives for a table that does not load, where this
/// crate writes the same; `None` where it does not (a byte that is not
/// UTF-8, which the input set has none of).
fn load_failure(e: &LoadError) -> (&'static str, Option<String>) {
    match e {
        LoadError::Utf8 { .. } => ("java.nio.charset.MalformedInputException", None),
        LoadError::BlankSource => ("java.lang.IllegalArgumentException", s(e)),
        _ => ("opennlp.tools.util.InvalidFormatException", s(e)),
    }
}

fn rebuild_load(kind: &str, id: &str, bytes: &[u8]) -> Rebuilt {
    let expected = format!("table-{kind}-");
    assert!(id.starts_with(&expected), "{id} is not a {kind} table");
    let mut rules = Vec::new();
    let loaded = match kind {
        "geonames" => GeoNamesGazetteer::from_bytes(bytes).map(drop),
        "overture" => OvertureGazetteer::from_bytes(bytes).map(drop),
        "user" => UserGazetteer::from_bytes(bytes, USER_SOURCE).map(drop),
        "suppressions" => UserGazetteer::suppressions_from_bytes(bytes).map(|r| rules = r),
        _ => panic!("unknown table kind {kind}"),
    };
    let mut out = Rebuilt::new();
    let row = match &loaded {
        Ok(()) => vec![s(1), s(""), s("")],
        Err(e) => {
            let (class, message) = load_failure(e);
            vec![s(0), s(class), message]
        }
    };
    out.insert("turbo:gazetteer-load", vec![row]);
    out.insert(
        "turbo:gazetteer-suppressions",
        rules
            .iter()
            .map(|r| {
                let country = r.country().map(|c| c.to_string());
                vec![
                    s(r.name()),
                    s(u32::from(country.is_some())),
                    s(country.unwrap_or_default()),
                    s(u32::from(r.feature_class().is_some())),
                    s(r.feature_class().unwrap_or("")),
                ]
            })
            .collect(),
    );
    out
}

/// Every golden of the annotator, as its run and file.
fn goldens() -> Vec<(String, PathBuf)> {
    let root = turbo_goldens::data_dir().join("gazetteer");
    let mut out = Vec::new();
    for run in std::fs::read_dir(&root).unwrap() {
        let run = run.unwrap().path();
        let name = run.file_name().unwrap().to_str().unwrap().to_owned();
        for file in std::fs::read_dir(run.join("gazetteer")).unwrap() {
            out.push((name.clone(), file.unwrap().path()));
        }
    }
    out.sort();
    out
}

fn compare(path: &Path, golden: &Golden, mut rebuilt: Rebuilt) {
    assert!(golden.header.error.is_none(), "{}: an error record", path.display());
    for layer in &golden.layers {
        let ours =
            rebuilt.remove(layer.id.as_str()).unwrap_or_else(|| panic!("{}: no layer {}", path.display(), layer.id));
        let theirs = cells(layer);
        assert_eq!(ours.len(), theirs.len(), "{}: {} has {} rows here", path.display(), layer.id, ours.len());
        for (row, (a, b)) in ours.iter().zip(&theirs).enumerate() {
            for (column, (x, y)) in a.iter().zip(b).enumerate() {
                if let Some(x) = x {
                    assert_eq!(
                        x,
                        y,
                        "{}: {} row {row} column {}",
                        path.display(),
                        layer.id,
                        layer.columns[column].name
                    );
                }
            }
        }
    }
    assert!(rebuilt.is_empty(), "{}: layers not in the golden: {:?}", path.display(), rebuilt.keys());
}

/// The input of a golden: the set's file the header names.
fn input(golden: &Golden) -> Vec<u8> {
    let data = turbo_goldens::data_dir();
    let text = golden.input_text(&data).unwrap();
    text.into_bytes()
}

#[test]
fn the_bundled_table_is_the_one_in_the_opennlp_jar() {
    let (_, path) = goldens()
        .into_iter()
        .find(|(run, _)| run == "naturalearth-populated-places")
        .expect("a golden of the bundled gazetteer");
    let golden = Golden::read_with(&path, &shapes()).unwrap();
    assert_eq!(golden.header.models.len(), 1);
    assert_eq!(golden.header.models[0].file, turbo_geo::BUNDLED_TABLE_NAME);
    assert_eq!(golden.header.models[0].sha512, turbo_goldens::sha512(BUNDLED_TABLE.as_bytes()));
}

#[test]
fn every_gazetteer_golden_matches() {
    let geonames = GeoNamesGazetteer::load(fixture("gazetteer-geonames.txt")).unwrap();
    let overture = OvertureGazetteer::load(fixture("gazetteer-overture.txt")).unwrap();
    let user = UserGazetteer::load(fixture("gazetteer-user.txt"), USER_SOURCE).unwrap();
    let rules = UserGazetteer::load_suppressions(fixture("gazetteer-suppressions.txt")).unwrap();
    let overlay = OverlayGazetteer::new(BundledGazetteer::get(), Some(&user), rules).unwrap();
    let shapes = shapes();
    let mut compared = BTreeMap::<String, usize>::new();
    for (run, path) in goldens() {
        let golden = Golden::read_with(&path, &shapes).unwrap();
        let id = path.file_stem().unwrap().to_str().unwrap();
        let bytes = input(&golden);
        let rebuilt = match run.as_str() {
            "naturalearth-populated-places" => {
                rebuild_queries(BundledGazetteer::get(), id, std::str::from_utf8(&bytes).unwrap())
            }
            "gazetteer-geonames" => rebuild_queries(&geonames, id, std::str::from_utf8(&bytes).unwrap()),
            "gazetteer-overture" => rebuild_queries(&overture, id, std::str::from_utf8(&bytes).unwrap()),
            "gazetteer-user" => rebuild_queries(&user, id, std::str::from_utf8(&bytes).unwrap()),
            "overlay" => rebuild_queries(&overlay, id, std::str::from_utf8(&bytes).unwrap()),
            load if load.starts_with("load-") => rebuild_load(&load["load-".len()..], id, &bytes),
            _ => panic!("unknown gazetteer run {run}"),
        };
        compare(&path, &golden, rebuilt);
        *compared.entry(run).or_default() += 1;
    }
    let runs: Vec<&str> = compared.keys().map(String::as_str).collect();
    assert_eq!(
        runs,
        [
            "gazetteer-geonames",
            "gazetteer-overture",
            "gazetteer-user",
            "load-geonames",
            "load-overture",
            "load-suppressions",
            "load-user",
            "naturalearth-populated-places",
            "overlay",
        ]
    );
    for (run, n) in &compared {
        let expected = if run.starts_with("load-") { 2 } else { 5 };
        assert!(*n >= expected, "{run}: {n} goldens");
    }
}
