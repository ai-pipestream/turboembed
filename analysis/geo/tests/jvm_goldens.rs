//! Reviewed gazetteer snapshots and load/query invariants.
//!
//! `tests/snapshots/gazetteer/<run>/<id>.snap` holds the rebuilt layers for
//! each input under `tests/inputs/gazetteer/`. Regenerate with
//! `TURBO_BLESS=1 cargo test -p turbo-geo --test jvm_goldens`.
//! JVM goldens are not an oracle. Doubles are compared bit for bit; load
//! failures keep the OpenNLP exception class the port mirrors.

use std::collections::BTreeMap;
use std::path::PathBuf;

use turbo_geo::{
    BUNDLED_TABLE, BundledGazetteer, Entry, Gazetteer, GeoNamesGazetteer, LoadError, OverlayGazetteer,
    OvertureGazetteer, RegionCode, UserGazetteer,
};
/// The source the user table is loaded with.
const USER_SOURCE: &str = "sites";

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

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn fixture(name: &str) -> PathBuf {
    root().join("fixtures").join(name)
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

fn render(rebuilt: &Rebuilt) -> String {
    let mut out = String::new();
    for (id, rows) in rebuilt {
        out.push_str(&format!("# {id}\n"));
        for row in rows {
            let cells: Vec<String> = row.iter().map(|c| c.clone().unwrap_or_default()).collect();
            out.push_str(&cells.join("\t"));
            out.push('\n');
        }
    }
    out
}

fn snap_path(run: &str, id: &str) -> PathBuf {
    root().join("snapshots/gazetteer").join(run).join(format!("{id}.snap"))
}

fn input_text(id: &str) -> String {
    std::fs::read_to_string(root().join("inputs/gazetteer").join(format!("{id}.txt"))).unwrap()
}

fn query_ids() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(root().join("inputs/gazetteer"))
        .unwrap()
        .filter_map(|e| {
            let name = e.unwrap().file_name().into_string().unwrap();
            let id = name.strip_suffix(".txt")?.to_string();
            if id.starts_with("names") || id == "ids" || id == "regions" { Some(id) } else { None }
        })
        .collect();
    ids.sort();
    ids
}

fn table_ids(kind: &str) -> Vec<String> {
    let prefix = format!("table-{kind}-");
    let mut ids: Vec<String> = std::fs::read_dir(root().join("inputs/gazetteer"))
        .unwrap()
        .filter_map(|e| {
            let name = e.unwrap().file_name().into_string().unwrap();
            let id = name.strip_suffix(".txt")?.to_string();
            if id.starts_with(&prefix) { Some(id) } else { None }
        })
        .collect();
    ids.sort();
    ids
}

fn check_snap(run: &str, id: &str, rebuilt: Rebuilt) {
    let bless = std::env::var_os("TURBO_BLESS").is_some();
    let body = render(&rebuilt);
    let path = snap_path(run, id);
    if bless {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &body).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing snapshot {} (TURBO_BLESS=1 to write): {e}", path.display()));
    assert_eq!(body, want, "{run}/{id}");
}

#[test]
fn the_bundled_table_name_and_bytes_are_stable() {
    assert_eq!(turbo_geo::BUNDLED_TABLE_NAME, "naturalearth-populated-places.txt");
    assert!(!BUNDLED_TABLE.is_empty());
    assert!(!BundledGazetteer::get().sources().is_empty());
}

#[test]
fn every_gazetteer_snapshot_matches() {
    let geonames = GeoNamesGazetteer::load(fixture("gazetteer-geonames.txt")).unwrap();
    let overture = OvertureGazetteer::load(fixture("gazetteer-overture.txt")).unwrap();
    let user = UserGazetteer::load(fixture("gazetteer-user.txt"), USER_SOURCE).unwrap();
    let rules = UserGazetteer::load_suppressions(fixture("gazetteer-suppressions.txt")).unwrap();
    let overlay = OverlayGazetteer::new(BundledGazetteer::get(), Some(&user), rules).unwrap();
    let mut compared = BTreeMap::<String, usize>::new();
    for id in query_ids() {
        let text = input_text(&id);
        for (run, g) in [
            ("naturalearth-populated-places", &BundledGazetteer::get() as &dyn Gazetteer),
            ("gazetteer-geonames", &geonames as &dyn Gazetteer),
            ("gazetteer-overture", &overture as &dyn Gazetteer),
            ("gazetteer-user", &user as &dyn Gazetteer),
            ("overlay", &overlay as &dyn Gazetteer),
        ] {
            check_snap(run, &id, rebuild_queries(g, &id, &text));
            *compared.entry(run.to_string()).or_default() += 1;
        }
    }
    for (kind, run) in [
        ("geonames", "load-geonames"),
        ("overture", "load-overture"),
        ("user", "load-user"),
        ("suppressions", "load-suppressions"),
    ] {
        for id in table_ids(kind) {
            let bytes = input_text(&id).into_bytes();
            check_snap(run, &id, rebuild_load(kind, &id, &bytes));
            *compared.entry(run.to_string()).or_default() += 1;
        }
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
        assert!(*n >= expected, "{run}: {n} snapshots");
    }
}
