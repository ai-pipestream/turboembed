//! Place profiles against the JVM: every `place-profiles` golden of
//! `analysis/goldens` (written by `turbo-goldens generate`, see
//! analysis/goldens-tool/README.md and `PlaceProfilesGolden.java`).
//!
//! Each golden is what `PlaceProfiles.load` made of one table of the
//! `place-profiles` input set: whether it loaded or OpenNLP's message, the
//! metric names, the standardized profile of every place, the similarity of
//! every ordered pair of places and `mostSimilar` of every place with the
//! counts 1, 2 and the number of places. The test loads the same table here
//! and rebuilds every layer; doubles are compared bit for bit.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use turbo_geo::{PlaceProfiles, ProfileError};
use turbo_goldens::{ColumnData, ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape};

const U32: ColumnType = ColumnType::U32;
const F64: ColumnType = ColumnType::F64;
const STR: ColumnType = ColumnType::Str;

fn shapes() -> Vec<Shape> {
    const LOAD: &[ColumnSpec] =
        &[ColumnSpec::new("loaded", U32), ColumnSpec::new("error-class", STR), ColumnSpec::new("error-message", STR)];
    const METRICS: &[ColumnSpec] = &[ColumnSpec::new("name", STR)];
    const PLACES: &[ColumnSpec] = &[ColumnSpec::new("id", STR)];
    const VALUES: &[ColumnSpec] =
        &[ColumnSpec::new("place", U32), ColumnSpec::new("metric", U32), ColumnSpec::new("value", F64)];
    const PAIRS: &[ColumnSpec] =
        &[ColumnSpec::new("a", U32), ColumnSpec::new("b", U32), ColumnSpec::new("similarity", F64)];
    const NEIGHBORS: &[ColumnSpec] = &[
        ColumnSpec::new("query", U32),
        ColumnSpec::new("count", U32),
        ColumnSpec::new("rank", U32),
        ColumnSpec::new("place", U32),
        ColumnSpec::new("similarity", F64),
    ];
    let doc = |id: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Document, columns };
    vec![
        doc("turbo:profiles-load", LOAD),
        doc("turbo:profiles-metrics", METRICS),
        doc("turbo:profiles-places", PLACES),
        doc("turbo:profiles-values", VALUES),
        doc("turbo:profiles-similarity", PAIRS),
        doc("turbo:profiles-neighbors", NEIGHBORS),
    ]
}

/// The layers rebuilt here, by id, each as rows of cells written as text.
type Rebuilt = BTreeMap<&'static str, Vec<Vec<String>>>;

fn bits(v: f64) -> String {
    format!("{:016x}", v.to_bits())
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
                    ColumnData::F64(v) => bits(v[i]),
                })
                .collect()
        })
        .collect()
}

/// What `PlaceProfilesGolden` writes for one table.
fn rebuild(bytes: &[u8]) -> Rebuilt {
    let mut out = Rebuilt::new();
    let profiles = match PlaceProfiles::from_bytes(bytes) {
        Ok(p) => p,
        Err(e) => {
            let class = match e {
                ProfileError::Io(_) => "java.io.IOException",
                _ => "opennlp.tools.util.InvalidFormatException",
            };
            out.insert("turbo:profiles-load", vec![vec!["0".into(), class.into(), e.to_string()]]);
            for layer in [
                "turbo:profiles-metrics",
                "turbo:profiles-places",
                "turbo:profiles-values",
                "turbo:profiles-similarity",
                "turbo:profiles-neighbors",
            ] {
                out.insert(layer, Vec::new());
            }
            return out;
        }
    };
    out.insert("turbo:profiles-load", vec![vec!["1".into(), String::new(), String::new()]]);
    out.insert("turbo:profiles-metrics", profiles.metrics().map(|m| vec![m.to_owned()]).collect());
    // The generator sorts the ids with String.compareTo.
    let mut places: Vec<_> = profiles.places().collect();
    places.sort_by(|&a, &b| turbo_jvm::chars::compare(profiles.id(a), profiles.id(b)));
    let index: BTreeMap<_, usize> = places.iter().enumerate().map(|(i, &p)| (p, i)).collect();
    out.insert("turbo:profiles-places", places.iter().map(|&p| vec![profiles.id(p).to_owned()]).collect());
    let mut values = Vec::new();
    let mut pairs = Vec::new();
    let mut neighbors = Vec::new();
    for (i, &a) in places.iter().enumerate() {
        for (m, &v) in profiles.profile(a).iter().enumerate() {
            values.push(vec![i.to_string(), m.to_string(), bits(v)]);
        }
        for (j, &b) in places.iter().enumerate() {
            pairs.push(vec![i.to_string(), j.to_string(), bits(profiles.similarity(a, b).get())]);
        }
        for count in [1, 2, places.len()] {
            let found = profiles.most_similar(a, NonZeroUsize::new(count).unwrap());
            for (rank, n) in found.iter().enumerate() {
                neighbors.push(vec![
                    i.to_string(),
                    count.to_string(),
                    rank.to_string(),
                    index[&n.place].to_string(),
                    bits(n.similarity.get()),
                ]);
            }
        }
    }
    out.insert("turbo:profiles-values", values);
    out.insert("turbo:profiles-similarity", pairs);
    out.insert("turbo:profiles-neighbors", neighbors);
    out
}

/// Every golden, as its input id and file.
fn goldens() -> Vec<(String, PathBuf)> {
    let dir = turbo_goldens::data_dir().join("place-profiles").join("place-profiles").join("place-profiles");
    let mut out: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|f| {
            let path = f.unwrap().path();
            (path.file_stem().unwrap().to_str().unwrap().to_owned(), path)
        })
        .collect();
    out.sort();
    out
}

#[test]
fn every_place_profile_golden_matches() {
    let shapes = shapes();
    let data = turbo_goldens::data_dir();
    let goldens = goldens();
    let mut loaded = 0;
    for (id, path) in &goldens {
        let golden = Golden::read_with(path, &shapes).unwrap();
        assert!(golden.header.error.is_none(), "{id}: an error record");
        let bytes = golden.input_text(&data).unwrap().into_bytes();
        let mut ours = rebuild(&bytes);
        for layer in &golden.layers {
            let rows = ours.remove(layer.id.as_str()).unwrap_or_else(|| panic!("{id}: no layer {}", layer.id));
            let theirs = cells(layer);
            assert_eq!(rows.len(), theirs.len(), "{id}: {} has {} rows here", layer.id, rows.len());
            for (r, (a, b)) in rows.iter().zip(&theirs).enumerate() {
                assert_eq!(a, b, "{id}: {} row {r}", layer.id);
            }
            if layer.id == "turbo:profiles-load" && theirs[0][0] == "1" {
                loaded += 1;
            }
        }
        assert!(ours.is_empty(), "{id}: layers not in the golden: {:?}", ours.keys());
    }
    assert!(goldens.len() >= 30, "{} goldens", goldens.len());
    assert!(loaded >= 10, "{loaded} tables loaded");
}
