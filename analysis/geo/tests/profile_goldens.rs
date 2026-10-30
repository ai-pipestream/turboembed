//! Reviewed place-profile snapshots and load/similarity invariants.
//!
//! `tests/snapshots/place-profiles/<id>.snap` holds the rebuilt layers for
//! each input under `tests/inputs/place-profiles/`. Regenerate with
//! `TURBO_BLESS=1 cargo test -p turbo-geo --test profile_goldens`.
//! JVM goldens are not an oracle. Doubles are compared bit for bit.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use turbo_geo::{PlaceProfiles, ProfileError};

/// The layers rebuilt here, by id, each as rows of cells written as text.
type Rebuilt = BTreeMap<&'static str, Vec<Vec<String>>>;

fn bits(v: f64) -> String {
    format!("{:016x}", v.to_bits())
}

/// What one table produces as snapshot layers.
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
    // Ids in code point order.
    let mut places: Vec<_> = profiles.places().collect();
    places.sort_by(|&a, &b| profiles.id(a).cmp(profiles.id(b)));
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

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn render(rebuilt: &Rebuilt) -> String {
    let mut out = String::new();
    for (id, rows) in rebuilt {
        out.push_str(&format!("# {id}\n"));
        for row in rows {
            out.push_str(&row.join("\t"));
            out.push('\n');
        }
    }
    out
}

fn input_ids() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(root().join("inputs/place-profiles"))
        .unwrap()
        .map(|e| {
            let name = e.unwrap().file_name().into_string().unwrap();
            name.strip_suffix(".txt").unwrap().to_string()
        })
        .collect();
    ids.sort();
    ids
}

#[test]
fn every_place_profile_snapshot_matches() {
    let bless = std::env::var_os("TURBO_BLESS").is_some();
    let ids = input_ids();
    let mut loaded = 0;
    for id in &ids {
        let bytes = std::fs::read(root().join("inputs/place-profiles").join(format!("{id}.txt"))).unwrap();
        let ours = rebuild(&bytes);
        if ours.get("turbo:profiles-load").and_then(|r| r.first()).map(|r| r[0].as_str()) == Some("1") {
            loaded += 1;
        }
        let body = render(&ours);
        let path = root().join("snapshots/place-profiles").join(format!("{id}.snap"));
        if bless {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, &body).unwrap();
            continue;
        }
        let want = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("missing snapshot {} (TURBO_BLESS=1 to write): {e}", path.display()));
        assert_eq!(body, want, "{id}");
    }
    assert!(ids.len() >= 30, "{} inputs", ids.len());
    assert!(loaded >= 10, "{loaded} tables loaded");
}
