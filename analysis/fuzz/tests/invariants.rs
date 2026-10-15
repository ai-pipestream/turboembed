//! The offset invariants (`turbo_goldens::invariants`) over every layer
//! every run writes, on a small seeded corpus of the fuzz generator and on
//! the checked in golden inputs of the sets the covered runs read. No
//! container is needed: the Rust side runs alone. Runs whose published
//! model is not downloaded are skipped, unless
//! `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`; the published MeCab dictionaries
//! are left out unless `TURBO_TEST_MECAB_DICTIONARIES=1`, as the lattice
//! crate's own tests on them are ignored by default.

use std::collections::BTreeSet;
use std::path::Path;

use turbo_fuzz::{Resources, Unavailable, runs};
use turbo_goldens::fuzz::corpus;
use turbo_goldens::invariants::check_document;

const SEED: u64 = 7;
const COUNT: u64 = 40;

/// The names of the subdirectories of `dir`, or none when it is missing.
fn subdirectories(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    entries
        .map(|e| e.unwrap())
        .filter(|e| e.file_type().unwrap().is_dir())
        .map(|e| e.file_name().into_string().unwrap())
        .collect()
}

/// The input sets the goldens of the given annotators were made from: the
/// set directories under `<annotator>/<run>/` in the golden data.
fn sets_read_by(annotators: impl IntoIterator<Item = (String, Option<String>)>) -> BTreeSet<String> {
    let data = turbo_goldens::data_dir();
    let mut sets = BTreeSet::new();
    for (annotator, run) in annotators {
        let runs = match run {
            Some(run) => vec![run],
            None => subdirectories(&data.join(&annotator)),
        };
        for run in runs {
            sets.extend(subdirectories(&data.join(&annotator).join(run)));
        }
    }
    sets
}

/// The input sets some run of [`runs::RUNS`] reads. The other sets are
/// read only by the annotators the differential does not cover (the
/// stemmers' word lists, the format readers' corpora, ...); no covered run
/// is compared on them, and every run below reading them all, the
/// stemmers' half a megabyte of words among them, takes many times as long
/// as the rest of the test.
fn covered_sets() -> BTreeSet<String> {
    sets_read_by(runs::RUNS.iter().map(|&(a, r)| (a.to_string(), Some(r.to_string()))))
}

/// Every checked in input text of the given sets, as `<set>/<id>` and its
/// text.
fn golden_inputs(sets: &BTreeSet<String>) -> Vec<(String, String)> {
    let inputs = turbo_goldens::data_dir().join("inputs");
    let mut out = Vec::new();
    for set in sets {
        let mut texts = 0;
        for file in std::fs::read_dir(inputs.join(set)).unwrap_or_else(|e| panic!("inputs/{set}: {e}")) {
            let file = file.unwrap().path();
            if file.extension().is_some_and(|x| x == "txt") {
                let name = format!("{set}/{}", file.file_stem().unwrap().to_string_lossy());
                out.push((name, std::fs::read_to_string(&file).unwrap()));
                texts += 1;
            }
        }
        assert!(texts > 0, "inputs/{set} has no text, but a covered run's goldens were made from it");
    }
    out.sort();
    out
}

/// Every input set is fed to the runs below exactly when a covered run
/// reads it, and a set no covered run reads must be read by an annotator
/// the differential does not cover: a new set with goldens of a covered
/// run is picked up from the goldens, and a set with no goldens at all is
/// an error here rather than a set silently left out.
#[test]
fn the_sets_fed_are_the_ones_the_covered_runs_read() {
    let covered = covered_sets();
    for set in ["en", "multi", "edge", "surrogate", "norm", "money", "zh", "ja", "ko"] {
        assert!(covered.contains(set), "{set} is not among {covered:?}");
    }
    let others = sets_read_by(runs::NOT_COVERED.iter().map(|a| (a.to_string(), None)));
    let on_disk: BTreeSet<String> = subdirectories(&turbo_goldens::data_dir().join("inputs")).into_iter().collect();
    for set in &on_disk {
        assert!(
            covered.contains(set) || others.contains(set),
            "inputs/{set}: no golden of any run was made from it, so this test would leave it out"
        );
    }
    assert!(covered.is_subset(&on_disk), "{covered:?} names a set that is not in inputs/");
    // The stemmers' word lists are the largest set and are read only by `stem`.
    assert!(!covered.contains("stem"), "{covered:?}");
    let texts = golden_inputs(&covered);
    let fed: BTreeSet<&str> = texts.iter().map(|(id, _)| id.split_once('/').unwrap().0).collect();
    assert_eq!(fed, covered.iter().map(String::as_str).collect(), "every covered set gives texts");
}

fn required(path: &Path) -> bool {
    let dictionary = path.to_string_lossy().contains("mecab-");
    !dictionary && std::env::var("TURBO_TEST_REQUIRE_OPENNLP_MODELS").is_ok_and(|v| v == "1")
}

#[test]
fn every_layer_of_every_run_keeps_the_offset_invariants() {
    let res = Resources::from_env();
    let mut texts: Vec<(String, String)> = corpus(SEED, COUNT).into_iter().map(|i| (i.id, i.text)).collect();
    texts.extend(golden_inputs(&covered_sets()));
    assert!(texts.len() > 300, "{} texts", texts.len());
    let (mut runs_checked, mut layers_checked, mut skipped) = (0, 0, Vec::new());
    let mut failures = Vec::new();
    let dictionaries = std::env::var("TURBO_TEST_MECAB_DICTIONARIES").is_ok_and(|v| v == "1");
    for &(annotator, run) in runs::RUNS {
        let name = format!("{annotator}/{run}");
        if annotator == "lattice" && !run.starts_with("mecab-fixture-") && !dictionaries {
            skipped.push(name);
            continue;
        }
        // Tags and lemmas have no spans of their own; without a golden these
        // runs tokenize with the simple tokenizer, which has its own run here.
        if ["postag", "lemmatize", "lemmatize-dictionary"].contains(&annotator) {
            continue;
        }
        let r = match runs::open(annotator, run, &res) {
            Ok(r) => r,
            Err(Unavailable::Missing(p)) if !required(&p) => {
                skipped.push(name);
                continue;
            }
            Err(e) => panic!("{name}: {e}"),
        };
        runs_checked += 1;
        let started = std::time::Instant::now();
        for (id, text) in &texts {
            // A run may refuse a text (the lattice tokenizer without a
            // candidate); that is the differential's business, not this test's.
            let Ok(doc) = r.annotate(text, None) else { continue };
            let (layers, violations) = check_document(&doc);
            layers_checked += layers;
            for v in violations {
                failures.push(format!("{name} on {id}: {v}"));
            }
        }
        eprintln!("{name}: {:.1} s", started.elapsed().as_secs_f64());
    }
    eprintln!("{runs_checked} runs over {} texts, {layers_checked} layers checked; skipped {skipped:?}", texts.len());
    assert!(runs_checked >= 70, "{runs_checked} runs; skipped {skipped:?}");
    assert!(layers_checked > 10_000, "{layers_checked} layers checked");
    assert!(failures.is_empty(), "{} offset invariants broken:\n{}", failures.len(), failures.join("\n"));
}
