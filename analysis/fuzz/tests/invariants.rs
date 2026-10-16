//! The offset invariants (`turbo_goldens::invariants`) over every layer
//! every run writes, on a small seeded corpus of the fuzz generator and on
//! the checked in golden inputs of the sets of [`FED`]. No
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

/// The input sets every run is given. A set added under
/// `analysis/goldens/data/inputs` goes here or in [`NOT_FED`]; the test
/// below fails until it is in one of them.
const FED: &[&str] = &[
    "abbreviations",
    "ad",
    "alphanumeric",
    "bpe",
    "codec",
    "conllu",
    "corpora",
    "descriptor-errors",
    "detokenizer",
    "doccat",
    "doccat-train",
    "edge",
    "en",
    "formats",
    "glossary",
    "head-rules",
    "ja",
    "ko",
    "langdetect",
    "money",
    "multi",
    "namefind",
    "ner",
    "norm",
    "num",
    "parse",
    "parse-treebank",
    "pii",
    "place-profiles",
    "quantity",
    "sentence-fixture",
    "sentiment",
    "sentiment-train",
    "sequence",
    "stopword",
    "subwords",
    "surrogate",
    "svm-doccat",
    "svm-doccat-train",
    "tag-format",
    "tagged",
    "temporal",
    "train",
    "zh",
];

/// The input sets the runs are not given, each with the reason.
const NOT_FED: &[(&str, &str)] = &[
    (
        "hunspell",
        "words, one per line, read only by the Hunspell goldens with the \
         dictionary each file is named after; the covered runs gain nothing from them",
    ),
    (
        "gazetteer",
        "about half a megabyte of place names, record ids, region codes and small \
         tables, read only by the gazetteer goldens; fed, the test runs for more \
         than half an hour instead of seconds and gives the covered runs nothing new",
    ),
    (
        "stem",
        "about half a megabyte of words, one per line, read only by the stemmers; \
         it more than doubles the test's time and gives the covered runs nothing new",
    ),
];

/// The names of the subdirectories of `dir`.
fn subdirectories(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap())
        .filter(|e| e.file_type().unwrap().is_dir())
        .map(|e| e.file_name().into_string().unwrap())
        .collect()
}

/// [`FED`] as owned names.
fn fed_sets() -> BTreeSet<String> {
    FED.iter().map(|s| s.to_string()).collect()
}

/// Every checked in input text of the given sets, as `<set>/<id>` and its
/// text.
fn golden_inputs(sets: &BTreeSet<String>) -> Vec<(String, String)> {
    let inputs = turbo_goldens::data_dir().join("inputs");
    let mut out = Vec::new();
    for set in sets {
        for file in std::fs::read_dir(inputs.join(set)).unwrap_or_else(|e| panic!("inputs/{set}: {e}")) {
            let file = file.unwrap().path();
            if file.extension().is_some_and(|x| x == "txt") {
                let name = format!("{set}/{}", file.file_stem().unwrap().to_string_lossy());
                out.push((name, std::fs::read_to_string(&file).unwrap()));
            }
        }
    }
    out.sort();
    out
}

/// Every input set on disk is in exactly one of [`FED`] and [`NOT_FED`],
/// both name only sets on disk, and no text of a set of [`NOT_FED`] is fed.
#[test]
fn every_input_set_is_fed_or_left_out_by_name() {
    let on_disk = subdirectories(&turbo_goldens::data_dir().join("inputs"));
    let fed = fed_sets();
    let not_fed: BTreeSet<String> = NOT_FED.iter().map(|(set, _)| set.to_string()).collect();
    assert_eq!(fed.len(), FED.len(), "a set is named twice in FED");
    assert_eq!(not_fed.len(), NOT_FED.len(), "a set is named twice in NOT_FED");
    assert!(fed.is_disjoint(&not_fed), "in both FED and NOT_FED: {:?}", fed.intersection(&not_fed));
    for set in fed.iter().chain(&not_fed) {
        assert!(on_disk.contains(set), "{set} is named but not in inputs/");
    }
    for set in &on_disk {
        assert!(
            fed.contains(set) || not_fed.contains(set),
            "inputs/{set} is new: add it to FED, or to NOT_FED with the reason"
        );
    }
    let texts = golden_inputs(&fed);
    let given: BTreeSet<String> = texts.iter().map(|(id, _)| id.split_once('/').unwrap().0.to_string()).collect();
    assert!(given.is_disjoint(&not_fed), "{given:?}");
    assert_eq!(given, fed, "every fed set gives texts");
}

fn required(path: &Path) -> bool {
    let dictionary = path.to_string_lossy().contains("mecab-");
    !dictionary && std::env::var("TURBO_TEST_REQUIRE_OPENNLP_MODELS").is_ok_and(|v| v == "1")
}

#[test]
fn every_layer_of_every_run_keeps_the_offset_invariants() {
    let res = Resources::from_env();
    let mut texts: Vec<(String, String)> = corpus(SEED, COUNT).into_iter().map(|i| (i.id, i.text)).collect();
    texts.extend(golden_inputs(&fed_sets()));
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
