//! The offset invariants (`turbo_goldens::invariants`) over every layer
//! every run writes, on a small seeded corpus of the fuzz generator and on
//! the checked in golden inputs. No container is needed: the Rust side
//! runs alone. Runs whose published model is not downloaded are skipped,
//! unless `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`; the published MeCab
//! dictionaries are left out unless `TURBO_TEST_MECAB_DICTIONARIES=1`, as
//! the lattice crate's own tests on them are ignored by default.

use std::path::Path;

use turbo_fuzz::{Resources, Unavailable, runs};
use turbo_goldens::fuzz::corpus;
use turbo_goldens::invariants::check_document;

const SEED: u64 = 7;
const COUNT: u64 = 40;

/// Every checked in input text, as `<set>/<id>` and its text.
fn golden_inputs() -> Vec<(String, String)> {
    let inputs = turbo_goldens::data_dir().join("inputs");
    let mut out = Vec::new();
    for set in std::fs::read_dir(&inputs).unwrap() {
        let set = set.unwrap().path();
        for file in std::fs::read_dir(&set).unwrap() {
            let file = file.unwrap().path();
            if file.extension().is_some_and(|x| x == "txt") {
                let name = format!(
                    "{}/{}",
                    set.file_name().unwrap().to_string_lossy(),
                    file.file_stem().unwrap().to_string_lossy()
                );
                out.push((name, std::fs::read_to_string(&file).unwrap()));
            }
        }
    }
    out.sort();
    out
}

fn required(path: &Path) -> bool {
    let dictionary = path.to_string_lossy().contains("mecab-");
    !dictionary && std::env::var("TURBO_TEST_REQUIRE_OPENNLP_MODELS").is_ok_and(|v| v == "1")
}

#[test]
fn every_layer_of_every_run_keeps_the_offset_invariants() {
    let res = Resources::from_env();
    let mut texts: Vec<(String, String)> = corpus(SEED, COUNT).into_iter().map(|i| (i.id, i.text)).collect();
    texts.extend(golden_inputs());
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
