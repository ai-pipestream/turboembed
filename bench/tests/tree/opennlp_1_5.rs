//! No OpenNLP 1.5 model is fetched, pinned, scored or run (#1509): no
//! committed file names one of their files, ids, addresses, recorded
//! constants or SHA-256 pins, outside the licence record that refuses them
//! and the negative tests that check the source lists.

use std::path::Path;

/// The names, address and constants of the 1.5 English models, and the
/// first twelve hex digits of the SHA-256 the former gates pinned.
const RETIRED: &[&str] = &[
    "en-pos-maxent",
    "en-pos-perceptron",
    "en-chunker",
    "en-ner-",
    "models-1.5",
    "EN_POS_MAXENT",
    "EN_CHUNKER",
    "EN_NER_",
    "645a094f45a8",
    "0b49b7d9bdb9",
    "7861a0c2f134",
    "687a9263d96b",
    "8fe39e48633f",
    "0136c12afe1a",
    "120703092385",
];

/// Files that may name them, each with its reason.
const ALLOWED: &[&str] = &[
    // The licence record that refuses the models for every use.
    "docs/licences.md",
    // Negative tests over the train-recipes and goldens-tool source lists.
    "analysis/train-recipes/src/release/tests.rs",
    "analysis/goldens-tool/src/recipe/tests.rs",
    // This check.
    "bench/tests/tree/opennlp_1_5.rs",
];

/// The retired names `text` holds.
fn named(text: &str) -> Vec<&'static str> {
    RETIRED.iter().copied().filter(|r| text.contains(r)).collect()
}

/// Every committed file outside [`ALLOWED`] naming a 1.5 model, with what
/// it names, sorted.
pub fn violations(root: &Path) -> Vec<String> {
    let out = std::process::Command::new("git")
        .current_dir(root)
        .args(["ls-files", "-z"])
        .output()
        .expect("git lists the committed files");
    assert!(out.status.success());
    let mut found = Vec::new();
    for p in String::from_utf8(out.stdout).unwrap().split('\0').filter(|s| !s.is_empty()) {
        if ALLOWED.contains(&p) {
            continue;
        }
        let Ok(bytes) = std::fs::read(root.join(p)) else { continue };
        let names = named(&String::from_utf8_lossy(&bytes));
        if !names.is_empty() {
            found.push(format!("{p}: {}", names.join(", ")));
        }
    }
    found.sort();
    found
}

#[test]
fn no_committed_file_names_an_opennlp_1_5_model() {
    let found = violations(&crate::common::workspace());
    assert!(found.is_empty(), "OpenNLP 1.5 models named in:\n  {}", found.join("\n  "));
}

#[test]
fn the_check_fires_on_each_kind_of_name_and_not_on_other_models() {
    assert_eq!(named("fetch en-pos-maxent.bin"), ["en-pos-maxent"]);
    assert_eq!(named("https://opennlp.sourceforge.net/models-1.5/x.bin"), ["models-1.5"]);
    assert_eq!(named("namefind::EN_NER_PERSON"), ["EN_NER_"]);
    assert_eq!(named("sha256 687a9263d96b37fced707c9f2ac0560f"), ["687a9263d96b"]);
    assert!(named("opennlp-en-ud-ewt-pos-1.3-2.5.4.bin chunker-fixture.bin").is_empty());
}
