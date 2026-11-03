//! No OpenNLP 1.5 model is fetched, pinned, scored or run, and no OpenNLP
//! 2.5 model is fetched or pinned (#1509). No committed file names a 1.5
//! model's files, ids, address, recorded constants or SHA-256 pins, and
//! none gives a 2.5 model's download address or SHA-256 pin, outside the
//! licence record that refuses them and the filters and negative tests
//! that keep them out. The 2.5 models' names stay in the JVM goldens, the
//! #172 eval records and the receipts that say which model made them, and
//! in the disabled classical code; #1532 removes those.

use std::path::Path;

/// The names, address and constants of the 1.5 English models, and the
/// first twelve hex digits of the SHA-256 the former gates pinned.
const RETIRED_1_5: &[&str] = &[
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

/// Where the 2.5 models (the UD 1.3 models built with OpenNLP 2.5.4 and
/// `langdetect-183.bin`) are published, on Apache's hosts and on Maven
/// Central, and the first twelve hex digits of the SHA-256 the recipes,
/// the model tables and masc-references.txt pinned.
const RETIRED_2_5: &[&str] = &[
    "opennlp/models/",
    "ud-models-1.3",
    "langdetect/1.8.3",
    "opennlp-models-",
    "4de58d12b094",
    "95139111197b",
    "2ddf585fac2e",
    "d38e645dc812",
    "10b25bc4506a",
    "510b89087e7e",
    "073fefbfaff2",
    "711ad051faba",
    "8850986f02df",
    "32d0a7ff84fd",
    "309de611ff7a",
    "a06dbc9b7260",
    "b85f42a6a1df",
];

/// Files that may name either, each with its reason.
const ALLOWED: &[&str] = &[
    // The licence record that refuses the models.
    "docs/licences.md",
    // Negative tests over the train-recipes and goldens-tool source lists.
    "analysis/train-recipes/src/release/tests.rs",
    "analysis/goldens-tool/src/recipe/tests.rs",
    // The model pins' filter that refuses the 1.5 address and the 2.5
    // files, and its negative tests (#1623).
    "analysis/pins/src/lib.rs",
    "analysis/pins/src/tests.rs",
    // This check.
    "bench/tests/tree/retired_opennlp.rs",
];

/// Directories whose files record which 2.5 model made them, by its
/// SHA-256: the JVM goldens, the #172 eval records and the receipts.
/// #1532 removes the goldens; the records and receipts are evidence. A
/// 1.5 name is refused there too.
const RECORDS: &[&str] = &["analysis/goldens/data/", "analysis/eval/records/", "docs/receipts/"];

/// The retired names, addresses and pins `text` holds, for a file under
/// [`RECORDS`] when `record`.
fn named(text: &str, record: bool) -> Vec<&'static str> {
    let v2_5: &[&str] = if record { &RETIRED_2_5[..4] } else { RETIRED_2_5 };
    RETIRED_1_5.iter().chain(v2_5).copied().filter(|r| text.contains(r)).collect()
}

/// Every committed file outside [`ALLOWED`] naming a 1.5 model or
/// pinning or addressing a 2.5 one, with what it names, sorted.
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
        let record = RECORDS.iter().any(|d| p.starts_with(d));
        let names = named(&String::from_utf8_lossy(&bytes), record);
        if !names.is_empty() {
            found.push(format!("{p}: {}", names.join(", ")));
        }
    }
    found.sort();
    found
}

#[test]
fn no_committed_file_names_a_retired_opennlp_model() {
    let found = violations(&crate::common::workspace());
    assert!(found.is_empty(), "retired OpenNLP models named, pinned or addressed in:\n  {}", found.join("\n  "));
}

#[test]
fn the_check_fires_on_each_kind_of_name_and_not_on_other_models() {
    assert_eq!(named("fetch en-pos-maxent.bin", false), ["en-pos-maxent"]);
    assert_eq!(named("https://opennlp.sourceforge.net/models-1.5/x.bin", false), ["models-1.5"]);
    assert_eq!(named("namefind::EN_NER_PERSON", false), ["EN_NER_"]);
    assert_eq!(named("sha256 687a9263d96b37fced707c9f2ac0560f", false), ["687a9263d96b"]);
    assert_eq!(
        named("https://archive.apache.org/dist/opennlp/models/ud-models-1.3/x.bin", false),
        ["opennlp/models/", "ud-models-1.3"]
    );
    assert_eq!(named("https://downloads.apache.org/opennlp/models/langdetect/1.8.3/", true).len(), 2);
    assert_eq!(named("org.apache.opennlp:opennlp-models-pos-en:1.3.0", false), ["opennlp-models-"]);
    assert_eq!(named("sha256: \"2ddf585fac2e02a9\"", false), ["2ddf585fac2e"]);
    // A record may say which 2.5 model made it, by name and SHA-256, but
    // not where to fetch it, and never a 1.5 model.
    assert!(named("model opennlp-en-ud-ewt-sentence-1.3-2.5.4 4de58d12b09421003c8b", true).is_empty());
    assert_eq!(named("en-chunker.bin", true), ["en-chunker"]);
    // A 2.5 model's name alone, and other models, are not refused.
    assert!(named("opennlp-en-ud-ewt-pos-1.3-2.5.4.bin langdetect-183.bin chunker-fixture.bin", false).is_empty());
}
