//! No committed source writes a legacy model id: each component's old
//! id is spelled only in the table that maps it (`src/ids.rs`), and
//! readers of stored provenance accept it through that table. Records
//! written before the rename (evaluation and benchmark records,
//! receipts) keep the ids they were written with.

use std::path::Path;

use turbo_analysis::ids::legacy_model_ids;

/// Committed data that predates the rename, read through the fallback.
const DATA: &[&str] = &["analysis/eval/records/", "benchmarks/records/", "docs/receipts/", "analysis/core/src/ids.rs"];

/// Whether `text` holds `id` as a whole id prefix: followed by the end
/// of the text, the colon before a language part, or any character that
/// cannot continue a component name.
fn holds(text: &str, id: &str) -> bool {
    text.match_indices(id).any(|(at, _)| {
        text[at + id.len()..].chars().next().is_none_or(|c| !(c.is_alphanumeric() || c == '-' || c == '_'))
    })
}

#[test]
fn no_committed_source_writes_a_legacy_model_id() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out =
        std::process::Command::new("git").current_dir(&root).args(["ls-files", "-z"]).output().expect("git ls-files");
    assert!(out.status.success(), "git could not list the tree");
    let paths = String::from_utf8(out.stdout).expect("UTF-8 paths");
    let legacy: Vec<_> = legacy_model_ids().collect();
    let mut seen = 0;
    let mut found = Vec::new();
    for path in paths.split('\0').filter(|p| !p.is_empty()) {
        if DATA.iter().any(|d| path.starts_with(d)) {
            continue;
        }
        // Binary files (graph fixtures, models) are not source text.
        let Ok(text) = std::fs::read_to_string(root.join(path)) else { continue };
        seen += 1;
        for (old, new) in &legacy {
            if holds(&text, old) {
                found.push(format!("{path}: {old} (now {new})"));
            }
        }
    }
    assert!(seen > 1000, "only {seen} files scanned");
    assert!(found.is_empty(), "legacy model ids in committed sources:\n  {}", found.join("\n  "));
}

#[test]
fn the_guard_sees_an_id_with_and_without_a_language_part() {
    assert!(holds("id: opennlp:SentenceDetectorME:en", "opennlp:SentenceDetectorME"));
    assert!(holds("\"opennlp:MecabDictionary\"", "opennlp:MecabDictionary"));
    assert!(!holds("opennlp:MecabDictionaryReader", "opennlp:MecabDictionary"));
}
