//! No committed source writes a legacy shared-layer id: each one is
//! spelled only in the table that maps it (`src/ids.rs`), and readers
//! accept it through that table. Records written before the rename
//! (evaluation and benchmark records, receipts, the stored legacy graph)
//! keep the ids they were written with.

use std::path::Path;

use turbo_analysis::ids::legacy_ids;

/// Committed data that predates the rename, read through the fallback.
const DATA: &[&str] = &[
    "analysis/eval/records/",
    "benchmarks/records/",
    "docs/receipts/",
    "analysis/core/src/ids.rs",
    "analysis/layers/tests/fixtures/legacy-layer-ids.tgrf",
];

/// Whether `text` holds `id` as a whole id, not as the start of a longer name.
fn holds(text: &str, id: &str) -> bool {
    text.match_indices(id).any(|(at, _)| {
        text[at + id.len()..].chars().next().is_none_or(|c| !(c.is_alphanumeric() || c == '-' || c == '_'))
    })
}

#[test]
fn no_committed_source_writes_a_legacy_layer_id() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out =
        std::process::Command::new("git").current_dir(&root).args(["ls-files", "-z"]).output().expect("git ls-files");
    assert!(out.status.success(), "git could not list the tree");
    let paths = String::from_utf8(out.stdout).expect("UTF-8 paths");
    let legacy: Vec<_> = legacy_ids().collect();
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
    assert!(found.is_empty(), "legacy layer ids in committed sources:\n  {}", found.join("\n  "));
}
