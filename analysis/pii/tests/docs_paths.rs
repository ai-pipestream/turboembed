//! `docs/pii.md` names only repository paths that exist.
//!
//! Every backticked path starting with `analysis/` must be on disk. A path
//! with a `<placeholder>` segment is checked up to the segment before it.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn named_paths(doc: &str) -> Vec<String> {
    doc.split('`')
        .skip(1)
        .step_by(2)
        .filter(|s| s.starts_with("analysis/") && !s.contains(char::is_whitespace))
        .map(str::to_string)
        .collect()
}

fn checked_prefix(path: &str) -> &str {
    match path.find('<') {
        Some(at) => path[..at].trim_end_matches('/'),
        None => path,
    }
}

#[test]
fn placeholder_paths_are_checked_up_to_the_placeholder() {
    assert_eq!(checked_prefix("analysis/pii/tests/snapshots/<pack>/<id>.tsv"), "analysis/pii/tests/snapshots");
    assert_eq!(checked_prefix("analysis/pii/tests/goldens.rs"), "analysis/pii/tests/goldens.rs");
    assert_eq!(named_paths("see `analysis/a.rs` and `cargo test` and `analysis/b/`"), ["analysis/a.rs", "analysis/b/"]);
}

#[test]
fn every_path_named_in_the_pii_doc_exists() {
    let root = repo_root();
    let doc = std::fs::read_to_string(root.join("docs/pii.md")).expect("docs/pii.md");
    let paths = named_paths(&doc);
    assert!(paths.len() > 10, "{} paths named", paths.len());
    let missing: Vec<&String> = paths.iter().filter(|p| !root.join(checked_prefix(p)).exists()).collect();
    assert!(missing.is_empty(), "docs/pii.md names missing paths: {missing:?}");
}
