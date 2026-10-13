//! What the reader tests share: the checked-in samples and scratch
//! directories.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use turbo_eval::set::LoadedSet;
use turbo_eval::spans::Span;

/// The sample of `set` under tests/data.
pub fn data(set: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(set)
}

/// An empty directory for one test, under the system's temporary
/// directory.
pub fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-eval-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Copies the directory `from` into `to`, recursively.
pub fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for e in fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let dest = to.join(e.file_name());
        if e.path().is_dir() {
            copy_dir(&e.path(), &dest);
        } else {
            fs::copy(e.path(), dest).unwrap();
        }
    }
}

/// The text of document `id`.
pub fn text<'a>(set: &'a LoadedSet, id: &str) -> &'a str {
    &set.docs.iter().find(|d| d.id == id).unwrap_or_else(|| panic!("no document {id}")).text
}

/// The texts at the relevant spans of `doc` for `query`.
pub fn quoted<'a>(set: &'a LoadedSet, query: &str, doc: &str) -> Vec<&'a str> {
    let t = text(set, doc);
    set.spans.as_ref().unwrap().spans(query, doc).iter().map(|s| &t[s.start..s.end]).collect()
}

/// The relevant spans of `doc` for `query`, as (start, end).
pub fn spans(set: &LoadedSet, query: &str, doc: &str) -> Vec<(usize, usize)> {
    set.spans.as_ref().unwrap().spans(query, doc).iter().map(|&Span { start, end }| (start, end)).collect()
}

pub fn query_ids(set: &LoadedSet) -> Vec<&str> {
    set.queries.iter().map(|q| q.id.as_str()).collect()
}
