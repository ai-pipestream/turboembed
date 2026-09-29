//! What is committed in the tree, checked as a reader would: the records
//! under benchmarks/records are what their names say, the README and the
//! docs point at files that exist, and no committed text names a home
//! directory on one of our machines.

mod common;

use std::path::{Path, PathBuf};

use common::workspace;
use turbo::record::{self, HOST_PATHS, Record};

/// The `*.json` files under benchmarks/records, by name.
fn committed_records() -> Vec<(String, Vec<u8>)> {
    let dir = workspace().join("benchmarks/records");
    let mut out: Vec<(String, Vec<u8>)> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| (p.file_name().unwrap().to_str().unwrap().to_owned(), std::fs::read(&p).unwrap()))
        .collect();
    out.sort();
    out
}

#[test]
fn the_committed_records_carry_the_name_of_their_contents() {
    let records = committed_records();
    assert!(!records.is_empty(), "benchmarks/records has no record");
    for (name, bytes) in &records {
        let r = Record::parse(name, bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        // The name is a function of the record: device, backend, task,
        // precision, bundle, commit. A renamed or edited file shows here.
        assert_eq!(record::file_name(&r).unwrap(), *name, "{name}: named for other contents");
        assert_eq!(r.library.commit.len(), 40, "{name}: commit {}", r.library.commit);
        assert!(r.library.commit.bytes().all(|b| b.is_ascii_hexdigit()), "{name}: commit {}", r.library.commit);
        assert!(!r.library.pushed_to.is_empty(), "{name}: made from a commit on no branch of origin");
    }
}

/// The paths a document names in backticks: a slash in them and one of
/// the extensions a reader would open, no placeholder characters.
fn named_paths(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for piece in text.split('`').skip(1).step_by(2) {
        let p = piece.trim_end_matches(':');
        let ext = Path::new(p).extension().and_then(|x| x.to_str()).unwrap_or("");
        if p.contains('/')
            && matches!(ext, "md" | "h" | "txt" | "json" | "rs" | "toml" | "yml")
            && !p.contains(['<', '>', '*', '$', '{', ' ', '~'])
            && !p.starts_with('/')
        {
            out.push(p.to_owned());
        }
    }
    out
}

/// README.md and docs/*.md, plus each crate's README where there is one.
fn documents() -> Vec<PathBuf> {
    let root = workspace();
    let mut docs = vec![root.join("README.md")];
    for crate_dir in ["bench", "bundle", "core", "server"] {
        let p = root.join(crate_dir).join("README.md");
        if p.is_file() {
            docs.push(p);
        }
    }
    let mut pages: Vec<PathBuf> = std::fs::read_dir(root.join("docs"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    pages.sort();
    docs.extend(pages);
    docs
}

#[test]
fn the_readme_and_docs_point_at_files_that_exist() {
    let root = workspace();
    let mut missing = Vec::new();
    let mut seen = 0;
    // Paths inside the reference programs the docs cite, not in this tree:
    // TEI's router and core crates (docs/benchmarks.md), HailoRT's header.
    const CITED: [&str; 4] = ["router/", "core/src/infer.rs", "core/src/queue.rs", "include/hailo/"];
    for doc in documents() {
        let text = std::fs::read_to_string(&doc).unwrap();
        for p in named_paths(&text) {
            if CITED.iter().any(|c| p.starts_with(c)) {
                continue;
            }
            seen += 1;
            // A path is from the tree's root, or from the document's directory.
            if !root.join(&p).exists() && !doc.parent().unwrap().join(&p).exists() {
                missing.push(format!("{}: `{p}`", doc.strip_prefix(&root).unwrap().display()));
            }
        }
    }
    assert!(seen > 20, "only {seen} paths named across the documents; the scan is too narrow");
    assert!(missing.is_empty(), "paths named in the documents that do not exist:\n  {}", missing.join("\n  "));
}

#[test]
fn the_committed_text_carries_no_host_path() {
    let root = workspace();
    let mut files: Vec<PathBuf> = documents();
    files.extend(committed_records().iter().map(|(n, _)| root.join("benchmarks/records").join(n)));
    for dir in ["bundle/recipes", "include/turbo", ".github/workflows"] {
        if let Ok(rd) = std::fs::read_dir(root.join(dir)) {
            files.extend(rd.map(|e| e.unwrap().path()).filter(|p| p.is_file()));
        }
    }
    let mut hits = Vec::new();
    for f in &files {
        let text = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        let doc = f.extension().is_some_and(|x| x == "md");
        for (n, line) in text.lines().enumerate() {
            // A document may quote a refused prefix on its own, in backticks,
            // to state the rule; a record or a recipe may not.
            let mut line = line.to_owned();
            if doc {
                for h in HOST_PATHS {
                    line = line.replace(&format!("`{h}`"), "");
                }
            }
            if let Some(h) = HOST_PATHS.iter().find(|h| line.contains(*h)) {
                hits.push(format!("{}:{}: {h}", f.strip_prefix(&root).unwrap().display(), n + 1));
            }
        }
    }
    assert!(files.len() > 40, "only {} files scanned", files.len());
    assert!(hits.is_empty(), "host paths in committed text:\n  {}", hits.join("\n  "));
}
