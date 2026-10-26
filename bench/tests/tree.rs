//! What is committed in the tree, checked as a reader would: the records
//! under benchmarks/records, benchmarks/records/postag,
//! benchmarks/records/tag, benchmarks/records/throughput and
//! benchmarks/ranges are what their names say, the README and the
//! docs point at files that exist, and no committed text names a home
//! directory on one of our machines.

mod common;
#[path = "tree/dependencies.rs"]
mod dependencies;

use std::path::{Path, PathBuf};

use common::workspace;
use turbo::record::{self, HOST_PATHS, Record};
use turbo_bench::postag::{self, PosRecord};
use turbo_bench::ranges::{self, RangeRecord};
use turbo_bench::tag::{self, TagRecord};
use turbo_bench::throughput::{self, ThroughputRecord};

#[test]
fn analysis_uses_shared_fnv_constants() {
    let root = workspace();
    let files = std::process::Command::new("git")
        .current_dir(&root)
        .args(["ls-files", "-z", "--", "analysis/**/*.rs"])
        .output()
        .expect("git lists analysis source files");
    assert!(files.status.success(), "git could not list analysis sources");
    let files = String::from_utf8(files.stdout).expect("UTF-8 source paths");
    let mut duplicate = Vec::new();
    let mut seen = 0;
    for path in files.split('\0').filter(|p| !p.is_empty()) {
        if path.starts_with("analysis/core/src/hash/") {
            continue;
        }
        seen += 1;
        let mut text = std::fs::read_to_string(root.join(path)).unwrap();
        text.retain(|c| c != '_');
        text.make_ascii_lowercase();
        for tail in text.split("0x").skip(1) {
            let end = tail.find(|c: char| !c.is_ascii_hexdigit()).unwrap_or(tail.len());
            let value = u64::from_str_radix(&tail[..end], 16);
            if matches!(value, Ok(1099511628211 | 14695981039346656037)) {
                duplicate.push(path.to_owned());
                break;
            }
        }
    }
    assert!(seen > 100, "analysis source scan is too narrow");
    assert!(duplicate.is_empty(), "use the shared hash builder: {}", duplicate.join(", "));
}

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

/// The `*.json` files under benchmarks/ranges, by name.
fn committed_range_records() -> Vec<(String, Vec<u8>)> {
    let dir = workspace().join("benchmarks/ranges");
    let Ok(rd) = std::fs::read_dir(&dir) else { return Vec::new() };
    let mut out: Vec<(String, Vec<u8>)> = rd
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| (p.file_name().unwrap().to_str().unwrap().to_owned(), std::fs::read(&p).unwrap()))
        .collect();
    out.sort();
    out
}

#[test]
fn the_committed_range_records_carry_the_name_of_their_contents() {
    for (name, bytes) in committed_range_records() {
        let r: RangeRecord = serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(ranges::file_name(&r), name, "{name}: named for other contents");
        ranges::check(&r).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

/// The `*.json` files under benchmarks/records/tag, by name.
fn committed_tag_records() -> Vec<(String, Vec<u8>)> {
    let Ok(rd) = std::fs::read_dir(workspace().join(tag::TAG_DIR)) else { return Vec::new() };
    let mut out: Vec<(String, Vec<u8>)> = rd
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| (p.file_name().unwrap().to_str().unwrap().to_owned(), std::fs::read(&p).unwrap()))
        .collect();
    out.sort();
    out
}

#[test]
fn the_committed_tag_records_carry_the_name_of_their_contents() {
    for (name, bytes) in committed_tag_records() {
        let r: TagRecord = serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(tag::file_name(&r), name, "{name}: named for other contents");
        tag::check(&r).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

/// The `*.json` files under benchmarks/records/postag, by name.
fn committed_postag_records() -> Vec<(String, Vec<u8>)> {
    let Ok(rd) = std::fs::read_dir(workspace().join(postag::POSTAG_DIR)) else { return Vec::new() };
    let mut out: Vec<(String, Vec<u8>)> = rd
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| (p.file_name().unwrap().to_str().unwrap().to_owned(), std::fs::read(&p).unwrap()))
        .collect();
    out.sort();
    out
}

#[test]
fn the_committed_postag_records_carry_the_name_of_their_contents() {
    for (name, bytes) in committed_postag_records() {
        let r: PosRecord = serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(postag::file_name(&r), name, "{name}: named for other contents");
        postag::check(&r).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

/// The paths a document names in backticks: a slash in them and one of
/// the extensions a reader would open, no placeholder characters.
/// The `*.json` files under benchmarks/records/throughput, by name.
fn committed_throughput_records() -> Vec<(String, Vec<u8>)> {
    let Ok(rd) = std::fs::read_dir(workspace().join(throughput::THROUGHPUT_DIR)) else { return Vec::new() };
    let mut out: Vec<(String, Vec<u8>)> = rd
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| (p.file_name().unwrap().to_str().unwrap().to_owned(), std::fs::read(&p).unwrap()))
        .collect();
    out.sort();
    out
}

#[test]
fn the_committed_throughput_records_carry_the_name_of_their_contents() {
    for (name, bytes) in committed_throughput_records() {
        let r: ThroughputRecord =
            serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("{name}: not a throughput record: {e}"));
        assert_eq!(throughput::file_name(&r), name, "{name}: named for other contents");
        throughput::check(&r).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

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

/// AGENTS.md, README.md and docs/*.md, plus each crate's README where there is one.
fn documents() -> Vec<PathBuf> {
    let root = workspace();
    let mut docs = vec![root.join("AGENTS.md"), root.join("README.md")];
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
    files.extend(committed_range_records().iter().map(|(n, _)| root.join("benchmarks/ranges").join(n)));
    files.extend(committed_postag_records().iter().map(|(n, _)| root.join(postag::POSTAG_DIR).join(n)));
    files.extend(committed_tag_records().iter().map(|(n, _)| root.join(tag::TAG_DIR).join(n)));
    files.extend(committed_throughput_records().iter().map(|(n, _)| root.join(throughput::THROUGHPUT_DIR).join(n)));
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

#[test]
fn workspace_manifests_do_not_depend_on_regex() {
    let root = workspace();
    let (seen, found) = dependencies::regex_dependencies(&root);
    assert!(seen > 50, "only {seen} packages scanned");
    assert!(found.is_empty(), "regex dependencies in {}", found.join(", "));
}

#[test]
fn obsolete_source_tree_is_absent() {
    assert!(!workspace().join("ai-slop-generated-shit").exists());
}

#[test]
fn committed_rust_and_markdown_have_no_literal_em_dash() {
    let root = workspace();
    let out = std::process::Command::new("git")
        .current_dir(&root)
        .args(["ls-files", "-z", "--", ":(glob)**/*.rs", ":(glob)**/*.md"])
        .output()
        .expect("git lists Rust and Markdown sources");
    assert!(out.status.success());
    let paths = String::from_utf8(out.stdout).unwrap();
    let mut found = Vec::new();
    let mut seen = 0;
    for path in paths.split('\0').filter(|p| !p.is_empty()) {
        seen += 1;
        if std::fs::read_to_string(root.join(path)).unwrap().contains('\u{2014}') {
            found.push(path);
        }
    }
    assert!(seen > 100, "only {seen} source files scanned");
    assert!(found.is_empty(), "literal em dashes in {}", found.join(", "));
}

#[test]
fn core_and_analysis_external_dependencies_are_allowlisted() {
    const ALLOWED: &[&str] = &[
        "flate2",
        "libc",
        "libloading",
        "memmap2",
        "serde",
        "serde_json",
        "serde_path_to_error",
        "sha2",
        "tempfile",
        "tokenizers",
        "turbovec",
        "ureq",
    ];
    let root = workspace();
    let mut paths = vec!["core/Cargo.toml".to_owned()];
    let out = std::process::Command::new("git")
        .current_dir(&root)
        .args(["ls-files", "-z", "--", ":(glob)analysis/*/Cargo.toml"])
        .output()
        .expect("git lists analysis manifests");
    assert!(out.status.success());
    paths.extend(String::from_utf8(out.stdout).unwrap().split('\0').filter(|p| !p.is_empty()).map(str::to_owned));
    let mut found = Vec::new();
    for path in &paths {
        let text = std::fs::read_to_string(root.join(path)).unwrap();
        let mut dependencies = false;
        for line in text.lines() {
            let line = line.split('#').next().unwrap().trim();
            if line.starts_with('[') {
                dependencies = matches!(line, "[dependencies]" | "[dev-dependencies]" | "[build-dependencies]")
                    || line.ends_with(".dependencies]");
                continue;
            }
            if !dependencies {
                continue;
            }
            let Some((name, value)) = line.split_once('=') else { continue };
            let name = name.trim();
            if !value.contains("path =") && !ALLOWED.contains(&name) {
                found.push(format!("{path}: {name}"));
            }
            if matches!(path.as_str(), "core/Cargo.toml" | "analysis/calibration/Cargo.toml")
                && matches!(
                    name,
                    "sha2" | "unicode-normalization-alignments" | "unicode_categories" | "unicode-segmentation"
                )
            {
                found.push(format!("{path}: removed dependency {name} returned"));
            }
        }
    }
    assert!(paths.len() > 50, "only {} manifests scanned", paths.len());
    assert!(found.is_empty(), "unauthorized external dependencies:\n  {}", found.join("\n  "));
}
