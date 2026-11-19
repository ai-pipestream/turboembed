//! Every committed Rust file is compiled by something (#2455): it is a
//! cargo target root, or a `mod` declaration, a `#[path]` attribute or an
//! `include!` in the tree names it. A split or a retirement that removes a
//! module root otherwise leaves the rest of its directory behind, where it
//! reads as live code and is never built.
//!
//! The check is by name and place, not by expanding the crate: a file is
//! taken as declared when a file in its directory or the one above says
//! `mod <its name>;`. Looking one directory up covers a module declared
//! inside an inline `mod { }` block. The check errs toward silence: a file
//! whose name some neighbour declares for another purpose is not reported.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

/// Directories whose direct `.rs` children cargo builds as targets.
const TARGET_DIRS: &[&str] = &["tests", "examples", "benches", "bin"];

/// Programs kept in the tree that the workspace does not build, each with
/// why. They compare our tokenizers with outside ones whose crates are kept
/// out of the workspace's dependencies; the file named beside each says how
/// it is built by hand.
const STANDALONE: &[(&str, &str)] = &[
    ("analysis/tokenize/comparisons/jieba.rs", "docs/unigram-chinese.md"),
    ("analysis/tokenize/comparisons/lattice.rs", "docs/lattice-prebuilt.md"),
    ("analysis/tokenize/comparisons/unigram_overhead.rs", "docs/receipts/unigram-984/prepare.mjs"),
];

/// `a/b/../c` as `a/c`, without touching the file system.
fn normal(path: &Path) -> String {
    let mut out: Vec<String> = Vec::new();
    for part in path.components() {
        match part {
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(s) => out.push(s.to_str().expect("UTF-8 path").to_owned()),
            _ => {}
        }
    }
    out.join("/")
}

/// The string literals of every `#[path = "..."]` and `include!("...")`.
fn path_literals(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    for marker in ["#[path", "include!("] {
        for (at, _) in text.match_indices(marker) {
            let rest = &text[at + marker.len()..];
            let Some(open) = rest.find('"') else { continue };
            // only white space and `=` may stand before the literal
            if !rest[..open].chars().all(|c| c.is_whitespace() || c == '=') {
                continue;
            }
            if let Some(len) = rest[open + 1..].find('"') {
                out.push(&rest[open + 1..open + 1 + len]);
            }
        }
    }
    out
}

/// The names `text` declares as modules kept in a file of their own:
/// `mod name;`. An inline `mod name { }` names no file, and a declaration
/// in a line comment is not one.
fn declared_modules(text: &str) -> Vec<&str> {
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    let mut out = Vec::new();
    for line in text.lines() {
        let code = line.split("//").next().unwrap();
        for (at, _) in code.match_indices("mod") {
            if code[..at].chars().next_back().is_some_and(ident) {
                continue;
            }
            let rest = &code[at + 3..];
            let name = rest.trim_start();
            if name.len() == rest.len() {
                continue;
            }
            let name = name.strip_prefix("r#").unwrap_or(name);
            let end = name.find(|c: char| !ident(c)).unwrap_or(name.len());
            if end > 0 && name[end..].trim_start().starts_with(';') {
                out.push(&name[..end]);
            }
        }
    }
    out
}

fn parent(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(dir, _)| dir)
}

/// The files of `files` (path and text) that nothing compiles, sorted.
/// `manifest_paths` are the `path = "..."` targets of the Cargo manifests,
/// from the workspace root.
pub fn orphans(files: &BTreeMap<String, String>, manifest_paths: &BTreeSet<String>) -> Vec<String> {
    // files named by a path literal, and the modules each directory declares
    let mut named: BTreeSet<String> = BTreeSet::new();
    let mut declared: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for (path, text) in files {
        let dir = parent(path);
        let stem = path.rsplit('/').next().unwrap().trim_end_matches(".rs");
        for literal in path_literals(text) {
            named.insert(normal(&Path::new(dir).join(literal)));
            // `#[path]` inside an inline module of `stem.rs` starts at `stem/`
            named.insert(normal(&Path::new(dir).join(stem).join(literal)));
        }
        declared.entry(dir).or_default().extend(declared_modules(text));
    }
    let declares = |dir: &str, name: &str| declared.get(dir).is_some_and(|names| names.contains(name));
    let mut out = Vec::new();
    for path in files.keys() {
        let dir = parent(path);
        let file = path.rsplit('/').next().unwrap();
        let in_target_dir = TARGET_DIRS.contains(&dir.rsplit('/').next().unwrap());
        if matches!(file, "lib.rs" | "main.rs" | "build.rs") || in_target_dir {
            continue;
        }
        if manifest_paths.contains(path) || named.contains(path) {
            continue;
        }
        // the module's name, and the directory its declaration is looked for in
        let (name, home) = match file {
            "mod.rs" => (dir.rsplit('/').next().unwrap(), parent(dir)),
            _ => (file.trim_end_matches(".rs"), dir),
        };
        if !declares(home, name) && !declares(parent(home), name) {
            out.push(path.clone());
        }
    }
    out
}

/// The committed Rust files of the workspace's crates, and the target
/// paths its manifests name. Receipts under `docs/` are kept as evidence
/// and are not built.
fn committed(root: &Path) -> (BTreeMap<String, String>, BTreeSet<String>) {
    let out = std::process::Command::new("git")
        .current_dir(root)
        .args(["ls-files", "-z", "--", "*.rs", "*Cargo.toml"])
        .output()
        .expect("git lists the committed files");
    assert!(out.status.success());
    let mut files = BTreeMap::new();
    let mut manifest_paths = BTreeSet::new();
    for p in String::from_utf8(out.stdout).unwrap().split('\0').filter(|s| !s.is_empty()) {
        if p.starts_with("docs/") {
            continue;
        }
        let text = std::fs::read_to_string(root.join(p)).unwrap_or_else(|e| panic!("{p}: {e}"));
        if !p.ends_with(".rs") {
            for line in text.lines().filter(|l| l.trim_start().starts_with("path")) {
                if let Some(target) = line.split('"').nth(1).filter(|t| t.ends_with(".rs")) {
                    manifest_paths.insert(normal(&PathBuf::from(parent(p)).join(target)));
                }
            }
            continue;
        }
        files.insert(p.to_owned(), text);
    }
    (files, manifest_paths)
}

#[test]
fn every_committed_rust_file_is_compiled() {
    let (files, manifest_paths) = committed(&crate::common::workspace());
    assert!(files.len() > 1000, "the source scan is too narrow: {} files", files.len());
    let mut found = orphans(&files, &manifest_paths);
    found.retain(|path| !STANDALONE.iter().any(|(program, _)| program == path));
    assert!(
        found.is_empty(),
        "no mod, #[path] or include! names these files, so nothing builds them:\n  {}",
        found.join("\n  ")
    );
}

#[test]
fn each_standalone_program_is_unbuilt_and_documented() {
    let root = crate::common::workspace();
    let (files, manifest_paths) = committed(&root);
    let found = orphans(&files, &manifest_paths);
    for (program, document) in STANDALONE {
        // an entry for a file that is gone, or that is built now, is stale
        assert!(found.iter().any(|path| path == program), "{program} is built or gone: drop it from STANDALONE");
        let text = std::fs::read_to_string(root.join(document)).unwrap_or_else(|e| panic!("{document}: {e}"));
        let name = program.rsplit('/').next().unwrap();
        assert!(text.contains(name), "{document} does not say how {program} is built");
    }
}

fn tree(files: &[(&str, &str)]) -> BTreeMap<String, String> {
    files.iter().map(|(p, t)| ((*p).to_owned(), (*t).to_owned())).collect()
}

#[test]
fn a_directory_left_without_its_root_is_reported() {
    // what #2455 found: the test root is gone, its support modules are not
    let files = tree(&[
        ("a/src/lib.rs", "pub mod kept;\n"),
        ("a/src/kept.rs", ""),
        ("a/tests/live/main.rs", "mod helper;\n"),
        ("a/tests/live/helper.rs", ""),
        ("a/tests/gone/check.rs", "use crate::golden::*;\n"),
        ("a/tests/gone/golden.rs", ""),
    ]);
    assert_eq!(orphans(&files, &BTreeSet::new()), ["a/tests/gone/check.rs", "a/tests/gone/golden.rs"]);
}

#[test]
fn each_way_of_naming_a_file_counts() {
    let files = tree(&[
        ("a/src/lib.rs", "mod plain;\npub(crate) mod nested;\nmod inline {\n    pub mod inner;\n}\n"),
        ("a/src/plain.rs", "#[path = \"elsewhere/by_path.rs\"]\nmod renamed;\ninclude!(\"parts/included.rs\");\n"),
        ("a/src/nested/mod.rs", "mod leaf;\n"),
        ("a/src/nested/leaf.rs", ""),
        ("a/src/inline/inner.rs", ""),
        ("a/src/elsewhere/by_path.rs", ""),
        ("a/src/parts/included.rs", ""),
        ("a/tools/gen.rs", ""),
        ("a/tests/t.rs", "#[path = \"../src/shared.rs\"]\nmod shared;\n"),
        ("a/src/shared.rs", ""),
        ("a/src/bin/tool.rs", ""),
        ("a/build.rs", ""),
    ]);
    let manifest: BTreeSet<String> = ["a/tools/gen.rs".to_owned()].into();
    assert_eq!(orphans(&files, &manifest), Vec::<String>::new());
    // without the manifest entry the tool is nobody's
    assert_eq!(orphans(&files, &BTreeSet::new()), ["a/tools/gen.rs"]);
}

#[test]
fn words_that_only_look_like_a_declaration_do_not_count() {
    let files = tree(&[
        ("a/src/lib.rs", "// see the model; the commod stray; modstray;\nfn f() { let model = 1; }\n"),
        ("a/src/stray.rs", ""),
        ("a/src/model.rs", ""),
        ("a/src/other.rs", "// #[path] is described here, not used: \"stray.rs\"\n"),
        // an inline module names no file, and a comment declares nothing
        ("a/src/unit.rs", "#[cfg(test)]\nmod tests {\n    // mod retired;\n}\n"),
        ("a/src/tests.rs", ""),
        ("a/src/retired.rs", ""),
    ]);
    assert_eq!(
        orphans(&files, &BTreeSet::new()),
        ["a/src/model.rs", "a/src/other.rs", "a/src/retired.rs", "a/src/stray.rs", "a/src/tests.rs", "a/src/unit.rs"]
    );
}
