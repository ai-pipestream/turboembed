//! Every committed Rust file is compiled by something (#2455). A split or
//! a retirement that removes a module root otherwise leaves the rest of
//! its directory behind, where it reads as live code and is never built.
//!
//! The check walks each crate as the compiler does. It starts at the
//! files cargo builds as targets, reads their tokens (`rust_lex.rs`) and
//! follows every `mod name;` to `name.rs` or `name/mod.rs`, through inline
//! `mod name { }` blocks however deep, every `#[path = "..."]`, and every
//! `include!("...")`. A tracked `.rs` file the walk never reaches is
//! reported. `cfg` is not evaluated: a module behind any `cfg` counts.
//!
//! One thing cannot be followed: a `mod` written by a macro, where the
//! name is not in the source as such. Files declared that way are listed
//! in [`DECLARED_BY_MACRO`], each with the file whose macro declares it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

use crate::rust_lex::{Kind, Token, closing, tokens};

/// Programs kept in the tree that the workspace does not build, each with
/// why. They compare our tokenizers with outside ones whose crates are kept
/// out of the workspace's dependencies; the file named beside each says how
/// it is built by hand.
const STANDALONE: &[(&str, &str)] = &[
    ("analysis/tokenize/comparisons/jieba.rs", "docs/unigram-chinese.md"),
    ("analysis/tokenize/comparisons/lattice.rs", "docs/lattice-prebuilt.md"),
    ("analysis/tokenize/comparisons/unigram_overhead.rs", "docs/receipts/unigram-984/prepare.mjs"),
];

/// Files a macro declares as modules (`mod $name;`), each with the file
/// that invokes the macro. The walk cannot see these declarations, so they
/// are stated here; the named file must hold a `macro_rules!` or a macro
/// call and the module's name, or the entry fails its test.
const DECLARED_BY_MACRO: &[(&str, &str)] = &[];

/// `a/b/../c` as `a/c`, without touching the file system.
fn normal(path: &Path) -> String {
    let mut out: Vec<&str> = Vec::new();
    for part in path.components() {
        match part {
            Component::ParentDir => drop(out.pop()),
            Component::Normal(s) => out.push(s.to_str().expect("UTF-8 path")),
            _ => {}
        }
    }
    out.join("/")
}

fn join(dir: &str, rest: &str) -> String {
    normal(&Path::new(dir).join(rest))
}

fn parent(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(dir, _)| dir)
}

/// The files cargo builds as targets of the crate at `dir`: its build
/// script, library and binaries, and each test, example and bench, as a
/// file or as a directory with a `main.rs`.
fn is_target(dir: &str, path: &str) -> bool {
    let inside = match path.strip_prefix(dir) {
        Some(rest) if dir.is_empty() => rest,
        Some(rest) if rest.starts_with('/') => &rest[1..],
        _ => return false,
    };
    let parts: Vec<&str> = inside.split('/').collect();
    let listed = |at: usize| ["tests", "examples", "benches"].contains(&parts[at]);
    match parts.len() {
        1 => parts[0] == "build.rs",
        2 => (parts[0] == "src" && ["lib.rs", "main.rs"].contains(&parts[1])) || listed(0),
        3 => (listed(0) && parts[2] == "main.rs") || (parts[..2] == ["src", "bin"]),
        4 => parts[..2] == ["src", "bin"] && parts[3] == "main.rs",
        _ => false,
    }
}

/// The string of a `path = "..."` inside the attribute that starts at
/// `at` (the `#`), and the index of the attribute's closing bracket.
fn path_attribute<'a>(tokens: &[Token<'a>], at: usize) -> (Option<&'a str>, usize) {
    let open = at + usize::from(tokens.get(at + 1).is_some_and(|t| t.is('!'))) + 1;
    if !tokens.get(open).is_some_and(|t| t.is('[')) {
        return (None, at);
    }
    let close = closing(tokens, open);
    let inside = &tokens[open..close];
    let found = inside.windows(3).find(|w| w[0].word("path") && w[1].is('=') && w[2].string().is_some());
    (found.and_then(|w| w[2].string()), close)
}

struct Walk<'f> {
    files: &'f BTreeMap<String, String>,
    /// Each file reached, with the directory its modules were looked for in.
    seen: BTreeSet<(String, String)>,
}

impl Walk<'_> {
    /// Reads `file`, whose `mod name;` declarations are files under
    /// `mod_dir`, and follows each one that is in the tree.
    fn visit(&mut self, file: &str, mod_dir: &str) {
        let Some(text) = self.files.get(file) else { return };
        if !self.seen.insert((file.to_owned(), mod_dir.to_owned())) {
            return;
        }
        let tokens = tokens(text);
        // one entry per open brace: the directory an inline module adds, or nothing
        let mut open: Vec<Option<String>> = Vec::new();
        let mut path: Option<&str> = None;
        let mut i = 0;
        while i < tokens.len() {
            let t = &tokens[i];
            // the directory the modules declared at this point are files of
            let inline = || open.iter().flatten().map(String::as_str).collect::<Vec<_>>().join("/");
            if t.is('#') {
                let (found, end) = path_attribute(&tokens, i);
                path = found.or(path);
                i = end;
            } else if t.word("mod") && tokens.get(i + 1).is_some_and(|n| n.kind == Kind::Word) {
                let name = tokens[i + 1].text;
                match tokens.get(i + 2) {
                    Some(end) if end.is(';') => match path.take() {
                        // a path is taken from the file's own directory, or from
                        // the inline module's when the declaration is inside one
                        Some(p) => {
                            let inside = inline();
                            let from = if inside.is_empty() { parent(file).to_owned() } else { join(mod_dir, &inside) };
                            let target = join(&from, p);
                            let dir = parent(&target).to_owned();
                            self.visit(&target, &dir);
                        }
                        None => {
                            let dir = join(&join(mod_dir, &inline()), name);
                            self.visit(&format!("{dir}.rs"), &dir);
                            self.visit(&format!("{dir}/mod.rs"), &dir);
                        }
                    },
                    Some(brace) if brace.is('{') => {
                        open.push(Some(path.take().unwrap_or(name).to_owned()));
                        i += 2;
                    }
                    _ => {}
                }
            } else if t.word("include") && tokens.get(i + 1).is_some_and(|n| n.is('!')) {
                if let Some(included) = tokens.get(i + 3).and_then(Token::string) {
                    let here = join(mod_dir, &inline());
                    self.visit(&join(parent(file), included), &here);
                }
            } else if t.is('{') {
                open.push(None);
            } else if t.is('}') {
                open.pop();
            } else if t.is(';') {
                path = None;
            }
            i += 1;
        }
    }
}

/// The files of `files` (path and text) that no target of the crates at
/// `crates` reaches, sorted. `manifest_paths` are the `path = "..."`
/// targets of the Cargo manifests, from the workspace root.
pub fn orphans(
    files: &BTreeMap<String, String>,
    crates: &BTreeSet<String>,
    manifest_paths: &BTreeSet<String>,
) -> Vec<String> {
    let mut walk = Walk { files, seen: BTreeSet::new() };
    for path in files.keys() {
        if manifest_paths.contains(path) || crates.iter().any(|dir| is_target(dir, path)) {
            walk.visit(path, parent(path));
        }
    }
    let reached: BTreeSet<&str> = walk.seen.iter().map(|(file, _)| file.as_str()).collect();
    files.keys().filter(|path| !reached.contains(path.as_str())).cloned().collect()
}

type Tree = (BTreeMap<String, String>, BTreeSet<String>, BTreeSet<String>);

/// The committed Rust files of the workspace's crates, the crates'
/// directories, and the target paths their manifests name. Receipts under
/// `docs/` are kept as evidence and are not built.
fn committed(root: &Path) -> Tree {
    let out = std::process::Command::new("git")
        .current_dir(root)
        .args(["ls-files", "-z", "--", "*.rs", "*Cargo.toml"])
        .output()
        .expect("git lists the committed files");
    assert!(out.status.success());
    let (mut files, mut crates, mut manifest_paths) = Tree::default();
    for p in String::from_utf8(out.stdout).unwrap().split('\0').filter(|s| !s.is_empty()) {
        if p.starts_with("docs/") {
            continue;
        }
        let text = std::fs::read_to_string(root.join(p)).unwrap_or_else(|e| panic!("{p}: {e}"));
        if p.ends_with(".rs") {
            files.insert(p.to_owned(), text);
            continue;
        }
        crates.insert(parent(p).to_owned());
        for line in text.lines().filter(|l| l.trim_start().starts_with("path")) {
            if let Some(target) = line.split('"').nth(1).filter(|t| t.ends_with(".rs")) {
                manifest_paths.insert(join(parent(p), target));
            }
        }
    }
    (files, crates, manifest_paths)
}

fn exempt(path: &str) -> bool {
    STANDALONE.iter().chain(DECLARED_BY_MACRO).any(|(file, _)| *file == path)
}

#[test]
fn every_committed_rust_file_is_compiled() {
    let (files, crates, manifest_paths) = committed(&crate::common::workspace());
    assert!(files.len() > 1000 && crates.len() > 50, "the scan is too narrow: {} files", files.len());
    let mut found = orphans(&files, &crates, &manifest_paths);
    found.retain(|path| !exempt(path));
    assert!(
        found.is_empty(),
        "no target's modules reach these files, so nothing builds them:\n  {}",
        found.join("\n  ")
    );
}

#[test]
fn each_exempt_file_is_unreached_and_accounted_for() {
    let root = crate::common::workspace();
    let (files, crates, manifest_paths) = committed(&root);
    let found = orphans(&files, &crates, &manifest_paths);
    for (file, because) in STANDALONE.iter().chain(DECLARED_BY_MACRO) {
        // an entry for a file that is gone, or that the walk reaches, is stale
        assert!(found.iter().any(|path| path == file), "{file} is built or gone: drop its entry");
        let text = std::fs::read_to_string(root.join(because)).unwrap_or_else(|e| panic!("{because}: {e}"));
        let name = file.rsplit('/').next().unwrap();
        let stem = name.trim_end_matches(".rs");
        if DECLARED_BY_MACRO.iter().any(|(f, _)| f == file) {
            assert!(text.contains('!') && text.contains(stem), "{because} has no macro call naming {stem}");
        } else {
            assert!(text.contains(name), "{because} does not say how {file} is built");
        }
    }
}

#[cfg(test)]
#[path = "orphans_tests.rs"]
mod tests;
