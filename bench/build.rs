//! The commit the tool, and the library linked into it, are built from,
//! and whether the working tree had changes then: `record` names that
//! commit only when it is the one --repo is at and the build was clean
//! (docs/benchmarks.md, "Provenance").
//!
//! Sets TURBO_BENCH_BUILD_COMMIT (40 hex, or empty outside a git tree) and
//! TURBO_BENCH_BUILD_CHANGES (the changed paths, as `git status
//! --porcelain` gives them, joined with `, `; empty when clean). New files
//! under benchmarks/records/ are not changes, as in git.rs.
//!
//! Cargo takes a watched path that does not exist as always changed, and
//! would run this, and rebuild the tool, on every call: so of a clean
//! tree only paths that exist are watched.

use std::path::{Path, PathBuf};
use std::process::Command;

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git").arg("-C").arg(dir).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim_end().to_owned())
}

/// Watch `path`; when it does not exist, the nearest directory above it
/// that does, below `top` (the file coming back changes it). A missing
/// path directly under `top` is watched as it is, so every build runs
/// this until it is back: `top` itself holds target/, which every build
/// changes.
fn watch(top: &Path, path: &Path) {
    let mut p = path;
    while !p.exists() {
        match p.parent() {
            Some(parent) if parent != top && parent.starts_with(top) => p = parent,
            _ => {
                p = path;
                break;
            }
        }
    }
    println!("cargo:rerun-if-changed={}", p.display());
}

/// The paths a `git status --porcelain=v1` line names: one, or both sides
/// of a rename. A path git quotes has its quotes taken off and its escapes
/// left, which at worst watches a directory above it.
fn status_paths(line: &str) -> Vec<String> {
    line.get(3..).unwrap_or("").split(" -> ").map(|p| p.trim_matches('"').to_owned()).collect()
}

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    // Canonical, so no path handed to cargo has `..` in it.
    let top = Path::new(&manifest).join("..").canonicalize().unwrap();

    // What moves the commit: HEAD itself (this worktree's, when .git is a
    // file), the branch it names, and the packed refs, each where
    // `--git-path` says git keeps it, so a worktree works. A branch only in
    // packed-refs has no file of its own: its directory is watched, where
    // the branch's next commit writes one.
    let mut refs = vec!["HEAD".to_owned(), "packed-refs".to_owned()];
    if let Some(branch) = git(&top, &["symbolic-ref", "--quiet", "HEAD"]) {
        refs.push(branch);
    }
    for name in refs {
        let Some(p) = git(&top, &["rev-parse", "--git-path", &name]) else { continue };
        let p: PathBuf = top.join(p);
        if p.exists() {
            println!("cargo:rerun-if-changed={}", p.display());
        } else if name != "packed-refs"
            && let Some(dir) = p.parent().filter(|d| d.is_dir())
        {
            println!("cargo:rerun-if-changed={}", dir.display());
        }
    }
    // What is built into the tool: a change to any of it runs this again,
    // so the changes below are never older than the binary.
    for part in ["core", "include", "bench", "benchmarks", "Cargo.toml", "Cargo.lock"] {
        let p = top.join(part);
        if p.exists() {
            println!("cargo:rerun-if-changed={}", p.display());
        }
    }

    let commit = git(&top, &["rev-parse", "--verify", "HEAD^{commit}"]).unwrap_or_default();
    // --no-optional-locks: status must not rewrite the index while cargo
    // watches the tree.
    let status = git(
        &top,
        &["--no-optional-locks", "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"],
    );
    let changes: Vec<String> = match status {
        Some(s) => s
            .lines()
            .filter(|l| !l.strip_prefix("?? ").is_some_and(|p| p.starts_with("benchmarks/records/")))
            .map(str::to_owned)
            .collect(),
        None => vec!["(git status failed)".to_owned()],
    };
    // A dirty build watches its own dirt: each changed path, wherever it
    // is, so reverting or removing it runs this again.
    for line in &changes {
        for p in status_paths(line) {
            watch(&top, &top.join(p));
        }
    }
    println!("cargo:rustc-env=TURBO_BENCH_BUILD_COMMIT={commit}");
    println!("cargo:rustc-env=TURBO_BENCH_BUILD_CHANGES={}", changes.join(", "));
}
