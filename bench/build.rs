//! The commit the tool, and the library linked into it, are built from,
//! and whether the working tree had changes then: `record` names that
//! commit only when it is the one --repo is at and the build was clean
//! (docs/benchmarks.md, "Provenance").
//!
//! Sets TURBO_BENCH_BUILD_COMMIT (40 hex, or empty outside a git tree) and
//! TURBO_BENCH_BUILD_CHANGES (the changed paths, as `git status
//! --porcelain` gives them, joined with `, `; empty when clean). New files
//! under benchmarks/records/ are not changes, as in git.rs.

use std::path::Path;
use std::process::Command;

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git").arg("-C").arg(dir).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim_end().to_owned())
}

/// A path git gives relative to `dir`, made absolute.
fn git_path(dir: &Path, name: &str) -> Option<String> {
    let p = git(dir, &["rev-parse", "--git-path", name])?;
    Some(dir.join(p).display().to_string())
}

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let top = Path::new(&manifest).join("..");
    // What moves the commit: HEAD itself (this worktree's, when .git is a
    // file), the branch it names, and the packed refs. `--git-path` finds
    // each where git keeps it, in a worktree too.
    let mut watch = vec!["HEAD".to_owned(), "packed-refs".to_owned()];
    if let Some(branch) = git(&top, &["symbolic-ref", "--quiet", "HEAD"]) {
        watch.push(branch);
    }
    for name in watch {
        if let Some(p) = git_path(&top, &name) {
            println!("cargo:rerun-if-changed={p}");
        }
    }
    // What is built into the tool: a change to any of it runs this again,
    // so the changes below are never older than the binary.
    for part in ["core", "include", "bench", "benchmarks", "Cargo.toml", "Cargo.lock"] {
        println!("cargo:rerun-if-changed={}", top.join(part).display());
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
    println!("cargo:rustc-env=TURBO_BENCH_BUILD_COMMIT={commit}");
    println!("cargo:rustc-env=TURBO_BENCH_BUILD_CHANGES={}", changes.join(", "));
}
