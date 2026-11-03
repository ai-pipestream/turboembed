//! The commit `turbo-eval` is built from, as TURBO_EVAL_BUILD_COMMIT (40
//! hex, or empty outside a git tree): `run` names a commit in a record
//! only when it is this one and the working tree is clean.
//!
//! What moves the commit is watched: HEAD, the branch it names and the
//! packed refs, each where `git rev-parse --git-path` puts it, so a
//! worktree works.

use std::path::Path;
use std::process::Command;

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git").arg("-C").arg(dir).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim_end().to_owned())
}

fn main() {
    for key in ["PROFILE", "OPT_LEVEL", "DEBUG"] {
        let value = std::env::var(key).expect("Cargo build setting");
        println!("cargo:rustc-env=TURBO_EVAL_{key}={value}");
    }
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let dir = Path::new(&manifest);
    let mut refs = vec!["HEAD".to_owned(), "packed-refs".to_owned()];
    if let Some(branch) = git(dir, &["symbolic-ref", "--quiet", "HEAD"]) {
        refs.push(branch);
    }
    for name in refs {
        let Some(p) = git(dir, &["rev-parse", "--path-format=absolute", "--git-path", &name]) else { continue };
        let p = Path::new(&p);
        if p.exists() {
            println!("cargo:rerun-if-changed={}", p.display());
        } else if name != "packed-refs"
            && let Some(parent) = p.parent().filter(|d| d.is_dir())
        {
            // A branch only in packed-refs: its next commit writes a file
            // in this directory.
            println!("cargo:rerun-if-changed={}", parent.display());
        }
    }
    let commit = git(dir, &["rev-parse", "--verify", "HEAD^{commit}"]).unwrap_or_default();
    println!("cargo:rustc-env=TURBO_EVAL_BUILD_COMMIT={commit}");
}
