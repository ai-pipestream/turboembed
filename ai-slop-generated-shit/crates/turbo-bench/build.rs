// The commit a benchmark receipt names comes from the tree the binary was
// built from, captured here so a receipt never carries the HEAD of whatever
// directory the tool happened to run in. A tree without git (an rsynced
// copy on a benchmark machine) leaves it unset and the run must be told
// the commit with --commit.
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let out = Command::new("git").arg("-C").arg(root).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
    if let Some(head) = git(&["rev-parse", "HEAD"]) {
        let dirty = git(&["status", "--porcelain", "--untracked-files=no"]).map(|s| !s.is_empty()).unwrap_or(true);
        println!("cargo:rustc-env=TURBO_BENCH_GIT_COMMIT={head}{}", if dirty { "-dirty" } else { "" });
    }
}
