//! Where the library came from: the commit a record names, and the
//! refusals that keep that name true (docs/benchmarks.md, "Provenance").

use std::path::Path;
use std::process::Command;

use crate::Result;

/// Records the tool writes, which may sit uncommitted in the tree between
/// runs: they are not what is measured.
pub const RECORDS_DIR: &str = "benchmarks/records/";

#[derive(Debug, Clone, PartialEq)]
pub struct Provenance {
    /// The working tree's top directory.
    pub top: String,
    pub commit: String,
    /// The branches of origin that contain the commit, `origin/<name>`.
    pub pushed_to: Vec<String>,
}

/// `git -C <dir> <args>`, its standard output trimmed, or an error with
/// the command and git's own message.
pub fn git(dir: &Path, args: &[&str]) -> Result<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .map_err(|e| format!("git {}: {e}", args.join(" ")))?;
    if !out.status.success() {
        return Err(format!(
            "git -C {} {}: {}",
            dir.display(),
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim_end().to_owned())
}

/// A `git status --porcelain=v1` line for an untracked file under
/// benchmarks/records/: a record made and not yet committed. A tracked
/// record changed or deleted is a change like any other.
pub fn is_new_record(line: &str) -> bool {
    line.strip_prefix("?? ").is_some_and(|p| p.starts_with(RECORDS_DIR))
}

/// The commit checked out in `dir`'s working tree, when the tree is
/// clean and the commit is on a branch of origin. Refused:
///
/// - a tree with any change, staged or not, or any untracked file that
///   is not ignored, except new files under benchmarks/records/ (`git
///   status --porcelain --untracked-files=all`);
/// - no remote named origin (`git remote get-url origin`);
/// - a commit no remote-tracking branch of origin contains (`git
///   for-each-ref --contains HEAD refs/remotes/origin/`). Those refs are
///   what the last push or fetch left; the tool does not reach the remote.
pub fn provenance(dir: &Path) -> Result<Provenance> {
    let top = git(dir, &["rev-parse", "--show-toplevel"])
        .map_err(|e| format!("{}: not a git working tree: {e}", dir.display()))?;
    let top_path = Path::new(&top);
    let status = git(top_path, &["status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"])?;
    let changed: Vec<&str> = status.lines().filter(|l| !is_new_record(l)).collect();
    if !changed.is_empty() {
        let shown: Vec<&str> = changed.iter().take(5).copied().collect();
        return Err(format!(
            "{top}: the working tree is not clean ({} path{}: {}); a record names a commit, and this tree is \
             not that commit: commit or stash the changes first",
            changed.len(),
            if changed.len() == 1 { "" } else { "s" },
            shown.join(", ")
        ));
    }
    let commit = git(top_path, &["rev-parse", "--verify", "HEAD^{commit}"])?;
    git(top_path, &["remote", "get-url", "origin"])
        .map_err(|e| format!("{top}: no remote named origin, so no commit here is pushed: {e}"))?;
    let refs =
        git(top_path, &["for-each-ref", "--contains", &commit, "--format=%(refname:short)", "refs/remotes/origin/"])?;
    let pushed_to: Vec<String> =
        refs.lines().filter(|r| !r.is_empty() && *r != "origin/HEAD" && *r != "origin").map(str::to_owned).collect();
    if pushed_to.is_empty() {
        return Err(format!(
            "{top}: commit {commit} is on no branch of origin as this clone last saw it (refs/remotes/origin/*): \
             push it, or fetch if it was pushed from elsewhere"
        ));
    }
    Ok(Provenance { top, commit, pushed_to })
}
