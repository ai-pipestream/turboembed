//! Where a record's pipeline came from: the commit it names and the
//! refusals that keep that name true, as turbo-bench refuses
//! (docs/benchmarks.md, "Provenance"), and the time it was recorded.

use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::Result;
use crate::pipeline::Source;

/// Records `turbo-eval run` writes, which may sit uncommitted in the tree
/// between runs: they are not what is measured.
pub const RECORDS_DIR: &str = "analysis/eval/records/";

/// The commit `turbo-eval` was built from (build.rs); empty outside a git
/// tree.
pub const BUILD_COMMIT: &str = env!("TURBO_EVAL_BUILD_COMMIT");

/// `git -C <dir> <args>`, its standard output trimmed.
fn git(dir: &Path, args: &[&str]) -> Result<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .map_err(|e| format!("git {}: {e}", args.join(" ")))?;
    if !out.status.success() {
        return Err(format!("git {}: {}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim_end().to_owned())
}

/// A `git status --porcelain=v1` line for a new file under RECORDS_DIR.
fn is_new_record(line: &str) -> bool {
    line.strip_prefix("?? ").is_some_and(|p| p.starts_with(RECORDS_DIR))
}

/// The commit checked out in `dir`'s working tree. Refused: a tree with
/// any change or untracked file except new records; a commit no
/// remote-tracking branch of origin contains; and a commit other than
/// the one this tool was built from.
pub fn source(dir: &Path) -> Result<Source> {
    let top = git(dir, &["rev-parse", "--show-toplevel"])?;
    let top = Path::new(&top);
    let status = git(top, &["status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"])?;
    let changed: Vec<&str> = status.lines().filter(|l| !is_new_record(l)).collect();
    if !changed.is_empty() {
        return Err(format!(
            "the working tree is not clean ({}): a record names a commit, and this tree is not that commit",
            changed.iter().take(5).copied().collect::<Vec<_>>().join(", ")
        ));
    }
    let commit = git(top, &["rev-parse", "--verify", "HEAD^{commit}"])?;
    if commit != BUILD_COMMIT {
        return Err(format!(
            "this tool was built from {:?} and the tree is at {commit}: build and run it from that tree \
             (cargo run --release -p turbo-eval)",
            BUILD_COMMIT
        ));
    }
    let refs = git(top, &["for-each-ref", "--contains", &commit, "--format=%(refname:short)", "refs/remotes/origin/"])?;
    let pushed_to: Vec<String> =
        refs.lines().filter(|r| !r.is_empty() && *r != "origin/HEAD" && *r != "origin").map(str::to_owned).collect();
    if pushed_to.is_empty() {
        return Err(format!("commit {commit} is on no branch of origin as this clone last saw it: push it first"));
    }
    Ok(Source { commit, pushed_to })
}

/// A time as `YYYY-MM-DDTHH:MM:SSZ`, UTC (turbo-bench's `utc`).
pub fn utc(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (days, rem) = ((secs / 86400) as i64, secs % 86400);
    // Days since 1970-01-01 to a civil date (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn utc_by_hand() {
        assert_eq!(utc(UNIX_EPOCH), "1970-01-01T00:00:00Z");
        // 2000-03-01 is day 11017; plus 1h 2m 3s.
        assert_eq!(utc(UNIX_EPOCH + Duration::from_secs(11017 * 86400 + 3723)), "2000-03-01T01:02:03Z");
    }

    #[test]
    fn new_records_are_not_changes() {
        assert!(is_new_record("?? analysis/eval/records/scifact.json"));
        assert!(!is_new_record(" M analysis/eval/records/scifact.json"));
        assert!(!is_new_record("?? analysis/eval/src/x.rs"));
    }
}
