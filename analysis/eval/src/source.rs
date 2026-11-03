//! Where a record's pipeline came from: the commit it names and the
//! refusals that keep that name true, as turbo-bench refuses
//! (docs/benchmarks.md, "Provenance"), and the time it was recorded.

use std::path::{Path, PathBuf};
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

/// The top of `dir`'s working tree and its `HEAD` commit, when the tree has
/// no change or untracked file except new records.
fn clean_commit(dir: &Path) -> Result<(PathBuf, String)> {
    let top = PathBuf::from(git(dir, &["rev-parse", "--show-toplevel"])?);
    let status = git(&top, &["status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"])?;
    let changed: Vec<&str> = status.lines().filter(|l| !is_new_record(l)).collect();
    if !changed.is_empty() {
        return Err(format!(
            "the working tree is not clean ({}): a record names a commit, and this tree is not that commit",
            changed.iter().take(5).copied().collect::<Vec<_>>().join(", ")
        ));
    }
    let commit = git(&top, &["rev-parse", "--verify", "HEAD^{commit}"])?;
    Ok((top, commit))
}

/// The commit checked out in `dir`'s working tree. Refused: a tree with
/// any change or untracked file except new records; a commit no
/// remote-tracking branch of origin contains; and a commit other than
/// the one this tool was built from.
pub fn source(dir: &Path) -> Result<Source> {
    source_for(dir, BUILD_COMMIT)
}

fn source_for(dir: &Path, build_commit: &str) -> Result<Source> {
    let (top, commit) = clean_commit(dir)?;
    if commit != build_commit {
        return Err(format!(
            "this tool was built from {:?} and the tree is at {commit}: build and run it from that tree \
             (cargo run --release -p turbo-eval)",
            build_commit
        ));
    }
    let refs =
        git(&top, &["for-each-ref", "--contains", &commit, "--format=%(refname:short)", "refs/remotes/origin/"])?;
    let pushed_to: Vec<String> =
        refs.lines().filter(|r| !r.is_empty() && *r != "origin/HEAD" && *r != "origin").map(str::to_owned).collect();
    if pushed_to.is_empty() {
        return Err(format!("commit {commit} is on no branch of origin as this clone last saw it: push it first"));
    }
    Ok(Source { commit, pushed_to })
}

/// Refuses when `after` names a commit other than `started`. Changes to
/// `pushed_to` (branches of `origin` fetched or pruned during a run) do
/// not move the working tree.
pub fn same_commit(started: &Source, after: &Source) -> Result<()> {
    if after.commit != started.commit {
        return Err("the working tree moved during the run; the records written name the commit it started at".into());
    }
    Ok(())
}

/// Checks after a run that `dir`'s working tree is still clean (except new
/// records) and still at `started.commit`, keeping `started.pushed_to` as
/// the provenance already written to the records.
pub fn check_unchanged(dir: &Path, started: &Source) -> Result<()> {
    let (_, commit) = clean_commit(dir)?;
    same_commit(started, &Source { commit, pushed_to: Vec::new() })
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
    use crate::analysis::{Analysis, Sentencer};
    use crate::chunkers::SentenceWindows;
    use crate::embedder::stub::Stub;
    use crate::pipeline::{Pipeline, Titles};
    use crate::record::EvalRecord;
    use crate::run::{record_path, run_sets};
    use crate::set::LoadedSet;
    use std::fs;
    use std::time::Duration;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("turbo-eval-source-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn git_ok(dir: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .env("GIT_AUTHOR_NAME", "test")
            .env("GIT_AUTHOR_EMAIL", "test@example.invalid")
            .env("GIT_COMMITTER_NAME", "test")
            .env("GIT_COMMITTER_EMAIL", "test@example.invalid")
            .output()
            .unwrap();
        assert!(out.status.success(), "git {:?}: {}", args, String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap().trim_end().to_owned()
    }

    fn pushed_clone(root: &Path) -> PathBuf {
        let origin = root.join("origin.git");
        let work = root.join("work");
        fs::create_dir_all(&origin).unwrap();
        fs::create_dir_all(&work).unwrap();
        git_ok(&origin, &["init", "--bare", "--quiet", "--initial-branch=main"]);
        git_ok(&work, &["init", "--quiet", "--initial-branch=main"]);
        git_ok(&work, &["remote", "add", "origin", origin.to_str().unwrap()]);
        fs::write(work.join("README"), "first\n").unwrap();
        git_ok(&work, &["add", "README"]);
        git_ok(&work, &["commit", "--quiet", "-m", "first"]);
        git_ok(&work, &["push", "--quiet", "-u", "origin", "main"]);
        git_ok(&work, &["push", "--quiet", "origin", "HEAD:refs/heads/mq-temp"]);
        work
    }

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

    #[test]
    fn changed_pushed_to_with_same_commit_does_not_fail_and_changed_commit_does() {
        let started = Source {
            commit: "0123456789abcdef0123456789abcdef01234567".into(),
            pushed_to: vec!["origin/main".into(), "origin/mq-temp".into()],
        };
        let refs_changed =
            Source { commit: started.commit.clone(), pushed_to: vec!["origin/main".into(), "origin/release".into()] };
        assert_ne!(started, refs_changed);
        same_commit(&started, &refs_changed).unwrap();

        let commit_changed =
            Source { commit: "fedcba9876543210fedcba9876543210fedcba98".into(), pushed_to: started.pushed_to.clone() };
        let err = same_commit(&started, &commit_changed).unwrap_err();
        assert_eq!(err, "the working tree moved during the run; the records written name the commit it started at");
    }

    fn load_while_origin_refs_change(_: &str, work: &Path) -> Result<LoadedSet> {
        git_ok(work, &["update-ref", "-d", "refs/remotes/origin/mq-temp"]);
        git_ok(work, &["push", "--quiet", "origin", "HEAD:refs/heads/release"]);
        Ok(crate::pipeline::tests::set(None))
    }

    #[test]
    fn run_in_clone_whose_remote_refs_change_midway_keeps_written_records() {
        let root = scratch("remote-refs-midway");
        let work = pushed_clone(&root);
        let head = git_ok(&work, &["rev-parse", "HEAD"]);
        let started = source_for(&work, &head).unwrap();
        assert_eq!(started.pushed_to, vec!["origin/main".to_owned(), "origin/mq-temp".to_owned()]);

        let mut pipeline = Pipeline {
            analysis: Analysis { sentences: Sentencer::Newline, structure: None },
            chunker: Box::new(SentenceWindows::new(1).unwrap()),
            embedder: Box::new(Stub::new(&["cat", "dog", "rust", "tea"])),
            titles: Titles::Ignore,
            lexical: None,
            hybrid: None,
            cascade: None,
            static_embedder: None,
            dump_cuts: None,
        };
        let out = work.join("analysis/eval/records/run-1");
        let written =
            run_sets(&mut pipeline, &["toy".into()], &work, load_while_origin_refs_change, &out, &started).unwrap();
        assert_eq!(written, [record_path(&out, "toy")]);

        // Remote refs changed during the run, so a fresh `source_for` has a
        // different `pushed_to`, while `check_unchanged` accepts the clean tree
        // at the same commit and the written record keeps start-of-run `pushed_to`.
        let after = source_for(&work, &head).unwrap();
        assert_eq!(after.pushed_to, vec!["origin/main".to_owned(), "origin/release".to_owned()]);
        check_unchanged(&work, &started).unwrap();

        let record = EvalRecord::read(&written[0]).unwrap();
        assert_eq!(record.provenance.commit, head);
        assert_eq!(record.provenance.pushed_to, started.pushed_to);

        // Dirtying a tracked file or moving HEAD during a run still fails.
        fs::write(work.join("README"), "second\n").unwrap();
        assert!(check_unchanged(&work, &started).unwrap_err().contains("the working tree is not clean"));
        git_ok(&work, &["commit", "--quiet", "-am", "second"]);
        assert_eq!(
            check_unchanged(&work, &started).unwrap_err(),
            "the working tree moved during the run; the records written name the commit it started at"
        );
        fs::remove_dir_all(&root).unwrap();
    }
}
