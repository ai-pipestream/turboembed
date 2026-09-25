//! The commit a record names: a clean tree, on a branch pushed to origin,
//! checked with git in repositories made for each test.

mod common;

use common::*;
use turbo_bench::git::provenance;

#[test]
fn a_clean_pushed_commit_is_named_with_its_branch() {
    let root = scratch("clean");
    let work = pushed_repo(&root);
    let p = provenance(&work).unwrap();
    assert_eq!(p.commit, git(&work, &["rev-parse", "HEAD"]));
    assert_eq!(p.pushed_to, vec!["origin/main".to_owned()]);
    // From a directory inside the tree too.
    std::fs::create_dir_all(work.join("sub")).unwrap();
    assert_eq!(provenance(&work.join("sub")).unwrap(), p);
}

#[test]
fn a_changed_tracked_file_is_refused() {
    let root = scratch("modified");
    let work = pushed_repo(&root);
    std::fs::write(work.join("README"), "changed\n").unwrap();
    let e = provenance(&work).unwrap_err();
    assert!(e.contains("the working tree is not clean (1 path:  M README)"), "{e}");
}

#[test]
fn a_staged_change_is_refused() {
    let root = scratch("staged");
    let work = pushed_repo(&root);
    std::fs::write(work.join("README"), "changed\n").unwrap();
    git(&work, &["add", "README"]);
    let e = provenance(&work).unwrap_err();
    assert!(e.contains("not clean") && e.contains("M  README"), "{e}");
}

#[test]
fn an_untracked_file_is_refused_unless_it_is_a_record_or_ignored() {
    let root = scratch("untracked");
    let work = pushed_repo(&root);
    std::fs::create_dir_all(work.join("benchmarks/records")).unwrap();
    std::fs::write(work.join("benchmarks/records/a.json"), "{}").unwrap();
    provenance(&work).unwrap();
    std::fs::write(work.join(".git/info/exclude"), "build/\n").unwrap();
    std::fs::create_dir_all(work.join("build")).unwrap();
    std::fs::write(work.join("build/out.o"), "").unwrap();
    provenance(&work).unwrap();
    std::fs::create_dir_all(work.join("src")).unwrap();
    std::fs::write(work.join("src/new.rs"), "").unwrap();
    let e = provenance(&work).unwrap_err();
    assert!(e.contains("not clean (1 path: ?? src/new.rs)"), "{e}");
}

#[test]
fn a_commit_on_no_branch_of_origin_is_refused() {
    let root = scratch("unpushed");
    let work = pushed_repo(&root);
    std::fs::write(work.join("README"), "second\n").unwrap();
    git(&work, &["commit", "--quiet", "-am", "second"]);
    let head = git(&work, &["rev-parse", "HEAD"]);
    let e = provenance(&work).unwrap_err();
    assert!(e.contains(&format!("commit {head} is on no branch of origin")), "{e}");
    // Pushed to another branch of origin, it is named with that branch.
    git(&work, &["push", "--quiet", "origin", "HEAD:refs/heads/topic"]);
    assert_eq!(provenance(&work).unwrap().pushed_to, vec!["origin/topic".to_owned()]);
}

#[test]
fn a_tree_without_origin_or_outside_git_is_refused() {
    let root = scratch("no-origin");
    let work = pushed_repo(&root);
    git(&work, &["remote", "remove", "origin"]);
    let e = provenance(&work).unwrap_err();
    assert!(e.contains("no remote named origin"), "{e}");
    let bare = scratch("not-git");
    let e = provenance(&bare).unwrap_err();
    assert!(e.contains("not a git working tree"), "{e}");
}
