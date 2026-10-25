//! Execute the actual build script against real Git states, so build-time
//! and runtime record exemptions cannot disagree unnoticed.
mod common;

use std::path::Path;
use std::process::Command;

use common::*;
use turbo_bench::git::provenance;

fn build_changes(script: &Path, work: &Path) -> String {
    let out = Command::new(script).env("CARGO_MANIFEST_DIR", work.join("bench")).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix("cargo:rustc-env=TURBO_BENCH_BUILD_CHANGES=").map(str::to_owned))
        .unwrap()
}

#[test]
fn rebuilding_with_new_records_preserves_both_provenance_gates() {
    let root = scratch("build-record-exemptions");
    let work = pushed_repo(&root);
    std::fs::create_dir_all(work.join("bench")).unwrap();
    let script = root.join(format!("build-provenance{}", std::env::consts::EXE_SUFFIX));
    let out = Command::new("rustc")
        .arg("--edition=2024")
        .arg(workspace().join("bench/build.rs"))
        .arg("-o")
        .arg(&script)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(build_changes(&script, &work).is_empty());
    for dir in ["benchmarks/records", "benchmarks/static"] {
        std::fs::create_dir_all(work.join(dir)).unwrap();
        std::fs::write(work.join(dir).join("a.json"), "{}\n").unwrap();
        assert!(provenance(&work).is_ok());
        assert!(build_changes(&script, &work).is_empty(), "new {dir} is not a source change");
    }
    git(&work, &["add", "benchmarks"]);
    assert!(provenance(&work).is_err());
    assert!(build_changes(&script, &work).contains("A  benchmarks/static/a.json"));
    git(&work, &["commit", "--quiet", "-m", "records"]);
    git(&work, &["push", "--quiet", "origin", "main"]);
    assert!(build_changes(&script, &work).is_empty());
    std::fs::write(work.join("benchmarks/static/a.json"), "{\"changed\":true}\n").unwrap();
    assert!(provenance(&work).is_err());
    assert!(build_changes(&script, &work).contains(" M benchmarks/static/a.json"));
    git(&work, &["add", "benchmarks/static/a.json"]);
    assert!(build_changes(&script, &work).contains("M  benchmarks/static/a.json"));
    std::fs::remove_file(work.join("benchmarks/static/a.json")).unwrap();
    assert!(provenance(&work).is_err());
    assert!(!build_changes(&script, &work).is_empty());
    git(&work, &["add", "benchmarks/static/a.json"]);
    assert!(build_changes(&script, &work).contains("D  benchmarks/static/a.json"));
    git(&work, &["commit", "--quiet", "-m", "remove record"]);
    git(&work, &["push", "--quiet", "origin", "main"]);
    std::fs::write(work.join("benchmarks/qa.json"), "{}\n").unwrap();
    assert!(provenance(&work).is_err());
    assert!(build_changes(&script, &work).contains("?? benchmarks/qa.json"));
}
