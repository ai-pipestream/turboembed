//! What the tool's tests share: the small sealed bundle, git working trees
//! made in temporary directories, and one real measurement of the CPU
//! backend on the bundle.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use turbo::TURBO_PRECISION_MODEL;
use turbo::record::{self, Measured, Record, ReferenceRun};
use turbo_bench::git::Provenance;
use turbo_bench::measure::{Measurement, Plan, measure};

pub fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

pub fn tiny_bundle() -> PathBuf {
    workspace().join("testdata/tiny-bert-bundle")
}

/// A copy of the small bundle at `dir`, outside testdata/, where the tool
/// takes it as it would any bundle.
pub fn bundle_copy(dir: &Path) -> PathBuf {
    fn copy(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for e in std::fs::read_dir(from).unwrap() {
            let e = e.unwrap();
            if e.file_type().unwrap().is_dir() {
                copy(&e.path(), &to.join(e.file_name()));
            } else {
                std::fs::copy(e.path(), to.join(e.file_name())).unwrap();
            }
        }
    }
    copy(&tiny_bundle(), dir);
    dir.to_owned()
}

/// A fresh, empty directory, removed with everything in it when this goes.
pub struct Scratch(PathBuf);

impl std::ops::Deref for Scratch {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A fresh, empty directory for `name`.
pub fn scratch(name: &str) -> Scratch {
    let d = std::env::temp_dir().join(format!("turbo-bench-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    Scratch(d)
}

/// git in `dir`, with an identity and no signing of its own, so the
/// user's configuration changes nothing; its output, or a panic.
pub fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "user.name=turbo-bench test", "-c", "user.email=test@invalid", "-c", "commit.gpgsign=false"])
        .args(args)
        .output()
        .unwrap();
    assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().to_owned()
}

/// A working tree under `root` with one commit on main, pushed to a bare
/// repository that is its origin. Returns the working tree.
pub fn pushed_repo(root: &Path) -> PathBuf {
    let origin = root.join("origin.git");
    let work = root.join("work");
    std::fs::create_dir_all(&origin).unwrap();
    std::fs::create_dir_all(&work).unwrap();
    git(&origin, &["init", "--quiet", "--bare", "--initial-branch=main"]);
    git(&work, &["init", "--quiet", "--initial-branch=main"]);
    std::fs::write(work.join("README"), "a library\n").unwrap();
    git(&work, &["add", "README"]);
    git(&work, &["commit", "--quiet", "-m", "first"]);
    git(&work, &["remote", "add", "origin", origin.to_str().unwrap()]);
    git(&work, &["push", "--quiet", "origin", "main"]);
    work
}

/// The reference programs' names, as records give them.
pub const TEI: &str = turbo_bench::tei::NAME;
pub const TRT: &str = turbo_bench::tensorrt::NAME;

/// The CPU backend on the small bundle, measured once for every test
/// here.
pub fn cpu_measurement() -> &'static Measurement {
    static M: OnceLock<Measurement> = OnceLock::new();
    M.get_or_init(|| {
        let plan = Plan {
            bundle: tiny_bundle(),
            device: "cpu".into(),
            precision: TURBO_PRECISION_MODEL,
            batch: None,
            seq: None,
            warmup: 3,
            iterations: 30,
        };
        measure(&plan).unwrap_or_else(|e| panic!("{e}"))
    })
}

/// The provenance of a pushed repository made for the test.
pub fn provenance(name: &str) -> Provenance {
    let root = scratch(name);
    turbo_bench::git::provenance(&pushed_repo(&root)).unwrap()
}

/// A record of the real CPU measurement with `references`.
pub fn cpu_record(name: &str, references: Vec<ReferenceRun>) -> Record {
    turbo_bench::record(cpu_measurement(), &provenance(name), references, "2026-01-02T03:04:05Z".into())
        .unwrap_or_else(|e| panic!("{e}"))
}

/// A reference entry for the SUPPORTED rule's tests. No reference program
/// runs here, so its figures are derived from the real CPU measurement
/// (twice its latency, half its rate) and it names an image nobody
/// pulls; it is never written where the core reads records.
pub fn measured_reference(name: &str) -> ReferenceRun {
    let m = cpu_measurement();
    let role = record::REFERENCES.iter().find(|r| r.0 == name).unwrap_or_else(|| panic!("{name}: no such reference")).1;
    ReferenceRun {
        name: name.into(),
        role: role.into(),
        pinned: format!("example.invalid/{name}@sha256:{}", "0".repeat(64)),
        version: "0.0.0".into(),
        commands: vec![vec!["turbo-bench".into(), "record".into()]],
        procedure: "figures derived from the library's own measurement, for the rule's tests".into(),
        measured: Some(Measured {
            iterations: m.timing.iterations as u64,
            p50_ms: m.timing.p50_ms * 2.0,
            p99_ms: m.timing.p99_ms * 2.0,
            rows_per_second: m.timing.rows_per_second / 2.0,
            min_cosine: Some(m.conformance.min_cosine),
        }),
        not_run: None,
    }
}
