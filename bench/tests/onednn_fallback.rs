//! #2263: a `levelzero-onednn` build of the tool says on stderr, in one
//! line, when oneDNN cannot open and the backend's own kernels run the
//! linear layers, on each command that measures a levelzero session:
//! `record`, `ranges`, `qa` and `rerank`.
//!
//! These run the built `turbo-bench` on the device. The library opens
//! `libturbo_onednn.so` where the build wrote it, by a path compiled in
//! (core/src/levelzero/onednn.rs), so taking the build directory off the
//! library path does not stop it; the test moves the built file aside for
//! the run and puts it back, as a machine that lost the file would have it.
//!
//! Built with the `levelzero-onednn` feature only.
//!
//! SKIPS, and passes, on any checkout that is not a clean pushed commit,
//! the only tree the tool measures from: a green run there has checked
//! nothing. TURBO_BENCH_REQUIRE_RECORD=1 makes that a failure instead.
//! Where the backend lists no device it fails, unless
//! TURBO_TEST_REQUIRE_LEVELZERO=0 lets it skip.

#![cfg(feature = "levelzero-onednn")]

mod common;
#[path = "common/heads.rs"]
mod heads;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::*;

const LIBRARY: &str = "libturbo_onednn.so";
const ASIDE: &str = "libturbo_onednn.so.aside";

/// Every file named `name` under `dir`.
fn find(dir: &Path, name: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.filter_map(Result::ok) {
        let path = e.path();
        if e.file_type().is_ok_and(|t| t.is_dir()) {
            find(&path, name, out);
        } else if e.file_name() == name {
            out.push(path);
        }
    }
}

/// The built oneDNN libraries of this profile, moved aside until dropped.
struct Aside(Vec<PathBuf>);

impl Aside {
    fn take(build: &Path) -> Self {
        let mut found = Vec::new();
        find(build, LIBRARY, &mut found);
        assert!(!found.is_empty(), "no {LIBRARY} under {}", build.display());
        for lib in &found {
            std::fs::rename(lib, lib.with_file_name(ASIDE)).unwrap();
        }
        Self(found)
    }
}

impl Drop for Aside {
    fn drop(&mut self) {
        for lib in &self.0 {
            let _ = std::fs::rename(lib.with_file_name(ASIDE), lib);
        }
    }
}

/// Put back a library a killed run of this test left aside.
fn restore_leftovers(build: &Path) {
    let mut left = Vec::new();
    find(build, ASIDE, &mut left);
    for aside in left {
        let lib = aside.with_file_name(LIBRARY);
        if !lib.exists() {
            std::fs::rename(&aside, &lib).unwrap();
        }
    }
}

/// The tool's stderr lines that say oneDNN was not used, after a run that
/// must succeed. The library path keeps no entry under `target`, so no
/// copy of the library is found there by name either.
fn fallback_lines(target: &Path, args: &[&str], choices: Option<&str>) -> Vec<String> {
    let kept: Vec<String> = std::env::var("LD_LIBRARY_PATH")
        .unwrap_or_default()
        .split(':')
        .filter(|entry| !entry.is_empty() && !Path::new(entry).starts_with(target))
        .map(str::to_owned)
        .collect();
    let mut c = Command::new(env!("CARGO_BIN_EXE_turbo-bench"));
    c.args(args).env("LD_LIBRARY_PATH", kept.join(":")).env_remove("TURBO_LEVELZERO_CHOICES");
    if let Some(choices) = choices {
        c.env("TURBO_LEVELZERO_CHOICES", choices);
    }
    let out = c.output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "turbo-bench {args:?}: {err}");
    err.lines().filter(|l| l.contains("oneDNN not used")).map(str::to_owned).collect()
}

#[test]
fn each_measuring_command_says_once_when_onednn_cannot_open() {
    let test = "each_measuring_command_says_once_when_onednn_cannot_open";
    if turbo_bench::api::Runtime::create().unwrap().find("levelzero").is_err() {
        assert!(
            turbo_pins::device_skip_allowed("levelzero"),
            "{test}: the levelzero backend lists no device; TURBO_TEST_REQUIRE_LEVELZERO=0 lets the test skip"
        );
        println!("{test}: skipped: the levelzero-onednn feature is on and the levelzero backend lists no device");
        return;
    }
    let unrecordable = turbo_bench::git::provenance(&workspace())
        .and_then(|p| turbo_bench::check_build(&p.commit, turbo_bench::BUILD_COMMIT, turbo_bench::BUILD_CHANGES));
    if let Err(why) = unrecordable {
        let required = std::env::var("TURBO_BENCH_REQUIRE_RECORD").as_deref() == Ok("1");
        assert!(!required, "{test}: TURBO_BENCH_REQUIRE_RECORD=1, and this checkout cannot be recorded from: {why}");
        println!("{test}: skipped: this checkout cannot be recorded from: {why}");
        return;
    }

    // target/<profile>/turbo-bench: the build scripts write under
    // target/<profile>/build.
    let bin = Path::new(env!("CARGO_BIN_EXE_turbo-bench"));
    let profile = bin.parent().expect("the binary is under target/<profile>");
    let target = profile.parent().expect("the binary is under target/<profile>");
    let build = profile.join("build");
    restore_leftovers(&build);

    let root = scratch("onednn-fallback");
    let repo = workspace().canonicalize().unwrap();
    let repo = repo.to_str().unwrap();
    let embed = bundle_copy(&root.join("embed"));
    let qa = heads::tiny_qa_bundle(&root.join("qa"));
    let rerank = heads::tiny_rerank_bundle(&root.join("rerank"));
    let questions = root.join("squad.json");
    let squad = serde_json::json!({"data": [{"paragraphs": [
        {"context": "Paris is in France.", "qas": [{"question": "Where is Paris?"}]},
        {"context": "Berlin is in Germany.", "qas": [{"question": "Where is Berlin?"}]}
    ]}]});
    std::fs::write(&questions, serde_json::to_vec(&squad).unwrap()).unwrap();
    let (embed, qa, rerank) = (embed.to_str().unwrap(), qa.to_str().unwrap(), rerank.to_str().unwrap());
    let questions = questions.to_str().unwrap();
    let out = |name: &str| root.join(name).to_str().unwrap().to_owned();

    let common = ["--device", "levelzero", "--repo", repo, "--warmup", "1", "--iterations", "2"];
    let commands: [(&str, Vec<&str>); 4] = [
        // `record` wants each reference program for the backend named or turned off: TEI and OpenVINO here.
        ("record", vec!["record", "--bundle", embed, "--batch", "12", "--no-tei", "--no-openvino"]),
        ("ranges", vec!["ranges", "--bundle", embed, "--batch", "2", "--seq", "32", "--chunk", "8"]),
        ("qa", vec!["qa", "--bundle", qa, "--questions", questions, "--batch", "1", "--per-bin", "1"]),
        (
            "rerank",
            vec![
                "rerank",
                "--bundle",
                rerank,
                "--questions",
                questions,
                "--requests",
                "1",
                "--passages",
                "2",
                "--no-tei",
            ],
        ),
    ];
    let run = |name: &str, args: &[&str], round: &str, choices: Option<&str>| {
        let dir = out(&format!("{name}-{round}"));
        let mut all = args.to_vec();
        all.extend(common);
        all.extend(["--out", &dir]);
        fallback_lines(target, &all, choices)
    };

    for (name, args) in &commands {
        // The library where the build wrote it: oneDNN opens, nothing is said.
        assert_eq!(run(name, args, "present", None), [""; 0], "{name}: oneDNN opens, so there is no line");

        let aside = Aside::take(&build);
        let lines = run(name, args, "aside", None);
        let forced = run(name, args, "forced", Some("linear=own"));
        drop(aside);

        // The library gone: one line, with the reason the library logged.
        assert_eq!(lines.len(), 1, "{name}: {lines:?}");
        let line = &lines[0];
        assert!(line.starts_with("turbo-bench: levelzero device "), "{name}: {line}");
        assert!(line.contains("the backend's own kernels run the linear layers"), "{name}: {line}");
        assert!(line.contains(LIBRARY), "{name}: the line names what could not open: {line}");
        // The own kernels asked for by name: nothing to say.
        assert_eq!(forced, [""; 0], "{name}: linear=own was forced, so there is no line");
    }
}
