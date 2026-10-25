//! Actual CLI refusals and records across Cargo profile configurations.
mod common;

use std::process::Command;

#[test]
fn cargo_build_settings_are_embedded_separately_from_assertions() {
    let profile = env!("TURBO_BENCH_CARGO_PROFILE");
    let level = env!("TURBO_BENCH_CARGO_OPT_LEVEL");
    assert!(matches!(profile, "debug" | "release"), "{profile}");
    assert!(matches!(level, "0" | "1" | "2" | "3" | "s" | "z"), "{level}");
}

#[test]
fn an_opt_level_zero_cli_refuses_before_any_measurement_or_output() {
    // This test is also run with assertions disabled and with a release
    // profile overridden to opt-level 0. The assertion cfg is irrelevant.
    if env!("TURBO_BENCH_CARGO_OPT_LEVEL") != "0" {
        return;
    }
    let root = common::scratch("throughput-opt-zero");
    for (mode, enabled) in [("grammar", cfg!(feature = "grammar")), ("lattice", cfg!(feature = "lattice"))] {
        let out = root.join(mode);
        let result = Command::new(env!("CARGO_BIN_EXE_turbo-bench"))
            .args([mode, "--rounds", "1", "--out"])
            .arg(&out)
            .output()
            .unwrap();
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(!result.status.success(), "{mode}: unexpectedly succeeded");
        if enabled {
            assert!(error.contains("opt-level=0"), "{mode}: {error}");
        } else {
            assert!(error.contains(&format!("without the {mode} feature")), "{error}");
        }
        assert!(!out.exists(), "{mode}: created output before refusal");
    }
}

#[test]
fn optimized_modes_reach_argument_validation_even_with_assertions_enabled() {
    if env!("TURBO_BENCH_CARGO_OPT_LEVEL") == "0" {
        return;
    }
    for (mode, enabled) in [("grammar", cfg!(feature = "grammar")), ("lattice", cfg!(feature = "lattice"))] {
        if !enabled {
            continue;
        }
        let result = Command::new(env!("CARGO_BIN_EXE_turbo-bench")).args([mode, "--rounds", "0"]).output().unwrap();
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(!result.status.success());
        assert!(error.contains("--rounds 0: nothing would be timed"), "{mode}: {error}");
    }
}

#[cfg(feature = "grammar")]
#[test]
fn an_optimized_clean_pushed_cli_records_real_settings_and_grammar_counts() {
    use turbo_bench::throughput::{CargoBuild, ThroughputRecord, check};

    if CargoBuild::current().opt_level == "0" {
        return;
    }
    let repo = common::workspace().canonicalize().unwrap();
    let ready = turbo_bench::git::provenance(&repo)
        .and_then(|p| turbo_bench::check_build(&p.commit, turbo_bench::BUILD_COMMIT, turbo_bench::BUILD_CHANGES));
    if let Err(why) = ready {
        assert_ne!(std::env::var("TURBO_BENCH_REQUIRE_THROUGHPUT").as_deref(), Ok("1"), "{why}");
        eprintln!("cannot record from this tree: {why}");
        return;
    }
    let root = common::scratch("throughput-optimized-grammar");
    let out = root.join("records");
    let result = Command::new(env!("CARGO_BIN_EXE_turbo-bench"))
        .args(["grammar", "--rounds", "1", "--repo"])
        .arg(&repo)
        .arg("--out")
        .arg(&out)
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let paths = std::fs::read_dir(&out).unwrap().map(|p| p.unwrap().path()).collect::<Vec<_>>();
    assert_eq!(paths.len(), 1);
    let r: ThroughputRecord = serde_json::from_slice(&std::fs::read(&paths[0]).unwrap()).unwrap();
    check(&r).unwrap();
    assert_eq!(r.library.commit, turbo_bench::BUILD_COMMIT);
    assert_eq!(r.cargo_build, Some(CargoBuild::current()));
    assert_eq!(r.command, "turbo-bench grammar --rounds 1 --repo <repo> --out <dir>");
    assert_eq!(r.text.bytes, 106407);
    assert_eq!(r.text.chars, 100282);
    assert_eq!(r.runs.iter().map(|r| r.units).collect::<Vec<_>>(), [11081, 31073, 32529, 45640, 1344]);
    assert!(r.runs.iter().all(|r| r.seconds > 0.0 && r.seconds.is_finite()));
}
