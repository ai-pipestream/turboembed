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
    assert_eq!(r.runs.iter().map(|r| r.units).collect::<Vec<_>>(), GRAMMAR_UNITS);
    assert!(r.runs.iter().all(|r| r.seconds > 0.0 && r.seconds.is_finite()));
}

/// What `turbo-bench grammar` counts over the goldens number inputs:
/// find, lex, parse, parse-units and times. The parse-units count went
/// from 45619 to 45640 with #1234 (rates other than speed): each of the
/// seven repeats of the text now also charts an information rate in MB
/// per second and a rupee rate per month, as unit, quantity and rate
/// nodes. #1233 pinned 45619 on a base from before #1234.
#[cfg(feature = "grammar")]
const GRAMMAR_UNITS: [u64; 5] = [11081, 31073, 32529, 45640, 1344];

/// The same counts from the grammar itself, on any tree, so a grammar
/// change that moves them fails in its own pull request. The CLI test
/// above only runs on a clean pushed tree.
#[cfg(feature = "grammar")]
#[test]
fn the_grammar_counts_hold_on_any_tree() {
    use turbo_analysis::{Layer, ViewId};
    use turbo_analysis_layers::DateTime;
    use turbo_grammar::engine::Lexed;
    use turbo_grammar::{ENGLISH, ENGLISH_UNITS, Grammar, GrammarAnnotator, Parser, TimeAnnotator};

    let mut one = String::new();
    for set in ["num", "quantity", "money", "temporal"] {
        let dir = common::workspace().join("analysis/goldens/data/inputs").join(set);
        let mut names = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect::<Vec<_>>();
        names.retain(|p| p.extension().is_some_and(|x| x == "txt"));
        names.sort();
        for p in &names {
            one.push_str(&std::fs::read_to_string(p).unwrap());
            one.push('\n');
        }
    }
    let mut text = String::new();
    while text.len() < 100 * 1024 {
        text.push_str(&one);
    }
    assert_eq!((text.len(), text.chars().count()), (106407, 100282));

    let found = GrammarAnnotator::english().find(&text, ViewId::SOURCE);
    let find =
        found.numbers.len() + found.percents.len() + found.ordinals.len() + found.quantities.len() + found.money.len();
    let mut lexed = Lexed::default();
    lexed.lex(&text, &ENGLISH.style());
    let mut parser = Parser::new(&ENGLISH);
    parser.parse(&text);
    let mut units = Parser::new(&ENGLISH_UNITS);
    units.parse(&text);
    let times = TimeAnnotator::english().find(&text, ViewId::SOURCE, DateTime::parse("2016-11-07T00:00:00"));
    let counts = [find, lexed.len(), parser.chart().nodes.len(), units.chart().nodes.len(), times.len()];
    assert_eq!(counts.map(|n| n as u64), GRAMMAR_UNITS);
}
