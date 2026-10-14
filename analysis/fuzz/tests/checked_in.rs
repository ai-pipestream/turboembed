//! The differential over the checked in goldens: every golden under
//! `analysis/goldens/data` the differential covers, compared with the Rust
//! side as `turbo-goldens fuzz` compares the goldens of a generated
//! corpus. The crates' own golden tests pass on this tree, so the walk
//! must find no divergence and no broken invariant: what it finds instead
//! is the documented differences, each classified, and the goldens it
//! cannot compare. Runs whose model is not downloaded are skipped unless
//! `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`; the published MeCab dictionaries
//! are left out unless `TURBO_TEST_MECAB_DICTIONARIES=1`, as the lattice
//! crate's own tests on them are ignored by default.

use turbo_fuzz::{Outcome, Resources, Unavailable, compare_tree};

#[test]
fn the_checked_in_goldens_show_no_divergence() {
    let res = Resources::from_env();
    let data = turbo_goldens::data_dir();
    // The time each run takes, so a slow one shows in the test output.
    let started = std::time::Instant::now();
    let mut last: Option<(String, std::time::Instant)> = None;
    let progress = |_: usize, _: usize, path: &std::path::Path| {
        let run = path.iter().rev().nth(2).map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
        match &last {
            Some((r, _)) if *r == run => {}
            _ => {
                if let Some((r, at)) = last.take() {
                    eprintln!("{r}: {:.1} s", at.elapsed().as_secs_f64());
                }
                last = Some((run, std::time::Instant::now()));
            }
        }
    };
    // IPADIC and mecab-ko-dic take a minute to load in a debug build; the
    // lattice tests on them are ignored by default too.
    let dictionaries = std::env::var("TURBO_TEST_MECAB_DICTIONARIES").is_ok_and(|v| v == "1");
    let wanted =
        |annotator: &str, run: &str| dictionaries || !(annotator == "lattice" && !run.starts_with("mecab-fixture-"));
    let report = compare_tree(&data, &data.join("inputs"), &res, wanted, progress).unwrap_or_else(|e| panic!("{e}"));
    eprintln!("the whole tree: {:.1} s", started.elapsed().as_secs_f64());
    let require = std::env::var("TURBO_TEST_REQUIRE_OPENNLP_MODELS").is_ok_and(|v| v == "1");
    for (run, why) in &report.unavailable {
        match why {
            Unavailable::Missing(p) if !require || p.to_string_lossy().contains("mecab-") => {}
            // The Turkish locale case fold has goldens and no counterpart in the port.
            Unavailable::NotCovered(_) if run == "normalize/case-fold-tr" => {}
            other => panic!("{run}: {other}"),
        }
    }
    let rendered = report.render();
    eprintln!("{rendered}");
    let problems: Vec<String> =
        report.problems().map(|f| format!("{} on {}: {}", f.run_name(), f.input_name(), f.outcome)).collect();
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
    let count = |f: fn(&Outcome) -> bool| report.findings.iter().filter(|x| f(&x.outcome)).count();
    let matched = count(|o| *o == Outcome::Match);
    let documented = count(|o| matches!(o, Outcome::Documented(_)));
    let skipped = count(|o| matches!(o, Outcome::Skipped(_)));
    eprintln!("{matched} matched, {documented} documented, {skipped} skipped");
    assert!(matched > 1000, "{matched} matched");
    // The documented differences the golden tests list: the normalizers
    // with a "Differences from OpenNLP" section, the WordPiece spans over
    // stripped marks, the keep new lines defects of SimpleTokenizer, the
    // language detector declining texts without evidence.
    assert!(documented > 20, "{documented} documented");
    // Skipped: goldens for runs whose model is absent, the goldens whose
    // JVM tokens cut a surrogate pair, the unreadable lemma golden.
    for f in report.findings.iter().filter(|f| matches!(f.outcome, Outcome::Skipped(_))) {
        let Outcome::Skipped(why) = &f.outcome else { unreachable!() };
        let known = why.contains("surrogate pair") || why.contains("not available") || why.contains("does not read");
        assert!(known, "{} on {}: {why}", f.run_name(), f.input_name());
    }
}
