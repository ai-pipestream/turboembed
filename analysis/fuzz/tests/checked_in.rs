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

use std::collections::BTreeMap;

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
    // The documented differences are exactly the ones the golden tests
    // list, run by run: the inputs of each normalizer with a "Differences
    // from OpenNLP" section that show it, the WordPiece inputs of
    // `analysis/embed/tests/goldens.rs`, the two keep new lines defects
    // of `SimpleTokenizer`, and the two texts the Rust language fixture
    // knows no n-gram of, which the annotator declines
    // (`analysis/langdetect/tests/fixture.rs`). A new golden input that shows a documented
    // difference is added here with its rule.
    let mut by_run: BTreeMap<String, usize> = BTreeMap::new();
    for f in report.findings.iter().filter(|f| matches!(f.outcome, Outcome::Documented(_))) {
        *by_run.entry(f.run_name()).or_default() += 1;
    }
    let expected: BTreeMap<String, usize> = [
        ("langdetect/langdetect-fixture-rust", 2),
        ("normalize/bounded-url", 1),
        ("normalize/bullet", 1),
        ("normalize/confusable-skeleton", 16),
        ("normalize/default-chain", 2),
        ("normalize/dehyphenation", 1),
        ("normalize/ellipsis", 1),
        ("normalize/english-contraction", 2),
        ("normalize/invisible", 1),
        ("normalize/quote", 1),
        ("normalize/symbol-joiner", 2),
        ("subwords/all-minilm-l6-v2-vocab", 11),
        ("tokenize/simple+keep-new-lines", 2),
        ("tokenize/simple+keep-new-lines+legacy", 2),
    ]
    .into_iter()
    .map(|(run, n)| (run.to_string(), n))
    .collect();
    assert_eq!(by_run, expected);
    // Every explanation names where the difference is written down.
    for f in report.findings.iter().filter(|f| matches!(f.outcome, Outcome::Documented(_))) {
        let Outcome::Documented(why) = &f.outcome else { unreachable!() };
        let named =
            why.contains("docs/normalizers.md") || why.contains("tests/goldens.rs") || why.contains("tests/golden.rs");
        assert!(named, "{} on {}: {why}", f.run_name(), f.input_name());
    }
    // Skipped: goldens for runs whose model is absent, the goldens whose
    // JVM tokens cut a surrogate pair, the unreadable lemma golden.
    for f in report.findings.iter().filter(|f| matches!(f.outcome, Outcome::Skipped(_))) {
        let Outcome::Skipped(why) = &f.outcome else { unreachable!() };
        let known = why.contains("surrogate pair") || why.contains("not available") || why.contains("does not read");
        assert!(known, "{} on {}: {why}", f.run_name(), f.input_name());
    }
}

#[test]
fn turkish_lowercase_is_compared_without_skips() {
    let data = turbo_goldens::data_dir();
    let report = compare_tree(
        &data,
        &data.join("inputs"),
        &Resources::from_env(),
        |annotator, run| annotator == "normalize" && run == "case-fold-tr",
        |_, _, _| {},
    )
    .unwrap();
    assert!(report.unavailable.is_empty(), "{}", report.render());
    assert!(!report.findings.is_empty());
    assert!(report.findings.iter().all(|f| f.outcome == Outcome::Match), "{}", report.render());
}
