//! The Spanish specifications: numbers, ordinals, percentages, and dates
//! and times, judged as the English ones are. The specifications write a
//! number's value with a decimal comma (`101231,2353`), read here as a
//! dot.

use std::fmt::Write as _;

use turbo_grammar::{Output, SPANISH, SPANISH_TIMES};

use super::{Case, Spec, check_cases, data, datetime, ratchet};

const LANGUAGE: &str = "Spanish";

fn cases(kind: &str, file: &str) -> Vec<Case> {
    let path = data().join("recognizers-text").join(LANGUAGE).join(kind).join(file);
    let json = std::fs::read_to_string(path).expect("a specification file");
    let mut cases: Vec<Case> =
        serde_json::from_str(json.trim_start_matches('\u{feff}')).expect("a specification parses");
    for e in cases.iter_mut().flat_map(|c| c.results.iter_mut()) {
        if let Some(v) = e.resolution.as_mut().and_then(|r| r.value.as_mut()) {
            *v = v.replace(',', ".");
        }
    }
    cases
}

/// Checks one number file against its pass list, or with `bless` and
/// `BLESS` set rewrites the list; returns (eligible, passing).
fn check(file: &'static str, output: Output, bless: bool) -> (usize, usize) {
    let spec = Spec { language: LANGUAGE, kind: "Number", file, output };
    check_cases(&SPANISH, &spec, &cases("Number", file), bless)
}

fn datetime_cases() -> Vec<datetime::TimeCase> {
    let path = data().join("recognizers-text").join(LANGUAGE).join("DateTime/DateTimeModel.json");
    let json = std::fs::read_to_string(path).expect("a specification file");
    serde_json::from_str(json.trim_start_matches('\u{feff}')).expect("a specification parses")
}

#[test]
fn spanish_numbers() {
    assert_eq!(check("NumberModel.json", Output::Number, true).0, 206);
}

#[test]
fn spanish_ordinals() {
    assert_eq!(check("OrdinalModel.json", Output::Ordinal, true).0, 82);
}

#[test]
fn spanish_percentages() {
    assert_eq!(check("PercentModel.json", Output::Percent, true).0, 66);
}

#[test]
fn spanish_datetimes() {
    let cases = datetime_cases();
    let run = datetime::run_all_with(&SPANISH_TIMES, &cases, std::env::var_os("SPEC_SHOW").is_some());
    println!("{LANGUAGE}/DateTime/DateTimeModel.json: {} of {} eligible cases pass", run.passing.len(), run.eligible);
    println!("spans match but values differ: {}", run.spans_only);
    for (name, (eligible, passing)) in &run.by_type {
        println!("  {name:<28}{eligible:>5}{passing:>6}");
    }
    ratchet(LANGUAGE, "DateTimeModel.json", &run.passing, true);
    assert_eq!(run.eligible, 744);
}

/// The pass rate of every Spanish file, in one table.
#[test]
fn spanish_summary() {
    let mut table = String::from("file                                eligible  passing\n");
    for (file, output) in [
        ("NumberModel.json", Output::Number),
        ("OrdinalModel.json", Output::Ordinal),
        ("PercentModel.json", Output::Percent),
    ] {
        let (eligible, passing) = check(file, output, false);
        let _ = writeln!(table, "{:<36}{eligible:>8}{passing:>9}", format!("{LANGUAGE}/Number/{file}"));
    }
    let run = datetime::run_all_with(&SPANISH_TIMES, &datetime_cases(), false);
    ratchet(LANGUAGE, "DateTimeModel.json", &run.passing, false);
    let _ = writeln!(table, "{:<36}{:>8}{:>9}", "Spanish/DateTime/DateTimeModel.json", run.eligible, run.passing.len());
    println!("{table}");
}
