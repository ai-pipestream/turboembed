//! The Dutch specifications: numbers, ordinals, percentages, and dates
//! and times, judged as the English ones are. The specifications write a
//! number's value with a decimal comma (`103,666666666667`), read here as a
//! dot.
//!
//! Many Dutch number cases give a result's `Text` and no `Start` or
//! `End`; the span is then where that text first occurs in the input,
//! ignoring case, after the result before it. A negative `Start` or
//! `End`, which only cases not run on .NET hold, is read as 0.

use std::fmt::Write as _;
use std::path::PathBuf;

use serde_json::Value;
use turbo_grammar::{DUTCH, DUTCH_TIMES, Output};

use super::{Case, Spec, check_cases, data, datetime, ratchet};

const LANGUAGE: &str = "Dutch";

/// The inclusive UTF-16 span of the first occurrence of `text` in `input`
/// at or after UTF-16 index `from`, ignoring case.
fn locate(input: &str, text: &str, from: u64) -> Option<(u64, u64)> {
    let low = |c: char| c.to_lowercase().next().unwrap_or(c);
    let want: Vec<char> = text.chars().map(low).collect();
    if want.is_empty() {
        return None;
    }
    let mut at = 0u64;
    for (i, c) in input.char_indices() {
        if at >= from && input[i..].chars().map(low).take(want.len()).eq(want.iter().copied()) {
            let len: u64 = input[i..].chars().take(want.len()).map(|c| c.len_utf16() as u64).sum();
            return Some((at, at + len - 1));
        }
        at += c.len_utf16() as u64;
    }
    None
}

/// A specification file as JSON, with a `Start` and an `End` on every
/// result (see the module comment). A text that is not found gets a span
/// past the input, which fails its case.
fn load(path: PathBuf) -> Value {
    let json = std::fs::read_to_string(path).expect("a specification file");
    let mut all: Value = serde_json::from_str(json.trim_start_matches('\u{feff}')).expect("a specification parses");
    for case in all.as_array_mut().into_iter().flatten() {
        let input = case.get("Input").and_then(Value::as_str).unwrap_or_default().to_string();
        let mut from = 0;
        for r in case.get_mut("Results").and_then(Value::as_array_mut).into_iter().flatten() {
            match r.get("Start").and_then(Value::as_i64) {
                None => {
                    let text = r.get("Text").and_then(Value::as_str).unwrap_or_default();
                    let (s, e) = locate(&input, text, from).unwrap_or((1 << 30, 1 << 30));
                    r["Start"] = s.into();
                    r["End"] = e.into();
                    from = e + 1;
                }
                Some(s) if s < 0 => r["Start"] = 0.into(),
                Some(_) => {}
            }
            if r.get("End").and_then(Value::as_i64).is_some_and(|e| e < 0) {
                r["End"] = 0.into();
            }
        }
    }
    all
}

fn cases(kind: &str, file: &str) -> Vec<Case> {
    let path = data().join("recognizers-text").join(LANGUAGE).join(kind).join(file);
    let mut cases: Vec<Case> = serde_json::from_value(load(path)).expect("a specification parses");
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
    check_cases(&DUTCH, &spec, &cases("Number", file), bless)
}

fn datetime_cases() -> Vec<datetime::TimeCase> {
    let path = data().join("recognizers-text").join(LANGUAGE).join("DateTime/DateTimeModel.json");
    serde_json::from_value(load(path)).expect("a specification parses")
}

#[test]
fn texts_are_located_after_the_last() {
    assert_eq!(locate("Nul is hetzelfde als 0", "nul", 0), Some((0, 2)));
    assert_eq!(locate("1 en 1", "1", 1), Some((5, 5)));
    assert_eq!(locate("\u{e9}\u{e9}n 10G", "10g", 0), Some((4, 6)));
    assert_eq!(locate("x", "y", 0), None);
}

#[test]
fn dutch_numbers() {
    assert_eq!(check("NumberModel.json", Output::Number, true).0, 161);
}

#[test]
fn dutch_ordinals() {
    assert_eq!(check("OrdinalModel.json", Output::Ordinal, true).0, 48);
}

#[test]
fn dutch_percentages() {
    assert_eq!(check("PercentModel.json", Output::Percent, true).0, 13);
}

#[test]
fn dutch_datetimes() {
    let cases = datetime_cases();
    let run = datetime::run_all_with(&DUTCH_TIMES, &cases, std::env::var_os("SPEC_SHOW").is_some());
    println!("{LANGUAGE}/DateTime/DateTimeModel.json: {} of {} eligible cases pass", run.passing.len(), run.eligible);
    println!("spans match but values differ: {}", run.spans_only);
    for (name, (eligible, passing)) in &run.by_type {
        println!("  {name:<28}{eligible:>5}{passing:>6}");
    }
    ratchet(LANGUAGE, "DateTimeModel.json", &run.passing, true);
    assert_eq!(run.eligible, 605);
}

/// The pass rate of every Dutch file, in one table.
#[test]
fn dutch_summary() {
    let mut table = String::from("file                                eligible  passing\n");
    for (file, output) in [
        ("NumberModel.json", Output::Number),
        ("OrdinalModel.json", Output::Ordinal),
        ("PercentModel.json", Output::Percent),
    ] {
        let (eligible, passing) = check(file, output, false);
        let _ = writeln!(table, "{:<36}{eligible:>8}{passing:>9}", format!("{LANGUAGE}/Number/{file}"));
    }
    let run = datetime::run_all_with(&DUTCH_TIMES, &datetime_cases(), false);
    ratchet(LANGUAGE, "DateTimeModel.json", &run.passing, false);
    let _ = writeln!(table, "{:<36}{:>8}{:>9}", "Dutch/DateTime/DateTimeModel.json", run.eligible, run.passing.len());
    println!("{table}");
}
