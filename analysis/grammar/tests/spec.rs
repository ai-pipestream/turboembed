//! Conformance with the Recognizers-Text specifications in
//! `tests/data/recognizers-text`, ratcheted by pass lists.
//!
//! A case is eligible unless its `NotSupported` or `NotSupportedByDesign`
//! field names `dotnet`. It passes when the grammar's mentions for the
//! model's output and the expected results are the same set: equal byte
//! spans (the specification's inclusive UTF-16 `Start` and `End`
//! converted through the input's offset table), type names and resolved
//! values. A number's value is equal within a relative 1e-12, since the
//! specifications print .NET doubles to 15 digits, and its subtype is
//! compared when the case gives one.
//!
//! Dates and times are compared by [`datetime`].
//!
//! `tests/data/pass/<Language>/<File>.txt` lists the indices of the cases
//! that pass. A listed case that fails fails the test; a case that passes
//! and is not listed is printed. `BLESS=1` rewrites the lists from the
//! run. `SPEC_SHOW=1` prints every failing case with what was found.

#[path = "spec/datetime.rs"]
mod datetime;
#[path = "spec/dutch.rs"]
mod dutch;

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::PathBuf;

use serde::Deserialize;
use turbo_analysis::{OffsetTable, Utf16Offset};
use turbo_grammar::{ENGLISH, Mention, Output, Parser, Value};

#[derive(Deserialize)]
struct Case {
    #[serde(rename = "Input")]
    input: String,
    #[serde(rename = "Results", default)]
    results: Vec<Expected>,
    #[serde(rename = "NotSupported", default)]
    not_supported: String,
    #[serde(rename = "NotSupportedByDesign", default)]
    not_supported_by_design: String,
}

#[derive(Deserialize)]
struct Expected {
    #[serde(rename = "Start")]
    start: u32,
    #[serde(rename = "End")]
    end: u32,
    #[serde(rename = "TypeName", alias = "Typename")]
    type_name: String,
    #[serde(rename = "Resolution")]
    resolution: Option<Resolution>,
}

#[derive(Deserialize)]
struct Resolution {
    value: Option<String>,
    subtype: Option<String>,
    offset: Option<String>,
    #[serde(rename = "relativeTo")]
    relative_to: Option<String>,
}

impl Case {
    fn eligible(&self) -> bool {
        let names = |s: &str| s.split(',').any(|p| p.trim().eq_ignore_ascii_case("dotnet"));
        !names(&self.not_supported) && !names(&self.not_supported_by_design)
    }
}

/// A specification file and the output it checks.
struct Spec {
    language: &'static str,
    kind: &'static str,
    file: &'static str,
    output: Output,
}

fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn same_number(expected: &str, got: f64) -> bool {
    let Ok(want) = expected.parse::<f64>() else { return false };
    want == got || (want - got).abs() <= 1e-12 * want.abs().max(got.abs())
}

/// Whether mention `got` is what `want` describes, spans aside.
fn matches(output: Output, want: &Expected, got: &Mention) -> bool {
    let res = want.resolution.as_ref();
    let field = |f: fn(&Resolution) -> &Option<String>| res.and_then(|r| f(r).as_deref());
    match (output, got.value) {
        (Output::Number, Value::Num(n)) => {
            want.type_name == "number"
                && field(|r| &r.value).is_some_and(|v| same_number(v, n.value.to_f64()))
                && field(|r| &r.subtype).is_none_or(|s| s == n.kind.name())
        }
        (Output::Percent, Value::Percent(p)) => {
            want.type_name == "percentage"
                && field(|r| &r.value).and_then(|v| v.strip_suffix('%')).is_some_and(|v| same_number(v, p.to_f64()))
        }
        (Output::Ordinal, Value::Ordinal(o)) => {
            want.type_name == "ordinal"
                && field(|r| &r.relative_to) == Some("start")
                && field(|r| &r.offset) == Some(o.value.to_string().as_str())
        }
        (Output::Ordinal, Value::Relative(r)) => {
            want.type_name == "ordinal.relative"
                && field(|x| &x.relative_to) == Some(r.to.name())
                && field(|x| &x.offset) == Some(r.offset.to_string().as_str())
        }
        _ => false,
    }
}

/// Runs one case; returns whether it passed and, for a report, what was
/// found.
fn run(parser: &mut Parser, output: Output, case: &Case) -> (bool, Vec<Mention>) {
    parser.parse(&case.input);
    let mut got = Vec::new();
    parser.select(&case.input, output, &mut got);
    let table = OffsetTable::build(&case.input);
    let mut want: Vec<(u32, u32, &Expected)> = Vec::new();
    for e in &case.results {
        let start = table.from_utf16(&case.input, Utf16Offset(e.start));
        let end = table.from_utf16(&case.input, Utf16Offset(e.end + 1));
        match (start, end) {
            (Ok(s), Ok(t)) => want.push((s, t, e)),
            _ => return (false, got),
        }
    }
    want.sort_by_key(|w| (w.0, w.1));
    let pass = want.len() == got.len()
        && want.iter().zip(&got).all(|(w, g)| w.0 == g.start && w.1 == g.end && matches(output, w.2, g));
    (pass, got)
}

fn describe(input: &str, got: &[Mention]) -> String {
    let mut s = String::new();
    for m in got {
        let v = match m.value {
            Value::Num(n) => format!("{} {}", n.value, n.kind.name()),
            Value::Percent(p) => format!("{p}%"),
            Value::Ordinal(o) => format!("ordinal {}", o.value),
            Value::Relative(r) => format!("{}{:+}", r.to.name(), r.offset),
            v => format!("{v:?}"),
        };
        let _ = write!(s, " [{:?} {v}]", &input[m.start as usize..m.end as usize]);
    }
    s
}

/// Checks the passing cases of a file against its pass list, or with
/// `bless` and `BLESS` set rewrites the list.
fn ratchet(language: &str, file: &str, passing: &BTreeSet<usize>, bless: bool) {
    let list_path = data().join("pass").join(language).join(file.replace(".json", ".txt"));
    let listed: BTreeSet<usize> = std::fs::read_to_string(&list_path)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse().expect("a case index per line"))
        .collect();
    if bless && std::env::var_os("BLESS").is_some() {
        let body: String = passing.iter().map(|i| format!("{i}\n")).collect();
        std::fs::create_dir_all(list_path.parent().expect("a directory")).expect("the pass directory");
        std::fs::write(&list_path, body).expect("the pass list is written");
    } else {
        let lost: Vec<&usize> = listed.difference(passing).collect();
        assert!(lost.is_empty(), "{file}: listed cases fail: {lost:?}");
        let gained: Vec<&usize> = passing.difference(&listed).collect();
        if !gained.is_empty() {
            println!("{file}: passing but not listed (BLESS=1 adds them): {gained:?}");
        }
    }
}

/// Checks one file against its pass list, or with `bless` and `BLESS`
/// set rewrites the list; returns (eligible, passing).
fn check(spec: &Spec, bless: bool) -> (usize, usize) {
    let dir = data().join("recognizers-text").join(spec.language).join(spec.kind);
    let json = std::fs::read_to_string(dir.join(spec.file)).expect("a specification file");
    let cases: Vec<Case> = serde_json::from_str(&json).expect("a specification parses");
    let show = std::env::var_os("SPEC_SHOW").is_some();
    let mut parser = Parser::new(&ENGLISH);
    let mut passing = BTreeSet::new();
    let mut eligible = 0;
    for (i, case) in cases.iter().enumerate() {
        if !case.eligible() {
            continue;
        }
        eligible += 1;
        let (pass, got) = run(&mut parser, spec.output, case);
        if pass {
            passing.insert(i);
        } else if show {
            println!("{} #{i} {:?} found{}", spec.file, case.input, describe(&case.input, &got));
        }
    }
    ratchet(spec.language, spec.file, &passing, bless);
    println!("{}/{}/{}: {} of {} eligible cases pass", spec.language, spec.kind, spec.file, passing.len(), eligible);
    (eligible, passing.len())
}

const NUMBER: Spec = Spec { language: "English", kind: "Number", file: "NumberModel.json", output: Output::Number };
const ORDINAL: Spec = Spec { language: "English", kind: "Number", file: "OrdinalModel.json", output: Output::Ordinal };
const PERCENT: Spec = Spec { language: "English", kind: "Number", file: "PercentModel.json", output: Output::Percent };

#[test]
fn english_numbers() {
    let (eligible, _) = check(&NUMBER, true);
    assert_eq!(eligible, 234);
}

#[test]
fn english_ordinals() {
    let (eligible, _) = check(&ORDINAL, true);
    assert_eq!(eligible, 54);
}

#[test]
fn english_percentages() {
    let (eligible, _) = check(&PERCENT, true);
    assert_eq!(eligible, 20);
}

fn datetime_cases() -> Vec<datetime::TimeCase> {
    let path = data().join("recognizers-text/English/DateTime/DateTimeModel.json");
    let json = std::fs::read_to_string(path).expect("a specification file");
    serde_json::from_str(json.trim_start_matches('\u{feff}')).expect("a specification parses")
}

/// Dates and times, with the cases whose spans alone match and the
/// counts by expected type.
#[test]
fn english_datetimes() {
    let cases = datetime_cases();
    let run = datetime::run_all(&cases, std::env::var_os("SPEC_SHOW").is_some());
    println!("English/DateTime/DateTimeModel.json: {} of {} eligible cases pass", run.passing.len(), run.eligible);
    println!("spans match but values differ: {}", run.spans_only);
    for (name, (eligible, passing)) in &run.by_type {
        println!("  {name:<28}{eligible:>5}{passing:>6}");
    }
    ratchet("English", "DateTimeModel.json", &run.passing, true);
    assert_eq!(run.eligible, 970);
}

/// The pass rate of every file, in one table.
#[test]
fn summary() {
    let mut table = String::from("file                                eligible  passing\n");
    for spec in [&NUMBER, &ORDINAL, &PERCENT] {
        let (eligible, passing) = check(spec, false);
        let name = format!("{}/{}/{}", spec.language, spec.kind, spec.file);
        let _ = writeln!(table, "{name:<36}{eligible:>8}{passing:>9}");
    }
    let run = datetime::run_all(&datetime_cases(), false);
    ratchet("English", "DateTimeModel.json", &run.passing, false);
    let _ = writeln!(table, "{:<36}{:>8}{:>9}", "English/DateTime/DateTimeModel.json", run.eligible, run.passing.len());
    println!("{table}");
}

/// Spans convert from the specification's inclusive UTF-16 indices to
/// bytes: U+202F is one UTF-16 unit and three bytes.
#[test]
fn spans_convert_from_utf16() {
    let case = |start, end| Case {
        input: "as 40\u{202f}000".into(),
        results: vec![Expected {
            start,
            end,
            type_name: "number".into(),
            resolution: Some(Resolution {
                value: Some("40000".into()),
                subtype: Some("integer".into()),
                offset: None,
                relative_to: None,
            }),
        }],
        not_supported: String::new(),
        not_supported_by_design: String::new(),
    };
    let mut parser = Parser::new(&ENGLISH);
    let (pass, got) = run(&mut parser, Output::Number, &case(3, 8));
    assert!(pass);
    assert_eq!((got[0].start, got[0].end), (3, 11));
    assert!(!run(&mut parser, Output::Number, &case(3, 7)).0);
    assert!(!run(&mut parser, Output::Number, &case(3, 40)).0, "an index past the input fails the case");
}

#[test]
fn eligibility_reads_both_fields() {
    let case = |ns: &str, nsd: &str| Case {
        input: String::new(),
        results: Vec::new(),
        not_supported: ns.into(),
        not_supported_by_design: nsd.into(),
    };
    assert!(case("javascript, python", "").eligible());
    assert!(!case("dotnet, java", "").eligible());
    assert!(!case("", "dotnet").eligible());
    assert!(case("", "").eligible());
}

#[test]
fn numbers_compare_within_double_precision() {
    assert!(same_number("0.333333333333333", 1.0 / 3.0));
    assert!(same_number("8.95430243255239", 8.954302432552373));
    assert!(same_number("1E-12", 1e-12));
    assert!(same_number("-1.2732E+15", -1.2732e15));
    assert!(!same_number("0.5", 0.51));
    assert!(!same_number("null", 0.0));
}
