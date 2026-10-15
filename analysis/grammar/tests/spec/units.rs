//! Numbers with units: the Recognizers-Text `NumberWithUnit` models
//! (currency, dimension, temperature, age) against the unit grammar.
//!
//! The grammar chooses all quantities at once, so each file is compared
//! with the mentions of its kind among them. A case passes when those and
//! the expected results are equal sets: the same byte spans, type names,
//! units (by their Recognizers-Text names), values (within a relative
//! 1e-12, or both absent for a unit named alone), a dimension's subtype
//! and, where the case gives one, the ISO currency code.
//!
//! Recognizers-Text files gallons under weight; the comparison takes its
//! subtype for a gallon as volume, which is what the grammar says. Its
//! `Ton` covers both a metric ton and a short ton (see [`spec_unit`]).

use std::collections::BTreeSet;
use std::fmt::Write as _;

use serde::Deserialize;
use turbo_analysis::{OffsetTable, Utf16Offset};
use turbo_grammar::unit::Weight;
use turbo_grammar::{CurrencyUnit, ENGLISH_UNITS, Measure, Mention, Output, Parser, Unit, Value};

use super::{data, same_number};

#[derive(Deserialize)]
pub struct UnitCase {
    #[serde(rename = "Input")]
    input: String,
    #[serde(rename = "Results", default)]
    results: Vec<UnitExpected>,
    #[serde(rename = "NotSupported", default)]
    not_supported: String,
    #[serde(rename = "NotSupportedByDesign", default)]
    not_supported_by_design: String,
}

#[derive(Deserialize)]
struct UnitExpected {
    #[serde(rename = "Start")]
    start: u32,
    #[serde(rename = "End")]
    end: u32,
    #[serde(rename = "TypeName", alias = "Typename")]
    type_name: String,
    #[serde(rename = "Resolution")]
    resolution: Option<UnitResolution>,
}

#[derive(Deserialize, Default)]
struct UnitResolution {
    value: Option<String>,
    unit: Option<String>,
    subtype: Option<String>,
    #[serde(rename = "isoCurrency")]
    iso_currency: Option<String>,
}

impl UnitCase {
    fn eligible(&self) -> bool {
        let names = |s: &str| s.split(',').any(|p| p.trim().eq_ignore_ascii_case("dotnet"));
        !names(&self.not_supported) && !names(&self.not_supported_by_design)
    }
}

/// The subtype Recognizers-Text gives a unit, where it differs from the
/// grammar's.
fn spec_subtype(unit: &str, subtype: &str) -> String {
    match (unit, subtype) {
        ("Gallon", "Weight") => "Volume".into(),
        _ => subtype.into(),
    }
}

/// Whether the expected unit name is `unit`'s. Recognizers-Text names
/// a tonne, `t` and an unqualified ton all `Ton`; the grammar reads the
/// first two as a metric ton and the last as a short ton.
fn spec_unit(want: Option<&str>, unit: Unit, name: &str) -> bool {
    match want {
        Some("Ton") => matches!(unit, Unit::Weight(Weight::MetricTon | Weight::ShortTon)),
        Some(w) => w == name,
        None => false,
    }
}

/// A mention's unit and value, if it is a quantity or a unit alone.
fn reading(m: &Mention) -> Option<(Unit, Option<f64>)> {
    match m.value {
        Value::Quantity(q) => Some((q.unit(), Some(q.value().to_f64()))),
        Value::Unit(u) => Some((u, None)),
        _ => None,
    }
}

fn matches(want: &UnitExpected, unit: Unit, value: Option<f64>) -> bool {
    let empty = UnitResolution::default();
    let res = want.resolution.as_ref().unwrap_or(&empty);
    let value_ok = match (res.value.as_deref(), value) {
        (Some(v), Some(got)) => same_number(v, got),
        (None, None) => true,
        _ => false,
    };
    let name = unit.to_string();
    let subtype_ok = match (res.subtype.as_deref(), unit.dimension()) {
        (Some(s), Some(d)) => spec_subtype(&name, s) == d.name(),
        (None, d) => d.is_none(),
        (Some(_), None) => false,
    };
    let iso_ok = res.iso_currency.as_deref().is_none_or(|iso| match unit {
        Unit::Currency(c) => CurrencyUnit::currency(c).is_some_and(|(code, e)| e == 0 && code.as_str() == iso),
        _ => false,
    });
    want.type_name == unit.measure().name()
        && spec_unit(res.unit.as_deref(), unit, &name)
        && value_ok
        && subtype_ok
        && iso_ok
}

/// Runs one case for `measure`; returns whether it passed and what was
/// found.
pub fn run(parser: &mut Parser, measure: Measure, case: &UnitCase) -> (bool, Vec<Mention>) {
    parser.parse(&case.input);
    let mut got = Vec::new();
    parser.select(&case.input, Output::Quantity, &mut got);
    got.retain(|m| reading(m).is_some_and(|(u, _)| u.measure() == measure));
    let table = OffsetTable::build(&case.input);
    let mut want: Vec<(u32, u32, &UnitExpected)> = Vec::new();
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
        && want.iter().zip(&got).all(|(w, g)| {
            let (u, v) = reading(g).expect("kept for its reading");
            w.0 == g.start && w.1 == g.end && matches(w.2, u, v)
        });
    (pass, got)
}

pub fn describe(input: &str, got: &[Mention]) -> String {
    let mut s = String::new();
    for m in got {
        if let Some((u, v)) = reading(m) {
            let v = v.map_or("-".to_string(), |v| v.to_string());
            let sub = u.dimension().map_or("", |d| d.name());
            let _ = write!(s, " [{:?} {v} {u} {sub}]", &input[m.start as usize..m.end as usize]);
        }
    }
    s
}

pub fn cases(file: &str) -> Vec<UnitCase> {
    let path = data().join("recognizers-text/English/NumberWithUnit").join(file);
    let json = std::fs::read_to_string(path).expect("a specification file");
    serde_json::from_str(json.trim_start_matches('\u{feff}')).expect("a specification parses")
}

/// Runs every eligible case of `file`; returns (eligible, passing).
pub fn run_file(file: &str, measure: Measure) -> (usize, BTreeSet<usize>) {
    let show = std::env::var_os("SPEC_SHOW").is_some();
    let mut parser = Parser::new(&ENGLISH_UNITS);
    let mut passing = BTreeSet::new();
    let mut eligible = 0;
    for (i, case) in cases(file).iter().enumerate() {
        if !case.eligible() {
            continue;
        }
        eligible += 1;
        let (pass, got) = run(&mut parser, measure, case);
        if pass {
            passing.insert(i);
        } else if show {
            println!("{file} #{i} {:?} found{}", case.input, describe(&case.input, &got));
        }
    }
    (eligible, passing)
}
