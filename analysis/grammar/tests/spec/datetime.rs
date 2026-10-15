//! The Recognizers-Text date and time model: `DateTimeModel.json`.
//!
//! Each case is run against its `Context.ReferenceDateTime`. A case
//! passes when the mentions and the expected results are the same set:
//! equal byte spans, type names and Recognizers-Text's resolution values,
//! compared as a set of alternatives each holding its `timex`, `type`,
//! `Mod`, `value`, `start` and `end` (other fields, such as
//! `sourceEntity`, are not compared).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use serde::Deserialize;
use turbo_analysis::{OffsetTable, Utf16Offset};
use turbo_analysis_layers::{DateTime, Relation, TimeKind};
use turbo_grammar::time::{Resolution, Resolved, resolve};
use turbo_grammar::{ENGLISH_TIMES, Mention, Output, Parser};

/// The reference of a case that gives none.
const DEFAULT_REFERENCE: &str = "2016-11-07T00:00:00";

/// The fields of a resolution value that are compared.
const FIELDS: [&str; 6] = ["timex", "type", "Mod", "value", "start", "end"];

#[derive(Deserialize)]
pub struct TimeCase {
    #[serde(rename = "Input")]
    pub input: String,
    #[serde(rename = "Context", default)]
    context: Option<Context>,
    #[serde(rename = "Results", default)]
    results: Vec<TimeExpected>,
    #[serde(rename = "NotSupported", default)]
    not_supported: String,
    #[serde(rename = "NotSupportedByDesign", default)]
    not_supported_by_design: String,
}

#[derive(Deserialize)]
struct Context {
    #[serde(rename = "ReferenceDateTime")]
    reference: Option<String>,
}

#[derive(Deserialize)]
struct TimeExpected {
    #[serde(rename = "Start")]
    start: u32,
    #[serde(rename = "End")]
    end: u32,
    #[serde(rename = "TypeName", alias = "Typename", default)]
    type_name: Option<String>,
    #[serde(rename = "Resolution")]
    resolution: Option<Values>,
}

#[derive(Deserialize)]
struct Values {
    #[serde(default)]
    values: Vec<BTreeMap<String, serde_json::Value>>,
}

/// One alternative as compared: the [`FIELDS`] it has.
type Alt = BTreeMap<&'static str, String>;

impl TimeCase {
    pub fn eligible(&self) -> bool {
        let names = |s: &str| s.split(',').any(|p| p.trim().eq_ignore_ascii_case("dotnet"));
        !names(&self.not_supported) && !names(&self.not_supported_by_design)
    }

    fn reference(&self) -> Option<DateTime> {
        let r = self.context.as_ref().and_then(|c| c.reference.as_deref()).unwrap_or(DEFAULT_REFERENCE);
        DateTime::parse(r)
    }

    /// The first expected type name, for the counts by type.
    pub fn type_name(&self) -> &str {
        self.results.first().and_then(|r| r.type_name.as_deref()).unwrap_or("none")
    }
}

/// A found mention: its span, its type name and its alternatives.
pub struct Found {
    start: u32,
    end: u32,
    type_name: String,
    alts: BTreeSet<Alt>,
}

fn expected_alts(e: &TimeExpected) -> BTreeSet<Alt> {
    let mut out = BTreeSet::new();
    for v in e.resolution.iter().flat_map(|r| &r.values) {
        let mut alt = Alt::new();
        for f in FIELDS {
            if let Some(x) = v.get(f) {
                let s = match x {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                alt.insert(f, s);
            }
        }
        out.insert(alt);
    }
    out
}

/// Writes seconds as Recognizers-Text does: `10800`, `5400.5`.
fn seconds(millis: i64) -> String {
    let (whole, frac) = (millis / 1000, (millis % 1000).abs());
    if frac == 0 {
        return whole.to_string();
    }
    let digits = format!("{frac:03}");
    format!("{whole}.{}", digits.trim_end_matches('0'))
}

/// A mention's type name and alternatives as Recognizers-Text writes them.
pub fn found_alts(r: &Resolved) -> (String, BTreeSet<Alt>) {
    let approx = r.modifier.relation == Relation::Approx;
    let value_kind = if approx {
        match r.kind {
            TimeKind::Date => TimeKind::DateRange,
            TimeKind::Time => TimeKind::TimeRange,
            TimeKind::DateTime => TimeKind::DateTimeRange,
            k => k,
        }
    } else {
        r.kind
    };
    let modifier = r.modifier.to_string();
    let mut alts = BTreeSet::new();
    for a in &r.alternatives {
        let mut alt = Alt::new();
        alt.insert("timex", a.timex.to_string());
        alt.insert("type", value_kind.name().to_string());
        if !modifier.is_empty() {
            alt.insert("Mod", modifier.clone());
        }
        match a.value {
            Resolution::Point(i) => {
                alt.insert("value", i.to_string());
            }
            Resolution::Range { start, end } => {
                if let Some(s) = start {
                    alt.insert("start", s.to_string());
                }
                if let Some(e) = end {
                    alt.insert("end", e.to_string());
                }
            }
            Resolution::Duration { millis } => {
                alt.insert("value", seconds(millis));
            }
            Resolution::NotResolved => {
                alt.insert("value", "not resolved".to_string());
            }
        }
        alts.insert(alt);
    }
    (format!("datetimeV2.{}", r.kind.name()), alts)
}

/// What one case gives: whether it passed, whether at least the spans
/// match, and what was found.
pub struct Outcome {
    pub pass: bool,
    pub spans: bool,
    pub found: Vec<Found>,
}

pub fn run(parser: &mut Parser, case: &TimeCase) -> Outcome {
    let fail = |found| Outcome { pass: false, spans: false, found };
    let Some(now) = case.reference() else { return fail(Vec::new()) };
    parser.parse(&case.input);
    let mut got: Vec<Mention> = Vec::new();
    parser.select(&case.input, Output::Time, &mut got);
    let mut found = Vec::new();
    for m in &got {
        let resolved = parser.times().get(&m.value).and_then(|v| resolve(&v, now));
        let (type_name, alts) = resolved.as_ref().map(found_alts).unwrap_or_default();
        found.push(Found { start: m.start, end: m.end, type_name, alts });
    }
    let table = OffsetTable::build(&case.input);
    let mut want = Vec::new();
    for e in &case.results {
        let start = table.from_utf16(&case.input, Utf16Offset(e.start));
        let end = table.from_utf16(&case.input, Utf16Offset(e.end + 1));
        match (start, end) {
            (Ok(s), Ok(t)) => want.push((s, t, e)),
            _ => return fail(found),
        }
    }
    want.sort_by_key(|w| (w.0, w.1));
    let spans = want.len() == found.len() && want.iter().zip(&found).all(|(w, g)| w.0 == g.start && w.1 == g.end);
    let pass = spans
        && want
            .iter()
            .zip(&found)
            .all(|(w, g)| w.2.type_name.as_deref() == Some(g.type_name.as_str()) && expected_alts(w.2) == g.alts);
    Outcome { pass, spans, found }
}

fn write_alts(s: &mut String, alts: &BTreeSet<Alt>) {
    for a in alts {
        let _ = write!(s, " {{");
        for (k, v) in a {
            let _ = write!(s, "{k}={v};");
        }
        let _ = write!(s, "}}");
    }
}

/// The case's expected and found mentions, for `SPEC_SHOW`.
pub fn describe(case: &TimeCase, found: &[Found]) -> String {
    let mut s = String::from("\n  want:");
    for e in &case.results {
        let _ = write!(s, " [{}..{} {}", e.start, e.end + 1, e.type_name.as_deref().unwrap_or("?"));
        write_alts(&mut s, &expected_alts(e));
        s.push(']');
    }
    s.push_str("\n  got: ");
    for f in found {
        let text = case.input.get(f.start as usize..f.end as usize).unwrap_or("?");
        let _ = write!(s, " [{text:?} {}", f.type_name);
        write_alts(&mut s, &f.alts);
        s.push(']');
    }
    s
}

/// Runs every eligible case: the passing indices, the eligible count,
/// the cases whose spans alone match, and (eligible, passing) by the
/// first expected type name.
pub struct Run {
    pub passing: BTreeSet<usize>,
    pub eligible: usize,
    pub spans_only: usize,
    pub by_type: BTreeMap<String, (usize, usize)>,
}

pub fn run_all(cases: &[TimeCase], show: bool) -> Run {
    let mut parser = Parser::new(&ENGLISH_TIMES);
    let mut out = Run { passing: BTreeSet::new(), eligible: 0, spans_only: 0, by_type: BTreeMap::new() };
    for (i, case) in cases.iter().enumerate() {
        if !case.eligible() {
            continue;
        }
        out.eligible += 1;
        let o = run(&mut parser, case);
        let counts = out.by_type.entry(case.type_name().to_string()).or_default();
        counts.0 += 1;
        if o.pass {
            out.passing.insert(i);
            counts.1 += 1;
        } else {
            out.spans_only += usize::from(o.spans);
            if show {
                println!("#{i} {:?}{}", case.input, describe(case, &o.found));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seconds_are_written_as_decimals() {
        assert_eq!(seconds(10_800_000), "10800");
        assert_eq!(seconds(5_400_500), "5400.5");
        assert_eq!(seconds(-86_400_000), "-86400");
    }
}
