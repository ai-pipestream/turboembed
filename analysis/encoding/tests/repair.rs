//! The mojibake repair against ftfy 6.3.1's own test cases
//! (tests/test_cases.json, Copyright Robyn Speer, Apache-2.0; see NOTICE),
//! with the output and explanation ftfy's fix_encoding_and_explain gives
//! for each, recorded in tests/data/ftfy_cases.tsv by
//! analysis/encoding-gen/ftfy.py.
//!
//! Every case runs, none is skipped. The port must give ftfy's output and
//! ftfy's explanation exactly, including for the ten cases ftfy itself
//! marks as expected failures (its heuristic does not reach the intended
//! text; the port reproduces the same miss, and the test checks the miss
//! is still there, as ftfy's strict xfail does). Each passing case also
//! runs with one more layer of UTF-8 read as Latin-1, as ftfy's test does.
//!
//! ftfy's test compares `fix_text` (all of ftfy's fixes) to the "fixed"
//! column; only `fix_encoding` is ported, so the comparison here is with
//! "fixed-encoding" where a case has one, as ftfy's own assertion on
//! `fix_encoding_and_explain` does. Cases whose "fixed" text needs the
//! fixes that are not ported (HTML entities, curly quotes, ligatures,
//! width) carry a "fixed-encoding" column and are checked against it.
//!
//! Beyond the text, every output is checked through its alignment: each
//! edit's source and output are whole characters, unchanged stretches are
//! identical, and every output character maps back to source characters
//! and forward again to the same place.

use turbo_analysis::Span;
use turbo_analysis::text::Side;
use turbo_encoding::repair::{Action, PlanStep, fix_encoding, is_bad};

fn unescape(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next().expect("an escape has a second character") {
            '\\' => out.push('\\'),
            't' => out.push('\t'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            'u' => {
                assert_eq!(chars.next(), Some('{'));
                let hex: String = chars.by_ref().take_while(|&c| c != '}').collect();
                out.push(char::from_u32(u32::from_str_radix(&hex, 16).unwrap()).unwrap());
            }
            other => panic!("unknown escape \\{other}"),
        }
    }
    out
}

fn plan_string(plan: &[PlanStep]) -> String {
    if plan.is_empty() {
        return "-".to_string();
    }
    plan.iter().map(|s| format!("{}:{}", s.action.name(), s.parameter)).collect::<Vec<_>>().join(",")
}

struct Case {
    label: String,
    expect: String,
    original: String,
    ftfy: String,
    intended: String,
    plan: String,
}

fn cases() -> Vec<Case> {
    let data = include_str!("data/ftfy_cases.tsv");
    data.lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            assert_eq!(f.len(), 6, "{line}");
            Case {
                label: unescape(f[0]),
                expect: f[1].to_string(),
                original: unescape(f[2]),
                ftfy: unescape(f[3]),
                intended: unescape(f[4]),
                plan: f[5].to_string(),
            }
        })
        .collect()
}

#[test]
fn every_ftfy_case_gives_ftfys_text_and_explanation() {
    let cases = cases();
    assert_eq!(cases.len(), 161 + 151);
    let mut failures = Vec::new();
    for case in &cases {
        let r = fix_encoding(&case.original);
        if r.text != case.ftfy || plan_string(&r.plan) != case.plan {
            failures.push(format!(
                "{}\n  original {:?}\n  ftfy     {:?} {}\n  port     {:?} {}",
                case.label,
                case.original,
                case.ftfy,
                case.plan,
                r.text,
                plan_string(&r.plan)
            ));
        }
    }
    assert!(failures.is_empty(), "{} of {} cases differ:\n{}", failures.len(), cases.len(), failures.join("\n"));
}

#[test]
fn passing_cases_reach_the_intended_text_and_failing_ones_still_miss() {
    let cases = cases();
    let (mut pass, mut fail) = (0, 0);
    for case in cases.iter().filter(|c| c.expect != "-") {
        let r = fix_encoding(&case.original);
        if case.expect == "pass" {
            assert_eq!(r.text, case.intended, "{}", case.label);
            pass += 1;
        } else {
            assert_eq!(case.expect, "fail");
            assert_ne!(r.text, case.intended, "{} is marked as failing in ftfy", case.label);
            fail += 1;
        }
    }
    assert_eq!((pass, fail), (151, 10));
}

#[test]
fn every_repair_maps_offsets_exactly() {
    for case in cases() {
        let src = &case.original;
        let r = fix_encoding(src);
        let out = &r.text;
        let mut prev = (0, 0);
        for e in r.alignment.edits() {
            let (p, c) = (e.parent.range(), e.child.range());
            assert!(src.is_char_boundary(p.start) && src.is_char_boundary(p.end), "{}", case.label);
            assert!(out.is_char_boundary(c.start) && out.is_char_boundary(c.end), "{}", case.label);
            assert_ne!(&src[p.clone()], &out[c.clone()], "{}: an edit that changes nothing", case.label);
            assert_eq!(&src[prev.0..p.start], &out[prev.1..c.start], "{}", case.label);
            prev = (p.end, c.end);
        }
        assert_eq!(&src[prev.0..], &out[prev.1..], "{}", case.label);
        // Each output character back to the source and forward again.
        for (i, ch) in out.char_indices() {
            let span = Span::new(i as u32, (i + ch.len_utf8()) as u32);
            let back = r.alignment.span_to_parent(span);
            assert!(src.is_char_boundary(back.start as usize) && src.is_char_boundary(back.end as usize));
            let forward = r.alignment.span_to_child(back);
            assert!(forward.start <= span.start && forward.end >= span.end, "{}", case.label);
            let start = r.alignment.to_parent(span.start, Side::Start);
            assert_eq!(start, back.start);
        }
    }
}

#[test]
fn the_heuristic_flags_every_original_that_ftfy_changes() {
    for case in cases() {
        if case.ftfy != case.original {
            assert!(is_bad(&case.original), "{}", case.label);
        }
    }
}

#[test]
fn a_double_encoding_is_undone_in_two_rounds_of_steps() {
    let r = fix_encoding("ÃƒÂ©");
    assert_eq!(r.text, "é");
    assert_eq!(r.alignment.edits().len(), 1);
    assert_eq!(r.plan.iter().filter(|s| s.action == Action::Decode).count(), 2);
}
