//! #1884: date truncation, shifting and relative dates (`DateRewrite`).
//! The written form stays on a shift, leap days and month ends clamp,
//! a keyed offset is the same for one context and never zero, what is not
//! a whole date is masked, and the strategy works as a rewrite and inside
//! a per-kind policy. Values are made up.

mod support;

use support::*;
use turbo_analysis::{Annotator, Span, ViewId};
use turbo_analysis_layers::{Date, Temporals};
use turbo_pii::rewrite::{
    AgeSafeHarbor, DateMode, DateRewrite, DateShift, DateShiftError, MaskPolicy, RedactedView, RewritePolicy,
    RewriteStrategy, source_view,
};
use turbo_pii::{Pack, PiiKind, PiiStrategyKind};

const KEY: [u8; 32] = [7; 32];

fn d(y: i32, m: u8, day: u8) -> Date {
    Date::new(y, m, day).unwrap()
}

#[test]
fn truncating_keeps_the_year_alone() {
    let year = DateRewrite::year();
    for (text, want) in [
        ("2026-07-14", "2026"),
        ("July 14, 2026", "2026"),
        ("14th July 2026", "2026"),
        ("Jul 4 1952", "1952"),
        ("July 2026", "2026"),
        ("2024-02-29", "2024"),
    ] {
        assert_eq!(year.apply(text), want, "{text:?}");
    }
}

#[test]
fn a_fixed_shift_keeps_the_written_form() {
    let shift = |y, m, day| DateRewrite::shift(DateShift::fixed(y, m, day));
    assert_eq!(shift(0, 0, 10).apply("2026-07-14"), "2026-07-24");
    assert_eq!(shift(0, 0, 10).apply("July 14, 2026"), "July 24, 2026");
    assert_eq!(shift(0, 0, -14).apply("14th July 2026"), "30th June 2026");
    assert_eq!(shift(1, 2, 3).apply("Jul 4 1952"), "Sep 7 1953");
    assert_eq!(shift(0, 0, 200).apply("JULY 14TH, 2026"), "JANUARY 30TH, 2027");
    assert_eq!(shift(0, 5, 0).apply("July 2026"), "December 2026");
    assert_eq!(shift(0, 0, 40).apply("July 2026"), "August 2026", "a month moves as its first day does");
}

#[test]
fn leap_days_and_month_ends_clamp() {
    let shift = |y, m, day| DateRewrite::shift(DateShift::fixed(y, m, day));
    assert_eq!(shift(1, 0, 0).apply("2024-02-29"), "2025-02-28", "no leap day in 2025");
    assert_eq!(shift(4, 0, 0).apply("2024-02-29"), "2028-02-29");
    assert_eq!(shift(0, 0, 1).apply("2024-02-28"), "2024-02-29");
    assert_eq!(shift(0, 0, 1).apply("2023-02-28"), "2023-03-01");
    assert_eq!(shift(0, 1, 0).apply("January 31, 2024"), "February 29, 2024");
    assert_eq!(shift(0, 1, 0).apply("January 31, 2023"), "February 28, 2023");
    assert_eq!(shift(0, 3, 0).apply("31st January 2026"), "30th April 2026");
    assert_eq!(shift(0, 0, 1).apply("2026-12-31"), "2027-01-01");
    assert_eq!(shift(-100, 0, 0).apply("2000-02-29"), "1900-02-28", "1900 is not a leap year");
}

#[test]
fn two_digit_years_and_other_unread_forms_are_masked_whole() {
    // The scanner reads four-digit years only, so a caller's mention with a
    // two-digit year is no date it can move, and nothing of it stays.
    for mode in
        [DateRewrite::year(), DateRewrite::shift(DateShift::fixed(0, 0, 3)), DateRewrite::relative(d(2026, 1, 1))]
    {
        for not in ["7/14/26", "July 14, 26", "14 Jul 26", "'26", "2026", "Q3 2026", "next week", ""] {
            assert_eq!(mode.apply(not), "*".repeat(not.chars().count()), "{not:?} under {:?}", mode.mode());
        }
    }
    assert_eq!(DateRewrite::year().with_mask('#').apply("7/14/26"), "#######");
    // A shift out of the four-digit years masks too.
    assert_eq!(DateRewrite::shift(DateShift::fixed(-10, 0, 0)).apply("0005-01-01"), "**********");
    assert_eq!(DateRewrite::shift(DateShift::fixed(1, 0, 0)).apply("December 9999"), "*************");
}

#[test]
fn a_keyed_offset_is_one_per_context_and_never_zero() {
    let a = DateShift::default().keyed(&KEY, "document-1", 30).unwrap();
    let again = DateShift::default().keyed(&KEY, "document-1", 30).unwrap();
    assert_eq!(a.keyed_days(), again.keyed_days());
    let mut seen = std::collections::BTreeSet::new();
    for i in 0..200 {
        let s = DateShift::default().keyed(&KEY, &format!("document-{i}"), 30).unwrap();
        let n = s.keyed_days();
        assert!(n != 0 && (-30..=30).contains(&n), "{n}");
        seen.insert(n);
    }
    assert!(seen.len() > 30, "contexts spread over the range: {seen:?}");
    assert!(seen.iter().any(|&n| n < 0) && seen.iter().any(|&n| n > 0));
    // The interval between two dates of one document stays.
    let r = DateRewrite::shift(a.clone());
    let (x, y) = (r.apply("2026-07-14"), r.apply("2026-07-24"));
    let (x, y) = (Date::parse(&x).unwrap(), Date::parse(&y).unwrap());
    assert_eq!(y.epoch_day() - x.epoch_day(), 10);
    assert_eq!(x.epoch_day() - d(2026, 7, 14).epoch_day(), a.keyed_days());
    // Another key moves the same context by another offset, for some context.
    let other = |c: &str| DateShift::default().keyed(&[8; 32], c, 3650).unwrap().keyed_days();
    let mine = |c: &str| DateShift::default().keyed(&KEY, c, 3650).unwrap().keyed_days();
    assert!((0..8).any(|i| other(&i.to_string()) != mine(&i.to_string())));
    // The key never prints.
    assert!(!format!("{a:?}").contains("document-1") && !format!("{a:?}").contains('7'));
}

#[test]
fn no_shift_writes_a_value_back_as_it_was() {
    // A keyed offset under a month still moves a month mention.
    for i in 0..200 {
        let s = DateShift::default().keyed(&KEY, &format!("document-{i}"), 30).unwrap();
        let n = s.keyed_days();
        let out = DateRewrite::shift(s).apply("July 2026");
        let want = if n > 0 { "August 2026" } else { "June 2026" };
        assert_eq!(out, want, "offset {n}");
    }
    // A net shift of nothing masks the date whole.
    assert_eq!(DateRewrite::shift(DateShift::fixed(0, 0, 0)).apply("2026-07-14"), "**********");
    assert_eq!(DateRewrite::shift(DateShift::fixed(0, 0, 0)).apply("July 2026"), "*********");
    let keyed = DateShift::default().keyed(&KEY, "document-1", 30).unwrap();
    let n = keyed.keyed_days();
    let cancel = DateShift::fixed(0, 0, i32::try_from(-n).unwrap()).keyed(&KEY, "document-1", 30).unwrap();
    assert_eq!(DateRewrite::shift(cancel.clone()).apply("July 14, 2026"), "*************");
    assert_eq!(DateRewrite::shift(cancel).apply("July 2026"), "*********");
}

#[test]
fn keyed_shifts_check_their_key_and_range() {
    assert_eq!(DateShift::default().keyed(&[1; 31], "c", 30).err(), Some(DateShiftError::ShortKey(31)));
    assert_eq!(DateShift::default().keyed(&KEY, "c", 0).err(), Some(DateShiftError::MaxDays(0)));
    assert_eq!(DateShift::default().keyed(&KEY, "c", 3651).err(), Some(DateShiftError::MaxDays(3651)));
    assert!(DateShift::default().keyed(&KEY, "c", 3650).is_ok());
}

#[test]
fn relative_dates_use_the_reference_passed_in_and_read_back() {
    let r = DateRewrite::relative(d(2026, 7, 14));
    assert_eq!(r.apply("2026-07-14"), "today");
    assert_eq!(r.apply("July 13, 2026"), "yesterday");
    assert_eq!(r.apply("14th April 2026"), "3 months ago");
    assert_eq!(r.apply("Jul 4 1952"), "74 years ago");
    assert_eq!(r.apply("2026-07-28"), "in 2 weeks");
    assert_eq!(r.apply("July 2026"), "this month");
    assert_eq!(r.apply("May 2026"), "2 months ago");
    // The temporal scanner reads each phrase back whole with the same
    // reference.
    let reference = d(2026, 7, 14);
    for text in ["2026-07-11", "2026-07-02", "2026-04-15", "2025-01-01", "2026-08-30", "2029-03-01", "May 2026"] {
        let phrase = r.apply(text);
        let mut found = Temporals::new(ViewId::SOURCE);
        turbo_numeric::temporal::scan(&phrase, Some(reference), &mut found);
        let (i, _) = found.spans.iter().next().unwrap_or_else(|| panic!("{phrase:?} reads back"));
        assert_eq!(found.get(i).span, Span::new(0, phrase.len() as u32), "{phrase:?}");
    }
}

#[test]
fn the_rewrite_reduces_detected_dates_and_keeps_the_words() {
    let text = "Admitted July 14, 2026; born 1952-07-04; seen 14th July 2026, follow-up August 2026.";
    let pii = layer(&pack(Pack::Dates), text);
    let dates: Vec<Span> = pii.mentions().filter(|m| m.kind == PiiKind::Date).map(|m| m.span).collect();
    assert_eq!(dates.len(), 4, "every date is found");
    let src = source_view(text);
    let year = DateRewrite::year().rewrite(&src, &pii).unwrap();
    assert_eq!(year.text(), "Admitted 2026; born 1952; seen 2026, follow-up 2026.");
    assert!(year.edits().edits().all(|e| e.strategy == PiiStrategyKind::DateYear));
    let shifted = DateRewrite::shift(DateShift::fixed(0, 0, -20)).rewrite(&src, &pii).unwrap();
    assert_eq!(shifted.text(), "Admitted June 24, 2026; born 1952-06-14; seen 24th June 2026, follow-up July 2026.");
    assert!(shifted.edits().edits().all(|e| e.strategy == PiiStrategyKind::DateShift));
    let rel = DateRewrite::relative(d(2026, 7, 20)).rewrite(&src, &pii).unwrap();
    assert_eq!(rel.text(), "Admitted 6 days ago; born 74 years ago; seen 6 days ago, follow-up in 1 month.");
    assert!(rel.edits().edits().all(|e| e.strategy == PiiStrategyKind::DateRelative));
    let sources: Vec<Span> = pii.mentions().map(|m| m.span).collect();
    for r in [&year, &shifted, &rel] {
        assert_offset_and_span_maps(text, &sources, r);
        assert_eq!(r.edits().edits().count(), 4);
    }
}

#[test]
fn a_policy_sends_dates_here_and_other_kinds_elsewhere() {
    let text = "Born July 4, 1952, aged 97.";
    let pii = manual(&[(5, 17, PiiKind::Date, "1952-07-04"), (24, 26, PiiKind::Age, "97")]);
    let policy = RewritePolicy::new(MaskPolicy::new('#').unwrap())
        .with(PiiKind::Date, DateRewrite::year())
        .with(PiiKind::Age, AgeSafeHarbor::new());
    assert_eq!(policy.strategy(PiiKind::Date).kind(), PiiStrategyKind::DateYear);
    let r = policy.rewrite(&source_view(text), &pii).unwrap();
    assert_eq!(r.text(), "Born 1952, aged 90+.");
    // A mention of another kind sent to the date strategy is masked whole.
    let all = RewritePolicy::new(RewriteStrategy::from(DateRewrite::year()));
    assert_eq!(all.rewrite(&source_view(text), &pii).unwrap().text(), "Born 1952, aged **.");
}

#[test]
fn overlapping_mentions_are_masked_whole() {
    let text = "on 2026-07-14";
    let pii = manual(&[(3, 13, PiiKind::Date, "2026-07-14"), (0, 13, PiiKind::Phone, "on 2026-07-14")]);
    let year = DateRewrite::year();
    assert_eq!(year.rewrite(&source_view(text), &pii).unwrap().text(), "*************");
    let policy = RewritePolicy::new(RewriteStrategy::from(year.clone()));
    assert_eq!(policy.rewrite(&source_view(text), &pii).unwrap().text(), "*************");
    assert!(year.refusing_overlaps().rewrite(&source_view(text), &pii).is_err());
}

#[test]
fn the_config_hash_covers_the_mode_and_its_parameters() {
    let hash = |s: DateRewrite| {
        RedactedView::new(RewritePolicy::new(MaskPolicy::new('*').unwrap()).with(PiiKind::Date, s)).config_hash()
    };
    let base = hash(DateRewrite::year());
    assert_eq!(base, hash(DateRewrite::year()));
    assert_ne!(base, hash(DateRewrite::year().with_mask('#')));
    assert_ne!(base, hash(DateRewrite::year().refusing_overlaps()));
    let fixed = |y, m, day| hash(DateRewrite::shift(DateShift::fixed(y, m, day)));
    assert_ne!(fixed(0, 0, 1), fixed(0, 0, 2));
    assert_ne!(fixed(0, 1, 0), fixed(1, 0, 0));
    let keyed = |key: &[u8], c: &str| hash(DateRewrite::shift(DateShift::default().keyed(key, c, 30).unwrap()));
    assert_ne!(keyed(&KEY, "a"), keyed(&KEY, "b"));
    assert_ne!(keyed(&KEY, "a"), keyed(&[8; 32], "a"));
    assert_ne!(keyed(&KEY, "a"), fixed(0, 0, 0));
    assert_ne!(hash(DateRewrite::relative(d(2026, 1, 1))), hash(DateRewrite::relative(d(2026, 1, 2))));
    assert_eq!(DateRewrite::relative(d(2026, 1, 1)).mode(), &DateMode::Relative(d(2026, 1, 1)));
}
