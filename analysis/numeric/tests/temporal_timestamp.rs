//! Dates with a clock suffix: the cases of OpenNLP's
//! `CursorTemporalTimestampTest`.
//!
//! `testGeneratedOffsetTimestamps` writes its texts with the JDK's
//! `OffsetDateTime.toString`; here they are written the same way by
//! [`offset_date_time`], from this crate's own generator.

mod common;
mod temporal_common;

use common::Rng;
use temporal_common::{absolute, analyze, day, found};
use turbo_analysis::Ix;
use turbo_numeric::{Date, DocumentDate, Granularity, TemporalExtractor, Temporals};

const DATE: &str = "2026-07-14";

#[test]
fn incomplete_suffixes_drop_the_date() {
    for suffix in [
        "T25:00",
        "T99:00",
        "T12:60",
        "T12:99",
        "T24:01",
        "T24:00:01",
        "T24:00:00.1",
        "T24:00.1",
        "T12:30:61",
        "T12:30:99",
        "T09:30:00:01",
        "T09:30:1",
        "T09:30:000",
        "T09:30+",
        "T09:30-",
        "T09:30+05",
        "T09:30+5:30",
        "T09:30+05:3",
        "T09:30+05:300",
        "T09:30+24:00",
        "T09:30+05:60",
        "T09:30-99:00",
        "T09:30+05:30:00",
        "T09:30+05:30Z",
        "T09:30Z+05:30",
        "T09:30+05:30+01:00",
        "T09:30:00.1.2",
        "T09:30:00.1x",
        "T09:30:00.1_",
        "T09:30:00.1\u{301}",
        "T09:30:00.Z",
        "T09:30:00..1",
        "T09:30:00,12,3",
    ] {
        assert_eq!(found(&format!("{DATE}{suffix}"), None), [], "{suffix}");
    }
}

/// A complete suffix gives the date alone. The prefix holds U+1F4DD, four
/// bytes, so the date starts at byte 9 where the Java's span starts at
/// unit 7.
#[test]
fn complete_suffixes_give_the_date() {
    let prefix = "Log \u{1F4DD} ";
    for suffix in [
        "T00:00",
        "T23:59",
        "T09:30:00",
        "T23:59:59Z",
        "T24:00",
        "T24:00:00.000Z",
        "T09:30.5",
        "T09:30,5",
        "T09:30:00.125",
        "T09:30:00,125+05:30",
        "T12:00-00:00",
        "T12:00+23:59",
        "T09:30+05:30",
        "T09:30-04:00",
        "t09:30z",
        "T09:30:00.123456789012345678901234567890Z",
    ] {
        let text = format!("{prefix}{DATE}{suffix} recorded.");
        let start = prefix.len();
        assert_eq!(found(&text, None), [absolute(start, start + 10, DATE, Granularity::Day)], "{suffix}");
        assert_eq!(&text[start..start + 10], DATE);
    }
}

#[test]
fn sentence_punctuation_stays_apart() {
    for p in [".", ",", ":", ";", "!", "?", ")", "]"] {
        let text = format!("{DATE}T09:30:00Z{p} Next event.");
        assert_eq!(found(&text, None), [absolute(0, 10, DATE, Granularity::Day)], "{p}");
    }
}

/// A date dropped for its suffix does not date the document.
#[test]
fn an_invalid_timestamp_does_not_set_the_document_date() {
    let doc = analyze(TemporalExtractor::new(), "Bad 2026-07-14T99:99; good 2026-08-20T12:00Z; yesterday.");
    let t = doc.get::<Temporals>().unwrap();
    let values: Vec<String> = (0..2).map(|i| t.value(Ix::new(i)).to_string()).collect();
    assert_eq!(values, ["2026-08-20", "2026-08-19"]);
    assert_eq!(doc.get::<DocumentDate>().unwrap().date, Some(day("2026-08-20")));
}

#[test]
fn a_leap_second_keeps_the_date() {
    assert_eq!(found("1990-12-31T23:59:60Z", None), [absolute(0, 10, "1990-12-31", Granularity::Day)]);
}

/// Each clock field takes its range and refuses the larger two-digit values.
#[test]
fn clock_field_ranges() {
    type Field = (fn(u32) -> String, u32);
    let fields: [Field; 5] = [
        (|v| format!("T{v:02}:00:00Z"), 24),
        (|v| format!("T12:{v:02}:00Z"), 59),
        (|v| format!("T12:00:{v:02}Z"), 60),
        (|v| format!("T12:00:00+{v:02}:00"), 23),
        (|v| format!("T12:00:00+05:{v:02}"), 59),
    ];
    for (suffix, maximum) in fields {
        for value in 0..100 {
            let text = format!("{DATE}{}", suffix(value));
            assert_eq!(found(&text, None).len(), usize::from(value <= maximum), "{text}");
        }
    }
}

/// `OffsetDateTime.toString`: `HH:mm`, then `:ss` when the seconds or the
/// nanoseconds are not zero, then the nanoseconds in groups of three
/// digits, then `Z` or the offset as `+hh:mm`.
fn offset_date_time(date: Date, hour: u32, minute: u32, second: u32, nano: u32, offset_minutes: i32) -> String {
    let mut s = format!("{date}T{hour:02}:{minute:02}");
    if second > 0 || nano > 0 {
        s.push_str(&format!(":{second:02}"));
        let fraction = if nano == 0 {
            String::new()
        } else if nano.is_multiple_of(1_000_000) {
            format!(".{:03}", nano / 1_000_000)
        } else if nano.is_multiple_of(1000) {
            format!(".{:06}", nano / 1000)
        } else {
            format!(".{nano:09}")
        };
        s.push_str(&fraction);
    }
    if offset_minutes == 0 {
        s.push('Z');
    } else {
        let sign = if offset_minutes < 0 { '-' } else { '+' };
        let m = offset_minutes.unsigned_abs();
        s.push_str(&format!("{sign}{:02}:{:02}", m / 60, m % 60));
    }
    s
}

#[test]
fn generated_offset_timestamps() {
    // The JDK writes these from `testGeneratedOffsetTimestamps`' first draws.
    let base = day("2000-01-01");
    assert_eq!(offset_date_time(day("2452-04-02"), 0, 11, 1, 282_318_746, 690), "2452-04-02T00:11:01.282318746+11:30");
    assert_eq!(
        offset_date_time(day("2062-10-07"), 3, 58, 10, 521_564_239, -840),
        "2062-10-07T03:58:10.521564239-14:00"
    );
    let mut rng = Rng(20_260_905);
    for _ in 0..500 {
        let date = base.plus_days(rng.below(300_000) as i64).unwrap();
        let offset = (rng.below(145) as i32 - 72) * 15;
        let nano = match rng.below(4) {
            0 => 0,
            1 => rng.below(1000) as u32 * 1_000_000,
            2 => rng.below(1_000_000) as u32 * 1000,
            _ => rng.below(1_000_000_000) as u32,
        };
        let second = if rng.below(4) == 0 { 0 } else { rng.below(60) as u32 };
        let text = offset_date_time(date, rng.below(24) as u32, rng.below(60) as u32, second, nano, offset);
        assert_eq!(found(&text, None), [absolute(0, 10, &date.to_string(), Granularity::Day)], "{text}");
    }
}
