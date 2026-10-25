//! Relative arithmetic and its output across the `LocalDate` range: the
//! cases of OpenNLP's `CursorTemporalRangeTest`.
//!
//! `testGeneratedCalendarShifts` draws its cases from `java.util.Random`
//! and checks them against the JDK's own arithmetic; the cases here are
//! the first ten of each of its five runs, with the values the JVM gave.
//! The locale cases have no counterpart, since nothing here reads a
//! locale; their values are checked as written.

mod temporal_common;

use temporal_common::{analyze, day, found, relative};
use turbo_analysis::Ix;
use turbo_numeric::{Date, DocumentDate, Granularity, Origin, TemporalExtractor, Temporals};

use Granularity::{Day, Month, Quarter, Week, Year};

/// Checks the one mention of `text` resolved against `reference`: its
/// value, granularity, origin and whole span.
#[track_caller]
fn assert_relative(text: &str, reference: Date, expected: &str, g: Granularity) {
    assert_eq!(found(text, Some(reference)), [relative(0, text.len(), expected, g)], "{text} at {reference}");
}

/// Years have at least four digits, with a sign below zero and past 9999.
#[test]
fn normalized_years() {
    for (year, expected) in [
        (-999_999_999, "-999999999"),
        (-10000, "-10000"),
        (-9999, "-9999"),
        (-1000, "-1000"),
        (-999, "-0999"),
        (-10, "-0010"),
        (-1, "-0001"),
        (0, "0000"),
        (1, "0001"),
        (9, "0009"),
        (99, "0099"),
        (999, "0999"),
        (1000, "1000"),
        (2026, "2026"),
        (9999, "9999"),
        (10000, "+10000"),
        (999_999_999, "+999999999"),
    ] {
        let reference = Date::new(year, 1, 4).unwrap();
        assert_relative("today", reference, &format!("{expected}-01-04"), Day);
        assert_relative("this week", reference, &format!("{expected}-W01"), Week);
        assert_relative("this month", reference, &format!("{expected}-01"), Month);
        assert_relative("this quarter", reference, &format!("{expected}-Q1"), Quarter);
        assert_relative("this year", reference, expected, Year);
    }
}

/// An expression whose date leaves the range gives nothing, and no panic.
#[test]
fn out_of_range_expressions() {
    for (maximum, text) in [
        (false, "yesterday"),
        (false, "last day"),
        (false, "last week"),
        (false, "last month"),
        (false, "last quarter"),
        (false, "last year"),
        (false, "1 day ago"),
        (false, "99 days ago"),
        (false, "99 weeks ago"),
        (false, "99 months ago"),
        (false, "99 quarters ago"),
        (false, "99 years ago"),
        (true, "tomorrow"),
        (true, "next day"),
        (true, "next week"),
        (true, "next month"),
        (true, "next quarter"),
        (true, "next year"),
        (true, "in 1 day"),
        (true, "in 99 days"),
        (true, "in 99 weeks"),
        (true, "in 99 months"),
        (true, "in 99 quarters"),
        (true, "in 99 years"),
    ] {
        let reference = if maximum { Date::MAX } else { Date::MIN };
        assert_eq!(found(text, Some(reference)), [], "{text} at {reference}");
    }
}

/// Unshifted units hold at both ends of the range. The JVM's
/// `ISO_WEEK_DATE` writes `LocalDate.MIN` as `-999999999-W01-1` and
/// `LocalDate.MAX` as `+999999999-W52-5`.
#[test]
fn reference_date_limits() {
    for (reference, week, quarter) in
        [(Date::MIN, "-999999999-W01", "-999999999-Q1"), (Date::MAX, "+999999999-W52", "+999999999-Q4")]
    {
        let iso = reference.to_string();
        let year = &iso[..iso.len() - 6];
        assert_relative("today", reference, &iso, Day);
        assert_relative("this day", reference, &iso, Day);
        assert_relative("in 0 days", reference, &iso, Day);
        assert_relative("0 days ago", reference, &iso, Day);
        assert_relative("this week", reference, week, Week);
        assert_relative("this month", reference, &iso[..iso.len() - 3], Month);
        assert_relative("this quarter", reference, quarter, Quarter);
        assert_relative("this year", reference, year, Year);
    }
}

/// A relative date out of range does not stop the other mentions or the
/// document date.
#[test]
fn the_pipeline_goes_on_after_an_out_of_range_date() {
    for (reference, shifted) in [(Date::MIN, "yesterday"), (Date::MAX, "tomorrow")] {
        let text = format!("today; {shifted}; 14 July 2026");
        let doc = analyze(TemporalExtractor::with_reference(reference), &text);
        let t = doc.get::<Temporals>().unwrap();
        assert_eq!(
            temporal_common::rows(t),
            [
                relative(0, 5, &reference.to_string(), Day),
                temporal_common::absolute(text.len() - 12, text.len(), "2026-07-14", Day)
            ]
        );
        assert_eq!(doc.get::<DocumentDate>().unwrap().date, Some(day("2026-07-14")));
    }
}

/// Leap days, week-based years and the years around zero.
#[test]
fn calendar_transitions() {
    for (date, text, expected, g) in [
        ("2024-03-01", "1 day ago", "2024-02-29", Day),
        ("2024-02-28", "in 1 day", "2024-02-29", Day),
        ("2024-02-29", "1 year ago", "2023", Year),
        ("2024-01-31", "in 1 month", "2024-02", Month),
        ("2021-01-01", "this week", "2020-W53", Week),
        ("2021-01-04", "last week", "2020-W53", Week),
        ("2020-12-31", "next week", "2021-W01", Week),
        ("2026-12-31", "next quarter", "2027-Q1", Quarter),
        ("0000-01-01", "yesterday", "-0001-12-31", Day),
        ("-0001-12-31", "tomorrow", "0000-01-01", Day),
        ("9999-12-31", "tomorrow", "+10000-01-01", Day),
        ("1000-01-01", "last year", "0999", Year),
    ] {
        assert_relative(text, day(date), expected, g);
    }
}

/// Counted shifts across the range, with the values the JVM computed.
#[test]
fn generated_calendar_shifts() {
    const CASES: [(Granularity, &str, i32, &str); 50] = [
        (Granularity::Day, "-863747151-11-24", -16, "-863747151-11-08"),
        (Granularity::Day, "+900988459-12-16", -66, "+900988459-10-11"),
        (Granularity::Day, "+813988476-12-19", 89, "+813988477-03-18"),
        (Granularity::Day, "-246692286-09-03", 62, "-246692286-11-04"),
        (Granularity::Day, "+670129549-10-27", 44, "+670129549-12-10"),
        (Granularity::Day, "+208610954-04-21", 50, "+208610954-06-10"),
        (Granularity::Day, "+553990973-05-30", 88, "+553990973-08-26"),
        (Granularity::Day, "-710232761-04-10", -13, "-710232761-03-28"),
        (Granularity::Day, "-332421788-10-21", -80, "-332421788-08-02"),
        (Granularity::Day, "+705047213-06-14", -9, "+705047213-06-05"),
        (Granularity::Week, "-863747151-11-24", -16, "-863747151-W31"),
        (Granularity::Week, "+900988459-12-16", -66, "+900988458-W37"),
        (Granularity::Week, "+813988476-12-19", 89, "+813988478-W35"),
        (Granularity::Week, "-246692286-09-03", 62, "-246692285-W46"),
        (Granularity::Week, "+670129549-10-27", 44, "+670129550-W35"),
        (Granularity::Week, "+208610954-04-21", 50, "+208610955-W14"),
        (Granularity::Week, "+553990973-05-30", 88, "+553990975-W05"),
        (Granularity::Week, "-710232761-04-10", -13, "-710232761-W01"),
        (Granularity::Week, "-332421788-10-21", -80, "-332421789-W15"),
        (Granularity::Week, "+705047213-06-14", -9, "+705047213-W15"),
        (Granularity::Month, "-863747151-11-24", -16, "-863747152-07"),
        (Granularity::Month, "+900988459-12-16", -66, "+900988454-06"),
        (Granularity::Month, "+813988476-12-19", 89, "+813988484-05"),
        (Granularity::Month, "-246692286-09-03", 62, "-246692281-11"),
        (Granularity::Month, "+670129549-10-27", 44, "+670129553-06"),
        (Granularity::Month, "+208610954-04-21", 50, "+208610958-06"),
        (Granularity::Month, "+553990973-05-30", 88, "+553990980-09"),
        (Granularity::Month, "-710232761-04-10", -13, "-710232762-03"),
        (Granularity::Month, "-332421788-10-21", -80, "-332421794-02"),
        (Granularity::Month, "+705047213-06-14", -9, "+705047212-09"),
        (Granularity::Quarter, "-863747151-11-24", -16, "-863747155-Q4"),
        (Granularity::Quarter, "+900988459-12-16", -66, "+900988443-Q2"),
        (Granularity::Quarter, "+813988476-12-19", 89, "+813988499-Q1"),
        (Granularity::Quarter, "-246692286-09-03", 62, "-246692270-Q1"),
        (Granularity::Quarter, "+670129549-10-27", 44, "+670129560-Q4"),
        (Granularity::Quarter, "+208610954-04-21", 50, "+208610966-Q4"),
        (Granularity::Quarter, "+553990973-05-30", 88, "+553990995-Q2"),
        (Granularity::Quarter, "-710232761-04-10", -13, "-710232764-Q1"),
        (Granularity::Quarter, "-332421788-10-21", -80, "-332421808-Q4"),
        (Granularity::Quarter, "+705047213-06-14", -9, "+705047211-Q1"),
        (Granularity::Year, "-863747151-11-24", -16, "-863747167"),
        (Granularity::Year, "+900988459-12-16", -66, "+900988393"),
        (Granularity::Year, "+813988476-12-19", 89, "+813988565"),
        (Granularity::Year, "-246692286-09-03", 62, "-246692224"),
        (Granularity::Year, "+670129549-10-27", 44, "+670129593"),
        (Granularity::Year, "+208610954-04-21", 50, "+208611004"),
        (Granularity::Year, "+553990973-05-30", 88, "+553991061"),
        (Granularity::Year, "-710232761-04-10", -13, "-710232774"),
        (Granularity::Year, "-332421788-10-21", -80, "-332421868"),
        (Granularity::Year, "+705047213-06-14", -9, "+705047204"),
    ];
    for (g, reference, count, expected) in CASES {
        let unit = match g {
            Day => "days",
            Week => "weeks",
            Month => "months",
            Quarter => "quarters",
            Year => "years",
        };
        let text = if count < 0 { format!("{} {unit} ago", -count) } else { format!("in {count} {unit}") };
        assert_relative(&text, day(reference), expected, g);
    }
}

/// The values of `testLocaleIndependentFormatting`, which no locale
/// changes here.
#[test]
fn values_are_ascii_iso_8601() {
    assert_relative("today", day("-0001-01-04"), "-0001-01-04", Day);
    assert_relative("this week", day("2021-01-01"), "2020-W53", Week);
    assert_relative("this quarter", day("+10000-01-04"), "+10000-Q1", Quarter);
    assert_eq!(found("January 2026", None)[0].1, "2026-01");
}

/// One extractor shared between threads resolves each text against its
/// own reference.
#[test]
fn concurrent_references() {
    let years: Vec<i32> = (-100..=100).collect();
    std::thread::scope(|s| {
        for chunk in years.chunks(50) {
            s.spawn(move || {
                for &year in chunk {
                    let reference = Date::new(year, 1, 4).unwrap();
                    let doc = analyze(TemporalExtractor::with_reference(reference), "today, last year");
                    let t = doc.get::<Temporals>().unwrap();
                    assert_eq!(t.value(Ix::new(0)).to_string(), reference.to_string());
                    let previous = reference.plus_years(-1).unwrap().to_string();
                    assert_eq!(t.value(Ix::new(1)).to_string(), previous[..previous.len() - 6]);
                    assert!(t.origin.iter().all(|&o| o == Origin::Relative));
                }
            });
        }
    });
}

// Edge cases of the port.

/// `LocalDate.plusMonths` and `plusYears` move a day past the end of the
/// new month back to its last day; a quarter is three months.
#[test]
fn month_ends_clamp() {
    assert_relative("in 1 month", day("2023-01-31"), "2023-02", Month);
    assert_eq!(day("2023-01-31").plus_months(1), Some(day("2023-02-28")));
    assert_eq!(day("2024-02-29").plus_years(4), Some(day("2028-02-29")));
    assert_eq!(day("2024-02-29").plus_years(1), Some(day("2025-02-28")));
    assert_eq!(day("2026-05-31").plus_months(-3), Some(day("2026-02-28")));
    assert_relative("last quarter", day("2026-05-31"), "2026-Q1", Quarter);
    assert_eq!(day("-0003-03-31").plus_months(-13), Some(day("-0004-02-29")));
    assert_eq!(day("-0001-03-31").plus_months(-13), Some(day("-0002-02-28")));
}

/// Week numbers across years whose first Thursday falls on each weekday.
#[test]
fn iso_weeks() {
    for (date, week) in [
        ("2015-12-31", "2015-W53"),
        ("2016-01-03", "2015-W53"),
        ("2016-01-04", "2016-W01"),
        ("2018-12-31", "2019-W01"),
        ("2019-12-30", "2020-W01"),
        ("2020-12-31", "2020-W53"),
        ("2027-01-03", "2026-W53"),
        ("2032-12-31", "2032-W53"),
        ("2004-12-31", "2004-W53"),
        ("2000-01-02", "1999-W52"),
        ("0000-01-01", "-0001-W52"),
    ] {
        assert_relative("this week", day(date), week, Week);
    }
}

/// Without a reference no relative expression is read, so `3 days ago`
/// alone is nothing; multi-digit counts up to nine digits are read by
/// default.
#[test]
fn counts_have_up_to_nine_digits() {
    let r = Some(day("2026-07-17"));
    assert_eq!(found("100 days ago", r), [relative(0, 12, "2026-04-08", Day)]);
    assert_eq!(found("in 100 days", r), [relative(0, 11, "2026-10-25", Day)]);
    assert_eq!(found("in 07 days", r), [relative(0, 10, "2026-07-24", Day)]);
    assert_eq!(found("3 days ago", None), []);
}
