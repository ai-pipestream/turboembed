//! The absolute and relative shapes: the cases of OpenNLP's
//! `CursorTemporalExtractorTest` and `CursorTemporalCalendarTest`.
//!
//! Spans are byte offsets where the Java's are UTF-16 units; the texts
//! are ASCII except where a case says otherwise. The Java cases that
//! check null arguments, a mention's own constructor and the default
//! locale have no counterpart: arguments cannot be null, a value is typed
//! (`Temporals::check` refuses one that is no date), and nothing here
//! reads a locale.

mod temporal_common;

use temporal_common::{absolute, analyze, day, found, relative, values};
use turbo_analysis::{Ix, Layer, Span};
use turbo_numeric::{Date, Granularity, Origin, TemporalExtractor, Temporals};

use Granularity::{Day, Month, Quarter, Week, Year};

/// The reference date every relative case resolves against; it falls in
/// ISO week 29.
fn reference() -> Option<Date> {
    Some(day("2026-07-17"))
}

/// The one mention in `text`, without a reference.
fn single(text: &str) -> (Span, String, Granularity, Origin) {
    let all = found(text, None);
    assert_eq!(all.len(), 1, "expected one mention in {text:?}: {all:?}");
    all.into_iter().next().unwrap()
}

#[test]
fn recognized_shapes_cover_the_full_mention() {
    for (text, value, g) in [
        ("2026-07-14", "2026-07-14", Day),
        ("July 14, 2026", "2026-07-14", Day),
        ("July 14 2026", "2026-07-14", Day),
        ("Jul 14 2026", "2026-07-14", Day),
        ("14 July 2026", "2026-07-14", Day),
        ("14th July 2026", "2026-07-14", Day),
        ("march 3rd, 1999", "1999-03-03", Day),
        ("July 2026", "2026-07", Month),
        ("SEPTEMBER 2024", "2024-09", Month),
        ("Q3 2024", "2024-Q3", Quarter),
        ("q1 1999", "1999-Q1", Quarter),
    ] {
        assert_eq!(single(text), absolute(0, text.len(), value, g), "{text}");
    }
}

/// The date of a timestamp is one day mention over the date alone.
#[test]
fn an_iso_timestamp_yields_the_date_at_day_granularity() {
    assert_eq!(found("2026-07-14T09:30:00Z", None), [absolute(0, 10, "2026-07-14", Day)]);
}

#[test]
fn an_iso_timestamp_with_an_offset_inside_a_sentence() {
    let text = "deployed at 2026-07-14T09:30+05:30 sharp";
    let (span, value, g, _) = single(text);
    assert_eq!((&text[span.range()], value.as_str(), g), ("2026-07-14", "2026-07-14", Day));
}

#[test]
fn a_mention_inside_a_sentence_has_its_exact_span() {
    let text = "the release shipped on 2026-07-14 after review";
    assert_eq!(&text[single(text).0.range()], "2026-07-14");
}

#[test]
fn several_mentions() {
    assert_eq!(values("from July 2025 to Q2 2026", None), ["2025-07", "2026-Q2"]);
}

#[test]
fn rejected_shapes() {
    for text in [
        "2026-13-01",         // no thirteenth month
        "2026-02-30",         // calendar-validated
        "2026-02-29",         // 2026 is not a leap year
        "Jul. 14, 2026",      // an abbreviation with a period
        "February 30, 2026",  // calendar-validated
        "July 32 2026",       // no thirty-second day
        "it may be 100",      // a month word without a calendar context
        "March onwards",      // a month word without a year
        "in 2026 alone",      // bare years
        "Q5 2024",            // no fifth quarter
        "Q3 24",              // two-digit years
        "2026-07",            // the ISO month form
        "12026-07-14",        // the year is part of a longer run of digits
        "2026-07-14Tomorrow", // a T followed by letters is no time of day
    ] {
        assert_eq!(found(text, None), [], "{text}");
    }
}

/// An impossible day is not repaired: the month and year after it are
/// the mention.
#[test]
fn an_impossible_day_leaves_only_the_month() {
    assert_eq!(found("31 April 2026", None), [absolute(3, 13, "2026-04", Month)]);
}

/// U+1F5D3 is four bytes (two UTF-16 units): a boundary, read whole.
#[test]
fn a_supplementary_neighbor_is_a_boundary() {
    assert_eq!(single("\u{1F5D3}2026-07-14"), absolute(4, 14, "2026-07-14", Day));
}

#[test]
fn the_annotator_writes_the_temporal_layer() {
    let doc = analyze(TemporalExtractor::new(), "earnings land on July 24, 2026 before Q3 2026 begins");
    let t = doc.get::<Temporals>().unwrap();
    assert_eq!(t.len(), 2);
    assert_eq!(t.value(Ix::new(0)).to_string(), "2026-07-24");
    assert_eq!(t.granularity[1], Quarter);
}

#[test]
fn relative_day_words_resolve_against_the_reference() {
    let all = found("filed Yesterday, due tomorrow, signed today", reference());
    assert_eq!(
        all,
        [relative(6, 15, "2026-07-16", Day), relative(21, 29, "2026-07-18", Day), relative(38, 43, "2026-07-17", Day)]
    );
}

#[test]
fn shifted_units_resolve_at_their_own_granularity() {
    let text = "last week we planned; next month it ships; last year it began; next quarter it scales";
    let all = found(text, reference());
    let got: Vec<(&str, Granularity)> = all.iter().map(|f| (f.1.as_str(), f.2)).collect();
    assert_eq!(got, [("2026-W28", Week), ("2026-08", Month), ("2025", Year), ("2026-Q4", Quarter)]);
    assert!(all.iter().all(|f| f.3 == Origin::Relative));
}

#[test]
fn counted_shifts_resolve_in_both_directions() {
    let text = "3 days ago it rained; in 2 weeks it ships; 14 months ago it started";
    let all = found(text, reference());
    let got: Vec<(&str, Granularity)> = all.iter().map(|f| (f.1.as_str(), f.2)).collect();
    assert_eq!(got, [("2026-07-14", Day), ("2026-W31", Week), ("2025-05", Month)]);
}

#[test]
fn relative_guard_rails() {
    assert_eq!(found("due yesterday and next month", None), []);
    assert_eq!(found("in cold water; last stand; 3 days late", reference()), []);
    assert_eq!(values("signed 2026-07-14, due tomorrow", reference()), ["2026-07-14", "2026-07-18"]);
}

/// Any `White_Space` character separates the parts as a space does. The
/// spans are bytes, so a wider gap moves the end.
#[test]
fn a_unicode_whitespace_gap_matches_like_a_space() {
    for gap in ["\u{A0}", "\u{202F}", "\u{2009}", "\u{3000}", "\t"] {
        for shape in [
            "July 14, 2026",
            "Jul 14 2026",
            "14 July 2026",
            "July 2026",
            "Q3 2024",
            "next month",
            "in 2 weeks",
            "3 days ago",
        ] {
            let expected = found(shape, reference());
            assert_eq!(expected.len(), 1, "{shape}");
            let text = shape.replace(' ', gap);
            let all = found(&text, reference());
            assert_eq!(all.len(), 1, "{text:?}");
            assert_eq!(all[0].0, Span::new(0, text.len() as u32), "{text:?}");
            assert_eq!((&all[0].1, all[0].2), (&expected[0].1, expected[0].2), "{text:?}");
        }
    }
}

// CursorTemporalCalendarTest

const MONTH_NAMES: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

fn ordinal(day: u8) -> &'static str {
    if (11..=13).contains(&day) {
        return "th";
    }
    match day % 10 {
        1 => "st",
        2 => "nd",
        3 => "rd",
        _ => "th",
    }
}

fn assert_day(text: &str, date: Date) {
    assert_eq!(found(text, None), [absolute(0, text.len(), &date.to_string(), Day)], "{text}");
}

/// Written dates follow the calendar, century leap years included.
#[test]
fn generated_calendar_dates() {
    for year in [1000, 1600, 1700, 1900, 2000, 2024, 2100, 2400, 2999] {
        for (m, name) in MONTH_NAMES.iter().enumerate() {
            for d in 1..=31u8 {
                let day_first = format!("{d} {name} {year}");
                let month_first = format!("{name} {d}, {year}");
                let Some(date) = Date::new(year, m as u8 + 1, d) else {
                    assert_eq!(found(&month_first, None), [], "{month_first}");
                    let month_start = day_first.find(name).unwrap();
                    let value = format!("{year}-{:02}", m + 1);
                    assert_eq!(
                        found(&day_first, None),
                        [absolute(month_start, day_first.len(), &value, Month)],
                        "{day_first}"
                    );
                    continue;
                };
                assert_day(&date.to_string(), date);
                assert_day(&day_first, date);
                assert_day(&month_first, date);
                let o = ordinal(d);
                assert_day(&format!("{d}{o} {name} {year}"), date);
                assert_day(&format!("{name} {d}{o}, {year}"), date);
                assert_day(&format!("{d}{} {name} {year}", o.to_uppercase()), date);
            }
        }
    }
}

/// Years outside 1000 to 2999 and dates that do not exist are not repaired.
#[test]
fn invalid_absolute_dates() {
    for text in [
        "0999-12-31",
        "3000-01-01",
        "January 0999",
        "January 3000",
        "Q1 0999",
        "Q4 3000",
        "2024-00-01",
        "2024-13-01",
        "2024-01-00",
        "2024-01-32",
        "February 0, 2024",
        "July 32, 2024",
    ] {
        assert_eq!(found(text, None), [], "{text}");
    }
}

/// An ordinal suffix that goes on with a digit, a mark or a letter is no
/// suffix, and the day before it is no day; the month and year are.
#[test]
fn ordinal_continuations() {
    for suffix in ["xx", "\u{17F}t", "st2", "st\u{301}", "st\u{10400}", "st_item"] {
        let text = format!("1{suffix} July 2026");
        let start = text.find("July").unwrap();
        assert_eq!(found(&text, None), [absolute(start, text.len(), "2026-07", Month)], "{text:?}");
    }
}

#[test]
fn month_names_in_any_case_keep_whole_spans() {
    for (text, date) in [
        ("29 FEB 2024", "2024-02-29"),
        ("29th fEb 2024", "2024-02-29"),
        ("FEB 29TH, 2024", "2024-02-29"),
        ("1st Jan 1000", "1000-01-01"),
        ("Dec 31st, 2999", "2999-12-31"),
    ] {
        assert_day(text, day(date));
    }
}

// Edge cases of the port.

/// U+0130 lowercases to `i` and the Kelvin sign to `k` with
/// `Character.toLowerCase`, so OpenNLP reads `İn` as `in` and `weeK` with
/// a Kelvin sign as `week`. Both are two or three bytes, which the spans
/// count.
#[test]
fn letters_outside_ascii_that_lowercase_into_a_keyword() {
    let r = reference();
    assert_eq!(found("\u{130}n 2 days", r), [relative(0, 10, "2026-07-19", Day)]);
    assert_eq!(found("next wee\u{212A}", r), [relative(0, 11, "2026-W30", Week)]);
    assert_eq!(found("3 wee\u{212A}s ago", r), [relative(0, 13, "2026-W26", Week)]);
    // No month or keyword starts with a letter that lowercases to `k`.
    assert_eq!(found("\u{212A}an 2026", None), []);
}

/// A letter outside the Basic Multilingual Plane is never the start of a
/// word, as the Java's `Character.isLetter` on a high surrogate is false,
/// but it ends the word before it: U+10400 is a letter.
#[test]
fn supplementary_letters() {
    assert_eq!(found("\u{10400}July 2026", None), []);
    assert_eq!(found("July\u{10400} 2026", None), []);
    assert_eq!(found("\u{10400} July 2026", None), [absolute(5, 14, "2026-07", Month)]);
}

/// A word of more than ten UTF-16 units is no word, and `September` is
/// nine: a suffix makes it no month.
#[test]
fn long_words_are_no_words() {
    assert_eq!(values("Septembers 2026", None), [] as [&str; 0]);
    assert_eq!(values("Septemberish 2026", None), [] as [&str; 0]);
    assert_eq!(values("in 2 quarters", reference()), ["2027-Q1"]);
}

/// After a mention the scan goes on at its end; after a failed candidate,
/// at the next character, so a mention can start inside one that failed.
#[test]
fn scanning_goes_on_after_a_failed_candidate() {
    // `14 July 2026` fails on the year and `July 20263` on its digits.
    assert_eq!(found("14 July 20263", None), []);
    // The day-first date fails, then `March 2026` is a month.
    assert_eq!(found("30 February March 2026", None), [absolute(12, 22, "2026-03", Month)]);
    // A digit that continues a number never starts a mention.
    assert_eq!(found("1.2026-07-14 and 1,14 July 2026", None), [absolute(22, 31, "2026-07", Month)]);
    assert_eq!(found("5e2026-07-14", None), []);
}

/// A gap is exactly one whitespace character.
#[test]
fn two_spaces_are_no_gap() {
    assert_eq!(found("July  2026", None), []);
    assert_eq!(found("Q3  2024", None), []);
    assert_eq!(found("next  month", reference()), []);
    assert_eq!(found("July\u{200B}2026", None), []);
}

/// The same text with and without a reference keeps its absolute mentions.
#[test]
fn absolute_mentions_do_not_depend_on_the_reference() {
    let text = "Filed 2026-07-10, amended last week; hearing on 2 Sept 2026 or Q4 2026.";
    let without: Vec<String> = values(text, None);
    let with: Vec<String> =
        found(text, reference()).into_iter().filter(|f| f.3 == Origin::Absolute).map(|f| f.1).collect();
    assert_eq!(without, with);
    assert_eq!(without, ["2026-07-10", "2026-Q4"]);
}
