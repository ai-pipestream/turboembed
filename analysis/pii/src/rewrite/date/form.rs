//! Reading a date mention's value and writing another value in the same
//! written form.
//!
//! A `date` mention is one of the absolute shapes the temporal scanner of
//! turbo-numeric finds: `2026-07-14`, `14 July 2026`, `14th July 2026`,
//! `July 14, 2026`, `Jul 14 2026`, `July 14th, 2026` and `July 2026`. Its
//! text is read again with that scanner, so the value is the one the
//! detector saw, and split into runs of ASCII digits, runs of letters and
//! everything else. Writing a new value replaces each run that holds a
//! field and keeps every other character as written.

use turbo_analysis::ViewId;
use turbo_analysis_layers::{Date, Origin, TemporalValue, Temporals};

/// Full English month names; the scanner also reads their first three
/// letters.
const MONTHS: [&str; 12] = [
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

/// The value a date mention states: a day, or a month of a year.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Stated {
    Day(Date),
    Month { year: i32, month: u8 },
}

impl Stated {
    pub(super) fn year(self) -> i32 {
        match self {
            Stated::Day(d) => d.year(),
            Stated::Month { year, .. } => year,
        }
    }
}

/// The value of `text` when the whole of it is one absolute day or month
/// mention, else `None`.
pub(super) fn read(text: &str) -> Option<Stated> {
    let mut found = Temporals::new(ViewId::SOURCE);
    turbo_numeric::temporal::scan(text, None, &mut found);
    let mut it = found.spans.iter();
    let (i, _) = it.next()?;
    if it.next().is_some() {
        return None;
    }
    let m = found.get(i);
    if m.origin != Origin::Absolute || m.span.start != 0 || m.span.end as usize != text.len() {
        return None;
    }
    match m.value {
        TemporalValue::Day(d) => Some(Stated::Day(d)),
        TemporalValue::Month { year, month } => Some(Stated::Month { year, month }),
        TemporalValue::Week { .. } | TemporalValue::Quarter { .. } | TemporalValue::Year(_) => None,
    }
}

/// One run of `text`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Run<'a> {
    Digits(&'a str),
    Letters(&'a str),
    Other(&'a str),
}

fn runs(text: &str) -> Vec<Run<'_>> {
    // 0 for an ASCII digit, 1 for a letter, 2 for anything else; a run of
    // anything else is one character long.
    let class = |c: char| u8::from(!c.is_ascii_digit()) + u8::from(!c.is_ascii_digit() && !c.is_alphabetic());
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        let k = class(c);
        let len = if k == 2 { c.len_utf8() } else { rest.find(|n| class(n) != k).unwrap_or(rest.len()) };
        let (piece, tail) = rest.split_at(len);
        out.push(match k {
            0 => Run::Digits(piece),
            1 => Run::Letters(piece),
            _ => Run::Other(piece),
        });
        rest = tail;
    }
    out
}

/// Whether `word` is an ordinal suffix, in any case.
fn is_ordinal(word: &str) -> bool {
    ["st", "nd", "rd", "th"].iter().any(|s| word.eq_ignore_ascii_case(s))
}

/// The month `word` names, full or by its first three letters, in any case.
fn month_of(word: &str) -> Option<usize> {
    MONTHS.iter().position(|m| word.eq_ignore_ascii_case(m) || word.eq_ignore_ascii_case(&m[..3]))
}

/// `word`'s case applied to `name`: all capitals, a capital first, or all
/// small letters.
fn cased(word: &str, name: &str, out: &mut String) {
    let upper = word.chars().all(|c| !c.is_lowercase());
    let first = word.chars().next().is_some_and(char::is_uppercase);
    if upper && word.chars().count() > 1 {
        out.push_str(&name.to_ascii_uppercase());
    } else if first {
        let mut cs = name.chars();
        out.extend(cs.next().map(|c| c.to_ascii_uppercase()));
        out.push_str(&cs.as_str().to_ascii_lowercase());
    } else {
        out.push_str(&name.to_ascii_lowercase());
    }
}

/// The English ordinal suffix of `day`: `st`, `nd`, `rd` or `th`.
fn suffix(day: u8) -> &'static str {
    match (day % 10, day % 100) {
        (_, 11..=13) => "th",
        (1, _) => "st",
        (2, _) => "nd",
        (3, _) => "rd",
        _ => "th",
    }
}

/// Appends `value` to `out` in the written form of `text`, a mention whose
/// value is `was`. `false`, with nothing appended, when the new year has
/// no four-digit form (before 1 or after 9999) or the forms differ in
/// granularity.
pub(super) fn write_like(text: &str, was: Stated, value: Stated, out: &mut String) -> bool {
    if !(1..=9999).contains(&value.year()) || std::mem::discriminant(&was) != std::mem::discriminant(&value) {
        return false;
    }
    let (month, day) = match value {
        Stated::Day(d) => (d.month(), Some(d.day())),
        Stated::Month { month, .. } => (month, None),
    };
    let mut piece = String::with_capacity(text.len() + 4);
    let mut digits_seen = 0;
    let rs = runs(text);
    let iso = matches!(rs.first(), Some(Run::Digits(d)) if d.len() == 4);
    for run in rs {
        match run {
            Run::Digits(d) if d.len() == 4 => {
                piece.push_str(&format!("{:04}", value.year()));
            }
            Run::Digits(_) if iso => {
                // `yyyy-MM-dd`: the month, then the day, two digits each.
                digits_seen += 1;
                let v = if digits_seen == 1 { month } else { day.unwrap_or(1) };
                piece.push_str(&format!("{v:02}"));
            }
            Run::Digits(d) => {
                let v = day.unwrap_or(1);
                // A day written with a leading zero keeps two digits.
                if d.starts_with('0') { piece.push_str(&format!("{v:02}")) } else { piece.push_str(&v.to_string()) }
            }
            Run::Letters(w) if is_ordinal(w) => cased(w, suffix(day.unwrap_or(1)), &mut piece),
            Run::Letters(w) => match month_of(w) {
                Some(_) => {
                    let name = MONTHS[usize::from(month) - 1];
                    let short = w.chars().count() == 3;
                    cased(w, if short { &name[..3] } else { name }, &mut piece);
                }
                None => piece.push_str(w),
            },
            Run::Other(o) => piece.push_str(o),
        }
    }
    out.push_str(&piece);
    true
}

/// The year of a mention read as `was`, in four digits.
pub(super) fn year_digits(was: Stated) -> String {
    format!("{:04}", was.year())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day(y: i32, m: u8, d: u8) -> Stated {
        Stated::Day(Date::new(y, m, d).unwrap())
    }

    fn like(text: &str, value: Stated) -> String {
        let was = read(text).unwrap_or_else(|| panic!("{text:?} is a date"));
        let mut out = String::new();
        assert!(write_like(text, was, value, &mut out), "{text:?}");
        out
    }

    #[test]
    fn every_scanned_shape_reads_and_keeps_its_form() {
        let v = day(2025, 3, 2);
        assert_eq!(like("2026-07-14", v), "2025-03-02");
        assert_eq!(like("14 July 2026", v), "2 March 2025");
        assert_eq!(like("14th July 2026", v), "2nd March 2025");
        assert_eq!(like("July 14, 2026", v), "March 2, 2025");
        assert_eq!(like("Jul 14 2026", v), "Mar 2 2025");
        assert_eq!(like("JULY 14TH, 2026", v), "MARCH 2ND, 2025");
        assert_eq!(like("july 04, 2026", v), "march 02, 2025");
        assert_eq!(like("July 2026", Stated::Month { year: 2025, month: 11 }), "November 2025");
    }

    #[test]
    fn ordinals_follow_the_new_day() {
        for (d, s) in [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12, "12th"), (13, "13th")] {
            assert_eq!(like("14th July 2026", day(2026, 7, d)), format!("{s} July 2026"));
        }
        for (d, s) in [(21, "21st"), (22, "22nd"), (23, "23rd"), (31, "31st")] {
            assert_eq!(like("14th July 2026", day(2026, 7, d)), format!("{s} July 2026"));
        }
    }

    #[test]
    fn what_is_not_one_whole_mention_reads_as_nothing() {
        for text in ["2026", "Q3 2026", "7/14/26", "July 14, 26", "the 2026-07-14", "2026-07-14 and 2026-07-15", ""] {
            assert_eq!(read(text), None, "{text:?}");
        }
    }

    #[test]
    fn a_year_with_no_four_digit_form_or_another_granularity_writes_nothing() {
        let was = read("2026-07-14").unwrap();
        let mut out = String::new();
        assert!(!write_like("2026-07-14", was, Stated::Month { year: 2026, month: 1 }, &mut out));
        assert!(!write_like("2026-07-14", was, day(10_000, 1, 1), &mut out));
        assert!(out.is_empty());
        assert_eq!(year_digits(was), "2026");
    }
}
