//! The mention shapes `CursorTemporalExtractor.matchAt` tries at one
//! position, and the relative arithmetic.
//!
//! The Java walks a `CharSequence` in UTF-16 units; this walks the same
//! characters in UTF-8 bytes. Every character a shape compares against is
//! ASCII except the one whitespace character between two parts, which is
//! always in the Basic Multilingual Plane and is stepped over whole
//! ([`gap`]), and the letters of a word, which [`word`] reads counting
//! UTF-16 units as the Java does. So both find the same mentions.

use turbo_analysis::Span;
use turbo_analysis_layers::{Date, Granularity, NumberNotation, Origin, TemporalMention, TemporalValue};
use turbo_unicode::java;

use super::clock::{digits, time_of_day_at};
use super::words::{Keyword, counted_unit, is_ago, is_ordinal, keyword, month, unit, word};
use crate::scan::{boundary_after, boundary_before, byte_at, char_at, continues_number, whitespace_at};

/// The years an absolute mention may have in OpenNLP parity mode, and by
/// default across four-digit years (`0001..=9999`).
const PARITY_MIN_YEAR: u32 = 1000;
const PARITY_MAX_YEAR: u32 = 2999;
const DEFAULT_MIN_YEAR: u32 = 1;
const DEFAULT_MAX_YEAR: u32 = 9999;

/// `matchAt`: the mention that starts at byte `start`, trying an ISO date,
/// a day-first date and then a counted relative form at a digit, and a
/// quarter, a month-first date and then a relative keyword at a letter.
/// The relative forms are tried only with a reference date.
pub(super) fn match_at(
    text: &str,
    start: usize,
    reference: Option<Date>,
    opennlp_parity: bool,
) -> Option<TemporalMention> {
    let c = char_at(text, start)?;
    let digit = c.is_ascii_digit();
    // In OpenNLP parity mode the Java tests one UTF-16 unit with
    // `Character.isLetter`, which is never true for half of a surrogate pair;
    // by default supplementary letters are treated as letters.
    let letter = if opennlp_parity { !java::is_supplementary(c) && java::is_letter(c) } else { java::is_letter(c) };
    if (!digit && !letter) || !boundary_before(text, start) {
        return None;
    }
    let b = text.as_bytes();
    if digit {
        if continues_number(text, start, NumberNotation::LatinUs) {
            return None;
        }
        return iso_date(b, text, start, opennlp_parity)
            .or_else(|| day_first(b, text, start, opennlp_parity))
            .or_else(|| count_ago(b, text, start, reference?, opennlp_parity));
    }
    if matches!(c, 'Q' | 'q') && byte_at(b, start + 1).is_ascii_digit() {
        return quarter(b, text, start, opennlp_parity);
    }
    month_first(b, text, start, opennlp_parity).or_else(|| relative_keyword(b, text, start, reference?, opennlp_parity))
}

/// `2026-07-14`, alone or as the date of a timestamp (`2026-07-14T09:30Z`),
/// whose span is the date's ten bytes.
fn iso_date(b: &[u8], text: &str, start: usize, opennlp_parity: bool) -> Option<TemporalMention> {
    let year = digits(b, start, 4)?;
    if byte_at(b, start + 4) != b'-' {
        return None;
    }
    let month = digits(b, start + 5, 2)?;
    if byte_at(b, start + 7) != b'-' {
        return None;
    }
    let day = digits(b, start + 8, 2)?;
    let end = start + 10;
    if !boundary_after(text, end) && !time_of_day_at(b, text, end) {
        return None;
    }
    day_mention(start, end, year, month, day, opennlp_parity)
}

/// `14 July 2026` and `14th July 2026`.
fn day_first(b: &[u8], text: &str, start: usize, opennlp_parity: bool) -> Option<TemporalMention> {
    let (day, day_end) = short_number(b, start)?;
    let month_start = gap(text, skip_ordinal(text, day_end))?;
    let month_end = word(text, month_start)?;
    let month = month(&text[month_start..month_end])?;
    let (year, end) = year_at(b, text, gap(text, month_end)?, opennlp_parity)?;
    day_mention(start, end, year, u32::from(month), day, opennlp_parity)
}

/// `July 2026`, `July 14, 2026`, `Jul 14 2026` and `July 14th, 2026`.
fn month_first(b: &[u8], text: &str, start: usize, opennlp_parity: bool) -> Option<TemporalMention> {
    let month_end = word(text, start)?;
    let month = month(&text[start..month_end])?;
    let after = gap(text, month_end)?;
    if let Some((year, end)) = year_at(b, text, after, opennlp_parity) {
        let value = TemporalValue::Month { year: year as i32, month };
        return Some(absolute(start, end, value));
    }
    let (day, day_end) = short_number(b, after)?;
    let mut i = skip_ordinal(text, day_end);
    if byte_at(b, i) == b',' {
        i += 1;
    }
    let (year, end) = year_at(b, text, gap(text, i)?, opennlp_parity)?;
    day_mention(start, end, year, u32::from(month), day, opennlp_parity)
}

/// `Q3 2024`; the caller has seen the `Q` and a digit.
fn quarter(b: &[u8], text: &str, start: usize, opennlp_parity: bool) -> Option<TemporalMention> {
    let quarter = byte_at(b, start + 1) - b'0';
    if !(1..=4).contains(&quarter) {
        return None;
    }
    let (year, end) = year_at(b, text, gap(text, start + 2)?, opennlp_parity)?;
    Some(absolute(start, end, TemporalValue::Quarter { year: year as i32, quarter }))
}

/// `today`, `yesterday` and `tomorrow`; `last`, `this` or `next` and a
/// unit; `in`, a count and a unit, plural or not.
fn relative_keyword(
    b: &[u8],
    text: &str,
    start: usize,
    reference: Date,
    opennlp_parity: bool,
) -> Option<TemporalMention> {
    let keyword_end = word(text, start)?;
    let keyword = keyword(&text[start..keyword_end])?;
    let steps = match keyword {
        Keyword::Today => return shifted(start, keyword_end, Granularity::Day, 0, reference),
        Keyword::Yesterday => return shifted(start, keyword_end, Granularity::Day, -1, reference),
        Keyword::Tomorrow => return shifted(start, keyword_end, Granularity::Day, 1, reference),
        Keyword::Last => -1,
        Keyword::This => 0,
        Keyword::Next => 1,
        Keyword::In => {
            let (count, count_end) = count_number(b, gap(text, keyword_end)?, opennlp_parity)?;
            let unit_start = gap(text, count_end)?;
            let unit_end = word(text, unit_start)?;
            let granularity = counted_unit(&text[unit_start..unit_end])?;
            return shifted(start, unit_end, granularity, count, reference);
        }
    };
    let unit_start = gap(text, keyword_end)?;
    let unit_end = word(text, unit_start)?;
    shifted(start, unit_end, unit(&text[unit_start..unit_end], opennlp_parity)?, steps, reference)
}

/// `3 days ago` and the same with weeks, months, quarters and years.
fn count_ago(b: &[u8], text: &str, start: usize, reference: Date, opennlp_parity: bool) -> Option<TemporalMention> {
    let (count, count_end) = count_number(b, start, opennlp_parity)?;
    let unit_start = gap(text, count_end)?;
    let unit_end = word(text, unit_start)?;
    let ago_start = gap(text, unit_end)?;
    let ago_end = word(text, ago_start)?;
    if !is_ago(&text[ago_start..ago_end]) {
        return None;
    }
    let granularity = counted_unit(&text[unit_start..unit_end])?;
    shifted(start, ago_end, granularity, -count, reference)
}

/// `shifted`: the reference moved by `steps` of the unit, at the unit's
/// granularity, or `None` when the date leaves the `LocalDate` range. A
/// quarter is three months.
fn shifted(start: usize, end: usize, granularity: Granularity, steps: i64, reference: Date) -> Option<TemporalMention> {
    let date = match granularity {
        Granularity::Day => reference.plus_days(steps),
        Granularity::Week => reference.plus_weeks(steps),
        Granularity::Month => reference.plus_months(steps),
        Granularity::Quarter => reference.plus_months(3 * steps),
        Granularity::Year => reference.plus_years(steps),
    }?;
    Some(TemporalMention {
        span: span(start, end),
        value: TemporalValue::of_date(date, granularity),
        origin: Origin::Relative,
    })
}

/// `day`: an absolute day mention, when the year is in range (`0001..=9999`
/// by default, `1000..=2999` in OpenNLP parity mode) and the date exists.
fn day_mention(
    start: usize,
    end: usize,
    year: u32,
    month: u32,
    day: u32,
    opennlp_parity: bool,
) -> Option<TemporalMention> {
    if !year_in_range(year, opennlp_parity) {
        return None;
    }
    // Month and day are at most two digits, so they fit.
    let date = Date::new(year as i32, month as u8, day as u8)?;
    Some(absolute(start, end, TemporalValue::Day(date)))
}

fn absolute(start: usize, end: usize, value: TemporalValue) -> TemporalMention {
    TemporalMention { span: span(start, end), value, origin: Origin::Absolute }
}

/// Offsets fit in `u32`: the scan's caller checks the text's length.
fn span(start: usize, end: usize) -> Span {
    Span::new(start as u32, end as u32)
}

fn year_in_range(year: u32, opennlp_parity: bool) -> bool {
    if opennlp_parity {
        (PARITY_MIN_YEAR..=PARITY_MAX_YEAR).contains(&year)
    } else {
        (DEFAULT_MIN_YEAR..=DEFAULT_MAX_YEAR).contains(&year)
    }
}

/// `yearAt`: four digits in range, not part of a longer run of digits and
/// followed by a word boundary.
fn year_at(b: &[u8], text: &str, start: usize, opennlp_parity: bool) -> Option<(u32, usize)> {
    let year = digits(b, start, 4)?;
    let end = start + 4;
    (year_in_range(year, opennlp_parity) && boundary_after(text, end)).then_some((year, end))
}

/// `shortNumber`: one or two digits, not part of a longer run.
fn short_number(b: &[u8], start: usize) -> Option<(u32, usize)> {
    let run = b.get(start..).unwrap_or_default().iter().take(3).take_while(|d| d.is_ascii_digit()).count();
    if !(1..=2).contains(&run) {
        return None;
    }
    let value = b[start..start + run].iter().fold(0, |v, &d| v * 10 + u32::from(d - b'0'));
    Some((value, start + run))
}

/// Reads a relative count: `1..=9` digits by default (`1..=2` digits in
/// OpenNLP parity mode), not part of a longer run of digits.
fn count_number(b: &[u8], start: usize, opennlp_parity: bool) -> Option<(i64, usize)> {
    let max_digits = if opennlp_parity { 2 } else { 9 };
    let run = b.get(start..).unwrap_or_default().iter().take(max_digits + 1).take_while(|d| d.is_ascii_digit()).count();
    if !(1..=max_digits).contains(&run) {
        return None;
    }
    let value = b[start..start + run].iter().fold(0i64, |v, &d| v * 10 + i64::from(d - b'0'));
    Some((value, start + run))
}

/// `skipOrdinal`: the end of an ordinal suffix word at `index` (`st`,
/// `nd`, `rd`, `th`), or `index` when there is none.
fn skip_ordinal(text: &str, index: usize) -> usize {
    match word(text, index) {
        Some(end) if is_ordinal(&text[index..end]) => end,
        _ => index,
    }
}

/// The byte after the one whitespace character at `i` that separates two
/// parts of a mention, or `None` when there is none. Past the end of the
/// text the Java reads a space and steps one unit; the part expected
/// after it then finds nothing.
fn gap(text: &str, i: usize) -> Option<usize> {
    whitespace_at(text, i).then(|| i + char_at(text, i).map_or(1, char::len_utf8))
}
