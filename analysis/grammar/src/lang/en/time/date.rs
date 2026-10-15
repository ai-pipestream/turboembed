//! Dates and periods: `May twenty nine`, `22/04`, `2016-11-07`,
//! `Friday`, `next Monday`, `the 18th`, `April 2017`, `this week`, `Q3
//! 2019`, `the 1990s`, `the last week of July`, holidays and seasons.

use turbo_analysis_layers::{Modifier, Portion, TimeGrain};

use crate::engine::{Ctx, Item, Kind, Rule, glued, glued_word, node, opt_word, word};
use crate::time::TimeValue;
use crate::time::spec::{DateSpec, Holiday, MonthRef, Point, Season, YearKind, YearRef};
use crate::value::Value;

use super::super::{INT, NUMERAL, ORD, ORD_SUFFIX};
use super::clock::{dig_len, dig_zero};
use super::lexicon::{self, rel};
use super::{
    ABBR, DATE, DAYWORD, DECADEW, DIG, DOM, HOL, HOLIDAY, MD, MDAY, MONTH, MONTHD, ORDINAL, PERIOD, PLURAL, QTR, RELW,
    SEASON, SEP, SHORT, SPELLED, TONIGHT, UNIT, WD, WEEKDAY, YEAR, date_of, emit, emit_date, r, term, term_value,
    year_ref,
};

// Years.

fn b_year_digits(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (y, _) = term(&v[0]);
    (dig_len(&v[0]) == 4 && !dig_zero(&v[0]) && (1000..3000).contains(&y)).then(|| term_value(y, 0))?
}

fn b_year_short(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    (dig_len(&v[0]) == 2).then(|| term_value(term(&v[0]).0, SHORT))?
}

fn spelled(v: &Value) -> Option<i64> {
    let n = v.num()?;
    (n.has(crate::value::flag::SPELLED) && !n.has(crate::value::flag::DIGITS) && n.value.is_integer()).then_some(())?;
    n.value.to_i64()
}

fn b_year_spelled(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let y = spelled(&v[0])?;
    (1000..3000).contains(&y).then(|| term_value(y, SPELLED))?
}

/// `nineteen ninety six`: two spelled pairs of digits.
fn b_year_pairs(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (a, b) = (spelled(&v[0])?, spelled(&v[1])?);
    ((10..30).contains(&a) && (10..100).contains(&b)).then(|| term_value(a * 100 + b, SPELLED))?
}

/// A year that stands for itself: four digits, or words after `in`.
fn full_year(v: &Value) -> Option<YearRef> {
    (term(v).1 & SHORT == 0).then(|| year_ref(v))
}

// Days of the month.

fn b_mday_digits(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let d = term(&v[0]).0;
    (dig_len(&v[0]) <= 2 && (1..=31).contains(&d)).then(|| term_value(d, 0))?
}

fn b_mday_ordinal(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let d = v[0].ordinal()?.value;
    (1..=31).contains(&d).then(|| term_value(d, ORDINAL))?
}

/// `2th`, `21th`: digits and any ordinal ending.
fn b_mday_suffix(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let d = term(&v[0]).0;
    (dig_len(&v[0]) <= 2 && (1..=31).contains(&d)).then(|| term_value(d, ORDINAL))?
}

fn b_mday_spelled(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let d = spelled(&v[0])?;
    (1..=31).contains(&d).then(|| term_value(d, SPELLED))?
}

fn day_fits(month: i64, day: i64) -> bool {
    let max = match month {
        2 => 30,
        4 | 6 | 9 | 11 => 30,
        1..=12 => 31,
        _ => 0,
    };
    (1..=max).contains(&day)
}

fn cal(year: YearRef, month: i64, day: i64, weekday: u8) -> Option<DateSpec> {
    (month == 0 || day == 0 || day_fits(month, day)).then_some(())?;
    Some(DateSpec::Cal { year, month: u8::try_from(month).ok()?, day: u8::try_from(day).ok()?, weekday })
}

fn md(c: &Ctx<'_>, month: i64, day: i64) -> Option<Value> {
    emit_date(c, cal(YearRef::Any, month, day, 0)?)
}

/// `one may` is no date: a month with another sense after a day takes
/// the day in digits or as an ordinal.
fn plain_words(month: &Value, day: &Value) -> bool {
    term(month).1 & super::AMBIG != 0 && term(day).1 & SPELLED != 0
}

/// `Apr. 1st`; a dot after a full month name ends a sentence.
fn b_md_month_day(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (month, flags) = term(&v[0]);
    if flags & ABBR == 0 && (c.start..c.end).any(|i| c.is_punct(i, '.')) {
        return None;
    }
    md(c, month, term(&v[1]).0)
}

fn b_md_day_month(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if plain_words(&v[1], &v[0]) {
        return None;
    }
    md(c, term(&v[1]).0, term(&v[0]).0)
}

fn b_md_month_digits(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (dig_len(&v[1]) <= 2).then(|| md(c, term(&v[0]).0, term(&v[1]).0))?
}

/// `11/20`, or `22/04` when the first number cannot be a month. After a
/// number or before `of` it is a fraction: `21 1/4`, `3/4 of a point`.
fn b_md_slash(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let fraction =
        c.before().is_some_and(|p| c.lexeme(p).kind == Kind::Numeral) || c.after().is_some_and(|a| c.word(a) == "of");
    if fraction || dig_len(&v[0]) > 2 || dig_len(&v[1]) > 2 {
        return None;
    }
    let (a, b) = (term(&v[0]).0, term(&v[1]).0);
    if a > 12 { md(c, b, a) } else { md(c, a, b) }
}

/// `5.12`, `7.6`: a month and a day written as a decimal.
fn dot_md(raw: &str) -> Option<(i64, i64)> {
    let (m, d) = raw.split_once('.')?;
    if m.is_empty()
        || m.len() > 2
        || d.is_empty()
        || d.len() > 2
        || !(m.bytes().chain(d.bytes())).all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let (m, d) = (m.parse().ok()?, d.parse().ok()?);
    ((1..=12).contains(&m) && day_fits(m, d)).then_some((m, d))
}

fn b_md_dot(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (m, d) = dot_md(c.raw(c.start))?;
    md(c, m, d)
}

fn b_weekday_dot(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (m, d) = dot_md(c.raw(c.end - 1))?;
    md(c, m, d)
}

/// The child that holds the value, the rule's others being words.
fn b_same(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(v[0])
}

/// A month and day with the year from the last child.
fn b_md_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let DateSpec::Cal { year: YearRef::Any, month, day, weekday } = date_of(c, &v[0])? else { return None };
    // `11/20, 12 of Nov` lists two days.
    let comma = (c.start..c.end).any(|i| c.word(i) == ",");
    if day == 0 || (comma && term(v.last()?).1 & SHORT != 0) {
        return None;
    }
    emit_date(c, DateSpec::Cal { year: year_ref(v.last()?), month, day, weekday })
}

fn numeric_year(v: &Value) -> Option<YearRef> {
    match dig_len(v) {
        4 => Some(YearRef::Abs(i16::try_from(term(v).0).ok()?)),
        2 => Some(year_ref(&Value::Term { value: term(v).0, flags: SHORT })),
        _ => None,
    }
}

/// `5/3/18`, `01/22/2016`, or `13/04/21` when the first number cannot be
/// a month.
fn b_mdy(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if dig_len(&v[0]) > 2 || dig_len(&v[1]) > 2 {
        return None;
    }
    let (a, b) = (term(&v[0]).0, term(&v[1]).0);
    let (m, d) = if a > 12 { (b, a) } else { (a, b) };
    emit_date(c, cal(numeric_year(&v[2])?, m, d, 0)?)
}

/// `23.04.2022`: day first.
fn b_dmy(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if dig_len(&v[0]) > 2 || dig_len(&v[1]) > 2 {
        return None;
    }
    let (a, b) = (term(&v[0]).0, term(&v[1]).0);
    let (m, d) = if b > 12 { (a, b) } else { (b, a) };
    emit_date(c, cal(numeric_year(&v[2])?, m, d, 0)?)
}

fn b_ymd(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if dig_len(&v[0]) != 4 || dig_len(&v[1]) > 2 || dig_len(&v[2]) > 2 {
        return None;
    }
    emit_date(c, cal(YearRef::Abs(i16::try_from(term(&v[0]).0).ok()?), term(&v[1]).0, term(&v[2]).0, 0)?)
}

/// `20161016`.
fn b_yyyymmdd(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if dig_len(&v[0]) != 8 {
        return None;
    }
    let n = term(&v[0]).0;
    let y = n / 10_000;
    if !(1900..2100).contains(&y) {
        return None;
    }
    emit_date(c, cal(YearRef::Abs(y as i16), n / 100 % 100, n % 100, 0)?)
}

fn year_digits(v: &Value) -> Option<YearRef> {
    (dig_len(v) == 4).then(|| YearRef::Abs(term(v).0 as i16))
}

fn small(v: &Value) -> Option<i64> {
    (dig_len(v) <= 2).then(|| term(v).0)
}

// `2019/aug/01`, `05-aug-2016`, `sep-23-2020` and the other orders of a
// year, a month name and a day.
fn b_y_mon_d(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(year_digits(&v[0])?, term(&v[2]).0, small(&v[4])?, 0)?)
}

fn b_d_mon_y(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(year_digits(&v[4])?, term(&v[2]).0, small(&v[0])?, 0)?)
}

fn b_mon_d_y(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(year_digits(&v[4])?, term(&v[0]).0, small(&v[2])?, 0)?)
}

fn b_mon_y_d(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(year_digits(&v[2])?, term(&v[0]).0, small(&v[4])?, 0)?)
}

fn b_y_d_mon(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(year_digits(&v[0])?, term(&v[4]).0, small(&v[2])?, 0)?)
}

fn b_d_y_mon(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(year_digits(&v[2])?, term(&v[4]).0, small(&v[0])?, 0)?)
}

/// `2016 10 16`.
fn b_ymd_spaced(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if dig_len(&v[1]) != 2 || dig_len(&v[2]) != 2 {
        return None;
    }
    emit_date(c, cal(year_digits(&v[0])?, term(&v[1]).0, term(&v[2]).0, 0)?)
}

/// A weekday before a full date says nothing more: `Tue May 9`.
fn b_weekday_date(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[0]).1 & PLURAL != 0 {
        return None;
    }
    match date_of(c, v.last()?)? {
        DateSpec::Cal { month, day, .. } if month > 0 && day > 0 => Some(*v.last()?),
        _ => None,
    }
}

// Weekdays.

fn weekday(v: &Value) -> Option<u8> {
    let (w, f) = term(v);
    (f & PLURAL == 0).then_some(w as u8)
}

fn b_wd(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::Cal { year: YearRef::Any, month: 0, day: 0, weekday: weekday(&v[0])? })
}

fn rel_weekday(c: &Ctx<'_>, rel_v: &Value, wd: u8) -> Option<Value> {
    let (n, kind) = term(rel_v);
    emit_date(
        c,
        match kind {
            rel::PAST => DateSpec::WeekdayPast { weekday: wd },
            rel::UPCOMING | rel::THIS => DateSpec::WeekdayRel { weekday: wd, weeks: 0 },
            _ => DateSpec::WeekdayRel { weekday: wd, weeks: n as i8 },
        },
    )
}

fn b_wd_rel(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    rel_weekday(c, &v[0], weekday(&v[1])?)
}

fn b_wd_this_rel(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if !matches!(term(&v[0]).1, rel::PAST | rel::UPCOMING) {
        return None;
    }
    rel_weekday(c, &v[0], weekday(&v[1])?)
}

fn b_wd_day(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::WeekdayDay { weekday: weekday(&v[0])?, day: term(&v[1]).0 as u8, this_month: false })
}

fn b_wd_the_day(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::WeekdayDay { weekday: weekday(&v[0])?, day: term(&v[1]).0 as u8, this_month: true })
}

fn weeks(v: &Value) -> i8 {
    match term(v).1 {
        rel::THIS | rel::UPCOMING => 0,
        _ => term(v).0 as i8,
    }
}

/// `next week Wednesday`.
fn b_week_wd(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::WeekdayRel { weekday: weekday(&v[1])?, weeks: weeks(&v[0]) })
}

/// `Tuesday of next week`.
fn b_wd_of_week(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::WeekdayRel { weekday: weekday(&v[0])?, weeks: weeks(&v[1]) })
}

fn b_dom(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(YearRef::Any, 0, term(&v[0]).0, 0)?)
}

fn b_dom_ordinal(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (term(&v[0]).1 & ORDINAL != 0).then(|| b_dom(v, c))?
}

// Days named from the reference.

fn b_dayword(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (n, f) = term(&v[0]);
    (f & TONIGHT == 0).then(|| emit_date(c, DateSpec::Rel { grain: TimeGrain::Day, n: n as i16 }))?
}

fn day_rel(c: &Ctx<'_>, n: i16) -> Option<Value> {
    emit_date(c, DateSpec::Rel { grain: TimeGrain::Day, n })
}

fn b_rel_month_day(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let months = match v.get(1) {
        Some(r) => weeks(r),
        None => 0,
    };
    emit_date(c, DateSpec::RelMonthDay { months, day: term(&v[0]).0 as u8 })
}

/// The month a child names, and its year: `July`, `July 2016`, `next
/// month`.
pub(crate) fn month_of(c: &Ctx<'_>, v: &Value) -> Option<(YearRef, MonthRef)> {
    match date_of(c, v)? {
        DateSpec::Cal { year, month, day: 0, weekday: 0 } if month > 0 => Some((year, MonthRef::Abs(month))),
        DateSpec::Rel { grain: TimeGrain::Month, n } => Some((YearRef::Rel(0), MonthRef::Rel(n as i8))),
        _ => None,
    }
}

/// The year a child names: `2016`, `next year`.
pub(crate) fn year_of(c: &Ctx<'_>, v: &Value) -> Option<YearRef> {
    match date_of(c, v)? {
        DateSpec::Cal { year, month: 0, day: 0, weekday: 0 } => Some(year),
        DateSpec::Rel { grain: TimeGrain::Year, n } => Some(YearRef::Rel(n)),
        _ => None,
    }
}

fn nth(v: &Value) -> Option<u8> {
    match v {
        Value::Ordinal(o) => u8::try_from(o.value).ok(),
        _ => None,
    }
}

fn b_nth_weekday(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (year, month) = month_of(c, &v[2])?;
    emit_date(c, DateSpec::NthWeekday { year, month, weekday: weekday(&v[1])?, n: nth(&v[0])? })
}

fn b_last_weekday(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (year, month) = month_of(c, &v[1])?;
    emit_date(c, DateSpec::NthWeekday { year, month, weekday: weekday(&v[0])?, n: 0 })
}

/// `1st week of Oct. on Friday`.
fn b_nth_week_weekday(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let month = MonthRef::Abs(term(&v[1]).0 as u8);
    emit_date(c, DateSpec::NthWeekday { year: YearRef::Any, month, weekday: weekday(&v[2])?, n: nth(&v[0])? })
}

// Months.

fn b_month(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(YearRef::Any, term(&v[0]).0, 0, 0)?)
}

fn b_rel_month(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (n, kind) = term(&v[0]);
    if !matches!(kind, rel::THIS | rel::NEXT | rel::LAST) {
        return None;
    }
    emit_date(c, cal(YearRef::Rel(n as i16), term(&v[1]).0, 0, 0)?)
}

/// `April 2017`, `June of 1992`, `April of 87`: two digits only after
/// `of`.
fn b_month_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let y = v.last()?;
    if term(y).1 & SHORT != 0 && c.word(c.start + 1) != "of" {
        return None;
    }
    emit_date(c, cal(year_ref(y), term(&v[0]).0, 0, 0)?)
}

fn b_month_sep_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(full_year(&v[2])?, term(&v[0]).0, 0, 0)?)
}

fn b_year_month(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(full_year(&v[0])?, term(&v[1]).0, 0, 0)?)
}

/// `2015-3`, `2015-12`.
fn b_ym_digits(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let m = small(&v[2])?;
    ((1..=12).contains(&m)).then(|| emit_date(c, cal(year_digits(&v[0])?, m, 0, 0)?))?
}

/// `12-2015`.
fn b_my_digits(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let m = small(&v[0])?;
    ((1..=12).contains(&m)).then(|| emit_date(c, cal(year_digits(&v[1])?, m, 0, 0)?))?
}

/// `the third month of 2021`.
fn b_nth_month(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let m = i64::from(nth(&v[0])?);
    emit_date(c, cal(year_of(c, &v[1])?, m, 0, 0)?)
}

// Periods.

fn b_period_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[0]).1 & (SHORT | SPELLED) != 0 {
        return None;
    }
    emit_date(c, cal(year_ref(&v[0]), 0, 0, 0)?)
}

fn b_period_year_spelled(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[0]).1 & SPELLED == 0 {
        return None;
    }
    emit_date(c, cal(year_ref(&v[0]), 0, 0, 0)?)
}

fn b_year_word(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(full_year(&v[0])?, 0, 0, 0)?)
}

/// The grain a unit word names as a period: `week`, `quarter`, `decade`.
pub(crate) fn unit_grain(v: &Value) -> Option<TimeGrain> {
    let u = term(v).0;
    Some(match (u & 255, u >> 8) {
        (0, 1) => TimeGrain::Year,
        (0, 10) => TimeGrain::Decade,
        (1, 1) => TimeGrain::Month,
        (1, 3) => TimeGrain::Quarter,
        (2, 1) => TimeGrain::Week,
        (2, 2) => TimeGrain::Week,
        (3, 1) => TimeGrain::Weekend,
        (6, 1) => TimeGrain::Day,
        _ => return None,
    })
}

fn is_fortnight(v: &Value) -> bool {
    term(v).0 == lexicon::unit(lexicon::WEEK, 2)
}

fn period(c: &Ctx<'_>, unit: &Value, n: i16) -> Option<Value> {
    if term(unit).1 & PLURAL != 0 {
        return None;
    }
    if is_fortnight(unit) {
        return emit_date(c, DateSpec::Fortnight { n: n as i8 });
    }
    let grain = unit_grain(unit)?;
    emit_date(c, DateSpec::Rel { grain, n })
}

fn b_period_rel(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let n = term(&v[0]).0 as i16;
    if unit_grain(&v[1]) == Some(TimeGrain::Day) {
        return None;
    }
    period(c, &v[1], n)
}

fn b_period_the(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    period(c, &v[0], 0)
}

fn b_weekend(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (unit_grain(&v[0]) == Some(TimeGrain::Weekend)).then(|| period(c, &v[0], 0))?
}

fn b_week_after_next(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::Rel { grain: TimeGrain::Week, n: 2 })
}

/// `the same week`, `that weekend`: a period the text does not place.
fn b_undefined(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[0]).1 & PLURAL != 0 {
        return None;
    }
    let grain = unit_grain(&v[0])?;
    if !matches!(grain, TimeGrain::Week | TimeGrain::Weekend | TimeGrain::Month | TimeGrain::Year)
        || is_fortnight(&v[0])
    {
        return None;
    }
    let point = Point::date(DateSpec::Undefined { grain });
    emit(c, TimeValue::Point { point, modifier: Modifier::portion(Portion::Undefined), clip: false })
}

fn b_week_n(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let w = small(&v[0])?;
    let year = match v.get(1) {
        Some(y) => year_of(c, y)?,
        None => YearRef::Rel(0),
    };
    ((1..=53).contains(&w)).then(|| emit_date(c, DateSpec::Week { year, week: w as u8 }))?
}

fn week_of(c: &Ctx<'_>, n: u8, of: &Value) -> Option<Value> {
    if let Some(year) = year_of(c, of) {
        return emit_date(c, DateSpec::Week { year, week: n });
    }
    let (year, month) = month_of(c, of)?;
    emit_date(c, DateSpec::MonthWeek { year, month, week: n })
}

fn b_nth_week(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let n = nth(&v[0])?;
    (1..=5).contains(&n).then(|| week_of(c, n, &v[1]))?
}

fn b_last_week(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    week_of(c, 0, &v[0])
}

// Quarters, halves, decades, centuries.

fn b_qtr(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let q = small(&v[0])?;
    ((1..=4).contains(&q)).then(|| term_value(q, 0))?
}

fn quarter(c: &Ctx<'_>, year: YearRef, q: i64) -> Option<Value> {
    emit_date(c, DateSpec::Quarter { year, quarter: q as u8 })
}

fn b_quarter(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    quarter(c, YearRef::Any, term(&v[0]).0)
}

fn b_year_quarter(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    quarter(c, full_year(&v[0])?, term(v.last()?).0)
}

fn b_quarter_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    quarter(c, full_year(v.last()?)?, term(&v[0]).0)
}

fn nth_quarter(v: &Value) -> Option<i64> {
    let q = i64::from(nth(v)?);
    (1..=4).contains(&q).then_some(q)
}

fn b_nth_quarter_of(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[1]).0 != lexicon::unit(lexicon::MONTH_U, 3) {
        return None;
    }
    quarter(c, year_of(c, &v[2])?, nth_quarter(&v[0])?)
}

fn b_nth_quarter_the_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[1]).0 != lexicon::unit(lexicon::MONTH_U, 3) {
        return None;
    }
    quarter(c, YearRef::Any, nth_quarter(&v[0])?)
}

/// `next year's first quarter`.
fn b_years_quarter(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[2]).0 != lexicon::unit(lexicon::MONTH_U, 3) {
        return None;
    }
    quarter(c, year_of(c, &v[0])?, nth_quarter(&v[1])?)
}

fn b_year_half(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let h = small(v.last()?)?;
    ((1..=2).contains(&h)).then(|| emit_date(c, DateSpec::Half { year: full_year(&v[0])?, half: h as u8 }))?
}

fn b_decade(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let y = term(&v[0]).0;
    if y % 10 != 0 {
        return None;
    }
    let year = match dig_len(&v[0]) {
        4 if !dig_zero(&v[0]) => YearRef::Abs(y as i16),
        2 => YearRef::Short(y as u8),
        _ => return None,
    };
    emit_date(c, DateSpec::Decade { year })
}

fn b_decade_word(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::Decade { year: YearRef::Short(term(&v[0]).0 as u8) })
}

/// `nineteen eighties`: a spelled century and a decade word.
fn b_century_decade(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let century = spelled(&v[0])?;
    let year = century * 100 + term(&v[1]).0;
    ((10..30).contains(&century)).then(|| emit_date(c, DateSpec::Decade { year: YearRef::Abs(year as i16) }))?
}

fn b_century(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let n = nth(&v[0])?;
    (1..=30).contains(&n).then(|| emit_date(c, DateSpec::Century { n }))?
}

// Seasons and other years.

const SEASONS: [Season; 4] = [Season::Spring, Season::Summer, Season::Fall, Season::Winter];

fn season(v: &Value) -> Option<Season> {
    SEASONS.get(usize::try_from(term(v).0).ok()?).copied()
}

fn b_season(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[0]).1 & super::AMBIG != 0 {
        return None;
    }
    emit_date(c, DateSpec::Season { year: YearRef::Any, season: season(&v[0])? })
}

fn b_rel_season(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::Season { year: YearRef::Rel(term(&v[0]).0 as i16), season: season(&v[1])? })
}

fn b_season_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::Season { year: full_year(v.last()?)?, season: season(&v[0])? })
}

fn other(c: &Ctx<'_>, kind: YearKind, year: YearRef) -> Option<Value> {
    emit_date(c, DateSpec::OtherYear { kind, year })
}

fn year_kind(c: &Ctx<'_>, i: usize) -> Option<YearKind> {
    match c.word(i) {
        "fiscal" | "fy" => Some(YearKind::Fiscal),
        "school" | "sy" => Some(YearKind::School),
        _ => None,
    }
}

fn b_other_year(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    other(c, year_kind(c, c.start)?, YearRef::Any)
}

fn b_other_year_n(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    other(c, year_kind(c, c.start)?, full_year(&v[0])?)
}

fn b_rel_other_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    other(c, year_kind(c, c.start + 1)?, YearRef::Rel(term(&v[0]).0 as i16))
}

/// `sy18`, `cy18`: two digits glued to the letters.
fn short_year(v: &Value) -> Option<YearRef> {
    (dig_len(v) == 2).then(|| year_ref(&Value::Term { value: term(v).0, flags: SHORT }))
}

fn b_other_year_glued(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    other(c, year_kind(c, c.start)?, short_year(&v[0])?)
}

fn b_calendar_year_glued(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(short_year(&v[0])?, 0, 0, 0)?)
}

fn b_rel_calendar_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::Rel { grain: TimeGrain::Year, n: term(&v[0]).0 as i16 })
}

// Holidays.

fn holiday(i: i64) -> Option<Holiday> {
    lexicon::HOLIDAYS.get(usize::try_from(i).ok()?).copied()
}

fn hol(c: &Ctx<'_>, i: i64) -> Option<Value> {
    emit_date(c, DateSpec::Holiday { holiday: holiday(i)?, year: YearRef::Any })
}

fn b_holiday(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    hol(c, term(&v[0]).0)
}

fn holiday_of(c: &Ctx<'_>, v: &Value) -> Option<Holiday> {
    match date_of(c, v)? {
        DateSpec::Holiday { holiday, year: YearRef::Any } => Some(holiday),
        _ => None,
    }
}

fn b_holiday_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::Holiday { holiday: holiday_of(c, &v[0])?, year: full_year(v.last()?)? })
}

fn b_holiday_of_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::Holiday { holiday: holiday_of(c, &v[0])?, year: year_of(c, &v[1])? })
}

fn b_holiday_near(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (n, kind) = term(&v[0]);
    if !matches!(kind, rel::NEXT | rel::LAST) {
        return None;
    }
    emit_date(c, DateSpec::HolidayNear { holiday: holiday_of(c, &v[1])?, future: n > 0 })
}

/// `the week of April 10th`, `w/c Feb 4`.
fn b_week_of(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    match date_of(c, v.last()?)? {
        DateSpec::Cal { year: YearRef::Any, month, day, weekday: 0 } if day > 0 => {
            emit_date(c, DateSpec::WeekOf { month, day })
        }
        _ => None,
    }
}

fn b_sep(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Empty)
}

const fn rr(name: &'static str, dim: crate::engine::Dim, items: &'static [Item], build: crate::engine::Build) -> Rule {
    r(name, dim, 0, items, build)
}

pub(super) static RULES: &[Rule] = &[
    rr("sep_slash", SEP, &[word("/")], b_sep),
    rr("sep_dash", SEP, &[word("-")], b_sep),
    // Years.
    rr("year_digits", YEAR, &[node(DIG)], b_year_digits),
    rr("year_short", YEAR, &[node(DIG)], b_year_short),
    rr("year_spelled", YEAR, &[node(INT)], b_year_spelled),
    rr("year_pairs", YEAR, &[node(INT), node(INT)], b_year_pairs),
    // Days of the month.
    rr("mday_digits", MDAY, &[node(DIG)], b_mday_digits),
    rr("mday_ordinal", MDAY, &[node(ORD)], b_mday_ordinal),
    rr("mday_suffix", MDAY, &[node(DIG), glued(ORD_SUFFIX)], b_mday_suffix),
    rr("mday_spelled", MDAY, &[node(INT)], b_mday_spelled),
    // A month and a day.
    rr("md_month_day", MD, &[node(MONTH), opt_word("."), opt_word(","), opt_word("the"), node(MDAY)], b_md_month_day),
    rr("md_day_month", MD, &[node(MDAY), opt_word("of"), node(MONTH)], b_md_day_month),
    rr("md_the_day_month", MD, &[word("the"), node(MDAY), opt_word("of"), node(MONTH)], b_md_day_month),
    rr("md_day_of_month", MD, &[node(MDAY), word("day"), word("of"), node(MONTH)], b_md_day_month),
    rr("md_month_sep_day", MD, &[node(MONTH), glued(SEP), glued(DIG)], |v, c| b_md_month_digits(&[v[0], v[2]], c)),
    rr("md_slash", MD, &[node(DIG), glued_word("/"), glued(DIG)], b_md_slash),
    rr("md_paren", MD, &[node(MD), word("("), node(WEEKDAY), word(")")], b_same),
    rr("md_dot", DATE, &[node(NUMERAL)], b_md_dot),
    // Days.
    rr("date_md", DATE, &[node(MD)], b_same),
    rr("date_wd", DATE, &[node(WD)], b_same),
    rr("date_dom", DATE, &[node(DOM)], b_same),
    rr("date_md_year", DATE, &[node(MD), opt_word(","), opt_word("of"), opt_word("in"), node(YEAR)], b_md_year),
    rr("date_mdy_slash", DATE, &[node(DIG), glued_word("/"), glued(DIG), glued_word("/"), glued(DIG)], |v, c| {
        b_mdy(&[v[0], v[1], v[2]], c)
    }),
    rr("date_mdy_dash", DATE, &[node(DIG), glued_word("-"), glued(DIG), glued_word("-"), glued(DIG)], |v, c| {
        b_mdy(&[v[0], v[1], v[2]], c)
    }),
    rr("date_dmy_dot", DATE, &[node(DIG), glued_word("."), glued(DIG), glued_word("."), glued(DIG)], |v, c| {
        b_dmy(&[v[0], v[1], v[2]], c)
    }),
    rr("date_ymd", DATE, &[node(DIG), glued(SEP), glued(DIG), glued(SEP), glued(DIG)], |v, c| {
        b_ymd(&[v[0], v[2], v[4]], c)
    }),
    rr("date_ymd_spaced", DATE, &[node(DIG), node(DIG), node(DIG)], b_ymd_spaced),
    rr("date_yyyymmdd", DATE, &[node(DIG)], b_yyyymmdd),
    rr("date_y_mon_d", DATE, &[node(DIG), glued(SEP), glued(MONTH), glued(SEP), glued(DIG)], b_y_mon_d),
    rr("date_d_mon_y", DATE, &[node(DIG), glued(SEP), glued(MONTH), glued(SEP), glued(DIG)], b_d_mon_y),
    rr("date_mon_d_y", DATE, &[node(MONTH), glued(SEP), glued(DIG), glued(SEP), glued(DIG)], b_mon_d_y),
    rr("date_mon_y_d", DATE, &[node(MONTH), glued(SEP), glued(DIG), glued(SEP), glued(DIG)], b_mon_y_d),
    rr("date_y_d_mon", DATE, &[node(DIG), glued(SEP), glued(DIG), glued(SEP), glued(MONTH)], b_y_d_mon),
    rr("date_d_y_mon", DATE, &[node(DIG), glued(SEP), glued(DIG), glued(SEP), glued(MONTH)], b_d_y_mon),
    rr("date_weekday_md", DATE, &[node(WEEKDAY), opt_word(","), opt_word("-"), node(MD)], b_weekday_date),
    rr("date_weekday_date", DATE, &[node(WEEKDAY), opt_word(","), node(DATE)], b_weekday_date),
    rr("date_weekday_paren", DATE, &[node(WEEKDAY), word("("), node(MD), word(")")], |v, c| {
        b_weekday_date(&[v[0], v[1]], c)
    }),
    rr("date_weekday_dot", DATE, &[node(WEEKDAY), node(NUMERAL)], |v, c| {
        weekday(&v[0]).and_then(|_| b_weekday_dot(v, c))
    }),
    rr("date_dayword", DATE, &[node(DAYWORD)], b_dayword),
    rr("date_the_day", DATE, &[word("the"), word("day")], |_, c| day_rel(c, 0)),
    rr("date_the_day_before", DATE, &[word("the"), word("day"), word("before")], |_, c| day_rel(c, -1)),
    rr("date_the_day_after", DATE, &[word("the"), word("day"), word("after")], |_, c| day_rel(c, 1)),
    rr("date_the_next_day", DATE, &[word("the"), word("next"), word("day")], |_, c| day_rel(c, 1)),
    rr("date_day_after_tomorrow", DATE, &[word("the"), word("day"), word("after"), word("tomorrow")], |_, c| {
        day_rel(c, 2)
    }),
    rr("date_day_after_tomorrow_bare", DATE, &[word("day"), word("after"), word("tomorrow")], |_, c| day_rel(c, 2)),
    rr("date_day_before_yesterday", DATE, &[word("the"), word("day"), word("before"), word("yesterday")], |_, c| {
        day_rel(c, -2)
    }),
    rr("date_day_before_yesterday_bare", DATE, &[word("day"), word("before"), word("yesterday")], |_, c| {
        day_rel(c, -2)
    }),
    rr(
        "date_the_day_of_rel_month",
        DATE,
        &[word("the"), node(MDAY), word("day"), word("of"), node(RELW), word("month")],
        b_rel_month_day,
    ),
    rr(
        "date_day_of_rel_month",
        DATE,
        &[node(MDAY), word("day"), word("of"), node(RELW), word("month")],
        b_rel_month_day,
    ),
    rr(
        "date_the_day_of_the_month",
        DATE,
        &[word("the"), node(MDAY), word("day"), word("of"), word("the"), word("month")],
        |v, c| b_rel_month_day(&v[..1], c),
    ),
    rr("date_day_of_the_month", DATE, &[node(MDAY), word("day"), word("of"), word("the"), word("month")], |v, c| {
        b_rel_month_day(&v[..1], c)
    }),
    rr("date_the_day_this_month", DATE, &[word("the"), node(MDAY), node(RELW), word("month")], b_rel_month_day),
    rr("date_nth_weekday", DATE, &[node(ORD), node(WEEKDAY), word("of"), node(MONTHD)], b_nth_weekday),
    rr("date_nth_weekday_rel", DATE, &[node(ORD), node(WEEKDAY), word("of"), node(PERIOD)], b_nth_weekday),
    rr("date_the_nth_weekday", DATE, &[word("the"), node(ORD), node(WEEKDAY), word("of"), node(MONTHD)], b_nth_weekday),
    rr(
        "date_the_nth_weekday_rel",
        DATE,
        &[word("the"), node(ORD), node(WEEKDAY), word("of"), node(PERIOD)],
        b_nth_weekday,
    ),
    rr("date_last_weekday", DATE, &[word("last"), node(WEEKDAY), word("of"), node(MONTHD)], b_last_weekday),
    rr("date_last_weekday_rel", DATE, &[word("last"), node(WEEKDAY), word("of"), node(PERIOD)], b_last_weekday),
    rr(
        "date_nth_week_weekday",
        DATE,
        &[node(ORD), word("week"), word("of"), node(MONTH), opt_word("."), word("on"), node(WEEKDAY)],
        b_nth_week_weekday,
    ),
    // Weekdays.
    rr("wd", WD, &[node(WEEKDAY)], b_wd),
    rr("wd_rel", WD, &[node(RELW), node(WEEKDAY)], b_wd_rel),
    rr("wd_this_rel", WD, &[word("this"), node(RELW), node(WEEKDAY)], b_wd_this_rel),
    rr("wd_day", WD, &[node(WEEKDAY), node(MDAY)], b_wd_day),
    rr("wd_the_day", WD, &[node(WEEKDAY), word("the"), node(MDAY)], b_wd_the_day),
    rr("wd_week", WD, &[node(RELW), word("week"), opt_word("on"), opt_word("-"), node(WEEKDAY)], b_week_wd),
    rr(
        "wd_on_of_week",
        WD,
        &[word("on"), node(WEEKDAY), word("of"), opt_word("the"), node(RELW), word("week")],
        b_wd_of_week,
    ),
    rr("wd_of_week", WD, &[node(WEEKDAY), word("of"), opt_word("the"), node(RELW), word("week")], b_wd_of_week),
    // Days of an open month.
    rr("dom_the", DOM, &[word("the"), node(MDAY)], b_dom),
    rr("dom_ordinal", DOM, &[node(MDAY)], b_dom_ordinal),
    // Months.
    rr("monthd_month", MONTHD, &[node(MONTH)], b_month),
    rr("monthd_rel", MONTHD, &[node(RELW), node(MONTH)], b_rel_month),
    rr("monthd_year", MONTHD, &[node(MONTH), opt_word(","), opt_word("of"), opt_word("in"), node(YEAR)], b_month_year),
    rr("monthd_sep_year", MONTHD, &[node(MONTH), glued(SEP), glued(YEAR)], b_month_sep_year),
    rr("monthd_year_month", MONTHD, &[node(YEAR), node(MONTH)], b_year_month),
    rr("monthd_ym", MONTHD, &[node(DIG), glued(SEP), glued(DIG)], b_ym_digits),
    rr("monthd_my", MONTHD, &[node(DIG), glued_word("-"), glued(DIG)], b_my_digits),
    rr("monthd_nth", MONTHD, &[node(ORD), word("month"), word("of"), node(PERIOD)], b_nth_month),
    rr("monthd_the_nth", MONTHD, &[word("the"), node(ORD), word("month"), word("of"), node(PERIOD)], b_nth_month),
    // Periods.
    rr("period_year", PERIOD, &[node(YEAR)], b_period_year),
    rr("period_year_spelled", PERIOD, &[node(YEAR)], b_period_year_spelled),
    rr("period_year_word", PERIOD, &[word("year"), node(YEAR)], b_year_word),
    rr("period_eq_year", PERIOD, &[word("="), node(YEAR)], b_year_word),
    rr("period_rel", PERIOD, &[node(RELW), node(UNIT)], b_period_rel),
    rr("period_the_rel", PERIOD, &[word("the"), node(RELW), word("decade")], |v, c| {
        emit_date(c, DateSpec::Rel { grain: TimeGrain::Decade, n: term(&v[0]).0 as i16 })
    }),
    rr("period_the_unit", PERIOD, &[word("the"), node(UNIT)], b_period_the),
    rr("period_weekend", PERIOD, &[node(UNIT)], b_weekend),
    rr("period_week_after_next", PERIOD, &[word("the"), word("week"), word("after"), word("next")], b_week_after_next),
    rr("period_same", PERIOD, &[word("same"), node(UNIT)], b_undefined),
    rr("period_that", PERIOD, &[word("that"), node(UNIT)], b_undefined),
    rr("period_week_n", PERIOD, &[word("week"), node(DIG)], b_week_n),
    rr("period_the_week_n", PERIOD, &[word("the"), word("week"), node(DIG)], b_week_n),
    rr("period_the_year_word", PERIOD, &[word("the"), word("year"), node(YEAR)], b_year_word),
    rr("period_week_n_of", PERIOD, &[word("week"), node(DIG), word("of"), node(PERIOD)], b_week_n),
    rr("period_nth_week", PERIOD, &[node(ORD), word("week"), word("of"), node(PERIOD)], b_nth_week),
    rr("period_nth_week_month", PERIOD, &[node(ORD), word("week"), word("of"), node(MONTHD)], b_nth_week),
    rr("period_the_nth_week", PERIOD, &[word("the"), node(ORD), word("week"), word("of"), node(PERIOD)], b_nth_week),
    rr(
        "period_the_nth_week_month",
        PERIOD,
        &[word("the"), node(ORD), word("week"), word("of"), node(MONTHD)],
        b_nth_week,
    ),
    rr("period_last_week", PERIOD, &[word("last"), word("week"), word("of"), node(PERIOD)], b_last_week),
    rr("period_last_week_month", PERIOD, &[word("last"), word("week"), word("of"), node(MONTHD)], b_last_week),
    rr(
        "period_the_last_week",
        PERIOD,
        &[word("the"), word("last"), word("week"), word("of"), node(PERIOD)],
        b_last_week,
    ),
    rr(
        "period_the_last_week_month",
        PERIOD,
        &[word("the"), word("last"), word("week"), word("of"), node(MONTHD)],
        b_last_week,
    ),
    rr("period_week_of", PERIOD, &[word("the"), word("week"), word("of"), node(DATE)], b_week_of),
    rr("period_week_beginning", PERIOD, &[word("the"), word("week"), word("beginning"), node(DATE)], b_week_of),
    rr(
        "period_week_starting",
        PERIOD,
        &[word("the"), word("week"), word("starting"), opt_word("on"), node(DATE)],
        b_week_of,
    ),
    rr("period_week_commencing", PERIOD, &[word("the"), word("week"), word("commencing"), node(DATE)], b_week_of),
    rr("period_w_c", PERIOD, &[word("w"), glued_word("/"), glued_word("c"), node(DATE)], b_week_of),
    rr("qtr", QTR, &[word("q"), glued(DIG)], b_qtr),
    rr("period_quarter", PERIOD, &[node(QTR)], b_quarter),
    rr("period_year_quarter", PERIOD, &[node(YEAR), node(QTR)], b_year_quarter),
    rr("period_year_sep_quarter", PERIOD, &[node(YEAR), glued_word("-"), glued(QTR)], b_year_quarter),
    rr("period_quarter_year", PERIOD, &[node(QTR), node(YEAR)], b_quarter_year),
    rr("period_quarter_sep_year", PERIOD, &[node(QTR), glued_word("-"), glued(YEAR)], b_quarter_year),
    rr("period_quarter_of_year", PERIOD, &[node(QTR), word("of"), node(YEAR)], b_quarter_year),
    rr("period_nth_quarter", PERIOD, &[node(ORD), node(UNIT), word("of"), node(PERIOD)], b_nth_quarter_of),
    rr(
        "period_the_nth_quarter",
        PERIOD,
        &[word("the"), node(ORD), node(UNIT), word("of"), node(PERIOD)],
        b_nth_quarter_of,
    ),
    rr(
        "period_the_nth_quarter_the_year",
        PERIOD,
        &[word("the"), node(ORD), node(UNIT), word("of"), word("the"), word("year")],
        b_nth_quarter_the_year,
    ),
    rr("period_years_quarter", PERIOD, &[node(PERIOD), word("'"), word("s"), node(ORD), node(UNIT)], b_years_quarter),
    rr(
        "period_years_quarter_curly",
        PERIOD,
        &[node(PERIOD), word("\u{2019}"), word("s"), node(ORD), node(UNIT)],
        b_years_quarter,
    ),
    rr("period_year_half", PERIOD, &[node(YEAR), word("h"), glued(DIG)], b_year_half),
    rr("period_year_sep_half", PERIOD, &[node(YEAR), glued_word("-"), glued_word("h"), glued(DIG)], b_year_half),
    rr("period_decade", PERIOD, &[node(DIG), word("s")], b_decade),
    rr("period_decade_word", PERIOD, &[node(DECADEW)], b_decade_word),
    rr("period_century_decade", PERIOD, &[node(INT), node(DECADEW)], b_century_decade),
    rr("period_century", PERIOD, &[node(ORD), opt_word("-"), word("century")], b_century),
    rr("period_season", PERIOD, &[node(SEASON)], b_season),
    rr("period_rel_season", PERIOD, &[node(RELW), node(SEASON)], b_rel_season),
    rr("period_season_year", PERIOD, &[node(SEASON), opt_word("of"), node(YEAR)], b_season_year),
    rr("period_fiscal_year", PERIOD, &[word("fiscal"), word("year")], b_other_year),
    rr("period_school_year", PERIOD, &[word("school"), word("year")], b_other_year),
    rr("period_fiscal_year_n", PERIOD, &[word("fiscal"), word("year"), node(YEAR)], b_other_year_n),
    rr("period_school_year_n", PERIOD, &[word("school"), word("year"), node(YEAR)], b_other_year_n),
    rr("period_fy", PERIOD, &[word("fy"), node(YEAR)], b_other_year_n),
    rr("period_sy", PERIOD, &[word("sy"), node(YEAR)], b_other_year_n),
    rr("period_fy_glued", PERIOD, &[word("fy"), glued(DIG)], b_other_year_glued),
    rr("period_sy_glued", PERIOD, &[word("sy"), glued(DIG)], b_other_year_glued),
    rr("period_rel_fiscal_year", PERIOD, &[node(RELW), word("fiscal"), word("year")], b_rel_other_year),
    rr("period_rel_school_year", PERIOD, &[node(RELW), word("school"), word("year")], b_rel_other_year),
    rr("period_calendar_year", PERIOD, &[word("calendar"), word("year"), node(YEAR)], b_year_word),
    rr("period_cy", PERIOD, &[word("cy"), node(YEAR)], b_year_word),
    rr("period_cy_glued", PERIOD, &[word("cy"), glued(DIG)], b_calendar_year_glued),
    rr("period_rel_calendar_year", PERIOD, &[node(RELW), word("calendar"), word("year")], b_rel_calendar_year),
    // Holidays.
    rr("hol", HOL, &[node(HOLIDAY)], b_holiday),
    rr("hol_christmas_eve", HOL, &[word("christmas"), word("eve")], |_, c| hol(c, lexicon::CHRISTMAS_EVE)),
    rr("hol_christmas_day", HOL, &[word("christmas"), word("day")], |_, c| hol(c, lexicon::CHRISTMAS)),
    rr("hol_new_year", HOL, &[word("new"), word("year")], |_, c| hol(c, lexicon::NEW_YEAR)),
    rr("hol_new_year_day", HOL, &[word("new"), word("year"), word("day")], |_, c| hol(c, lexicon::NEW_YEAR)),
    rr("hol_new_years", HOL, &[word("new"), word("years"), opt_word("day")], |_, c| hol(c, lexicon::NEW_YEAR)),
    rr("hol_new_year_s", HOL, &[word("new"), word("year"), word("'"), word("s"), opt_word("day")], |_, c| {
        hol(c, lexicon::NEW_YEAR)
    }),
    rr("hol_new_year_s_eve", HOL, &[word("new"), word("year"), word("'"), word("s"), word("eve")], |_, c| {
        hol(c, lexicon::NEW_YEARS_EVE)
    }),
    rr("hol_new_years_eve", HOL, &[word("new"), word("years"), word("eve")], |_, c| hol(c, lexicon::NEW_YEARS_EVE)),
    rr("hol_independence", HOL, &[word("independence"), word("day")], |_, c| hol(c, lexicon::INDEPENDENCE)),
    rr("hol_labor", HOL, &[word("labor"), word("day")], |_, c| hol(c, lexicon::LABOR)),
    rr("hol_labour", HOL, &[word("labour"), word("day")], |_, c| hol(c, lexicon::LABOR)),
    rr("hol_black_friday", HOL, &[word("black"), word("friday")], |_, c| hol(c, lexicon::BLACK_FRIDAY)),
    rr("hol_earth", HOL, &[word("earth"), word("day")], |_, c| hol(c, lexicon::EARTH)),
    rr("hol_easter_monday", HOL, &[word("easter"), word("monday")], |_, c| hol(c, lexicon::EASTER_MONDAY)),
    rr("hol_easter_sunday", HOL, &[word("easter"), word("sunday")], |_, c| hol(c, lexicon::EASTER)),
    rr("hol_good_friday", HOL, &[word("good"), word("friday")], |_, c| hol(c, lexicon::GOOD_FRIDAY)),
    rr("hol_saint_patrick", HOL, &[word("saint"), word("patrick")], |_, c| hol(c, lexicon::PATRICK)),
    rr("hol_st_patrick", HOL, &[word("st"), opt_word("."), word("patrick")], |_, c| hol(c, lexicon::PATRICK)),
    rr("hol_patricks_day", HOL, &[word("patrick"), word("'"), word("s"), word("day")], |_, c| hol(c, lexicon::PATRICK)),
    rr("hol_workers", HOL, &[word("international"), word("workers"), opt_word("'"), word("day")], |_, c| {
        hol(c, lexicon::WORKERS)
    }),
    rr("hol_workers_curly", HOL, &[word("international"), word("workers"), word("\u{2019}"), word("day")], |_, c| {
        hol(c, lexicon::WORKERS)
    }),
    rr("hol_freedom", HOL, &[word("freedom"), word("day")], |_, c| hol(c, lexicon::JUNETEENTH)),
    rr("hol_jubilee", HOL, &[word("jubilee"), word("day")], |_, c| hol(c, lexicon::JUNETEENTH)),
    rr("hol_memorial", HOL, &[word("memorial"), word("day")], |_, c| hol(c, lexicon::MEMORIAL)),
    rr("hol_veterans", HOL, &[word("veterans"), word("day")], |_, c| hol(c, lexicon::VETERANS)),
    rr("hol_valentine", HOL, &[word("valentine"), word("'"), word("s"), word("day")], |_, c| {
        hol(c, lexicon::VALENTINE)
    }),
    rr("hol_mothers", HOL, &[word("mother"), word("'"), word("s"), word("day")], |_, c| hol(c, lexicon::MOTHERS)),
    rr("hol_fathers", HOL, &[word("father"), word("'"), word("s"), word("day")], |_, c| hol(c, lexicon::FATHERS)),
    rr("hol_year", HOL, &[node(HOL), opt_word(","), node(YEAR)], b_holiday_year),
    rr("hol_of_year", HOL, &[node(HOL), word("of"), node(PERIOD)], b_holiday_of_year),
    rr("hol_near", HOL, &[node(RELW), node(HOL)], b_holiday_near),
];
