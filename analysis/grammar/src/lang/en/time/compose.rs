//! Dates with times of day: `tomorrow 8:00am`, `9am on Tuesday`, `this
//! evening`, `tuesday in the morning`, `five-thirty tomorrow evening`;
//! the present (`now`); and points moved by a duration: `in 5 minutes`,
//! `2 days ago`, `3 days after January 12th`.

use turbo_analysis_layers::{Modifier, Portion, TimeGrain};

use crate::engine::{Ctx, Item, Rule, node, opt_word, word};
use crate::time::TimeValue;
use crate::time::duration::Duration;
use crate::time::spec::{Clock, DateSpec, HolidayOrRel, Part, Point, Shown, TimeOfDay, YearRef};
use crate::value::Value;

use super::clock::{is_meal, part};
use super::duration::duration_of;
use super::lexicon::rel;
use super::{
    BARE, CLOCK, DATE, DAYWORD, DT, DTP, DUR, HOL, PART, PARTD, RELW, TONIGHT, WEEKDAY, clock_of, date_of, emit,
    emit_point, plain, r, term, tval,
};

/// A child's date when it names one day: `tomorrow`, `May 5`, `Friday`.
pub(crate) fn day_of(c: &Ctx<'_>, v: &Value) -> Option<DateSpec> {
    date_of(c, v).filter(|d| d.is_day())
}

/// A part of the day and its modifier: `in the morning`, `late evening`.
pub(crate) fn part_of(c: &Ctx<'_>, v: &Value) -> Option<(Part, Modifier)> {
    match tval(c, v)? {
        TimeValue::Point {
            point: Point { date: DateSpec::None, time: TimeOfDay::Part(p), shift }, modifier, ..
        } if shift.is_zero() && !is_meal(p) => Some((p, modifier)),
        _ => None,
    }
}

/// A day with a part of the day: `tomorrow night`, `tonight`.
pub(crate) fn day_part_of(c: &Ctx<'_>, v: &Value) -> Option<(DateSpec, Part)> {
    match plain(c, v)? {
        Point { date, time: TimeOfDay::Part(p), shift } if shift.is_zero() && date.is_day() => Some((date, p)),
        _ => None,
    }
}

pub(crate) fn at(c: &Ctx<'_>, date: DateSpec, k: Clock) -> Option<Value> {
    emit_point(c, Point { date, time: TimeOfDay::Clock(k), shift: Duration::ZERO })
}

pub(crate) fn b_date_clock(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    at(c, day_of(c, &v[0])?, clock_of(c, v.last()?)?)
}

pub(crate) fn b_clock_date(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    at(c, day_of(c, v.last()?)?, clock_of(c, &v[0])?)
}

/// A date and an hour with minutes, which stands as a time with no `at`:
/// `march eighteenth nine thirty`.
pub(crate) fn b_date_bare_minutes(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let k = clock_of(c, &v[1])?;
    (k.shown != Shown::Hour).then(|| at(c, day_of(c, &v[0])?, k))?
}

/// A clock in a day's part: `tonight at 1am`, `five-thirty tomorrow
/// evening`, `10, tonight`.
pub(crate) fn clock_in_day_part(c: &Ctx<'_>, dp: &Value, clock: &Value) -> Option<Value> {
    let (date, p) = day_part_of(c, dp)?;
    let k = clock_of(c, clock)?;
    at(c, date, k.with(p.meridiem(k.hour)))
}

pub(crate) fn b_dtp_clock(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    clock_in_day_part(c, &v[0], v.last()?)
}

pub(crate) fn b_clock_dtp(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    clock_in_day_part(c, v.last()?, &v[0])
}

/// An hour before a day's part needs a comma (`10, tonight`); a time
/// with minutes does not (`five-thirty tomorrow evening`).
pub(crate) fn b_bare_dtp(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let comma = c.word(c.start + 1) == ",";
    (comma || clock_of(c, &v[0])?.shown != Shown::Hour).then(|| b_clock_dtp(v, c))?
}

pub(crate) fn day_part(c: &Ctx<'_>, date: DateSpec, p: Part, modifier: Modifier) -> Option<Value> {
    let point = Point { date, time: TimeOfDay::Part(p), shift: Duration::ZERO };
    emit(c, TimeValue::Point { point, modifier, clip: false })
}

pub(crate) fn b_date_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (p, m) = part_of(c, v.last()?)?;
    day_part(c, day_of(c, &v[0])?, p, m)
}

pub(crate) fn b_part_date(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (p, m) = part_of(c, &v[0])?;
    day_part(c, day_of(c, v.last()?)?, p, m)
}

pub(crate) fn rel_day(rel_v: &Value) -> Option<DateSpec> {
    let (n, kind) = term(rel_v);
    matches!(kind, rel::THIS | rel::NEXT | rel::LAST).then_some(DateSpec::Rel { grain: TimeGrain::Day, n: n as i16 })
}

/// A part of the day that a date or `this` can take: not a plural, not
/// a meal.
pub(crate) fn single_part(v: &Value) -> Option<Part> {
    (term(v).1 & super::PLURAL == 0).then(|| part(v).filter(|&p| !is_meal(p)))?
}

/// `this evening`, `last night`, `next morning`.
pub(crate) fn b_rel_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    day_part(c, rel_day(&v[0])?, single_part(&v[1])?, Modifier::NONE)
}

pub(crate) fn portion(c: &Ctx<'_>, v: &[Value], p: Portion) -> Option<Value> {
    let part = single_part(&v[1])?;
    if !matches!(part, Part::Dawn | Part::Morning | Part::Afternoon | Part::Evening | Part::Night) {
        return None;
    }
    day_part(c, rel_day(&v[0])?, part, Modifier::portion(p))
}

/// `early this evening`, `late this morning`.
pub(crate) fn b_early_rel_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    portion(c, v, Portion::Start)
}

pub(crate) fn b_late_rel_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    portion(c, v, Portion::End)
}

pub(crate) fn b_tonight(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (term(&v[0]).1 & TONIGHT != 0)
        .then(|| day_part(c, DateSpec::Rel { grain: TimeGrain::Day, n: 0 }, Part::Night, Modifier::NONE))?
}

/// `today pm`, `tomorrow am`: the afternoon or the morning.
pub(crate) fn b_day_ampm(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if !matches!(day_of(c, &v[0])?, DateSpec::Rel { .. }) {
        return None;
    }
    let p = match c.word(c.end - 1) {
        "am" => Part::Morning,
        "pm" => Part::Afternoon,
        _ => return None,
    };
    day_part(c, day_of(c, &v[0])?, p, Modifier::NONE)
}

/// `Wednesday evening next week`.
pub(crate) fn b_weekday_part_week(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (wd, flags) = term(&v[0]);
    if flags & super::PLURAL != 0 {
        return None;
    }
    let weeks = match term(&v[2]).1 {
        rel::THIS => 0,
        rel::NEXT | rel::LAST => term(&v[2]).0 as i8,
        _ => return None,
    };
    day_part(c, DateSpec::WeekdayRel { weekday: wd as u8, weeks }, single_part(&v[1])?, Modifier::NONE)
}

pub(crate) fn b_now(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_point(c, Point::time(TimeOfDay::Now))
}

// Shifts.

pub(crate) fn today() -> DateSpec {
    DateSpec::Rel { grain: TimeGrain::Day, n: 0 }
}

/// The reference moved by `d`: a clock duration from now, a longer one
/// from today.
pub(crate) fn from_now(c: &Ctx<'_>, d: Duration) -> Option<Value> {
    let clock = d.parts().iter().any(|(u, _)| u.is_clock());
    let point = if clock {
        Point { time: TimeOfDay::Now, shift: d, ..Point::default() }
    } else {
        Point { date: today(), shift: d, ..Point::default() }
    };
    emit_point(c, point)
}

pub(crate) fn b_in(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    from_now(c, duration_of(c, v.last()?)?)
}

pub(crate) fn b_later(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    from_now(c, duration_of(c, &v[0])?)
}

pub(crate) fn b_ago(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    from_now(c, duration_of(c, &v[0])?.negated())
}

/// A point moved from another: `3 days after January 12th`, `two weeks
/// before Christmas`.
pub(crate) fn shifted(c: &Ctx<'_>, d: Duration, base: &Value) -> Option<Value> {
    let p = plain(c, base)?;
    // A day moves by days or longer: `3 minutes from today` is two
    // mentions.
    let clock = d.parts().iter().any(|(u, _)| u.is_clock());
    if !p.shift.is_zero() || p.date == DateSpec::None || (clock && p.time == TimeOfDay::None) {
        return None;
    }
    emit_point(c, Point { shift: d, ..p })
}

pub(crate) fn b_after(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    shifted(c, duration_of(c, &v[0])?, v.last()?)
}

pub(crate) fn b_before(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    shifted(c, duration_of(c, &v[0])?.negated(), v.last()?)
}

fn with_shifted_weekday(c: &Ctx<'_>, wd_v: &Value, dt_v: &Value) -> Option<Value> {
    let (wd, flags) = term(wd_v);
    if flags & super::PLURAL != 0 {
        return None;
    }
    let p = plain(c, dt_v)?;
    if p.shift.is_zero() || p.time != TimeOfDay::None || p.shift.parts().iter().any(|(u, _)| u.is_clock()) {
        return None;
    }
    let base = match p.date {
        DateSpec::Rel { grain: TimeGrain::Day, n: 0 } => HolidayOrRel::Today,
        DateSpec::Cal { year: YearRef::Any, month: m @ 1.., day: d @ 1.., weekday: 0 } => {
            HolidayOrRel::MonthDay { month: m, day: d }
        }
        DateSpec::Holiday { holiday, year: YearRef::Any } => HolidayOrRel::Holiday(holiday),
        _ => return None,
    };
    emit_point(c, Point { date: DateSpec::ShiftedWeekday { base, weekday: wd as u8 }, ..p })
}

pub(crate) fn b_weekday_dt(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    with_shifted_weekday(c, &v[0], v.last()?)
}

pub(crate) fn b_dt_weekday(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    with_shifted_weekday(c, v.last()?, &v[0])
}

const fn rr(name: &'static str, dim: crate::engine::Dim, items: &'static [Item], build: crate::engine::Build) -> Rule {
    r(name, dim, 0, items, build)
}

pub(super) static RULES: &[Rule] = &[
    // A date and a clock time.
    rr("dt_date_clock", DT, &[node(DATE), opt_word(","), node(CLOCK)], b_date_clock),
    rr("dt_date_at_clock", DT, &[node(DATE), opt_word(","), word("at"), node(CLOCK)], b_date_clock),
    rr("dt_date_at_bare", DT, &[node(DATE), opt_word(","), word("at"), node(BARE)], b_date_clock),
    rr("dt_date_around_clock", DT, &[node(DATE), word("around"), node(CLOCK)], b_date_clock),
    rr("dt_date_around_bare", DT, &[node(DATE), word("around"), node(BARE)], b_date_clock),
    rr("dt_date_for_clock", DT, &[node(DATE), word("for"), node(CLOCK)], b_date_clock),
    rr("dt_date_sign_clock", DT, &[node(DATE), word("@"), node(CLOCK)], b_date_clock),
    rr("dt_date_dash_clock", DT, &[node(DATE), word("-"), node(CLOCK)], b_date_clock),
    rr("dt_date_bare", DT, &[node(DATE), node(BARE)], b_date_bare_minutes),
    rr("dt_clock_date", DT, &[node(CLOCK), opt_word(","), node(DATE)], b_clock_date),
    rr("dt_clock_on_date", DT, &[node(CLOCK), word("on"), node(DATE)], b_clock_date),
    // A clock in a day's part.
    rr("dt_dtp_clock", DT, &[node(DTP), opt_word("at"), node(CLOCK)], b_dtp_clock),
    rr("dt_dtp_at_bare", DT, &[node(DTP), word("at"), node(BARE)], b_dtp_clock),
    rr("dt_dtp_around_bare", DT, &[node(DTP), word("around"), node(BARE)], b_dtp_clock),
    rr("dt_clock_dtp", DT, &[node(CLOCK), opt_word(","), node(DTP)], b_clock_dtp),
    rr("dt_bare_dtp", DT, &[node(BARE), opt_word(","), node(DTP)], b_bare_dtp),
    // A day and a part of it.
    rr("dtp_date_part", DTP, &[node(DATE), node(PARTD)], b_date_part),
    rr("dtp_part_date", DTP, &[node(PARTD), opt_word("of"), opt_word("on"), node(DATE)], b_part_date),
    rr("dtp_rel_part", DTP, &[node(RELW), node(PART)], b_rel_part),
    rr("dtp_early_rel_part", DTP, &[word("early"), node(RELW), node(PART)], b_early_rel_part),
    rr("dtp_late_rel_part", DTP, &[word("late"), node(RELW), node(PART)], b_late_rel_part),
    rr("dtp_later_rel_part", DTP, &[word("later"), node(RELW), node(PART)], b_late_rel_part),
    rr("dtp_tonight", DTP, &[node(DAYWORD)], b_tonight),
    rr("dtp_day_am", DTP, &[node(DATE), word("am")], b_day_ampm),
    rr("dtp_day_pm", DTP, &[node(DATE), word("pm")], b_day_ampm),
    rr("dtp_weekday_part_week", DTP, &[node(WEEKDAY), node(PART), node(RELW), word("week")], b_weekday_part_week),
    // The present.
    rr("dt_now", DT, &[word("now")], b_now),
    rr("dt_right_now", DT, &[word("right"), word("now")], b_now),
    rr("dt_at_the_moment", DT, &[word("at"), word("the"), word("moment")], b_now),
    rr("dt_at_the_minute", DT, &[word("at"), word("the"), word("minute")], b_now),
    rr("dt_at_present", DT, &[word("at"), word("present")], b_now),
    rr("dt_at_the_present_time", DT, &[word("at"), word("the"), word("present"), word("time")], b_now),
    rr("dt_at_this_time", DT, &[word("at"), word("this"), word("time")], b_now),
    // Shifts.
    rr("dt_in", DT, &[word("in"), node(DUR)], b_in),
    rr("dt_later", DT, &[node(DUR), word("later")], b_later),
    rr("dt_from_now", DT, &[node(DUR), word("from"), word("now")], b_later),
    rr("dt_ago", DT, &[node(DUR), word("ago")], b_ago),
    rr("dt_after", DT, &[node(DUR), word("after"), node(DATE)], b_after),
    rr("dt_from", DT, &[node(DUR), word("from"), node(DATE)], b_after),
    rr("dt_before", DT, &[node(DUR), word("before"), node(DATE)], b_before),
    rr("dt_after_hol", DT, &[node(DUR), word("after"), node(HOL)], b_after),
    rr("dt_before_hol", DT, &[node(DUR), word("before"), node(HOL)], b_before),
    rr("dt_weekday_shift", DT, &[node(WEEKDAY), opt_word(","), node(DT)], b_weekday_dt),
    rr("dt_shift_on_weekday", DT, &[node(DT), word("on"), node(WEEKDAY)], b_dt_weekday),
];
