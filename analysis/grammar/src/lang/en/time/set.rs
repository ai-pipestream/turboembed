//! Recurrences: `weekly`, `every two days`, `once a week`, `every
//! Monday at 4pm`, `3pm each day`, `mornings`, `Tuesday afternoons`,
//! `19th of every month`.

use crate::engine::{Ctx, Item, Rule, node, opt_word, word};
use crate::time::TimeValue;
use crate::time::duration::{Duration, MILLI, Unit};
use crate::time::spec::{DateSpec, Part, Point, Recurrence, TimeOfDay, YearRef};
use crate::value::Value;

use super::clock::part;
use super::duration::{length, unit_of};
use super::lexicon::NIGHT;
use super::{AMOUNT, CLOCK, MDAY, PART, PLURAL, SET, SETWORD, UNIT, WEEKDAY, clock_of, emit, r, term, tval};

fn set_of(c: &Ctx<'_>, v: &Value) -> Option<Recurrence> {
    match tval(c, v)? {
        TimeValue::Set(r) => Some(r),
        _ => None,
    }
}

fn emit_set(c: &Ctx<'_>, r: Recurrence) -> Option<Value> {
    emit(c, TimeValue::Set(r))
}

fn every(c: &Ctx<'_>, d: Duration) -> Option<Value> {
    emit_set(c, Recurrence::Every(d))
}

fn at(c: &Ctx<'_>, p: Point) -> Option<Value> {
    emit_set(c, Recurrence::At(p))
}

fn weekday_point(wd: u8) -> Point {
    Point::date(DateSpec::Cal { year: YearRef::Any, month: 0, day: 0, weekday: wd })
}

fn is_daily(r: Recurrence) -> bool {
    matches!(r, Recurrence::Every(d) if d.parts() == [(Unit::Day, MILLI)])
}

const UNITS: [Unit; 10] = [
    Unit::Year,
    Unit::Month,
    Unit::Week,
    Unit::Weekend,
    Unit::BusinessDay,
    Unit::Weekday,
    Unit::Day,
    Unit::Hour,
    Unit::Minute,
    Unit::Second,
];

/// A set word: its unit index and amount, `amount << 8 | unit`.
fn set_word(v: &Value) -> Option<(Unit, i32)> {
    let x = term(v).0;
    Some((*UNITS.get(usize::try_from(x & 255).ok()?)?, i32::try_from(x >> 8).ok()?))
}

fn b_word(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (u, milli) = set_word(&v[0])?;
    every(c, Duration::of(u, milli))
}

fn b_semi(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (u, milli) = set_word(&v[0])?;
    every(c, Duration::of(u, milli / 2))
}

fn b_bi(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (u, milli) = set_word(&v[0])?;
    (milli == MILLI).then(|| every(c, Duration::of(u, milli * 2)))?
}

/// `every day`, `each week`, `any weekend`; `every night` is the night.
fn b_every_unit(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (_, flags) = term(&v[0]);
    if flags & NIGHT != 0 {
        return at(c, Point::time(TimeOfDay::Part(Part::Night)));
    }
    if flags & PLURAL != 0 {
        return None;
    }
    every(c, length(i64::from(MILLI), &v[0])?)
}

fn b_every_n_unit(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    every(c, length(term(&v[0]).0, &v[1])?)
}

fn b_every_other_unit(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    every(c, length(i64::from(2 * MILLI), &v[0])?)
}

/// `weekends`: every weekend.
fn b_weekends(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (flags, (unit, _)) = (term(&v[0]).1, unit_of(&v[0])?);
    (flags & PLURAL != 0 && unit == Unit::Weekend).then(|| every(c, Duration::of(Unit::Weekend, MILLI)))?
}

fn b_every_weekday(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    at(c, weekday_point(term(&v[0]).0 as u8))
}

fn b_weekdays(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (wd, flags) = term(&v[0]);
    (flags & PLURAL != 0).then(|| at(c, weekday_point(wd as u8)))?
}

/// A point of a set at a clock time: `every Monday at 4pm`, `every day
/// at 7:13 p.m.`, `every evening at 7:13 p.m.`.
fn with_clock(c: &Ctx<'_>, set: &Value, clock: &Value) -> Option<Value> {
    let k = clock_of(c, clock)?;
    let r = set_of(c, set)?;
    let p = match r {
        _ if is_daily(r) => Point::default(),
        Recurrence::At(p) => p,
        Recurrence::Every(_) => return None,
    };
    let k = match p.time {
        TimeOfDay::None => k,
        TimeOfDay::Part(part) => k.with(part.meridiem(k.hour)),
        _ => return None,
    };
    at(c, Point { time: TimeOfDay::Clock(k), ..p })
}

fn b_set_clock(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    with_clock(c, &v[0], &v[1])
}

fn b_clock_set(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let r = set_of(c, &v[1])?;
    is_daily(r).then(|| with_clock(c, &v[1], &v[0]))?
}

fn b_weekdays_clock(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (wd, flags) = term(&v[0]);
    if flags & PLURAL == 0 {
        return None;
    }
    let k = clock_of(c, &v[1])?;
    at(c, Point { time: TimeOfDay::Clock(k), ..weekday_point(wd as u8) })
}

fn b_every_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    at(c, Point::time(TimeOfDay::Part(part(&v[0])?)))
}

fn b_parts(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (term(&v[0]).1 & PLURAL != 0).then(|| b_every_part(v, c))?
}

/// `Tuesday afternoons`.
fn b_weekday_parts(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (wd, flags) = term(&v[0]);
    if flags & PLURAL != 0 || term(&v[1]).1 & PLURAL == 0 {
        return None;
    }
    at(c, Point { time: TimeOfDay::Part(part(&v[1])?), ..weekday_point(wd as u8) })
}

fn b_day_of_every_month(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let d = term(&v[0]).0 as u8;
    at(c, Point::date(DateSpec::Cal { year: YearRef::Any, month: 0, day: d, weekday: 0 }))
}

const fn rr(name: &'static str, dim: crate::engine::Dim, items: &'static [Item], build: crate::engine::Build) -> Rule {
    r(name, dim, 0, items, build)
}

pub(super) static RULES: &[Rule] = &[
    rr("set_word", SET, &[node(SETWORD)], b_word),
    rr("set_semi", SET, &[word("semi"), opt_word("-"), node(SETWORD)], b_semi),
    rr("set_bi", SET, &[word("bi"), opt_word("-"), node(SETWORD)], b_bi),
    rr("set_every_unit", SET, &[word("every"), node(UNIT)], b_every_unit),
    rr("set_each_unit", SET, &[word("each"), node(UNIT)], b_every_unit),
    rr("set_any_unit", SET, &[word("any"), node(UNIT)], b_every_unit),
    rr("set_every_n_unit", SET, &[word("every"), node(AMOUNT), node(UNIT)], b_every_n_unit),
    rr("set_each_n_unit", SET, &[word("each"), node(AMOUNT), node(UNIT)], b_every_n_unit),
    rr("set_every_other_unit", SET, &[word("every"), word("other"), node(UNIT)], b_every_other_unit),
    rr("set_once_a", SET, &[word("once"), word("a"), node(UNIT)], b_every_unit),
    rr("set_one_a", SET, &[word("one"), word("a"), node(UNIT)], b_every_unit),
    rr("set_weekends", SET, &[node(UNIT)], b_weekends),
    rr("set_every_weekday", SET, &[word("every"), opt_word("other"), node(WEEKDAY)], b_every_weekday),
    rr("set_each_weekday", SET, &[word("each"), node(WEEKDAY)], b_every_weekday),
    rr("set_weekdays", SET, &[node(WEEKDAY)], b_weekdays),
    rr("set_weekdays_clock", SET, &[node(WEEKDAY), opt_word("at"), node(CLOCK)], b_weekdays_clock),
    rr("set_clock", SET, &[node(SET), opt_word("at"), node(CLOCK)], b_set_clock),
    rr("set_clock_first", SET, &[node(CLOCK), node(SET)], b_clock_set),
    rr("set_every_part", SET, &[word("every"), node(PART)], b_every_part),
    rr("set_each_part", SET, &[word("each"), node(PART)], b_every_part),
    rr("set_parts", SET, &[node(PART)], b_parts),
    rr("set_weekday_parts", SET, &[node(WEEKDAY), node(PART)], b_weekday_parts),
    rr("set_day_of_every_month", SET, &[node(MDAY), word("of"), word("every"), word("month")], b_day_of_every_month),
    rr(
        "set_the_day_of_every_month",
        SET,
        &[word("the"), node(MDAY), word("of"), word("every"), word("month")],
        b_day_of_every_month,
    ),
];
