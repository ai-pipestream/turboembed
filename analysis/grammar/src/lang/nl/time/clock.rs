//! Times of day: `19:56:30`, `15.00`, `3u`, `20.15u`, `7 uur`, `half
//! acht` (7:30), `kwart over acht`, `10 voor half negen` (8:20), `het
//! middaguur`, and the parts of the day: `'s avonds`, `in de ochtend`,
//! `in de vroege ochtend`.

use crate::engine::{Ctx, Item, Rule, glued, glued_word, node, opt_word, word};
use crate::lang::en::time::clock as en;
use crate::lang::en::time::{clock_of, emit_clock, r, term};
use crate::time::spec::{Clock, Meridiem, Part, Shown};
use crate::value::Value;

use super::super::{INT, NUMERAL};
use super::{APOS, BARE, CLOCK, DIG, GENITIVE, HALF, OF_PART, PART, PARTD};

/// The clock `delta` minutes from an hour or a half hour: `kwart over
/// acht` is 8:15, `half acht` 7:30, `10 voor half negen` 8:20.
fn moved(c: &Ctx<'_>, v: &Value, delta: i64) -> Option<Value> {
    let k = clock_of(c, v)?;
    if !(k.shown == Shown::Hour || k.minute == 30 && k.second == 0) {
        return None;
    }
    let total = (i64::from(k.hour) * 60 + i64::from(k.minute) + delta).rem_euclid(24 * 60);
    let (hour, minute) = ((total / 60) as u8, (total % 60) as u8);
    // `half een` is 12:30, of the half of the day the hour leaves open.
    let hour = if hour == 0 && k.meridiem != Meridiem::Day { 12 } else { hour };
    let meridiem = if hour > 12 { Meridiem::Day } else { k.meridiem };
    emit_clock(c, Clock::new(hour, minute, 0, Shown::Minute, meridiem)?)
}

/// A count of minutes before or after an hour, 1 to 29.
fn minutes(v: &Value) -> Option<i64> {
    let m = v.num()?.value.to_i64()?;
    (1..30).contains(&m).then_some(m)
}

fn b_half(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    moved(c, &v[0], -30)
}

fn b_quarter_over(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    moved(c, v.last()?, 15)
}

fn b_quarter_to(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    moved(c, v.last()?, -15)
}

fn b_minutes_over(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    moved(c, v.last()?, minutes(&v[0])?)
}

fn b_minutes_to(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    moved(c, v.last()?, -minutes(&v[0])?)
}

/// A clock time said with `uur` after it that has its minutes: `15.00
/// uur`.
fn b_minutes_uur(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (clock_of(c, &v[0])?.shown != Shown::Hour).then_some(v[0])
}

fn b_same(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(v[0])
}

/// A part of the day after `'s`, in its `-s` form: `'s avonds`.
fn b_genitive_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (term(v.last()?).1 & GENITIVE != 0).then(|| en::b_part_any(v, c))?
}

/// A part of the day in its plain form: `in de ochtend`.
fn b_plain_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (term(v.last()?).1 == 0).then(|| en::b_part_any(v, c))?
}

/// A meal said with its article: `het ontbijt`, `de lunch`.
fn b_meal(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let p = en::part(v.last()?)?;
    en::is_meal(p).then(|| en::b_part_any(v, c))?
}

/// The half of the day of an open hour in a part of the day; 12 in the
/// morning is noon, 12 at night midnight.
pub(super) fn part_meridiem(p: Part, hour: u8) -> Meridiem {
    match (p, hour) {
        (Part::Morning, 12) => Meridiem::Pm,
        (Part::Night, 12) => Meridiem::Am,
        _ => p.meridiem(hour),
    }
}

/// A clock and a part of the day said after it: `5 uur 's middags`, `4u
/// 's morgens`, `12 uur 's ochtends` (noon).
fn b_clock_of_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let k = clock_of(c, &v[0])?;
    let p = en::part_value(c, &v[1])?;
    if k.hour > 12 && k.meridiem != Meridiem::Day {
        return None;
    }
    emit_clock(c, k.with(part_meridiem(p, k.hour)))
}

/// A spelled hour, not the article `een`: `een uur` is an hour long.
fn b_bare_word(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if super::super::plain_een(c) {
        return None;
    }
    en::b_bare_word(v, c)
}

/// `12.23:59`: hours and minutes with a dot, then seconds.
fn b_dot_seconds(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let k = clock_of(c, &en::b_bare_dot(v, c)?)?;
    let s = term(&v[1]).0;
    if en::dig_len(&v[1]) != 2 || s > 59 {
        return None;
    }
    emit_clock(c, Clock { second: s as u8, shown: Shown::Second, ..k })
}

fn b_noon(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_noon(v, c)
}

/// The rest of a part's rules: `in de vroege ochtend`, `laat in de
/// avond`.
fn b_early(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (term(v.last()?).1 == 0).then(|| en::b_early_part(v, c))?
}

fn b_late(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (term(v.last()?).1 == 0).then(|| en::b_late_part(v, c))?
}

/// `het middaguur`, `rond de middag`: noon.
fn b_afternoon_noon(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (en::part(v.last()?)? == Part::Afternoon).then(|| b_noon(v, c))?
}

const fn rr(name: &'static str, dim: crate::engine::Dim, items: &'static [Item], build: crate::engine::Build) -> Rule {
    r(name, dim, 0, items, build)
}

pub(super) static RULES: &[Rule] = &[
    rr("dig", DIG, &[node(NUMERAL)], en::b_dig),
    // Hours that need context.
    rr("bare_dig", BARE, &[node(DIG)], en::b_bare_dig),
    rr("bare_word", BARE, &[node(INT)], b_bare_word),
    rr("bare_dot", BARE, &[node(NUMERAL)], en::b_bare_dot),
    rr("apos", APOS, &[word("'"), glued_word("s")], |_, _| Some(Value::Empty)),
    rr("apos_open", APOS, &[word("\u{2018}"), glued_word("s")], |_, _| Some(Value::Empty)),
    rr("apos_close", APOS, &[word("\u{2019}"), glued_word("s")], |_, _| Some(Value::Empty)),
    // Clock times.
    rr("clock_hm", CLOCK, &[node(DIG), glued_word(":"), glued(DIG)], en::b_hm),
    rr("clock_hms", CLOCK, &[node(DIG), glued_word(":"), glued(DIG), glued_word(":"), glued(DIG)], en::b_hms),
    rr("clock_hhmm", CLOCK, &[node(DIG)], en::b_hhmm_zero),
    rr("clock_dot", CLOCK, &[node(NUMERAL)], en::b_bare_dot),
    rr("clock_dot_seconds", CLOCK, &[node(NUMERAL), glued_word(":"), glued(DIG)], b_dot_seconds),
    rr("clock_om_dot", CLOCK, &[word("om"), node(CLOCK)], en::b_at_dot),
    rr("clock_uur", CLOCK, &[node(BARE), word("uur")], en::b_same_clock),
    rr("clock_c_uur", CLOCK, &[node(CLOCK), word("uur")], b_minutes_uur),
    rr("clock_u", CLOCK, &[node(BARE), glued_word("u")], en::b_same_clock),
    rr("clock_c_u", CLOCK, &[node(CLOCK), glued_word("u")], b_minutes_uur),
    rr("half", HALF, &[word("half"), node(BARE)], b_half),
    rr("clock_half", CLOCK, &[node(HALF)], b_same),
    rr("clock_quarter_over", CLOCK, &[word("kwart"), word("over"), node(BARE)], b_quarter_over),
    rr("clock_quarter_over_half", CLOCK, &[word("kwart"), word("over"), node(HALF)], b_quarter_over),
    rr("clock_quarter_to", CLOCK, &[word("kwart"), word("voor"), node(BARE)], b_quarter_to),
    rr("clock_quarter_to_half", CLOCK, &[word("kwart"), word("voor"), node(HALF)], b_quarter_to),
    rr(
        "clock_minutes_over",
        CLOCK,
        &[node(INT), opt_word("minuten"), opt_word("min"), word("over"), node(BARE)],
        b_minutes_over,
    ),
    rr(
        "clock_minutes_over_half",
        CLOCK,
        &[node(INT), opt_word("minuten"), opt_word("min"), word("over"), node(HALF)],
        b_minutes_over,
    ),
    rr(
        "clock_minutes_to",
        CLOCK,
        &[node(INT), opt_word("minuten"), opt_word("min"), word("voor"), node(BARE)],
        b_minutes_to,
    ),
    rr(
        "clock_minutes_to_half",
        CLOCK,
        &[node(INT), opt_word("minuten"), opt_word("min"), word("voor"), node(HALF)],
        b_minutes_to,
    ),
    rr("clock_middaguur", CLOCK, &[word("middaguur")], en::b_noon),
    rr("clock_middaguur_het", CLOCK, &[word("het"), word("middaguur")], en::b_noon),
    rr("clock_around_noon", CLOCK, &[word("rond"), word("de"), node(PART)], b_afternoon_noon),
    rr("clock_middernacht", CLOCK, &[word("middernacht")], en::b_midnight),
    // Parts of the day.
    rr("of_part_apos", OF_PART, &[node(APOS), node(PART)], b_genitive_part),
    rr("of_part_in", OF_PART, &[word("in"), word("de"), node(PART)], b_plain_part),
    rr("partd", PARTD, &[node(PART)], en::b_part),
    rr("partd_of", PARTD, &[node(OF_PART)], b_same),
    rr("partd_meal", PARTD, &[word("het"), node(PART)], b_meal),
    rr("partd_meal_de", PARTD, &[word("de"), node(PART)], b_meal),
    rr("partd_during", PARTD, &[word("tijdens"), opt_word("de"), opt_word("het"), node(PART)], en::b_part_any),
    rr("partd_early_in", PARTD, &[word("in"), word("de"), word("vroege"), node(PART)], b_early),
    rr("partd_early", PARTD, &[word("vroeg"), word("in"), word("de"), node(PART)], b_early),
    rr("partd_late_in", PARTD, &[word("in"), word("de"), word("late"), node(PART)], b_late),
    rr("partd_late", PARTD, &[word("laat"), word("in"), word("de"), node(PART)], b_late),
    rr("partd_later", PARTD, &[word("later"), word("in"), word("de"), node(PART)], b_late),
    // A clock in a part of the day.
    rr("clock_of_part", CLOCK, &[node(BARE), node(OF_PART)], b_clock_of_part),
    rr("clock_c_of_part", CLOCK, &[node(CLOCK), node(OF_PART)], b_clock_of_part),
    rr("clock_part_at", CLOCK, &[node(PARTD), word("om"), node(CLOCK)], en::b_part_clock),
    rr("clock_part_at_bare", CLOCK, &[node(PARTD), word("om"), node(BARE)], en::b_part_clock),
];
