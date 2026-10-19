//! Times of day: `7:56:30`, `5 p.m.`, `14h`, `siete y media`, `ocho y
//! veinte de la noche`, `seis menos cinco`, `8 menos cuarto`, `mediodía`,
//! and the parts of the day: `por la mañana`, `temprano en la mañana`,
//! `pasado el mediodía`.

use turbo_analysis_layers::Modifier;

use crate::engine::{Ctx, Item, Rule, glued, glued_word, node, opt_word, word};
use crate::lang::en::time::clock as en;
use crate::lang::en::time::{clock_of, emit_clock, r};
use crate::time::spec::{Clock, Meridiem, Part, Shown};
use crate::value::Value;

use super::super::{INT, NUMERAL};
use super::{AMPM, ART, BARE, CLOCK, DE, DIG, OF_PART, PART, PARTD, after_article};

/// An hour in one or two digits; `00` alone names no hour.
fn b_bare_dig(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (c.raw(c.start) != "00").then(|| en::b_bare_dig(v, c))?
}

/// A spelled count of minutes, 1 to 59.
fn minutes(v: &Value) -> Option<i64> {
    let n = v.num()?;
    (n.has(crate::value::flag::SPELLED) && !n.has(crate::value::flag::DIGITS)).then_some(())?;
    let m = n.value.to_i64()?;
    (1..60).contains(&m).then_some(m)
}

/// An hour and `m` minutes after it: `siete y media`.
fn after(c: &Ctx<'_>, v: &Value, m: i64) -> Option<Value> {
    let k = clock_of(c, v)?;
    if k.shown != Shown::Hour || !(1..60).contains(&m) {
        return None;
    }
    emit_clock(c, Clock { minute: m as u8, shown: Shown::Minute, ..k })
}

/// `m` minutes before an hour: `seis menos cinco` is 5:55.
fn before(c: &Ctx<'_>, v: &Value, m: i64) -> Option<Value> {
    let k = clock_of(c, v)?;
    if k.shown != Shown::Hour || !(1..60).contains(&m) {
        return None;
    }
    let hour = if k.hour == 0 { 23 } else { k.hour - 1 };
    // `1 menos cuarto` is 12:45; a known half stays the half of the hour
    // said.
    let hour = if hour == 0 && k.meridiem != Meridiem::Day { 12 } else { hour };
    let meridiem = if k.hour == 12 && k.meridiem == Meridiem::Pm { Meridiem::Am } else { k.meridiem };
    let meridiem = if hour > 12 { Meridiem::Day } else { meridiem };
    emit_clock(c, Clock::new(hour, (60 - m) as u8, 0, Shown::Minute, meridiem)?)
}

fn b_half(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    after(c, &v[0], 30)
}

fn b_quarter(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    after(c, &v[0], 15)
}

/// `una y veintidós`: minutes in words after `y`; small ones only after
/// `las`, as `dos y tres` more often counts.
fn b_and_minutes(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let m = minutes(&v[1])?;
    (m >= 10 || after_article(c)).then(|| after(c, &v[0], m))?
}

fn b_less_quarter(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    before(c, &v[0], 15)
}

fn b_less_minutes(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    before(c, &v[0], minutes(&v[1])?)
}

/// A clock and a part of the day said after it: `5 de la tarde`.
fn b_clock_of_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let k = en::clock_in_part(c, &v[0], &v[1])?;
    if k.hour > 12 && k.meridiem != Meridiem::Day {
        return None;
    }
    emit_clock(c, k)
}

/// A part of the day, from the `PART` term last among the children.
fn b_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_part_any(v, c)
}

fn b_afternoon(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::emit_part(c, Part::Afternoon, Modifier::NONE)
}

/// The end of a part of the day said as a reading: `al final del pasado
/// mediodía`, `más tarde pasado el mediodía`.
fn b_late_partd(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let p = en::part_value(c, v.last()?)?;
    if !matches!(p, Part::Morning | Part::Afternoon | Part::Evening | Part::Night) {
        return None;
    }
    en::emit_part(c, p, Modifier::portion(turbo_analysis_layers::Portion::End))
}

const fn rr(name: &'static str, dim: crate::engine::Dim, items: &'static [Item], build: crate::engine::Build) -> Rule {
    r(name, dim, 0, items, build)
}

pub(super) static RULES: &[Rule] = &[
    // Halves of the day.
    rr("ampm_am", AMPM, &[word("am")], en::b_am),
    rr("ampm_pm", AMPM, &[word("pm")], en::b_pm),
    rr("ampm_a_m", AMPM, &[word("a"), word("."), word("m"), opt_word(".")], en::b_am),
    rr("ampm_p_m", AMPM, &[word("p"), word("."), word("m"), opt_word(".")], en::b_pm),
    // Digits.
    rr("dig", DIG, &[node(NUMERAL)], en::b_dig),
    // Hours that need `las` before them.
    rr("bare_dig", BARE, &[node(DIG)], b_bare_dig),
    rr("bare_word", BARE, &[node(INT)], en::b_bare_word),
    rr("bare_dot", BARE, &[node(NUMERAL)], en::b_bare_dot),
    // Clock times.
    rr("clock_hm", CLOCK, &[node(DIG), glued_word(":"), glued(DIG)], en::b_hm),
    rr("clock_hms", CLOCK, &[node(DIG), glued_word(":"), glued(DIG), glued_word(":"), glued(DIG)], en::b_hms),
    rr("clock_hhmm", CLOCK, &[node(DIG)], en::b_hhmm_zero),
    rr("clock_bare_ampm", CLOCK, &[node(BARE), node(AMPM)], en::b_clock_ampm),
    rr("clock_ampm", CLOCK, &[node(CLOCK), node(AMPM)], en::b_clock_ampm),
    rr("clock_bare_h", CLOCK, &[node(BARE), glued_word("h")], en::b_same_clock),
    rr("clock_h", CLOCK, &[node(CLOCK), word("h")], |v, _| Some(v[0])),
    rr("clock_en_punto", CLOCK, &[node(BARE), word("en"), word("punto")], en::b_same_clock),
    rr("clock_y_media", CLOCK, &[node(BARE), word("y"), word("media")], b_half),
    rr("clock_y_cuarto", CLOCK, &[node(BARE), word("y"), word("cuarto")], b_quarter),
    rr("clock_y_minutes", CLOCK, &[node(BARE), word("y"), node(INT)], b_and_minutes),
    rr("clock_word_minutes", CLOCK, &[node(INT), node(INT)], en::b_bare_word_minutes),
    rr("clock_menos_cuarto", CLOCK, &[node(BARE), word("menos"), word("cuarto")], b_less_quarter),
    rr("clock_menos_minutes", CLOCK, &[node(BARE), word("menos"), node(INT)], b_less_minutes),
    rr("clock_c_menos_cuarto", CLOCK, &[node(CLOCK), word("menos"), word("cuarto")], b_less_quarter),
    rr("clock_mediodia", CLOCK, &[word("mediodia")], en::b_noon),
    rr("clock_medio_dia", CLOCK, &[word("medio"), word("dia")], en::b_noon),
    rr("clock_medianoche", CLOCK, &[word("medianoche")], en::b_midnight),
    // Parts of the day.
    rr("of_part", OF_PART, &[node(DE), node(ART), node(PART)], b_part),
    rr("of_part_a_la", OF_PART, &[word("a"), word("la"), node(PART)], b_part),
    rr("partd_of", PARTD, &[node(OF_PART)], |v, _| Some(v[0])),
    rr("partd_por_la", PARTD, &[word("por"), word("la"), node(PART)], b_part),
    rr("partd_en_la", PARTD, &[word("en"), word("la"), node(PART)], b_part),
    rr("partd_early", PARTD, &[word("temprano"), word("en"), word("la"), node(PART)], en::b_early_part),
    rr("partd_later_en", PARTD, &[word("mas"), word("tarde"), word("en"), word("la"), node(PART)], en::b_late_part),
    rr("partd_later_por", PARTD, &[word("mas"), word("tarde"), word("por"), word("la"), node(PART)], en::b_late_part),
    rr("partd_end", PARTD, &[word("final"), word("de"), word("la"), node(PART)], en::b_late_part),
    rr("partd_afternoon", PARTD, &[word("pasado"), opt_word("el"), word("mediodia")], b_afternoon),
    rr("partd_afternoon_dia", PARTD, &[word("pasado"), opt_word("el"), word("medio"), word("dia")], b_afternoon),
    rr("partd_later_partd", PARTD, &[word("mas"), word("tarde"), node(PARTD)], b_late_partd),
    rr("partd_later_por_el", PARTD, &[word("mas"), word("tarde"), word("por"), word("el"), node(PARTD)], b_late_partd),
    rr("partd_end_partd", PARTD, &[word("final"), node(DE), node(PARTD)], b_late_partd),
    // A clock in a part of the day.
    rr("clock_of_part", CLOCK, &[node(BARE), node(OF_PART)], b_clock_of_part),
    rr("clock_c_of_part", CLOCK, &[node(CLOCK), node(OF_PART)], b_clock_of_part),
    rr("clock_part_at", CLOCK, &[node(PARTD), word("a"), word("las"), node(CLOCK)], en::b_part_clock),
    rr("clock_part_at_bare", CLOCK, &[node(PARTD), word("a"), word("las"), node(BARE)], en::b_part_clock),
];
