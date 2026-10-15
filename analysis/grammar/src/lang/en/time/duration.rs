//! Durations: `3 minutes`, `3.5years`, `an hour`, `a few days`, `all
//! week`, `one hour and a half`, `1 hour and 30 minutes`, `more than 4
//! days`.

use turbo_analysis_layers::{Modifier, Relation};

use crate::engine::{Ctx, Item, Rule, glued_word, node, opt_word, word};
use crate::time::TimeValue;
use crate::time::duration::{Duration, MILLI, Unit};
use crate::value::Value;

use super::super::{ARTICLE, NUM, VULGAR};
use super::lexicon::{self, NIGHT};
use super::{AMOUNT, DUR, PLURAL, UNIT, emit, r, term, term_value, tval};

/// An amount said with an article, which a lone `a second` is not.
pub(crate) const ARTICLE_AMOUNT: u8 = 1;
/// An amount said vaguely: `a few`, `some`, `several`.
pub(crate) const VAGUE: u8 = 1 << 1;

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

/// The unit a `UNIT` word counts in and how many of it the word is: a
/// decade is ten years, a quarter three months.
pub(crate) fn unit_of(v: &Value) -> Option<(Unit, i32)> {
    let u = term(v).0;
    Some((*UNITS.get(usize::try_from(u & 255).ok()?)?, i32::try_from(u >> 8).ok()?))
}

/// `amount` thousandths of the unit word `v`.
pub(crate) fn length(amount: i64, v: &Value) -> Option<Duration> {
    let (unit, times) = unit_of(v)?;
    let milli = i32::try_from(amount).ok()?.checked_mul(times)?;
    (milli > 0).then(|| Duration::of(unit, milli))
}

/// A child's duration when it carries no modifier.
pub(crate) fn duration_of(c: &Ctx<'_>, v: &Value) -> Option<Duration> {
    match tval(c, v)? {
        TimeValue::Duration { length, modifier } if modifier.is_none() => Some(length),
        _ => None,
    }
}

pub(crate) fn emit_duration(c: &Ctx<'_>, length: Duration, modifier: Modifier) -> Option<Value> {
    emit(c, TimeValue::Duration { length, modifier })
}

// Amounts, in thousandths.

fn b_amount_num(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let n = v[0].num()?.value;
    if n.is_negative() || n.is_zero() {
        return None;
    }
    term_value(n.shift(3)?.to_i64()?, 0)
}

fn b_amount_article(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    term_value(i64::from(MILLI), ARTICLE_AMOUNT)
}

fn b_amount_three(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    term_value(3000, VAGUE)
}

fn b_amount_two(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    term_value(2000, VAGUE)
}

fn b_amount_half(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    term_value(500, ARTICLE_AMOUNT)
}

/// `one and half`: a number and a half the number rules leave apart.
fn b_amount_and_half(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (n, _) = term(&v[0]);
    (n % 1000 == 0).then(|| term_value(n + 500, 0))?
}

// Durations.

fn b_dur(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (amount, flags) = term(&v[0]);
    let (unit, _) = unit_of(&v[1])?;
    // `a second` is an ordinal more often than a duration.
    if flags & ARTICLE_AMOUNT != 0 && unit == Unit::Second && amount == 1000 {
        return None;
    }
    // `one night` counts days, but `a night` is a part of the day.
    if term(&v[1]).1 & NIGHT != 0 && flags & ARTICLE_AMOUNT != 0 {
        return None;
    }
    emit_duration(c, length(amount, &v[1])?, Modifier::NONE)
}

/// `3h`, `10d`, `2w`, `3y`: a number and a unit letter.
fn b_dur_letter(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (amount, _) = term(&v[0]);
    let unit = match c.word(c.end - 1) {
        "h" => Unit::Hour,
        "d" => Unit::Day,
        "w" => Unit::Week,
        "y" => Unit::Year,
        _ => return None,
    };
    emit_duration(c, Duration::of(unit, i32::try_from(amount).ok()?), Modifier::NONE)
}

/// `all day`, `another day`: one of the unit.
fn b_dur_one(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(v.last()?).1 & PLURAL != 0 {
        return None;
    }
    emit_duration(c, length(i64::from(MILLI), v.last()?)?, Modifier::NONE)
}

/// `for the hour`.
fn b_dur_the_hour(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (term(&v[0]).0 == lexicon::unit(lexicon::HOUR, 1)).then(|| b_dur_one(v, c))?
}

fn b_dur_sum(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (a, b) = (duration_of(c, &v[0])?, duration_of(c, &v[1])?);
    // Largest unit first: `1 hour and 30 minutes`, not `30 minutes 1 hour`.
    if a.finest()? >= b.parts().first()?.0 {
        return None;
    }
    emit_duration(c, a.plus(&b)?, Modifier::NONE)
}

/// `one hour and a half`: a fraction more of the duration's one unit.
fn and_fraction(c: &Ctx<'_>, v: &Value, milli: i32) -> Option<Value> {
    let d = duration_of(c, v)?;
    let &[(unit, n)] = d.parts() else { return None };
    emit_duration(c, Duration::of(unit, n.checked_add(milli)?), Modifier::NONE)
}

fn b_dur_and_half(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    and_fraction(c, &v[0], 500)
}

fn b_dur_and_quarter(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    and_fraction(c, &v[0], 250)
}

fn modified(c: &Ctx<'_>, v: &Value, relation: Relation) -> Option<Value> {
    emit_duration(c, duration_of(c, v)?, Modifier::relation(relation))
}

fn b_more(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    modified(c, v.last()?, Relation::More)
}

fn b_less(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    modified(c, v.last()?, Relation::Less)
}

/// `business day`: a working day.
fn b_business(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (u, f) = term(&v[0]);
    (u == lexicon::unit(lexicon::DAY, 1)).then(|| term_value(lexicon::unit(lexicon::BUSINESS_DAY, 1), f))?
}

const fn rr(name: &'static str, dim: crate::engine::Dim, items: &'static [Item], build: crate::engine::Build) -> Rule {
    r(name, dim, 0, items, build)
}

pub(super) static RULES: &[Rule] = &[
    rr("unit_business", UNIT, &[word("business"), node(UNIT)], b_business),
    rr("unit_working", UNIT, &[word("working"), node(UNIT)], b_business),
    rr("amount_num", AMOUNT, &[node(NUM)], b_amount_num),
    rr("amount_vulgar", AMOUNT, &[node(VULGAR)], b_amount_num),
    rr("amount_article", AMOUNT, &[node(ARTICLE)], b_amount_article),
    rr("amount_few", AMOUNT, &[word("few")], b_amount_three),
    rr("amount_a_few", AMOUNT, &[word("a"), word("few")], b_amount_three),
    rr("amount_some", AMOUNT, &[word("some")], b_amount_three),
    rr("amount_several", AMOUNT, &[word("several")], b_amount_three),
    rr("amount_couple", AMOUNT, &[word("couple"), opt_word("of")], b_amount_two),
    rr("amount_a_couple", AMOUNT, &[word("a"), word("couple"), opt_word("of")], b_amount_two),
    rr("amount_half_a", AMOUNT, &[word("half"), node(ARTICLE)], b_amount_half),
    rr("amount_and_half", AMOUNT, &[node(AMOUNT), word("and"), word("half")], b_amount_and_half),
    rr("dur", DUR, &[node(AMOUNT), opt_word("-"), node(UNIT)], b_dur),
    rr("dur_letter_h", DUR, &[node(AMOUNT), glued_word("h")], b_dur_letter),
    rr("dur_letter_d", DUR, &[node(AMOUNT), glued_word("d")], b_dur_letter),
    rr("dur_letter_w", DUR, &[node(AMOUNT), glued_word("w")], b_dur_letter),
    rr("dur_letter_y", DUR, &[node(AMOUNT), glued_word("y")], b_dur_letter),
    rr("dur_all", DUR, &[word("all"), node(UNIT)], b_dur_one),
    rr("dur_another", DUR, &[word("another"), node(UNIT)], b_dur_one),
    rr("dur_the_hour", DUR, &[word("the"), node(UNIT)], b_dur_the_hour),
    rr("dur_sum", DUR, &[node(DUR), opt_word(","), opt_word("and"), node(DUR)], b_dur_sum),
    rr("dur_and_half", DUR, &[node(DUR), word("and"), opt_word("a"), word("half")], b_dur_and_half),
    rr("dur_and_quarter", DUR, &[node(DUR), word("and"), word("a"), word("quarter")], b_dur_and_quarter),
    rr("dur_more", DUR, &[word("more"), word("than"), node(DUR)], b_more),
    rr("dur_less", DUR, &[word("less"), word("than"), node(DUR)], b_less),
];
