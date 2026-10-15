//! Recurrences: `semanalmente`, `cada dos días`, `todos los lunes`, `cada
//! lunes a las 4 p.m.`, `cada finde`, `una vez por semana`.

use crate::engine::{Ctx, Item, Rule, node, word};
use crate::lang::en::time::set as en;
use crate::lang::en::time::{r, term};
use crate::time::duration::MILLI;
use crate::value::Value;

use super::{AMOUNT, ART, BARE, CLOCK, PART, SET, SETWORD, UNIT, WEEKDAY};

/// `todos los días`, `todas las semanas`: every one of a plural unit.
fn b_all_units(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let unit = v.last()?;
    (term(unit).1 & super::PLURAL != 0).then(|| {
        let d = crate::lang::en::time::duration::length(i64::from(MILLI), unit)?;
        en::every(c, d)
    })?
}

fn b_last(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_every_weekday(&v[v.len() - 1..], c)
}

const fn rr(name: &'static str, dim: crate::engine::Dim, items: &'static [Item], build: crate::engine::Build) -> Rule {
    r(name, dim, 0, items, build)
}

pub(super) static RULES: &[Rule] = &[
    rr("set_word", SET, &[node(SETWORD)], en::b_word),
    rr("set_cada_unit", SET, &[word("cada"), node(UNIT)], en::b_every_unit),
    rr("set_cada_n_unit", SET, &[word("cada"), node(AMOUNT), node(UNIT)], en::b_every_n_unit),
    rr("set_todos_los_unit", SET, &[word("todos"), node(ART), node(UNIT)], b_all_units),
    rr("set_todas_las_unit", SET, &[word("todas"), node(ART), node(UNIT)], b_all_units),
    rr("set_una_vez_por", SET, &[word("una"), word("vez"), word("por"), node(UNIT)], en::b_every_unit),
    rr("set_una_vez_al", SET, &[word("una"), word("vez"), word("al"), node(UNIT)], en::b_every_unit),
    rr("set_cada_weekday", SET, &[word("cada"), node(WEEKDAY)], en::b_every_weekday),
    rr("set_todos_los_weekday", SET, &[word("todos"), node(ART), node(WEEKDAY)], b_last),
    rr("set_cada_part", SET, &[word("cada"), node(PART)], en::b_every_part),
    rr("set_todas_las_parts", SET, &[word("todas"), node(ART), node(PART)], |v, c| en::b_parts(&v[1..], c)),
    rr("set_clock", SET, &[node(SET), word("a"), node(ART), node(CLOCK)], |v, c| en::b_set_clock(&[v[0], v[2]], c)),
    rr("set_bare", SET, &[node(SET), word("a"), node(ART), node(BARE)], |v, c| en::b_set_clock(&[v[0], v[2]], c)),
];
