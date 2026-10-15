//! Recurrences: `wekelijks`, `elke dag`, `elke twee dagen`, `eenmaal per
//! week`, `elke maandag om 16.00`, `elke avond om 19.13`, `maandags`,
//! `om de week vrijdag`.

use crate::engine::{Item, Rule, node, opt_word, word};
use crate::lang::en::time::r;
use crate::lang::en::time::set as en;

use super::{AMOUNT, BARE, CLOCK, PART, SET, SETWORD, UNIT, WEEKDAY};

const fn rr(name: &'static str, dim: crate::engine::Dim, items: &'static [Item], build: crate::engine::Build) -> Rule {
    r(name, dim, 0, items, build)
}

pub(super) static RULES: &[Rule] = &[
    rr("set_word", SET, &[node(SETWORD)], en::b_word),
    rr("set_every_unit", SET, &[word("elke"), node(UNIT)], en::b_every_unit),
    rr("set_every_unit_elk", SET, &[word("elk"), node(UNIT)], en::b_every_unit),
    rr("set_each_unit", SET, &[word("iedere"), node(UNIT)], en::b_every_unit),
    rr("set_each_unit_ieder", SET, &[word("ieder"), node(UNIT)], en::b_every_unit),
    rr("set_every_n_unit", SET, &[word("elke"), node(AMOUNT), node(UNIT)], en::b_every_n_unit),
    rr("set_each_n_unit", SET, &[word("iedere"), node(AMOUNT), node(UNIT)], en::b_every_n_unit),
    rr("set_once_per", SET, &[word("eenmaal"), word("per"), node(UNIT)], en::b_every_unit),
    rr("set_once_a", SET, &[word("een"), word("keer"), word("per"), node(UNIT)], en::b_every_unit),
    rr("set_every_weekday", SET, &[word("elke"), node(WEEKDAY)], en::b_every_weekday),
    rr("set_each_weekday", SET, &[word("iedere"), node(WEEKDAY)], en::b_every_weekday),
    rr("set_every_n_unit_om", SET, &[word("om"), word("de"), node(AMOUNT), node(UNIT)], en::b_every_n_unit),
    rr("set_other_week_weekday", SET, &[word("om"), word("de"), word("week"), node(WEEKDAY)], en::b_every_weekday),
    rr("set_weekdays", SET, &[node(WEEKDAY)], en::b_weekdays),
    rr("set_clock", SET, &[node(SET), opt_word("om"), node(CLOCK)], en::b_set_clock),
    rr("set_bare", SET, &[node(SET), word("om"), node(BARE)], en::b_set_clock),
    rr("set_every_part", SET, &[word("elke"), node(PART)], en::b_every_part),
    rr("set_each_part", SET, &[word("iedere"), node(PART)], en::b_every_part),
];
