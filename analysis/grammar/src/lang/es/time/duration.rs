//! Durations: `30 minutos`, `123,45 segundos`, `una hora y media`, `unos
//! días`, `todo el día`, `un mes entero`, `tres fines de semana`, `más de
//! 4 días`.

use crate::engine::{Item, Rule, node, opt_word, word};
use crate::lang::en::time::duration as en;
use crate::lang::en::time::{r, term_value};
use crate::time::duration::MILLI;

use super::super::{NUM, VULGAR};
use super::{AMOUNT, DUR, UNIT};

const fn rr(name: &'static str, dim: crate::engine::Dim, items: &'static [Item], build: crate::engine::Build) -> Rule {
    r(name, dim, 0, items, build)
}

pub(super) static RULES: &[Rule] = &[
    rr("amount_num", AMOUNT, &[node(NUM)], en::b_amount_num),
    rr("amount_vulgar", AMOUNT, &[node(VULGAR)], en::b_amount_num),
    rr("amount_unos", AMOUNT, &[word("unos")], en::b_amount_three),
    rr("amount_unas", AMOUNT, &[word("unas")], en::b_amount_three),
    rr("amount_algunos", AMOUNT, &[word("algunos")], en::b_amount_three),
    rr("amount_algunas", AMOUNT, &[word("algunas")], en::b_amount_three),
    rr("amount_varios", AMOUNT, &[word("varios")], en::b_amount_three),
    rr("amount_varias", AMOUNT, &[word("varias")], en::b_amount_three),
    rr("amount_par", AMOUNT, &[word("un"), word("par"), word("de")], en::b_amount_two),
    rr("amount_media", AMOUNT, &[word("media")], |_, _| term_value(i64::from(MILLI / 2), en::ARTICLE_AMOUNT)),
    rr("dur", DUR, &[node(AMOUNT), node(UNIT)], en::b_dur),
    rr("dur_letter_h", DUR, &[node(AMOUNT), word("h")], en::b_dur_letter),
    rr("dur_todo_el", DUR, &[word("todo"), word("el"), node(UNIT)], en::b_dur_one),
    rr("dur_toda_la", DUR, &[word("toda"), word("la"), node(UNIT)], en::b_dur_one),
    rr("dur_otro", DUR, &[word("otro"), node(UNIT)], en::b_dur_one),
    rr("dur_otra", DUR, &[word("otra"), node(UNIT)], en::b_dur_one),
    rr("dur_entero", DUR, &[node(DUR), word("entero")], |v, _| Some(v[0])),
    rr("dur_entera", DUR, &[node(DUR), word("entera")], |v, _| Some(v[0])),
    rr("dur_sum", DUR, &[node(DUR), opt_word(","), opt_word("y"), node(DUR)], en::b_dur_sum),
    rr("dur_and_half", DUR, &[node(DUR), word("y"), word("media")], en::b_dur_and_half),
    rr("dur_and_half_m", DUR, &[node(DUR), word("y"), word("medio")], en::b_dur_and_half),
    rr("dur_and_quarter", DUR, &[node(DUR), word("y"), word("cuarto")], en::b_dur_and_quarter),
    rr("dur_more", DUR, &[word("mas"), word("de"), node(DUR)], en::b_more),
    rr("dur_less", DUR, &[word("menos"), word("de"), node(DUR)], en::b_less),
];
