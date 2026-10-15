//! Durations: `3 minuten`, `3,5 jaar`, `een uur`, `anderhalf uur`, `een
//! paar minuten`, `enige weken`, `de hele dag`, `1 uur en 30 minuten`,
//! `een jaar en een kwart`, `meer dan 4 dagen`.

use crate::engine::{Item, Rule, node, opt_word, word};
use crate::lang::en::time::duration as en;
use crate::lang::en::time::r;

use super::super::NUM;
use super::{AMOUNT, DUR, UNIT};

const fn rr(name: &'static str, dim: crate::engine::Dim, items: &'static [Item], build: crate::engine::Build) -> Rule {
    r(name, dim, 0, items, build)
}

pub(super) static RULES: &[Rule] = &[
    rr("amount_num", AMOUNT, &[node(NUM)], en::b_amount_num),
    rr("amount_paar", AMOUNT, &[word("paar")], en::b_amount_two),
    rr("amount_paar_een", AMOUNT, &[word("een"), word("paar")], en::b_amount_two),
    rr("amount_wat", AMOUNT, &[word("wat")], en::b_amount_three),
    rr("amount_enige", AMOUNT, &[word("enige")], en::b_amount_three),
    rr("amount_enkele", AMOUNT, &[word("enkele")], en::b_amount_three),
    rr("amount_meerdere", AMOUNT, &[word("meerdere")], en::b_amount_three),
    rr("amount_een_aantal", AMOUNT, &[word("een"), word("aantal")], en::b_amount_three),
    rr("amount_half", AMOUNT, &[word("een"), word("half")], en::b_amount_half),
    rr("amount_halve", AMOUNT, &[word("een"), word("halve")], en::b_amount_half),
    rr("dur", DUR, &[node(AMOUNT), opt_word("-"), node(UNIT)], en::b_dur),
    rr("dur_whole", DUR, &[word("hele"), node(UNIT)], en::b_dur_one),
    rr("dur_whole_het", DUR, &[word("het"), word("hele"), node(UNIT)], en::b_dur_one),
    rr("dur_whole_de", DUR, &[word("de"), word("hele"), node(UNIT)], en::b_dur_one),
    rr("dur_whole_heel", DUR, &[word("heel"), word("de"), node(UNIT)], en::b_dur_one),
    rr("dur_another", DUR, &[word("andere"), node(UNIT)], en::b_dur_one),
    rr("dur_sum", DUR, &[node(DUR), opt_word(","), opt_word("en"), node(DUR)], en::b_dur_sum),
    rr("dur_and_half", DUR, &[node(DUR), word("en"), opt_word("een"), word("half")], en::b_dur_and_half),
    rr("dur_and_quarter", DUR, &[node(DUR), word("en"), word("een"), word("kwart")], en::b_dur_and_quarter),
    rr("dur_more", DUR, &[word("meer"), word("dan"), node(DUR)], en::b_more),
    rr("dur_longer", DUR, &[word("langer"), word("dan"), node(DUR)], en::b_more),
    rr("dur_less", DUR, &[word("minder"), word("dan"), node(DUR)], en::b_less),
    rr("dur_shorter", DUR, &[word("korter"), word("dan"), node(DUR)], en::b_less),
];
