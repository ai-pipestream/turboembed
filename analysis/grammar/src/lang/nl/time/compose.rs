//! Dates with times of day: `morgen 8:00`, `vandaag om 9:00`, `20.30
//! vandaag`, `dinsdag in de ochtend`, `morgenavond`, `vanmiddag`, `deze
//! avond`, `afgelopen nacht om 8`; the present (`nu`); and points moved
//! by a duration: `over 5 minuten`, `een paar maanden geleden`, `twee
//! dagen na vandaag`.

use turbo_analysis_layers::{Modifier, TimeGrain};

use crate::engine::{Ctx, Item, Rule, glued, glued_word, node, opt_word, word};
use crate::lang::en::time::compose as en;
use crate::lang::en::time::lexicon::rel;
use crate::lang::en::time::{clock_of, r, term};
use crate::time::spec::{DateSpec, Part, Shown};
use crate::value::Value;

use super::{BARE, CLOCK, DATE, DT, DTP, DUR, HOL, PART, PARTD, RELW};

/// A day and a part of it written as one word: `morgenavond`,
/// `maandagmiddag`.
fn b_date_glued_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let p = en::single_part(&v[1])?;
    en::day_part(c, en::day_of(c, &v[0])?, p, Modifier::NONE)
}

/// `maandagmorgen`, `zondagmorgen`: the morning.
fn b_date_morning(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::day_part(c, en::day_of(c, &v[0])?, Part::Morning, Modifier::NONE)
}

/// `vanavond`, `vanochtend`: today's part.
fn b_today_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::day_part(c, en::today(), en::single_part(&v[0])?, Modifier::NONE)
}

fn b_this_morning(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::day_part(c, en::today(), Part::Morning, Modifier::NONE)
}

/// `deze avond`, `komende avond`, `afgelopen nacht`.
fn b_rel_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (n, kind) = term(&v[0]);
    let n = match kind {
        rel::THIS => 0,
        rel::NEXT | rel::COMING | rel::UPCOMING | rel::LAST | rel::PAST => n,
        _ => return None,
    };
    let p = en::single_part(&v[1])?;
    en::day_part(c, DateSpec::Rel { grain: TimeGrain::Day, n: n as i16 }, p, Modifier::NONE)
}

/// An hour before a day's part written as one word or after a comma: `8
/// vanmorgen`, `10, vanavond`, not `gebouw 4 deze middag`.
fn b_bare_dtp(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let comma = c.word(c.start + 1) == ",";
    let from = if comma { c.start + 2 } else { c.start + 1 };
    let one_word = (from + 1..c.end).all(|i| c.lexeme(i).glued);
    (comma || one_word || clock_of(c, &v[0])?.shown != Shown::Hour).then(|| en::b_clock_dtp(v, c))?
}

const fn rr(name: &'static str, dim: crate::engine::Dim, items: &'static [Item], build: crate::engine::Build) -> Rule {
    r(name, dim, 0, items, build)
}

pub(super) static RULES: &[Rule] = &[
    // A date and a clock time.
    rr("dt_date_clock", DT, &[node(DATE), opt_word(","), node(CLOCK)], en::b_date_clock),
    rr("dt_date_at_clock", DT, &[node(DATE), opt_word(","), word("om"), node(CLOCK)], en::b_date_clock),
    rr("dt_date_at_bare", DT, &[node(DATE), opt_word(","), word("om"), node(BARE)], en::b_date_clock),
    rr("dt_date_around_clock", DT, &[node(DATE), word("rond"), node(CLOCK)], en::b_date_clock),
    rr("dt_date_around_bare", DT, &[node(DATE), word("rond"), node(BARE)], en::b_date_clock),
    rr("dt_date_dash_clock", DT, &[node(DATE), word("-"), node(CLOCK)], en::b_date_clock),
    rr("dt_date_bare", DT, &[node(DATE), node(BARE)], en::b_date_bare_minutes),
    rr("dt_clock_date", DT, &[node(CLOCK), opt_word(","), opt_word("op"), node(DATE)], en::b_clock_date),
    // A clock in a day's part.
    rr("dt_dtp_clock", DT, &[node(DTP), opt_word("om"), node(CLOCK)], en::b_dtp_clock),
    rr("dt_dtp_at_bare", DT, &[node(DTP), word("om"), node(BARE)], en::b_dtp_clock),
    rr("dt_dtp_around_clock", DT, &[node(DTP), word("rond"), node(CLOCK)], en::b_dtp_clock),
    rr("dt_dtp_around_bare", DT, &[node(DTP), word("rond"), node(BARE)], en::b_dtp_clock),
    rr("dt_clock_dtp", DT, &[node(CLOCK), opt_word(","), node(DTP)], en::b_clock_dtp),
    rr("dt_bare_dtp", DT, &[node(BARE), opt_word(","), node(DTP)], b_bare_dtp),
    // A day and a part of it.
    rr("dtp_date_part", DTP, &[node(DATE), node(PARTD)], en::b_date_part),
    rr("dtp_part_date", DTP, &[node(PARTD), opt_word("van"), node(DATE)], en::b_part_date),
    rr("dtp_date_glued_part", DTP, &[node(DATE), glued(PART)], b_date_glued_part),
    rr("dtp_date_morning", DTP, &[node(DATE), glued_word("morgen")], b_date_morning),
    rr("dtp_today_part", DTP, &[word("van"), glued(PART)], b_today_part),
    rr("dtp_this_morning", DTP, &[word("van"), glued_word("morgen")], b_this_morning),
    rr("dtp_rel_part", DTP, &[node(RELW), node(PART)], b_rel_part),
    // The present.
    rr("dt_now", DT, &[word("nu")], en::b_now),
    rr("dt_right_now", DT, &[word("op"), word("dit"), word("moment")], en::b_now),
    rr("dt_currently", DT, &[word("momenteel")], en::b_now),
    // Shifts.
    rr("dt_in", DT, &[word("over"), node(DUR)], en::b_in),
    rr("dt_in_in", DT, &[word("in"), node(DUR)], en::b_in),
    rr("dt_later", DT, &[node(DUR), word("later")], en::b_later),
    rr("dt_from_now", DT, &[node(DUR), word("vanaf"), word("nu")], en::b_later),
    rr("dt_ago", DT, &[node(DUR), word("geleden")], en::b_ago),
    rr("dt_after", DT, &[node(DUR), word("na"), node(DATE)], en::b_after),
    rr("dt_from", DT, &[node(DUR), word("vanaf"), node(DATE)], en::b_after),
    rr("dt_counted_from", DT, &[node(DUR), word("gerekend"), word("vanaf"), node(DATE)], en::b_after),
    rr("dt_before", DT, &[node(DUR), word("voor"), node(DATE)], en::b_before),
    rr("dt_after_hol", DT, &[node(DUR), word("na"), node(HOL)], en::b_after),
    rr("dt_before_hol", DT, &[node(DUR), word("voor"), node(HOL)], en::b_before),
];
