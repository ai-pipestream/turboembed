//! Dates with times of day: `mañana a las 8:00 a.m.`, `el viernes a las
//! 5`, `13:00 28 de febrero de 2013`, `siete del 15`, `esta tarde`,
//! `mañana por la noche`, `anoche a las 8`; the present (`ahora`); and
//! points moved by a duration: `en 5 minutos`, `hace 30 minutos`, `dos
//! días después de hoy`.

use turbo_analysis_layers::{Modifier, TimeGrain};

use crate::engine::{Ctx, Item, Rule, node, opt_word, word};
use crate::lang::en::time::compose as en;
use crate::lang::en::time::{clock_of, date_of, r, term};
use crate::time::spec::{DateSpec, Part, YearRef};
use crate::value::Value;

use super::lexicon::LAST_NIGHT;
use super::{ADPOINT, ART, BARE, CLOCK, DATE, DAYWORD, DE, DT, DTP, DUR, HOL, MDAY, PART, PARTD, RELW, after_article};

/// A clock said with `las` or `la` before it, which an hour needs.
fn clock_said(c: &Ctx<'_>, v: &Value, bare_first: bool) -> Option<()> {
    clock_of(c, v)?;
    (!bare_first || after_article(c)).then_some(())
}

/// `siete del 15`: an hour of a day of the month.
fn b_clock_of_day(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    clock_said(c, &v[0], true)?;
    let day = u8::try_from(term(v.last()?).0).ok()?;
    let date = DateSpec::Cal { year: YearRef::Any, month: 0, day, weekday: 0 };
    en::at(c, date, clock_of(c, &v[0])?)
}

fn b_bare_date(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    clock_said(c, &v[0], true)?;
    en::b_clock_date(v, c)
}

fn b_bare_dtp(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    clock_said(c, &v[0], true)?;
    en::b_clock_dtp(v, c)
}

/// `anoche`: last night.
fn b_last_night(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (term(&v[0]).1 == LAST_NIGHT)
        .then(|| en::day_part(c, DateSpec::Rel { grain: TimeGrain::Day, n: -1 }, Part::Night, Modifier::NONE))?
}

/// `esta noche`, `esta tarde`; `estas noches` recur.
fn b_rel_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_rel_part(v, c)
}

/// `este pasado mediodía`: this afternoon.
fn b_this_afternoon(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::day_part(c, en::today(), Part::Afternoon, Modifier::NONE)
}

fn b_late_this_afternoon(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::day_part(c, en::today(), Part::Afternoon, Modifier::portion(turbo_analysis_layers::Portion::End))
}

/// `la noche del 21 de septiembre`.
fn b_part_of_date(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let p = crate::lang::en::time::clock::part(&v[0])?;
    if term(&v[0]).1 & super::PLURAL != 0 {
        return None;
    }
    let d = date_of(c, v.last()?).filter(|d| d.is_day())?;
    en::day_part(c, d, p, Modifier::NONE)
}

fn b_ago(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_ago(&v[v.len() - 1..], c)
}

const fn rr(name: &'static str, dim: crate::engine::Dim, items: &'static [Item], build: crate::engine::Build) -> Rule {
    r(name, dim, 0, items, build)
}

pub(super) static RULES: &[Rule] = &[
    // A date and a clock time.
    rr("dt_date_clock", DT, &[node(DATE), opt_word(","), node(CLOCK)], en::b_date_clock),
    rr("dt_date_a_clock", DT, &[node(DATE), opt_word(","), word("a"), node(ART), node(CLOCK)], en::b_date_clock),
    rr("dt_date_a_bare", DT, &[node(DATE), opt_word(","), word("a"), node(ART), node(BARE)], en::b_date_clock),
    rr("dt_date_around", DT, &[node(DATE), word("alrededor"), node(DE), node(ART), node(CLOCK)], en::b_date_clock),
    rr("dt_date_around_bare", DT, &[node(DATE), word("alrededor"), node(DE), node(ART), node(BARE)], en::b_date_clock),
    rr("dt_clock_date", DT, &[node(CLOCK), opt_word(","), node(DATE)], en::b_clock_date),
    rr("dt_clock_de_date", DT, &[node(CLOCK), node(DE), node(DATE)], en::b_clock_date),
    rr("dt_bare_de_date", DT, &[node(BARE), node(DE), node(DATE)], b_bare_date),
    rr("dt_clock_el_date", DT, &[node(CLOCK), word("el"), node(DATE)], en::b_clock_date),
    rr("dt_bare_de_day", DT, &[node(BARE), node(DE), node(MDAY)], b_clock_of_day),
    rr("dt_clock_de_day", DT, &[node(CLOCK), node(DE), node(MDAY)], |v, c| {
        let day = u8::try_from(term(v.last()?).0).ok()?;
        en::at(c, DateSpec::Cal { year: YearRef::Any, month: 0, day, weekday: 0 }, clock_of(c, &v[0])?)
    }),
    // A clock in a day's part.
    rr("dt_dtp_clock", DT, &[node(DTP), opt_word(","), node(CLOCK)], en::b_dtp_clock),
    rr("dt_dtp_a_clock", DT, &[node(DTP), word("a"), node(ART), node(CLOCK)], en::b_dtp_clock),
    rr("dt_dtp_a_bare", DT, &[node(DTP), word("a"), node(ART), node(BARE)], en::b_dtp_clock),
    rr("dt_clock_dtp", DT, &[node(CLOCK), opt_word(","), node(DTP)], en::b_clock_dtp),
    rr("dt_bare_dtp", DT, &[node(BARE), opt_word(","), node(DTP)], b_bare_dtp),
    rr("dt_bare_de_dtp", DT, &[node(BARE), node(DE), node(DTP)], b_bare_dtp),
    rr("dt_clock_de_dtp", DT, &[node(CLOCK), node(DE), node(DTP)], en::b_clock_dtp),
    // A day and a part of it.
    rr("dtp_date_part", DTP, &[node(DATE), node(PARTD)], en::b_date_part),
    rr("dtp_rel_part", DTP, &[node(RELW), node(PART)], b_rel_part),
    rr("dtp_part_of_date", DTP, &[node(PART), node(DE), node(DATE)], b_part_of_date),
    rr("dtp_anoche", DTP, &[node(DAYWORD)], b_last_night),
    rr("dtp_early_rel", DTP, &[word("temprano"), node(RELW), node(PART)], en::b_early_rel_part),
    rr("dtp_later_rel", DTP, &[word("mas"), word("tarde"), node(RELW), node(PART)], en::b_late_rel_part),
    rr("dtp_end_rel", DTP, &[word("fin"), node(DE), node(RELW), node(PART)], |v, c| en::b_late_rel_part(&v[1..], c)),
    rr("dtp_this_afternoon", DTP, &[word("este"), word("pasado"), word("mediodia")], b_this_afternoon),
    rr(
        "dtp_later_this_afternoon",
        DTP,
        &[word("mas"), word("tarde"), word("este"), word("pasado"), word("mediodia")],
        b_late_this_afternoon,
    ),
    // The present.
    rr("dt_now", DT, &[word("ahora")], en::b_now),
    rr("dt_right_now", DT, &[word("ahora"), word("mismo")], en::b_now),
    // Shifts.
    rr("dt_in", DT, &[word("en"), node(DUR)], en::b_in),
    rr("dt_ago", DT, &[word("hace"), node(DUR)], b_ago),
    rr("dt_later", DT, &[node(DUR), word("despues")], en::b_later),
    rr("dt_from_now", DT, &[node(DUR), word("a"), word("partir"), word("de"), word("ahora")], en::b_later),
    rr("dt_after", DT, &[node(DUR), word("despues"), node(DE), node(DATE)], en::b_after),
    rr("dt_from", DT, &[node(DUR), word("a"), word("partir"), node(DE), node(DATE)], en::b_after),
    rr("dt_before", DT, &[node(DUR), word("antes"), node(DE), node(DATE)], en::b_before),
    rr("dt_after_hol", DT, &[node(DUR), word("despues"), node(DE), node(HOL)], en::b_after),
    rr("dt_before_hol", DT, &[node(DUR), word("antes"), node(DE), node(HOL)], en::b_before),
    rr("dt_in_since", DT, &[word("en"), node(DUR), word("desde"), node(ADPOINT)], |v, c| {
        en::b_after(&[v[0], *v.last()?], c)
    }),
];
