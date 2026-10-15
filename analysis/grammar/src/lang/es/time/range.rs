//! Ranges and modified readings: `de 2014 a 2018`, `entre el 1 de
//! diciembre y el 4 de febrero`, `de 1 a 7 de mayo`, `de las 5 a las 6 de
//! la tarde`, `lunes 8-9 a.m.`, `los próximos tres días`, `dentro de 9
//! meses`, `antes de 2018`, `a partir de mañana`, `a finales de 2000`,
//! `más tarde esta semana`, `alrededor de las 2 p.m.`.

use turbo_analysis_layers::{Portion, TimeGrain};

use crate::engine::{Ctx, Item, Rule, node, opt_word, word};
use crate::lang::en::time::date::unit_grain;
use crate::lang::en::time::range as en;
use crate::lang::en::time::{date_of, r};
use crate::time::spec::{DateSpec, YearRef};
use crate::value::Value;

use super::{
    ADPOINT, AMOUNT, ART, ATPOINT, BARE, CLOCK, DATE, DE, DPOINT, DT, DTP, DUR, HOL, LRANGE, MDAY, MODED, MONTH,
    MONTHD, PARTD, PERIOD, RANGE, RELW, TO, TPOINT, TRANGE, UNIT, YEAR,
};

/// The children with the connector and articles left out: the values
/// that are time readings.
fn days_of(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    // MDAY, TO, MDAY, DE, DPOINT or MDAY, MDAY, DE, DPOINT.
    let first = v[0];
    let month = *v.last()?;
    let second = if v.len() == 5 { v[2] } else { v[1] };
    en::b_days_of(&[first, second, month], c)
}

/// `5 próximos años`, `2 días siguientes`: a count, a unit and the
/// relative word in either order.
fn b_n_unit_next(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_n_next(&[v[0], v[2], v[1]], c)
}

/// The end of a day said with `al`: `al final de mañana`.
fn b_end_of_day_al(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    date_of(c, v.last()?).filter(|d| d.is_day())?;
    en::b_end(v, c)
}

/// `principios de año`, `inicio de mes`: a part of the reference's own
/// period.
fn b_start_of_unit(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if unit_grain(v.last()?) == Some(TimeGrain::Day) {
        return None;
    }
    en::this_period(c, v.last()?, Portion::Start, false)
}

fn b_end_of_unit(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_end_of_unit(&v[v.len() - 1..], c)
}

/// Whether a clock range says its bounds alike: `a las` in the middle
/// (`5 a las 7`) wants `las` or a `de` before the start, as in `de las 5
/// a las 7`; otherwise the `a las` starts a time of its own.
fn paired(c: &Ctx<'_>) -> bool {
    if matches!(c.word(c.start), "las" | "la" | "de" | "desde" | "entre") || super::after_article(c) {
        return true;
    }
    !(c.start..c.end.saturating_sub(1)).any(|i| c.word(i) == "a" && matches!(c.word(i + 1), "las" | "la"))
}

fn b_trange(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    paired(c).then(|| en::b_trange(v, c))?
}

fn b_lrange(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    paired(c).then(|| en::b_lrange(v, c))?
}

/// Two bounds of a kind that can run from one to the other: a day of an
/// open year does not run to a bare year (`31-dic-1994` is one date).
fn b_range(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (a, b) = (date_of(c, &v[0]), date_of(c, v.last()?));
    if let (Some(DateSpec::Cal { year: YearRef::Any, day: 1.., .. }), Some(DateSpec::Cal { month: 0, day: 0, .. })) =
        (a, b)
    {
        return None;
    }
    // A weekday and a month make no range: `de lunes a mar` runs to
    // Tuesday.
    let weekday = |d: Option<DateSpec>| matches!(d, Some(DateSpec::Cal { month: 0, day: 0, weekday: 1.., .. }));
    let month = |d: Option<DateSpec>| matches!(d, Some(DateSpec::Cal { month: 1.., day: 0, weekday: 0, .. }));
    if (weekday(a) && month(b)) || (month(a) && weekday(b)) {
        return None;
    }
    en::b_range(v, c)
}

/// `más tarde de este mes` is its end; `más tarde de las diez`, after.
fn b_later_or_after(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_later(v, c).or_else(|| en::b_after(v, c))
}

const fn rr(name: &'static str, dim: crate::engine::Dim, items: &'static [Item], build: crate::engine::Build) -> Rule {
    r(name, dim, 0, items, build)
}

pub(super) static RULES: &[Rule] = &[
    rr("to_a", TO, &[word("a")], en::b_empty),
    rr("to_al", TO, &[word("al")], en::b_empty),
    rr("to_hasta", TO, &[word("hasta")], en::b_empty),
    rr("to_dash", TO, &[word("-")], en::b_empty),
    rr("to_tilde", TO, &[word("~")], en::b_empty),
    // Bounds.
    rr("dpoint_date", DPOINT, &[node(DATE)], en::b_point),
    rr("dpoint_month", DPOINT, &[node(MONTHD)], en::b_point),
    rr("dpoint_period", DPOINT, &[node(PERIOD)], en::b_point),
    rr("dpoint_hol", DPOINT, &[node(HOL)], en::b_point),
    rr("dpoint_dt", DPOINT, &[node(DT)], en::b_point),
    rr("dpoint_dtp", DPOINT, &[node(DTP)], en::b_point),
    rr("dpoint_clock", DPOINT, &[node(CLOCK)], en::b_point),
    rr("dpoint_partd", DPOINT, &[node(PARTD)], en::b_point),
    rr("dpoint_edge", DPOINT, &[node(MODED)], en::b_edge_point),
    rr("tpoint_clock", TPOINT, &[node(CLOCK)], en::b_tpoint),
    rr("tpoint_bare", TPOINT, &[node(BARE)], en::b_tpoint),
    rr("adpoint", ADPOINT, &[node(DPOINT)], en::b_point),
    rr("adpoint_art", ADPOINT, &[node(ART), node(DPOINT)], |v, c| en::b_point(&v[1..], c)),
    rr("adpoint_art_bare", ADPOINT, &[node(ART), node(BARE)], |v, c| en::b_point(&v[1..], c)),
    rr("atpoint", ATPOINT, &[node(TPOINT)], en::b_tpoint),
    rr("atpoint_art", ATPOINT, &[node(ART), node(TPOINT)], |v, c| en::b_tpoint(&v[1..], c)),
    // Date ranges.
    rr("range", RANGE, &[node(DPOINT), node(TO), node(ADPOINT)], b_range),
    rr("range_de", RANGE, &[word("de"), node(ADPOINT), node(TO), node(ADPOINT)], b_range),
    rr("range_desde", RANGE, &[word("desde"), node(ADPOINT), node(TO), node(ADPOINT)], b_range),
    rr("range_entre", RANGE, &[word("entre"), node(ADPOINT), word("y"), node(ADPOINT)], b_range),
    rr("range_entre_to", RANGE, &[word("entre"), node(ADPOINT), node(TO), node(ADPOINT)], b_range),
    rr("range_days_of", RANGE, &[node(MDAY), node(TO), node(MDAY), node(DE), node(DPOINT)], days_of),
    rr("range_de_days_of", RANGE, &[word("de"), node(MDAY), node(TO), node(MDAY), node(DE), node(DPOINT)], days_of),
    rr(
        "range_desde_days_of",
        RANGE,
        &[word("desde"), node(MDAY), node(TO), node(MDAY), node(DE), node(DPOINT)],
        days_of,
    ),
    rr(
        "range_entre_days_of",
        RANGE,
        &[word("entre"), node(MDAY), word("y"), node(MDAY), node(DE), node(DPOINT)],
        days_of,
    ),
    rr("range_month_days", RANGE, &[node(MONTH), node(MDAY), node(TO), node(MDAY)], en::b_month_days),
    rr(
        "range_month_days_year",
        RANGE,
        &[node(MONTH), node(MDAY), node(TO), node(MDAY), opt_word(","), node(YEAR)],
        en::b_month_days,
    ),
    // Clock ranges.
    rr("trange", TRANGE, &[node(TPOINT), node(TO), node(ATPOINT)], b_trange),
    rr("trange_de", TRANGE, &[word("de"), node(ATPOINT), node(TO), node(ATPOINT)], b_trange),
    rr("trange_desde", TRANGE, &[word("desde"), node(ATPOINT), node(TO), node(ATPOINT)], b_trange),
    rr("trange_entre", TRANGE, &[word("entre"), node(ATPOINT), word("y"), node(ATPOINT)], b_trange),
    rr("trange_entre_to", TRANGE, &[word("entre"), node(ATPOINT), node(TO), node(ATPOINT)], b_trange),
    rr("lrange", LRANGE, &[node(TPOINT), node(TO), node(ATPOINT)], b_lrange),
    rr("lrange_de", LRANGE, &[word("de"), node(ATPOINT), node(TO), node(ATPOINT)], b_lrange),
    rr("lrange_desde", LRANGE, &[word("desde"), node(ATPOINT), node(TO), node(ATPOINT)], b_lrange),
    rr("lrange_entre", LRANGE, &[word("entre"), node(ATPOINT), word("y"), node(ATPOINT)], b_lrange),
    rr("lrange_entre_to", LRANGE, &[word("entre"), node(ATPOINT), node(TO), node(ATPOINT)], b_lrange),
    rr("trange_part", TRANGE, &[node(TRANGE), node(PARTD)], en::b_trange_date),
    rr("trange_lpart", TRANGE, &[node(LRANGE), node(PARTD)], en::b_trange_date),
    r("range_date_trange", RANGE, 2, &[node(DATE), opt_word(","), node(TRANGE)], en::b_date_trange),
    r("range_date_lrange", RANGE, 2, &[node(DATE), opt_word(","), node(LRANGE)], en::b_date_trange),
    rr("range_dtp_trange", RANGE, &[node(DTP), node(TRANGE)], en::b_date_trange),
    rr("range_dtp_lrange", RANGE, &[node(DTP), node(LRANGE)], en::b_date_trange),
    rr("range_trange_date", RANGE, &[node(TRANGE), opt_word("el"), node(DATE)], en::b_trange_date),
    rr("range_lrange_date", RANGE, &[node(LRANGE), opt_word("el"), node(DATE)], en::b_trange_date),
    rr("range_trange_de_date", RANGE, &[node(TRANGE), node(DE), node(DATE)], en::b_trange_date),
    rr("range_lrange_de_date", RANGE, &[node(LRANGE), node(DE), node(DATE)], en::b_trange_date),
    rr("range_trange_dtp", RANGE, &[node(TRANGE), opt_word("de"), node(DTP)], en::b_trange_date),
    rr("range_lrange_dtp", RANGE, &[node(LRANGE), opt_word("de"), node(DTP)], en::b_trange_date),
    // Stretches from the reference.
    rr("range_next_n", RANGE, &[node(RELW), node(AMOUNT), node(UNIT)], en::b_next_n),
    rr("range_n_next", RANGE, &[node(AMOUNT), node(RELW), node(UNIT)], en::b_n_next),
    rr("range_n_unit_next", RANGE, &[node(AMOUNT), node(UNIT), node(RELW)], b_n_unit_next),
    rr("range_next_dur", RANGE, &[node(RELW), node(DUR)], en::b_next_dur),
    rr("range_next_unit", RANGE, &[node(RELW), node(UNIT)], en::b_next_unit),
    rr("range_within", RANGE, &[word("dentro"), word("de"), node(DUR)], en::b_within),
    rr("range_within_next", RANGE, &[word("dentro"), word("de"), node(RELW), node(DUR)], en::b_within),
    // Bounded and modified points.
    rr("moded_antes", MODED, &[word("antes"), node(DE), node(ADPOINT)], en::b_before),
    rr("moded_anterior", MODED, &[word("anterior"), node(DE), node(ADPOINT)], en::b_before),
    rr("moded_anterior_a", MODED, &[word("anterior"), word("a"), node(ADPOINT)], en::b_before),
    rr("moded_hasta", MODED, &[word("hasta"), node(ADPOINT)], en::b_before),
    rr(
        "moded_no_mas_tarde",
        MODED,
        &[word("no"), word("mas"), word("tarde"), word("que"), node(ADPOINT)],
        en::b_before,
    ),
    rr("moded_despues", MODED, &[word("despues"), node(DE), node(ADPOINT)], en::b_after),
    rr("moded_posterior", MODED, &[word("posterior"), node(DE), node(ADPOINT)], en::b_after),
    rr("moded_posterior_a", MODED, &[word("posterior"), word("a"), node(ADPOINT)], en::b_after),
    rr("moded_mas_tarde_de", MODED, &[word("mas"), word("tarde"), node(DE), node(ADPOINT)], b_later_or_after),
    rr("moded_desde", MODED, &[word("desde"), node(ADPOINT)], en::b_since),
    rr("moded_a_partir", MODED, &[word("a"), word("partir"), node(DE), node(ADPOINT)], en::b_since),
    rr("moded_tan_pronto", MODED, &[word("tan"), word("pronto"), word("como"), node(ADPOINT)], en::b_since),
    rr(
        "moded_tan_temprano",
        MODED,
        &[word("tan"), word("temprano"), word("como"), opt_word("a"), node(ADPOINT)],
        en::b_since,
    ),
    rr("moded_tan_tarde", MODED, &[word("tan"), word("tarde"), word("como"), node(ADPOINT)], en::b_until),
    rr("moded_comenzando", MODED, &[word("comenzando"), opt_word("el"), node(ADPOINT)], en::b_after),
    rr("moded_empezando", MODED, &[word("empezando"), opt_word("el"), node(ADPOINT)], en::b_after),
    rr("moded_ge", MODED, &[word(">"), opt_word("="), node(DPOINT)], en::b_since),
    rr("moded_le", MODED, &[word("<"), opt_word("="), node(DPOINT)], en::b_until),
    rr("moded_y_mas_tarde", MODED, &[node(DPOINT), word("y"), word("mas"), word("tarde")], en::b_since_after),
    rr("moded_o_mas_tarde", MODED, &[node(DPOINT), word("o"), word("mas"), word("tarde")], en::b_since_after),
    rr("moded_y_despues", MODED, &[node(DPOINT), word("y"), word("despues")], en::b_since_after),
    rr("moded_o_despues", MODED, &[node(DPOINT), word("o"), word("despues")], en::b_since_after),
    rr("moded_o_posterior", MODED, &[node(DPOINT), word("o"), word("posterior")], en::b_since_after),
    rr("moded_alrededor", MODED, &[word("alrededor"), node(DE), node(ADPOINT)], en::b_approx),
    rr("moded_cerca", MODED, &[word("cerca"), node(DE), node(ADPOINT)], en::b_approx),
    rr("moded_date", MODED, &[node(DATE), node(MODED)], en::b_date_moded),
    // Portions.
    rr("moded_principios", MODED, &[word("principios"), node(DE), node(ADPOINT)], en::b_start),
    rr("moded_inicios", MODED, &[word("inicios"), node(DE), node(ADPOINT)], en::b_start),
    rr("moded_inicio", MODED, &[word("inicio"), node(DE), node(ADPOINT)], en::b_start),
    rr("moded_comienzos", MODED, &[word("comienzos"), node(DE), node(ADPOINT)], en::b_start),
    rr("moded_mediados", MODED, &[word("mediados"), node(DE), node(ADPOINT)], en::b_mid),
    rr("moded_medianos", MODED, &[word("medianos"), node(DE), node(ADPOINT)], en::b_mid),
    rr("moded_finales", MODED, &[word("finales"), node(DE), node(ADPOINT)], en::b_end),
    rr("moded_final", MODED, &[word("final"), node(DE), node(ADPOINT)], en::b_end),
    rr("moded_fin", MODED, &[word("fin"), node(DE), node(ADPOINT)], en::b_end),
    rr("moded_al_final", MODED, &[word("al"), word("final"), node(DE), node(ADPOINT)], b_end_of_day_al),
    rr("moded_al_fin", MODED, &[word("al"), word("fin"), node(DE), node(ADPOINT)], b_end_of_day_al),
    rr("moded_mas_tarde", MODED, &[word("mas"), word("tarde"), node(DPOINT)], en::b_later),
    rr("moded_mas_temprano", MODED, &[word("mas"), word("temprano"), node(DPOINT)], en::b_earlier),
    rr("moded_mas_temprano_de", MODED, &[word("mas"), word("temprano"), node(DE), node(ADPOINT)], en::b_earlier),
    rr("moded_mas_tarde_en", MODED, &[word("mas"), word("tarde"), word("en"), node(ART), node(UNIT)], |v, c| {
        en::b_later_in_the(&v[1..], c)
    }),
    rr("moded_mas_temprano_en", MODED, &[word("mas"), word("temprano"), word("en"), node(ART), node(UNIT)], |v, c| {
        en::b_earlier_in_the(&v[1..], c)
    }),
    rr("moded_principios_unit", MODED, &[word("principios"), node(DE), node(UNIT)], b_start_of_unit),
    rr("moded_inicio_unit", MODED, &[word("inicio"), node(DE), node(UNIT)], b_start_of_unit),
    rr("moded_comienzos_unit", MODED, &[word("comienzos"), node(DE), node(UNIT)], b_start_of_unit),
    rr("moded_fin_unit", MODED, &[word("fin"), node(DE), node(UNIT)], b_end_of_unit),
    rr("moded_final_unit", MODED, &[word("final"), node(DE), node(UNIT)], b_end_of_unit),
    rr("moded_finales_unit", MODED, &[word("finales"), node(DE), node(UNIT)], b_end_of_unit),
    rr("moded_final_de_la_unit", MODED, &[word("final"), node(DE), node(ART), node(UNIT)], b_end_of_unit),
    rr("moded_end_of_day", MODED, &[word("al"), word("final"), word("del"), word("dia")], en::b_end_of_day),
    rr("moded_fin_de_dia", MODED, &[word("fin"), word("de"), word("dia")], en::b_end_of_day),
    rr("moded_el_fin_de_dia", MODED, &[word("el"), word("fin"), word("de"), word("dia")], en::b_end_of_day),
];
