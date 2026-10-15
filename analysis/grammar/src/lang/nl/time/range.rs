//! Ranges and modified readings: `van 2014 tot 2018`, `tussen 3 en 12
//! sept`, `van 5 tot 6 's middags`, `maandag 8-9 's ochtends`, `de
//! komende drie dagen`, `binnen 9 maanden`, `voor 2010`, `sinds
//! augustus`, `rond 13.00`, `eind van de maand`, `later deze week`.

use turbo_analysis_layers::{Modifier, Portion, TimeGrain};

use crate::engine::{Ctx, Item, Rule, node, opt_word, word};
use crate::lang::en::time::clock::part_value;
use crate::lang::en::time::date::unit_grain;
use crate::lang::en::time::duration::{duration_of, length};
use crate::lang::en::time::range as en;
use crate::lang::en::time::{clock_of, date_of, emit, r, term, tval};
use crate::time::TimeValue;
use crate::time::duration::MILLI;
use crate::time::spec::{Clock, DateSpec, Part, Point, Range, TimeOfDay, YearRef};
use crate::value::Value;

use super::clock::part_meridiem;
use super::{
    AMOUNT, BARE, CLOCK, DATE, DPOINT, DT, DTP, DUR, HOL, LRANGE, MDAY, MODED, MONTHD, OF_PART, PARTD, PERIOD, PLURAL,
    RANGE, RELW, TO, TPOINT, TRANGE, UNIT, YEAR,
};

/// Two bounds; a day of an open year does not run to a bare year:
/// `18 nov-2014` is no range.
fn b_range(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (a, b) = (date_of(c, &v[0]), date_of(c, v.last()?));
    if let (Some(DateSpec::Cal { year: YearRef::Any, day: 1.., .. }), Some(DateSpec::Cal { month: 0, day: 0, .. })) =
        (a, b)
    {
        return None;
    }
    en::b_range(v, c)
}

/// Two hours, the second in `uur`, read as they are said: `tussen 3 en 5
/// uur`, `van 9 tot 11 uur`.
fn b_hour_range(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (s, e) = (clock_of(c, &v[0])?, clock_of(c, v.last()?)?);
    (c.word(c.end - 1) == "uur" && !en::stands(&s, &e)).then(|| en::clock_range(c, DateSpec::None, s, e))?
}

/// `volgende maand van 4-23`: two days of the month said first.
fn b_month_days(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::days_of(c, &v[1], &v[v.len() - 1], &v[0])
}

/// Two clocks in a part of the day.
fn in_part(s: Clock, e: Clock, p: Part) -> (Clock, Clock) {
    (s.with(part_meridiem(p, s.hour)), e.with(part_meridiem(p, e.hour)))
}

/// A clock range and a part of the day after it, which sets the half of
/// the day of both bounds: `8-9 's ochtends`, `5 tot 7 's avonds`.
fn b_range_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (s, e) = en::clocks(c, &v[0])?;
    let (s, e) = in_part(s, e, part_value(c, &v[1])?);
    en::clock_range(c, DateSpec::None, s, e)
}

/// A clock range on a day, or on a day's part.
fn dated_clocks(c: &Ctx<'_>, day: &Value, range: &Value) -> Option<Value> {
    let (date, part) = en::day_and_part(c, day)?;
    if let Some((k, length)) = en::clock_length(c, range) {
        let start = Point { date, time: TimeOfDay::Clock(k), ..Point::default() };
        return emit(c, TimeValue::range(Range::For { start, length }));
    }
    let (s, e) = en::clocks(c, range)?;
    let (s, e) = match part {
        Some(p) => in_part(s, e, p),
        None => (s, e),
    };
    en::clock_range(c, date, s, e)
}

/// Whether the match ends in two numbers joined by a hyphen, the first
/// the larger: `maandag 12-4` is a date, `maandag 3-8` hours.
fn falling_dash(c: &Ctx<'_>) -> bool {
    let e = c.end;
    e >= c.start + 3
        && c.is_punct(e - 2, '-')
        && c.lexeme(e - 2).glued
        && c.lexeme(e - 1).glued
        && matches!((c.raw(e - 3).parse::<u32>(), c.raw(e - 1).parse::<u32>()), (Ok(a), Ok(b)) if a > b)
}

/// A clock range on a day; a day of an open month only after `op`: `op
/// de 17 van 14.00 tot 16.00`.
fn b_date_trange(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if date_of(c, &v[0]).is_some_and(en::open_day) && c.before().is_none_or(|p| c.word(p) != "op") {
        return None;
    }
    if falling_dash(c) {
        return None;
    }
    dated_clocks(c, &v[0], v.last()?)
}

fn b_trange_date(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    dated_clocks(c, v.last()?, &v[0])
}

/// `de komende weken`: the coming one of the plural unit.
fn b_next_units(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[1]).1 & PLURAL == 0 {
        return None;
    }
    let d = length(i64::from(MILLI), &v[1])?;
    en::next(c, en::direction(&v[0])?, d, en::align_of(&v[1], &d)?)
}

/// `2 uur in de toekomst`: the coming duration.
fn b_in_future(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let d = duration_of(c, &v[0])?;
    en::next(c, true, d, crate::time::resolve::unit_grain(d.finest()?))
}

/// A bound of an open year in `year`.
fn with_year(p: Point, year: YearRef) -> Option<Point> {
    match p.date {
        DateSpec::Cal { year: YearRef::Any, month: month @ 1.., day, weekday } => {
            Some(Point { date: DateSpec::Cal { year, month, day, weekday }, ..p })
        }
        _ => None,
    }
}

/// `26 juni tot 28 juni in 2020`: a year for both bounds.
fn b_range_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let TimeValue::Range { range: Range::Between { start, start_edge, end, end_edge }, modifier } = tval(c, &v[0])?
    else {
        return None;
    };
    let year = crate::lang::en::time::year_ref(&v[1]);
    let (start, end) = (with_year(start, year)?, with_year(end, year)?);
    emit(c, TimeValue::Range { range: Range::Between { start, start_edge, end, end_edge }, modifier })
}

/// A date with a bounded clock time said after it: `om 15.00 of later op
/// maandag`.
fn b_moded_date(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_date_moded(&[v[1], v[0]], c)
}

/// `later op de dag`: the second half of today.
fn b_later_in_day(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if unit_grain(&v[0]) != Some(TimeGrain::Day) || term(&v[0]).1 & PLURAL != 0 {
        return None;
    }
    let point = Point::date(DateSpec::Rel { grain: TimeGrain::Day, n: 0 });
    emit(c, TimeValue::Point { point, modifier: Modifier::portion(Portion::End), clip: true })
}

const fn rr(name: &'static str, dim: crate::engine::Dim, items: &'static [Item], build: crate::engine::Build) -> Rule {
    r(name, dim, 0, items, build)
}

pub(super) static RULES: &[Rule] = &[
    rr("to", TO, &[word("tot")], en::b_empty),
    rr("to_including", TO, &[word("tot"), word("en"), word("met")], en::b_empty),
    rr("to_tm", TO, &[word("t"), word("/"), word("m")], en::b_empty),
    rr("to_dash", TO, &[word("-")], en::b_empty),
    rr("to_en_dash", TO, &[word("\u{2013}")], en::b_empty),
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
    // Date ranges.
    rr("range", RANGE, &[node(DPOINT), node(TO), node(DPOINT)], b_range),
    rr("range_from", RANGE, &[word("van"), node(DPOINT), node(TO), node(DPOINT)], b_range),
    rr("range_since", RANGE, &[word("vanaf"), node(DPOINT), node(TO), node(DPOINT)], b_range),
    rr("range_between", RANGE, &[word("tussen"), node(DPOINT), word("en"), node(DPOINT)], b_range),
    rr("range_between_to", RANGE, &[word("tussen"), node(DPOINT), node(TO), node(DPOINT)], b_range),
    rr("range_days_of", RANGE, &[node(MDAY), node(TO), node(MDAY), opt_word("van"), node(DPOINT)], en::b_days_of),
    rr(
        "range_from_days_of",
        RANGE,
        &[word("van"), node(MDAY), node(TO), node(MDAY), opt_word("van"), node(DPOINT)],
        en::b_days_of,
    ),
    rr(
        "range_on_days_of",
        RANGE,
        &[word("op"), node(MDAY), node(TO), node(MDAY), opt_word("van"), node(DPOINT)],
        en::b_days_of,
    ),
    rr(
        "range_between_days_of",
        RANGE,
        &[
            word("tussen"),
            opt_word("de"),
            node(MDAY),
            word("en"),
            opt_word("de"),
            node(MDAY),
            opt_word("van"),
            node(DPOINT),
        ],
        en::b_days_of,
    ),
    rr("range_month_days_of", RANGE, &[node(DPOINT), word("van"), node(MDAY), node(TO), node(MDAY)], b_month_days),
    rr("range_month_days", RANGE, &[node(super::MONTH), node(MDAY), node(TO), node(MDAY)], en::b_month_days),
    rr("range_year", RANGE, &[node(RANGE), word("in"), node(YEAR)], b_range_year),
    rr("range_year_van", RANGE, &[node(RANGE), word("van"), node(YEAR)], b_range_year),
    // Clock ranges.
    rr("trange", TRANGE, &[node(TPOINT), node(TO), node(TPOINT)], en::b_trange),
    rr("trange_from", TRANGE, &[word("van"), node(TPOINT), node(TO), node(TPOINT)], en::b_trange),
    rr("trange_between", TRANGE, &[word("tussen"), node(TPOINT), word("en"), node(TPOINT)], en::b_trange),
    rr("trange_between_to", TRANGE, &[word("tussen"), node(TPOINT), node(TO), node(TPOINT)], en::b_trange),
    rr("trange_hours_from", TRANGE, &[word("van"), node(TPOINT), word("tot"), node(TPOINT)], b_hour_range),
    rr("trange_hours_between", TRANGE, &[word("tussen"), node(TPOINT), word("en"), node(TPOINT)], b_hour_range),
    rr("lrange", LRANGE, &[node(TPOINT), node(TO), node(TPOINT)], en::b_lrange),
    rr("lrange_from", LRANGE, &[word("van"), node(TPOINT), node(TO), node(TPOINT)], en::b_lrange),
    rr("lrange_between", LRANGE, &[word("tussen"), node(TPOINT), word("en"), node(TPOINT)], en::b_lrange),
    rr("lrange_between_to", LRANGE, &[word("tussen"), node(TPOINT), node(TO), node(TPOINT)], en::b_lrange),
    r("trange_part", TRANGE, 1, &[node(LRANGE), node(OF_PART)], b_range_part),
    r("trange_t_part", TRANGE, 1, &[node(TRANGE), node(OF_PART)], b_range_part),
    r("range_date_trange", RANGE, 2, &[node(DATE), opt_word(","), opt_word("om"), node(TRANGE)], b_date_trange),
    r("range_date_lrange", RANGE, 2, &[node(DATE), opt_word("om"), node(LRANGE)], b_date_trange),
    rr("range_dtp_trange", RANGE, &[node(DTP), node(TRANGE)], b_date_trange),
    rr("range_dtp_lrange", RANGE, &[node(DTP), node(LRANGE)], b_date_trange),
    rr("range_trange_date", RANGE, &[node(TRANGE), opt_word("op"), node(DATE)], b_trange_date),
    rr("range_trange_date_om", RANGE, &[word("om"), node(TRANGE), opt_word("op"), node(DATE)], b_trange_date),
    rr("range_lrange_date", RANGE, &[node(LRANGE), opt_word("op"), node(DATE)], b_trange_date),
    rr("range_lrange_date_om", RANGE, &[word("om"), node(LRANGE), opt_word("op"), node(DATE)], b_trange_date),
    rr("range_trange_dtp", RANGE, &[node(TRANGE), node(DTP)], b_trange_date),
    rr("range_lrange_dtp", RANGE, &[node(LRANGE), node(DTP)], b_trange_date),
    // Stretches from the reference.
    rr("range_next_n", RANGE, &[node(RELW), node(AMOUNT), node(UNIT)], en::b_next_n),
    rr("range_n_next", RANGE, &[node(AMOUNT), node(RELW), node(UNIT)], en::b_n_next),
    rr("range_next_dur", RANGE, &[node(RELW), node(DUR)], en::b_next_dur),
    rr("range_next_unit", RANGE, &[node(RELW), node(UNIT)], en::b_next_unit),
    rr("range_next_units", RANGE, &[node(RELW), node(UNIT)], b_next_units),
    rr("range_within", RANGE, &[word("binnen"), node(DUR)], en::b_within),
    rr("range_within_next", RANGE, &[word("binnen"), word("de"), node(RELW), node(DUR)], en::b_within),
    rr("range_in_future", RANGE, &[node(DUR), word("in"), word("de"), word("toekomst")], b_in_future),
    // Bounded and modified points.
    rr("moded_before", MODED, &[word("voor"), node(DPOINT)], en::b_before),
    rr("moded_by", MODED, &[word("tegen"), node(DPOINT)], en::b_before),
    rr("moded_until", MODED, &[word("tot"), node(DPOINT)], en::b_before),
    rr("moded_no_later_than", MODED, &[word("niet"), word("later"), word("dan"), node(DPOINT)], en::b_before),
    rr("moded_at_the_latest", MODED, &[word("uiterlijk"), node(DPOINT)], en::b_before),
    rr("moded_after", MODED, &[word("na"), node(DPOINT)], en::b_after),
    rr("moded_later_than", MODED, &[word("later"), word("dan"), node(DPOINT)], en::b_after),
    rr(
        "moded_year_higher_than",
        MODED,
        &[word("het"), word("jaar"), word("hoger"), word("dan"), node(DPOINT)],
        en::b_after,
    ),
    rr("moded_since", MODED, &[word("sinds"), node(DPOINT)], en::b_since),
    rr("moded_from_clock", MODED, &[word("vanaf"), node(TPOINT)], en::b_since),
    rr("moded_as_early_as", MODED, &[word("zo"), word("vroeg"), word("als"), node(DPOINT)], en::b_since),
    rr("moded_as_early_as_al", MODED, &[word("al"), word("zo"), word("vroeg"), word("als"), node(DPOINT)], en::b_since),
    rr("moded_as_late_as", MODED, &[word("zo"), word("laat"), word("als"), node(DPOINT)], en::b_until),
    rr("moded_as_late_as_al", MODED, &[word("al"), word("zo"), word("laat"), word("als"), node(DPOINT)], en::b_until),
    rr("moded_or_later", MODED, &[node(DPOINT), word("of"), word("later")], en::b_since_after),
    rr("moded_and_later", MODED, &[node(DPOINT), word("en"), word("later")], en::b_since_after),
    rr("moded_and_after", MODED, &[node(DPOINT), word("en"), word("daarna")], en::b_since_after),
    rr("moded_and_higher", MODED, &[node(DPOINT), word("en"), word("hoger")], en::b_since_after),
    rr("moded_around", MODED, &[word("rond"), node(DPOINT)], en::b_approx),
    rr("moded_about", MODED, &[word("omstreeks"), node(DPOINT)], en::b_approx),
    rr("moded_before_around", MODED, &[word("voor"), word("rond"), node(DPOINT)], en::b_before_around),
    rr("moded_after_around", MODED, &[word("na"), word("rond"), node(DPOINT)], en::b_after_around),
    rr("moded_date", MODED, &[node(DATE), node(MODED)], en::b_date_moded),
    rr("moded_on_date", MODED, &[node(MODED), word("op"), node(DATE)], b_moded_date),
    rr(
        "moded_more_before",
        MODED,
        &[word("meer"), word("dan"), node(DUR), word("voor"), node(DPOINT)],
        en::b_more_before,
    ),
    rr(
        "moded_longer_before",
        MODED,
        &[word("langer"), word("dan"), node(DUR), word("voor"), node(DPOINT)],
        en::b_more_before,
    ),
    rr("moded_more_after", MODED, &[word("meer"), word("dan"), node(DUR), word("na"), node(DPOINT)], en::b_more_after),
    rr(
        "range_less_after",
        RANGE,
        &[word("minder"), word("dan"), node(DUR), word("na"), node(DPOINT)],
        en::b_less_after,
    ),
    // Portions.
    rr("moded_start", MODED, &[word("begin"), opt_word("van"), node(DPOINT)], en::b_start),
    rr("moded_the_start", MODED, &[word("het"), word("begin"), word("van"), node(DPOINT)], en::b_start),
    rr(
        "moded_the_start_aan",
        MODED,
        &[word("aan"), word("het"), word("begin"), word("van"), node(DPOINT)],
        en::b_start,
    ),
    rr("moded_the_start_de", MODED, &[word("de"), word("start"), word("van"), node(DPOINT)], en::b_start),
    rr("moded_early_in_day", MODED, &[node(DPOINT), word("vroeg"), word("op"), word("de"), word("dag")], en::b_start),
    rr("moded_mid_month", MODED, &[word("half"), node(MONTHD)], en::b_mid),
    rr("moded_mid", MODED, &[word("halverwege"), node(DPOINT)], en::b_mid),
    rr("moded_medio", MODED, &[word("medio"), node(DPOINT)], en::b_mid),
    rr("moded_middle", MODED, &[word("midden"), word("van"), node(DPOINT)], en::b_mid),
    rr("moded_middle_het", MODED, &[word("het"), word("midden"), word("van"), node(DPOINT)], en::b_mid),
    rr("moded_end", MODED, &[word("eind"), opt_word("van"), node(DPOINT)], en::b_end),
    rr("moded_end_einde", MODED, &[word("einde"), opt_word("van"), node(DPOINT)], en::b_end),
    rr("moded_the_end", MODED, &[word("het"), word("eind"), word("van"), node(DPOINT)], en::b_end),
    rr("moded_the_end_aan", MODED, &[word("aan"), word("het"), word("eind"), word("van"), node(DPOINT)], en::b_end),
    rr("moded_the_end_einde", MODED, &[word("het"), word("einde"), word("van"), node(DPOINT)], en::b_end),
    rr(
        "moded_the_end_einde_aan",
        MODED,
        &[word("aan"), word("het"), word("einde"), word("van"), node(DPOINT)],
        en::b_end,
    ),
    rr("moded_end_the", MODED, &[word("eind"), word("van"), word("de"), node(DPOINT)], en::b_end),
    rr("moded_end_of_unit", MODED, &[word("eind"), word("van"), word("de"), node(UNIT)], en::b_end_of_unit),
    rr("moded_end_of_unit_einde", MODED, &[word("einde"), word("van"), word("de"), node(UNIT)], en::b_end_of_unit),
    rr(
        "moded_the_end_of_unit",
        MODED,
        &[word("het"), word("eind"), word("van"), word("de"), node(UNIT)],
        en::b_end_of_unit,
    ),
    rr(
        "moded_the_end_of_unit_einde",
        MODED,
        &[word("het"), word("einde"), word("van"), word("de"), node(UNIT)],
        en::b_end_of_unit,
    ),
    rr("moded_end_of_day", MODED, &[word("eind"), word("van"), word("de"), word("dag")], en::b_end_of_day),
    rr(
        "moded_end_of_day_het",
        MODED,
        &[word("het"), word("eind"), word("van"), word("de"), word("dag")],
        en::b_end_of_day,
    ),
    rr(
        "moded_end_of_day_aan",
        MODED,
        &[word("aan"), opt_word("het"), word("eind"), word("van"), word("de"), word("dag")],
        en::b_end_of_day,
    ),
    rr("moded_earlier", MODED, &[word("eerder"), node(DPOINT)], en::b_earlier),
    rr("moded_later", MODED, &[word("later"), node(DPOINT)], en::b_later),
    rr("moded_earlier_in_de", MODED, &[word("eerder"), word("in"), word("de"), node(UNIT)], en::b_earlier_in_the),
    rr("moded_earlier_in_het", MODED, &[word("eerder"), word("in"), word("het"), node(UNIT)], en::b_earlier_in_the),
    rr("moded_later_in_de", MODED, &[word("later"), word("in"), word("de"), node(UNIT)], en::b_later_in_the),
    rr("moded_later_in_het", MODED, &[word("later"), word("in"), word("het"), node(UNIT)], en::b_later_in_the),
    rr("moded_later_in_day", MODED, &[word("later"), word("op"), word("de"), node(UNIT)], b_later_in_day),
];
