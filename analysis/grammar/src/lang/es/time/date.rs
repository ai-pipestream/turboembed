//! Dates and periods: `3 de enero del 2019`, `27/11`, `31-dic-1994`,
//! `el día 18`, `viernes 6 de julio`, `el lunes que viene`, `marzo de
//! 2015`, `la semana pasada`, `el primer trimestre de 2017`, `los años
//! noventa`, `el año fiscal 2008`, holidays and seasons. Days come before
//! months in digits: `3/5/2018` is the third of May.

use turbo_analysis_layers::TimeGrain;

use crate::engine::{Ctx, Item, Kind, Rule, glued, glued_word, node, opt_word, word};
use crate::lang::en::time::clock::dig_len;
use crate::lang::en::time::date as en;
use crate::lang::en::time::lexicon::{self as words, rel};
use crate::lang::en::time::{date_of, emit_date, r, term, term_value, year_ref};
use crate::time::spec::{DateSpec, YearKind, YearRef};
use crate::value::Value;

use super::super::{INT, ORD, TENS};
use super::{
    ABBR, ART, DATE, DAYWORD, DE, DIG, DOM, HOL, HOLIDAY, KIND, MD, MDAY, MONTH, MONTHD, PERIOD, PLURAL, QTR, RELW,
    SEASON, SEP, SHORT, UNIT, WD, WEEKDAY, YEAR,
};
use crate::lang::en::time::lexicon::{BLACK_FRIDAY, EARTH, INDEPENDENCE, PATRICK, VALENTINE, WORKERS};

/// A relative word made up by a phrase: `que viene` is `next`.
const NEXT: Value = Value::Term { value: 1, flags: rel::NEXT };
/// `este pasado`: the most recent.
const PAST: Value = Value::Term { value: -1, flags: rel::PAST };
/// `este próximo`: the coming one.
const UPCOMING: Value = Value::Term { value: 1, flags: rel::UPCOMING };

/// An abbreviation with its period: `sept.`, `sáb.`.
fn b_abbr_dot(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    (term(&v[0]).1 & ABBR != 0).then_some(v[0])
}

/// A day first and a month second in digits, `27/11`; the other way
/// round only when the second cannot be a month, `11/27`.
fn b_dm(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[0]).0 == 0 || term(&v[1]).0 == 0 {
        return None;
    }
    en::b_md_slash(&[v[1], v[0]], c)
}

/// `9 de 8 de 1971`, `17 del 11`: a month in digits after `de`.
fn b_day_of_month_digits(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let m = en::small(v.last()?)?;
    ((1..=12).contains(&m)).then(|| en::md(c, m, term(&v[0]).0))?
}

/// `4-jul-95`: a day, a month name and a year of two or four digits.
fn b_d_mon_y(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, en::cal(en::numeric_year(&v[4])?, term(&v[2]).0, en::small(&v[0])?, 0)?)
}

/// `6-jul`, `30-sept`.
fn b_d_mon(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::md(c, term(&v[2]).0, en::small(&v[0])?)
}

/// `abril de 2017`, `abril 2017`; two digits only after a word, as in
/// `abril del 87`.
fn month_year(c: &Ctx<'_>, month: &Value, year: &Value, said: bool) -> Option<Value> {
    if term(year).1 & SHORT != 0 && !said {
        return None;
    }
    emit_date(c, en::cal(year_ref(year), term(month).0, 0, 0)?)
}

/// `el lunes pasado`: the relative word after the weekday.
fn b_wd_after(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_wd_rel(&[v[1], v[0]], c)
}

fn b_wd_next(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_wd_rel(&[NEXT, v[0]], c)
}

fn b_wd_past(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_wd_this_rel(&[PAST, *v.last()?], c)
}

fn b_wd_upcoming(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_wd_this_rel(&[UPCOMING, *v.last()?], c)
}

/// `este pasado miércoles`, `este próximo viernes`: the most recent or
/// the coming one.
fn b_wd_this_rel(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    match term(&v[0]).1 {
        rel::LAST => b_wd_past(&v[1..], c),
        rel::NEXT => b_wd_upcoming(&v[1..], c),
        _ => None,
    }
}

/// `lunes de la próxima semana`, `lunes de la semana siguiente`: the
/// weekday first and the relative word last among the children.
fn b_wd_of_week(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let rel = v.last()?;
    if !matches!(term(rel).1, rel::THIS | rel::NEXT | rel::LAST) {
        return None;
    }
    en::b_wd_of_week(&[v[0], *rel], c)
}

fn b_wd_of_week_next(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_wd_of_week(&[v[0], NEXT], c)
}

/// `la semana anterior - lunes`.
fn b_week_wd(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_week_wd(&[v[0], *v.last()?], c)
}

/// `la semana pasada`, `el mes que viene`, `el trimestre actual`.
fn b_period_after(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_period_rel(&[v[1], v[0]], c)
}

fn b_period_next(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_period_rel(&[NEXT, v[0]], c)
}

/// `el año`, `la semana`: the reference's own, a year, month or week.
fn b_period_the(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let unit = v.last()?;
    if !matches!(en::unit_grain(unit)?, TimeGrain::Year | TimeGrain::Month | TimeGrain::Week) {
        return None;
    }
    en::b_period_the(&[*unit], c)
}

/// `el día anterior`, `el día siguiente`.
fn b_day_rel(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (n, kind) = term(v.last()?);
    if term(&v[0]).0 != words::unit(words::DAY, 1) || !matches!(kind, rel::NEXT | rel::LAST) {
        return None;
    }
    en::day_rel(c, n as i16)
}

/// `semana 27 del año pasado`.
fn b_week_n_of(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_week_n(&[v[0], *v.last()?], c)
}

/// `la primera semana de 2015`.
fn b_nth_week(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_nth_week(&[v[0], *v.last()?], c)
}

/// `el primer trimestre de 2017`, `el segundo semestre de 2019`.
fn b_nth_part_of(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    nth_part(c, &v[0], &v[1], en::year_of(c, v.last()?)?)
}

/// `1º trimestre 2020`, `1º trimestre-2019`.
fn b_nth_part_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    nth_part(c, &v[0], &v[1], en::full_year(v.last()?)?)
}

/// `el primer trimestre del año`, `2º semestre`: of no year said.
fn b_nth_part_the_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let year = if term(&v[1]).0 == words::unit(words::MONTH_U, 6) { YearRef::Rel(0) } else { YearRef::Any };
    nth_part(c, &v[0], &v[1], year)
}

/// A quarter or a half of a year, by its ordinal.
fn nth_part(c: &Ctx<'_>, ord: &Value, unit: &Value, year: YearRef) -> Option<Value> {
    let n = i64::from(en::nth(ord)?);
    match term(unit) {
        (u, 0) if u == words::unit(words::MONTH_U, 3) && (1..=4).contains(&n) => en::quarter(c, year, n),
        (u, 0) if u == words::unit(words::MONTH_U, 6) && (1..=2).contains(&n) => {
            emit_date(c, DateSpec::Half { year, half: n as u8 })
        }
        _ => None,
    }
}

/// `década de 1990`.
fn b_decade_of(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (dig_len(v.last()?) == 4).then(|| en::b_decade(&[*v.last()?], c))?
}

/// `los años noventa`, `los años 90`.
fn b_years_tens(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let n = term(v.last()?).0;
    (n % 10 == 0 && (10..100).contains(&n)).then(|| emit_date(c, DateSpec::Decade { year: YearRef::Short(n as u8) }))?
}

/// The kind of year a word names: `fiscal`, `escolar`; `calendario` is
/// the plain year.
fn year_kind(c: &Ctx<'_>) -> Option<Option<YearKind>> {
    (c.start..c.end).find_map(|i| match c.word(i) {
        "fiscal" => Some(Some(YearKind::Fiscal)),
        "escolar" => Some(Some(YearKind::School)),
        "calendario" => Some(None),
        _ => None,
    })
}

fn other_year(c: &Ctx<'_>, year: YearRef) -> Option<Value> {
    match year_kind(c)? {
        Some(kind) => emit_date(c, DateSpec::OtherYear { kind, year }),
        None => emit_date(c, en::cal(year, 0, 0, 0)?),
    }
}

fn b_other_year(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    other_year(c, YearRef::Any).filter(|_| year_kind(c) != Some(None))
}

/// `año fiscal 2008`, `año calendario 18`.
fn b_other_year_n(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    other_year(c, year_ref(v.last()?))
}

/// `este año escolar`, `el último año fiscal`.
fn b_rel_other_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (n, kind) = term(v.iter().find(|x| matches!(x, Value::Term { flags: 1..=6, .. }))?);
    if !matches!(kind, rel::THIS | rel::NEXT | rel::LAST) {
        return None;
    }
    match year_kind(c)? {
        Some(k) => emit_date(c, DateSpec::OtherYear { kind: k, year: YearRef::Rel(n as i16) }),
        None => emit_date(c, DateSpec::Rel { grain: TimeGrain::Year, n: n as i16 }),
    }
}

fn b_holiday_of(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_holiday_of_year(&[v[0], *v.last()?], c)
}

/// `la semana del 4 de febrero`.
fn b_week_of(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    match date_of(c, v.last()?)? {
        DateSpec::Cal { year: YearRef::Any, month, day, weekday: 0 } if day > 0 => {
            emit_date(c, DateSpec::WeekOf { month, day })
        }
        _ => None,
    }
}

/// `el día 21 de este mes`.
fn b_day_this_month(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_rel_month_day(&[v[0], *v.last()?], c)
}

fn b_season_the(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_season(&[*v.last()?], c)
}

fn b_season_after(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_rel_season(&[v[1], v[0]], c)
}

/// `el 4.º`: an ordinal day in digits; `el primero` is more often the
/// first of something else.
fn b_dom_ordinal(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (c.lexeme(c.start).kind == Kind::Numeral).then(|| en::b_dom_ordinal(v, c))?
}

fn hol(c: &Ctx<'_>, i: i64) -> Option<Value> {
    en::hol(c, i)
}

const fn rr(name: &'static str, dim: crate::engine::Dim, items: &'static [Item], build: crate::engine::Build) -> Rule {
    r(name, dim, 0, items, build)
}

pub(super) static RULES: &[Rule] = &[
    rr("sep_slash", SEP, &[word("/")], en::b_sep),
    rr("sep_dash", SEP, &[word("-")], en::b_sep),
    rr("unit_fin_de_semana", UNIT, &[word("fin"), word("de"), word("semana")], |_, _| {
        term_value(words::unit(words::WEEKEND, 1), 0)
    }),
    rr("unit_fines_de_semana", UNIT, &[word("fines"), word("de"), word("semana")], |_, _| {
        term_value(words::unit(words::WEEKEND, 1), PLURAL)
    }),
    // Years and days of the month.
    rr("year_digits", YEAR, &[node(DIG)], en::b_year_digits),
    rr("year_short", YEAR, &[node(DIG)], en::b_year_short),
    rr("year_spelled", YEAR, &[node(INT)], en::b_year_spelled),
    rr("mday_digits", MDAY, &[node(DIG)], en::b_mday_digits),
    rr("mday_ordinal", MDAY, &[node(ORD)], en::b_mday_ordinal),
    rr("mday_spelled", MDAY, &[node(INT)], en::b_mday_spelled),
    // A day and a month.
    rr("month_abbr_dot", MONTH, &[node(MONTH), glued_word(".")], b_abbr_dot),
    rr("weekday_abbr_dot", WEEKDAY, &[node(WEEKDAY), glued_word(".")], b_abbr_dot),
    rr("md_day_de_month", MD, &[node(MDAY), node(DE), node(MONTH)], |v, c| en::b_md_day_month(&[v[0], v[2]], c)),
    rr("md_day_month", MD, &[node(MDAY), node(MONTH)], en::b_md_day_month),
    rr("md_day_de_digits", MD, &[node(MDAY), node(DE), node(DIG)], b_day_of_month_digits),
    rr("md_month_day", MD, &[node(MONTH), node(MDAY)], en::b_md_month_day),
    rr("md_slash", MD, &[node(DIG), glued_word("/"), glued(DIG)], b_dm),
    rr("md_dash", MD, &[node(DIG), glued_word("-"), glued(DIG)], b_dm),
    rr("md_d_mon", MD, &[node(DIG), glued(SEP), glued(MONTH)], b_d_mon),
    rr("md_el_dia", MD, &[word("el"), word("dia"), node(MD)], en::b_same),
    rr("md_primer_dia", MD, &[node(ORD), word("dia"), node(DE), node(MONTH)], |v, c| {
        en::b_md_day_month(&[en::b_mday_ordinal(&v[..1], c)?, v[2]], c)
    }),
    // Dates.
    rr("date_md", DATE, &[node(MD)], en::b_same),
    rr("date_wd", DATE, &[node(WD)], en::b_same),
    rr("date_dom", DATE, &[node(DOM)], en::b_same),
    rr("date_md_year", DATE, &[node(MD), opt_word(","), node(YEAR)], en::b_md_year),
    rr("date_md_de_year", DATE, &[node(MD), opt_word(","), node(DE), node(YEAR)], en::b_md_year),
    rr("date_dmy_slash", DATE, &[node(DIG), glued_word("/"), glued(DIG), glued_word("/"), glued(DIG)], |v, c| {
        en::b_dmy(&[v[0], v[1], v[2]], c)
    }),
    rr("date_dmy_dash", DATE, &[node(DIG), glued_word("-"), glued(DIG), glued_word("-"), glued(DIG)], |v, c| {
        en::b_dmy(&[v[0], v[1], v[2]], c)
    }),
    rr("date_dmy_dot", DATE, &[node(DIG), glued_word("."), glued(DIG), glued_word("."), glued(DIG)], |v, c| {
        en::b_dmy(&[v[0], v[1], v[2]], c)
    }),
    rr("date_ymd", DATE, &[node(DIG), glued(SEP), glued(DIG), glued(SEP), glued(DIG)], |v, c| {
        en::b_ymd(&[v[0], v[2], v[4]], c)
    }),
    rr("date_y_mon_d", DATE, &[node(DIG), glued(SEP), glued(MONTH), glued(SEP), glued(DIG)], en::b_y_mon_d),
    rr("date_d_mon_y", DATE, &[node(DIG), glued(SEP), glued(MONTH), glued(SEP), glued(DIG)], b_d_mon_y),
    rr("date_mon_d_y", DATE, &[node(MONTH), glued(SEP), glued(DIG), glued(SEP), glued(DIG)], en::b_mon_d_y),
    rr("date_mon_y_d", DATE, &[node(MONTH), glued(SEP), glued(DIG), glued(SEP), glued(DIG)], en::b_mon_y_d),
    rr("date_y_d_mon", DATE, &[node(DIG), glued(SEP), glued(DIG), glued(SEP), glued(MONTH)], en::b_y_d_mon),
    rr("date_d_y_mon", DATE, &[node(DIG), glued(SEP), glued(DIG), glued(SEP), glued(MONTH)], en::b_d_y_mon),
    rr("date_weekday_md", DATE, &[node(WEEKDAY), opt_word(","), node(MD)], en::b_weekday_date),
    rr("date_weekday_date", DATE, &[node(WEEKDAY), opt_word(","), node(DATE)], en::b_weekday_date),
    rr("date_dayword", DATE, &[node(DAYWORD)], en::b_dayword),
    rr("date_pasado_manana", DATE, &[word("pasado"), word("mañana")], |_, c| en::day_rel(c, 2)),
    rr("date_el_dia", DATE, &[word("el"), node(UNIT)], |v, c| {
        (term(&v[0]) == (words::unit(words::DAY, 1), 0)).then(|| en::day_rel(c, 0))?
    }),
    rr("date_el_dia_rel", DATE, &[word("el"), node(UNIT), node(RELW)], b_day_rel),
    rr(
        "date_day_this_month",
        DATE,
        &[word("el"), word("dia"), node(MDAY), node(DE), node(RELW), word("mes")],
        |v, c| b_day_this_month(&[v[0], v[2]], c),
    ),
    // Weekdays.
    rr("wd", WD, &[node(WEEKDAY)], en::b_wd),
    rr("wd_rel", WD, &[node(RELW), node(WEEKDAY)], en::b_wd_rel),
    rr("wd_rel_after", WD, &[node(WEEKDAY), node(RELW)], b_wd_after),
    rr("wd_que_viene", WD, &[node(WEEKDAY), word("que"), word("viene")], b_wd_next),
    rr("wd_este_rel", WD, &[word("este"), node(RELW), node(WEEKDAY)], b_wd_this_rel),
    rr("wd_este_pasado", WD, &[word("este"), node(WEEKDAY), word("pasado")], b_wd_past),
    rr("wd_day", WD, &[node(WEEKDAY), node(MDAY)], en::b_wd_day),
    rr("wd_el_day", WD, &[node(WEEKDAY), word("el"), opt_word("dia"), node(MDAY)], en::b_wd_the_day),
    rr("wd_de_la_rel_semana", WD, &[node(WEEKDAY), node(DE), node(ART), node(RELW), word("semana")], b_wd_of_week),
    rr("wd_de_la_semana_rel", WD, &[node(WEEKDAY), node(DE), node(ART), word("semana"), node(RELW)], b_wd_of_week),
    rr("wd_de_rel_semana", WD, &[node(WEEKDAY), node(DE), node(RELW), word("semana")], b_wd_of_week),
    rr("wd_de_semana_rel", WD, &[node(WEEKDAY), node(DE), word("semana"), node(RELW)], b_wd_of_week),
    rr("wd_la_rel_semana", WD, &[node(WEEKDAY), node(ART), node(RELW), word("semana")], b_wd_of_week),
    rr("wd_rel_semana", WD, &[node(WEEKDAY), node(RELW), word("semana")], b_wd_of_week),
    rr(
        "wd_de_la_semana_que_viene",
        WD,
        &[node(WEEKDAY), node(DE), node(ART), word("semana"), word("que"), word("viene")],
        b_wd_of_week_next,
    ),
    rr("wd_semana_rel_dash", WD, &[node(ART), word("semana"), node(RELW), word("-"), node(WEEKDAY)], |v, c| {
        b_week_wd(&v[1..], c)
    }),
    rr("dom_el_dia", DOM, &[word("el"), word("dia"), node(MDAY)], en::b_dom),
    rr("dom_ordinal", DOM, &[node(MDAY)], b_dom_ordinal),
    // Months.
    rr("monthd_month", MONTHD, &[node(MONTH)], en::b_month),
    rr("monthd_rel", MONTHD, &[node(RELW), node(MONTH)], en::b_rel_month),
    rr("monthd_year", MONTHD, &[node(MONTH), opt_word(","), node(YEAR)], |v, c| month_year(c, &v[0], &v[1], false)),
    rr("monthd_de_year", MONTHD, &[node(MONTH), node(DE), node(YEAR)], |v, c| month_year(c, &v[0], &v[2], true)),
    rr("monthd_en_year", MONTHD, &[node(MONTH), word("en"), node(YEAR)], |v, c| month_year(c, &v[0], &v[1], true)),
    rr("monthd_sep_year", MONTHD, &[node(MONTH), glued(SEP), glued(YEAR)], en::b_month_sep_year),
    rr("monthd_my", MONTHD, &[node(DIG), glued_word("-"), glued(DIG)], en::b_my_digits),
    rr("monthd_nth", MONTHD, &[node(ORD), word("mes"), node(DE), node(PERIOD)], |v, c| {
        en::b_nth_month(&[v[0], v[2]], c)
    }),
    rr("monthd_el_nth", MONTHD, &[word("el"), node(ORD), word("mes"), node(DE), node(PERIOD)], |v, c| {
        en::b_nth_month(&[v[0], v[2]], c)
    }),
    // Periods.
    rr("period_year", PERIOD, &[node(YEAR)], en::b_period_year),
    rr("period_year_word", PERIOD, &[word("año"), node(YEAR)], en::b_year_word),
    rr("period_del_year_word", PERIOD, &[word("del"), word("año"), node(YEAR)], en::b_year_word),
    rr("period_rel", PERIOD, &[node(RELW), node(UNIT)], en::b_period_rel),
    rr("period_art_rel", PERIOD, &[node(ART), node(RELW), node(UNIT)], |v, c| en::b_period_rel(&v[1..], c)),
    rr("period_rel_after", PERIOD, &[node(UNIT), node(RELW)], b_period_after),
    rr("period_art_rel_after", PERIOD, &[node(ART), node(UNIT), node(RELW)], |v, c| b_period_after(&v[1..], c)),
    rr("period_que_viene", PERIOD, &[node(UNIT), word("que"), word("viene")], b_period_next),
    rr("period_art_que_viene", PERIOD, &[node(ART), node(UNIT), word("que"), word("viene")], |v, c| {
        b_period_next(&v[1..], c)
    }),
    rr("period_the_unit", PERIOD, &[node(UNIT)], b_period_the),
    rr("period_same", PERIOD, &[word("mismo"), node(UNIT)], en::b_undefined),
    rr("period_same_f", PERIOD, &[word("misma"), node(UNIT)], en::b_undefined),
    rr("period_ese", PERIOD, &[word("ese"), node(UNIT)], en::b_undefined),
    rr("period_esa", PERIOD, &[word("esa"), node(UNIT)], en::b_undefined),
    rr(
        "period_week_after_next",
        PERIOD,
        &[word("la"), word("semana"), word("despues"), word("de"), word("la"), word("proxima")],
        en::b_week_after_next,
    ),
    rr("period_week_n", PERIOD, &[word("semana"), node(DIG)], en::b_week_n),
    rr("period_week_n_of", PERIOD, &[word("semana"), node(DIG), node(DE), node(PERIOD)], b_week_n_of),
    rr("period_week_n_of_year", PERIOD, &[word("semana"), node(DIG), node(DE), node(YEAR)], |v, c| {
        let w = en::small(&v[0])?;
        ((1..=53).contains(&w)).then(|| emit_date(c, DateSpec::Week { year: year_ref(v.last()?), week: w as u8 }))?
    }),
    rr("period_nth_week", PERIOD, &[node(ORD), word("semana"), node(DE), node(PERIOD)], b_nth_week),
    rr("period_nth_week_month", PERIOD, &[node(ORD), word("semana"), node(DE), node(MONTHD)], b_nth_week),
    rr("period_la_nth_week", PERIOD, &[word("la"), node(ORD), word("semana"), node(DE), node(PERIOD)], b_nth_week),
    rr(
        "period_la_nth_week_month",
        PERIOD,
        &[word("la"), node(ORD), word("semana"), node(DE), node(MONTHD)],
        b_nth_week,
    ),
    rr("period_last_week", PERIOD, &[word("la"), word("ultima"), word("semana"), node(DE), node(PERIOD)], |v, c| {
        en::b_last_week(&v[1..], c)
    }),
    rr(
        "period_last_week_month",
        PERIOD,
        &[word("la"), word("ultima"), word("semana"), node(DE), node(MONTHD)],
        |v, c| en::b_last_week(&v[1..], c),
    ),
    rr("period_week_of", PERIOD, &[word("la"), word("semana"), node(DE), node(DATE)], b_week_of),
    rr("period_week_of_dom", PERIOD, &[word("la"), word("semana"), node(DE), node(MDAY)], |v, c| {
        emit_date(c, DateSpec::WeekOf { month: 0, day: u8::try_from(term(v.last()?).0).ok()? })
    }),
    rr(
        "period_week_que_inicia",
        PERIOD,
        &[word("la"), word("semana"), word("que"), word("inicia"), word("el"), node(DATE)],
        b_week_of,
    ),
    rr(
        "period_week_que_comienza",
        PERIOD,
        &[word("la"), word("semana"), word("que"), word("comienza"), word("el"), node(DATE)],
        b_week_of,
    ),
    rr(
        "period_week_que_va",
        PERIOD,
        &[word("la"), word("semana"), word("que"), word("va"), node(DE), node(DATE)],
        b_week_of,
    ),
    rr(
        "period_week_a_partir",
        PERIOD,
        &[word("la"), word("semana"), word("a"), word("partir"), node(DE), node(DATE)],
        b_week_of,
    ),
    rr("qtr", QTR, &[word("t"), glued(DIG)], en::b_qtr),
    rr("period_quarter", PERIOD, &[node(QTR)], en::b_quarter),
    rr("period_year_quarter", PERIOD, &[node(YEAR), node(QTR)], en::b_year_quarter),
    rr("period_year_sep_quarter", PERIOD, &[node(YEAR), glued_word("-"), glued(QTR)], en::b_year_quarter),
    rr("period_nth_part", PERIOD, &[node(ORD), node(UNIT), node(DE), node(PERIOD)], b_nth_part_of),
    rr("period_el_nth_part", PERIOD, &[word("el"), node(ORD), node(UNIT), node(DE), node(PERIOD)], b_nth_part_of),
    rr("period_nth_part_year", PERIOD, &[node(ORD), node(UNIT), node(YEAR)], b_nth_part_year),
    rr("period_nth_part_sep_year", PERIOD, &[node(ORD), node(UNIT), glued_word("-"), glued(YEAR)], b_nth_part_year),
    rr(
        "period_nth_part_the_year",
        PERIOD,
        &[word("el"), node(ORD), node(UNIT), word("del"), word("año")],
        b_nth_part_the_year,
    ),
    rr("period_nth_part_alone", PERIOD, &[node(ORD), node(UNIT)], |v, c| {
        (term(&v[1]).0 == words::unit(words::MONTH_U, 6)).then(|| b_nth_part_the_year(v, c))?
    }),
    rr("period_decade_of", PERIOD, &[word("decada"), node(DE), node(DIG)], b_decade_of),
    rr("period_years_tens", PERIOD, &[word("los"), word("años"), node(TENS)], b_years_tens),
    rr("period_years_digits", PERIOD, &[word("los"), word("años"), node(DIG)], b_years_tens),
    rr("period_anos_tens", PERIOD, &[word("los"), word("anos"), node(TENS)], b_years_tens),
    rr("period_anos_digits", PERIOD, &[word("los"), word("anos"), node(DIG)], b_years_tens),
    rr("period_season", PERIOD, &[node(SEASON)], en::b_season),
    rr("period_the_season", PERIOD, &[node(ART), node(SEASON)], b_season_the),
    rr("period_rel_season", PERIOD, &[node(RELW), node(SEASON)], en::b_rel_season),
    rr("period_season_rel", PERIOD, &[node(SEASON), node(RELW)], b_season_after),
    rr("period_season_year", PERIOD, &[node(SEASON), opt_word("de"), node(YEAR)], en::b_season_year),
    rr("period_year_kind", PERIOD, &[word("año"), node(KIND)], b_other_year),
    rr("period_kind_year", PERIOD, &[node(KIND), word("año")], b_other_year),
    rr("period_year_kind_n", PERIOD, &[word("año"), node(KIND), opt_word("de"), node(YEAR)], b_other_year_n),
    rr("period_kind_year_n", PERIOD, &[node(KIND), word("año"), node(YEAR)], b_other_year_n),
    rr("period_rel_year_kind", PERIOD, &[node(RELW), word("año"), node(KIND)], b_rel_other_year),
    rr("period_rel_kind_year", PERIOD, &[node(RELW), node(KIND), word("año")], b_rel_other_year),
    rr("period_el_rel_year_kind", PERIOD, &[word("el"), node(RELW), word("año"), node(KIND)], b_rel_other_year),
    rr("period_el_rel_kind_year", PERIOD, &[word("el"), node(RELW), node(KIND), word("año")], b_rel_other_year),
    // Holidays.
    rr("hol", HOL, &[node(HOLIDAY)], en::b_holiday),
    rr("hol_trabajador", HOL, &[word("dia"), word("del"), word("trabajador")], |_, c| hol(c, WORKERS)),
    rr("hol_trabajo", HOL, &[word("dia"), word("del"), word("trabajo")], |_, c| hol(c, WORKERS)),
    rr(
        "hol_trabajadores",
        HOL,
        &[word("dia"), word("internacional"), word("de"), word("los"), word("trabajadores")],
        |_, c| hol(c, WORKERS),
    ),
    rr("hol_independencia", HOL, &[word("dia"), word("de"), opt_word("la"), word("independencia")], |_, c| {
        hol(c, INDEPENDENCE)
    }),
    rr("hol_viernes_negro", HOL, &[word("viernes"), word("negro")], |_, c| hol(c, BLACK_FRIDAY)),
    rr("hol_tierra", HOL, &[word("dia"), word("de"), word("la"), word("tierra")], |_, c| hol(c, EARTH)),
    rr("hol_san_patricio", HOL, &[word("dia"), word("de"), word("san"), word("patricio")], |_, c| hol(c, PATRICK)),
    rr("hol_san_valentin", HOL, &[word("dia"), word("de"), word("san"), word("valentin")], |_, c| hol(c, VALENTINE)),
    rr("hol_ano_nuevo", HOL, &[word("año"), word("nuevo")], |_, c| hol(c, words::NEW_YEAR)),
    rr("hol_dia_ano_nuevo", HOL, &[word("dia"), word("de"), word("año"), word("nuevo")], |_, c| {
        hol(c, words::NEW_YEAR)
    }),
    rr("hol_viernes_santo", HOL, &[word("viernes"), word("santo")], |_, c| hol(c, words::GOOD_FRIDAY)),
    rr("hol_year", HOL, &[node(HOL), opt_word(","), node(YEAR)], en::b_holiday_year),
    rr("hol_of_year", HOL, &[node(HOL), node(DE), node(PERIOD)], b_holiday_of),
];
