//! Dates and periods, the day before the month: `22 april`, `22/04`,
//! `1-1-2015`, `de 9e van mei`, `januari de eerste`, `vrijdag`,
//! `volgende week maandag`, `de 5e`, `april 2017`, `deze week`, `de
//! eerste week van 2015`, `het eerste kwartaal van volgend jaar`, `de
//! jaren '90`, `15e eeuw`, holidays and seasons.

use turbo_analysis_layers::TimeGrain;

use crate::engine::{Ctx, Grammar, Item, Kind, Rule, glued, glued_word, node, opt_word, word};
use crate::lang::en::time::clock::{dig_len, dig_zero};
use crate::lang::en::time::date as en;
use crate::lang::en::time::lexicon::rel;
use crate::lang::en::time::{emit_date, r, term, term_value};
use crate::time::spec::{DateSpec, YearRef};
use crate::value::Value;

use super::super::{INT, ORD};
use super::lexicon::{self, HOLIDAYS};
use super::{
    ABBR, DATE, DAYWORD, DIG, DOM, HOL, HOLIDAY, MD, MDAY, MONTH, MONTHD, PERIOD, QTR, RELW, SEASON, SEP, SHORT, UNIT,
    WD, WEEKDAY, YEAR,
};

/// `'18`, `'90`: two digits after an apostrophe.
fn b_year_apos(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    (dig_len(&v[0]) == 2).then(|| term_value(term(&v[0]).0, SHORT))?
}

/// `20 ste`: a day of the month and its ordinal ending apart.
fn b_mday_spaced(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_mday_suffix(v, c)
}

/// A day and a month: `22 april`, `de 9e van mei`; a dot after the month
/// only for an abbreviation, `31 dec.`.
fn b_day_month(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (day, month) = (&v[0], &v[1]);
    if c.word(c.end - 1) == "." && term(month).1 & ABBR == 0 {
        return None;
    }
    en::b_md_day_month(&[*day, *month], c)
}

/// `de 9e van mei` takes its article with a day in digits; `de tweede van
/// augustus` leaves it out, as Recognizers-Text does.
fn b_the_day_month(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (c.lexeme(c.start + 1).kind == Kind::Numeral).then(|| b_day_month(v, c))?
}

/// `januari de eerste`, `november de 15e`.
fn b_month_the_day(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::md(c, term(&v[0]).0, term(&v[1]).0)
}

fn dm(c: &Ctx<'_>, a: &Value, b: &Value) -> Option<Value> {
    if dig_len(a) > 2 || dig_len(b) > 2 {
        return None;
    }
    let (d, m) = (term(a).0, term(b).0);
    ((1..=12).contains(&m)).then(|| en::md(c, m, d))?
}

/// Whether the word before the match is a weekday.
fn after_weekday(c: &Ctx<'_>) -> bool {
    c.before().is_some_and(|p| super::DUTCH_TIMES.lexicon().lookup(c.word(p)).iter().any(|e| e.dim == WEEKDAY))
}

/// `22/04`, `12/5`: the day first. After a number it is a fraction.
fn b_dm_slash(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if c.before().is_some_and(|p| c.lexeme(p).kind == Kind::Numeral) {
        return None;
    }
    dm(c, &v[0], &v[1])
}

/// `20-11`, `1-10`; after a weekday two hours in order are a clock range,
/// `maandag 3-8`.
fn b_dm_dash(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (a, b) = (term(&v[0]).0, term(&v[1]).0);
    if after_weekday(c) && a < b && b <= 12 {
        return None;
    }
    dm(c, &v[0], &v[1])
}

/// `2/okt`, `3/jan`.
fn b_day_sep_month(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (dig_len(&v[0]) <= 2).then(|| en::md(c, term(&v[2]).0, term(&v[0]).0))?
}

fn b_same(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(v[0])
}

/// A month after a relative word: `deze september` is this year's,
/// `aanstaande september` and `volgende september` next year's.
fn b_rel_month(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (n, kind) = term(&v[0]);
    let n = match kind {
        rel::THIS => 0,
        rel::NEXT | rel::COMING | rel::UPCOMING => 1,
        _ => n,
    };
    emit_date(c, en::cal(YearRef::Rel(n as i16), term(&v[1]).0, 0, 0)?)
}

/// `deze 12/5`, `volgende 12/5`: a day of this year, the next or the
/// last.
fn b_rel_md(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (n, kind) = term(&v[0]);
    let n = match kind {
        rel::THIS => 0,
        rel::NEXT | rel::LAST => n,
        _ => return None,
    };
    let DateSpec::Cal { year: YearRef::Any, month: month @ 1.., day: day @ 1.., weekday } =
        crate::lang::en::time::date_of(c, &v[1])?
    else {
        return None;
    };
    emit_date(c, DateSpec::Cal { year: YearRef::Rel(n as i16), month, day, weekday })
}

/// `deze dag`, `de volgende dag`.
fn b_rel_day(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (en::unit_grain(&v[1]) == Some(TimeGrain::Day) && term(&v[1]).1 == 0)
        .then(|| en::day_rel(c, term(&v[0]).0 as i16))?
}

/// `jaren negentig`: a decade in words.
fn b_decade_spelled(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let y = en::spelled(&v[0])?;
    (y % 10 == 0 && (10..100).contains(&y)).then(|| emit_date(c, DateSpec::Decade { year: YearRef::Short(y as u8) }))?
}

/// `jaren 1990`, `de jaren '90`.
fn b_decade(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if dig_zero(v.last()?) {
        return None;
    }
    en::b_decade(&v[v.len() - 1..], c)
}

fn hol(c: &Ctx<'_>, i: i64) -> Option<Value> {
    let holiday = *HOLIDAYS.get(usize::try_from(i).ok()?)?;
    emit_date(c, DateSpec::Holiday { holiday, year: YearRef::Any })
}

fn b_holiday(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    hol(c, term(&v[0]).0)
}

/// `het eerste kwartaal van volgend jaar`.
fn b_nth_quarter_of(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_nth_quarter_of(v, c)
}

const fn rr(name: &'static str, dim: crate::engine::Dim, items: &'static [Item], build: crate::engine::Build) -> Rule {
    r(name, dim, 0, items, build)
}

pub(super) static RULES: &[Rule] = &[
    rr("sep_slash", SEP, &[word("/")], en::b_sep),
    rr("sep_dash", SEP, &[word("-")], en::b_sep),
    // Years.
    rr("year_digits", YEAR, &[node(DIG)], en::b_year_digits),
    rr("year_short", YEAR, &[node(DIG)], en::b_year_short),
    rr("year_apos", YEAR, &[word("'"), glued(DIG)], b_year_apos),
    rr("year_spelled", YEAR, &[node(INT)], en::b_year_spelled),
    // Days of the month.
    rr("mday_digits", MDAY, &[node(DIG)], en::b_mday_digits),
    rr("mday_ordinal", MDAY, &[node(ORD)], en::b_mday_ordinal),
    rr("mday_spelled", MDAY, &[node(INT)], en::b_mday_spelled),
    rr("mday_spaced", MDAY, &[node(DIG), word("ste")], b_mday_spaced),
    // A day and a month.
    rr("md_day_month", MD, &[node(MDAY), opt_word("van"), node(MONTH), opt_word(".")], b_day_month),
    rr("md_the_day_month", MD, &[word("de"), node(MDAY), opt_word("van"), node(MONTH), opt_word(".")], b_the_day_month),
    rr("md_day_of_month", MD, &[node(MDAY), word("dag"), word("van"), node(MONTH)], b_day_month),
    rr("md_day_of_month_de", MD, &[word("de"), node(MDAY), word("dag"), word("van"), node(MONTH)], b_day_month),
    rr("md_month_the_day", MD, &[node(MONTH), word("de"), node(MDAY)], b_month_the_day),
    rr("md_month_day", MD, &[node(MONTH), opt_word("."), node(MDAY)], en::b_md_month_day),
    rr("md_slash", MD, &[node(DIG), glued_word("/"), glued(DIG)], b_dm_slash),
    rr("md_dash", MD, &[node(DIG), glued_word("-"), glued(DIG)], b_dm_dash),
    rr("md_day_sep_month", MD, &[node(DIG), glued(SEP), glued(MONTH)], b_day_sep_month),
    rr("md_paren", MD, &[node(MD), word("("), node(WEEKDAY), opt_word("."), word(")")], b_same),
    // Days.
    rr("date_md", DATE, &[node(MD)], b_same),
    rr("date_wd", DATE, &[node(WD)], b_same),
    rr("date_dom", DATE, &[node(DOM)], b_same),
    rr("date_md_year", DATE, &[node(MD), opt_word(","), opt_word("van"), opt_word("in"), node(YEAR)], en::b_md_year),
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
    rr("date_yyyymmdd", DATE, &[node(DIG)], en::b_yyyymmdd),
    rr("date_d_mon_y", DATE, &[node(DIG), glued(SEP), glued(MONTH), glued(SEP), glued(DIG)], en::b_d_mon_y),
    rr("date_y_mon_d", DATE, &[node(DIG), glued(SEP), glued(MONTH), glued(SEP), glued(DIG)], en::b_y_mon_d),
    rr("date_weekday_md", DATE, &[node(WEEKDAY), opt_word(","), node(MD)], en::b_weekday_date),
    rr("date_weekday_date", DATE, &[node(WEEKDAY), opt_word(","), node(DATE)], en::b_weekday_date),
    rr("date_dayword", DATE, &[node(DAYWORD)], en::b_dayword),
    rr("date_the_day_before", DATE, &[word("de"), word("dag"), word("ervoor")], |_, c| en::day_rel(c, -1)),
    rr("date_the_day_before_that", DATE, &[word("de"), word("dag"), word("daarvoor")], |_, c| en::day_rel(c, -1)),
    rr("date_the_day_after", DATE, &[word("de"), word("dag"), word("erna")], |_, c| en::day_rel(c, 1)),
    rr("date_the_day_after_that", DATE, &[word("de"), word("dag"), word("daarna")], |_, c| en::day_rel(c, 1)),
    rr("date_rel_day", DATE, &[node(RELW), node(UNIT)], b_rel_day),
    rr("date_rel_md", DATE, &[node(RELW), node(MD)], b_rel_md),
    rr(
        "date_day_of_rel_month",
        DATE,
        &[node(MDAY), word("van"), word("de"), node(RELW), word("maand")],
        en::b_rel_month_day,
    ),
    rr("date_nth_weekday", DATE, &[node(ORD), node(WEEKDAY), word("van"), node(MONTHD)], en::b_nth_weekday),
    rr(
        "date_nth_weekday_de",
        DATE,
        &[word("de"), node(ORD), node(WEEKDAY), word("van"), node(MONTHD)],
        en::b_nth_weekday,
    ),
    rr("date_nth_weekday_rel", DATE, &[node(ORD), node(WEEKDAY), word("van"), node(PERIOD)], en::b_nth_weekday),
    rr(
        "date_nth_weekday_rel_de",
        DATE,
        &[word("de"), node(ORD), node(WEEKDAY), word("van"), node(PERIOD)],
        en::b_nth_weekday,
    ),
    rr("date_last_weekday", DATE, &[word("laatste"), node(WEEKDAY), word("van"), node(MONTHD)], en::b_last_weekday),
    rr(
        "date_last_weekday_de",
        DATE,
        &[word("de"), word("laatste"), node(WEEKDAY), word("van"), node(MONTHD)],
        en::b_last_weekday,
    ),
    // Weekdays.
    rr("wd", WD, &[node(WEEKDAY)], en::b_wd),
    rr("wd_dot", WD, &[node(WEEKDAY), glued_word(".")], |v, c| (term(&v[0]).1 & ABBR != 0).then(|| en::b_wd(v, c))?),
    rr("wd_rel", WD, &[node(RELW), node(WEEKDAY)], en::b_wd_rel),
    rr("wd_this_rel", WD, &[word("deze"), node(RELW), node(WEEKDAY)], en::b_wd_this_rel),
    rr("wd_day", WD, &[node(WEEKDAY), node(MDAY)], en::b_wd_day),
    rr("wd_the_day", WD, &[node(WEEKDAY), word("de"), node(MDAY)], en::b_wd_the_day),
    rr("wd_week", WD, &[node(RELW), word("week"), opt_word("op"), opt_word("-"), node(WEEKDAY)], en::b_week_wd),
    rr(
        "wd_week_op",
        WD,
        &[word("op"), node(RELW), word("week"), opt_word("op"), opt_word("-"), node(WEEKDAY)],
        en::b_week_wd,
    ),
    rr("wd_of_week", WD, &[node(WEEKDAY), word("van"), node(RELW), word("week")], en::b_wd_of_week),
    // Days of an open month.
    rr("dom_the", DOM, &[word("de"), node(MDAY)], en::b_dom),
    rr("dom_ordinal", DOM, &[node(MDAY)], en::b_dom_ordinal),
    // Months.
    rr("monthd_month", MONTHD, &[node(MONTH)], en::b_month),
    rr("monthd_rel", MONTHD, &[node(RELW), node(MONTH)], b_rel_month),
    rr(
        "monthd_year",
        MONTHD,
        &[node(MONTH), opt_word("."), opt_word(","), opt_word("van"), opt_word("in"), node(YEAR)],
        en::b_month_year,
    ),
    rr("monthd_sep_year", MONTHD, &[node(MONTH), glued(SEP), glued(YEAR)], en::b_month_sep_year),
    rr("monthd_year_month", MONTHD, &[node(YEAR), node(MONTH)], en::b_year_month),
    rr("monthd_ym", MONTHD, &[node(DIG), glued(SEP), glued(DIG)], en::b_ym_digits),
    rr("monthd_my", MONTHD, &[node(DIG), glued_word("-"), glued(DIG)], en::b_my_digits),
    rr("monthd_nth", MONTHD, &[node(ORD), word("maand"), word("van"), node(PERIOD)], en::b_nth_month),
    rr("monthd_nth_de", MONTHD, &[word("de"), node(ORD), word("maand"), word("van"), node(PERIOD)], en::b_nth_month),
    // Periods.
    rr("period_year", PERIOD, &[node(YEAR)], en::b_period_year),
    rr("period_year_spelled", PERIOD, &[node(YEAR)], en::b_period_year_spelled),
    rr("period_year_word", PERIOD, &[word("jaar"), node(YEAR)], en::b_year_word),
    rr("period_eq_year", PERIOD, &[word("="), node(YEAR)], en::b_year_word),
    rr("period_rel", PERIOD, &[node(RELW), node(UNIT)], en::b_period_rel),
    rr("period_the_unit", PERIOD, &[word("de"), node(UNIT)], en::b_period_the),
    rr("period_the_unit_het", PERIOD, &[word("het"), node(UNIT)], en::b_period_the),
    rr("period_weekend", PERIOD, &[node(UNIT)], en::b_weekend),
    rr(
        "period_week_after_next",
        PERIOD,
        &[word("de"), word("week"), word("na"), word("volgende")],
        en::b_week_after_next,
    ),
    rr("period_same", PERIOD, &[word("dezelfde"), node(UNIT)], en::b_undefined),
    rr("period_same_het", PERIOD, &[word("hetzelfde"), node(UNIT)], en::b_undefined),
    rr("period_that", PERIOD, &[word("dat"), node(UNIT)], en::b_undefined),
    rr("period_that_die", PERIOD, &[word("die"), node(UNIT)], en::b_undefined),
    rr("period_week_n", PERIOD, &[word("week"), node(DIG)], en::b_week_n),
    rr("period_week_n_of", PERIOD, &[word("week"), node(DIG), word("van"), node(PERIOD)], en::b_week_n),
    rr("period_decade", PERIOD, &[word("jaren"), node(DIG)], b_decade),
    rr("period_decade_de", PERIOD, &[word("de"), word("jaren"), node(DIG)], b_decade),
    rr("period_decade_apos", PERIOD, &[word("jaren"), word("'"), glued(DIG)], b_decade),
    rr("period_decade_apos_de", PERIOD, &[word("de"), word("jaren"), word("'"), glued(DIG)], b_decade),
    rr("period_decade_spelled", PERIOD, &[word("jaren"), node(INT)], b_decade_spelled),
    rr("period_decade_spelled_de", PERIOD, &[word("de"), word("jaren"), node(INT)], b_decade_spelled),
    rr("period_nth_week", PERIOD, &[node(ORD), word("week"), word("van"), node(PERIOD)], en::b_nth_week),
    rr("period_nth_week_de", PERIOD, &[word("de"), node(ORD), word("week"), word("van"), node(PERIOD)], en::b_nth_week),
    rr("period_nth_week_month", PERIOD, &[node(ORD), word("week"), word("van"), node(MONTHD)], en::b_nth_week),
    rr(
        "period_nth_week_month_de",
        PERIOD,
        &[word("de"), node(ORD), word("week"), word("van"), node(MONTHD)],
        en::b_nth_week,
    ),
    rr("period_last_week", PERIOD, &[word("laatste"), word("week"), word("van"), node(PERIOD)], en::b_last_week),
    rr(
        "period_last_week_de",
        PERIOD,
        &[word("de"), word("laatste"), word("week"), word("van"), node(PERIOD)],
        en::b_last_week,
    ),
    rr("period_last_week_month", PERIOD, &[word("laatste"), word("week"), word("van"), node(MONTHD)], en::b_last_week),
    rr(
        "period_last_week_month_de",
        PERIOD,
        &[word("de"), word("laatste"), word("week"), word("van"), node(MONTHD)],
        en::b_last_week,
    ),
    rr("period_week_of", PERIOD, &[word("de"), word("week"), word("van"), node(DATE)], en::b_week_of),
    rr(
        "period_week_beginning",
        PERIOD,
        &[word("de"), word("week"), word("beginnend"), opt_word("op"), node(DATE)],
        en::b_week_of,
    ),
    rr(
        "period_week_starting",
        PERIOD,
        &[word("de"), word("week"), word("startend"), opt_word("op"), node(DATE)],
        en::b_week_of,
    ),
    rr(
        "period_week_commencing",
        PERIOD,
        &[word("de"), word("week"), word("aanvangend"), opt_word("op"), node(DATE)],
        en::b_week_of,
    ),
    rr("qtr", QTR, &[word("q"), glued(DIG)], en::b_qtr),
    rr("qtr_k", QTR, &[word("k"), glued(DIG)], en::b_qtr),
    rr("period_quarter", PERIOD, &[node(QTR)], en::b_quarter),
    rr("period_year_quarter", PERIOD, &[node(YEAR), node(QTR)], en::b_year_quarter),
    rr("period_quarter_year", PERIOD, &[node(QTR), node(YEAR)], en::b_quarter_year),
    rr("period_nth_quarter", PERIOD, &[node(ORD), node(UNIT), word("van"), node(PERIOD)], b_nth_quarter_of),
    rr(
        "period_nth_quarter_het",
        PERIOD,
        &[word("het"), node(ORD), node(UNIT), word("van"), node(PERIOD)],
        b_nth_quarter_of,
    ),
    rr(
        "period_nth_quarter_the_year",
        PERIOD,
        &[node(ORD), node(UNIT), word("van"), word("het"), word("jaar")],
        en::b_nth_quarter_the_year,
    ),
    rr(
        "period_nth_quarter_the_year_het",
        PERIOD,
        &[word("het"), node(ORD), node(UNIT), word("van"), word("het"), word("jaar")],
        en::b_nth_quarter_the_year,
    ),
    rr("period_century", PERIOD, &[node(ORD), word("eeuw")], en::b_century),
    rr("period_century_the", PERIOD, &[word("de"), node(ORD), word("eeuw")], en::b_century),
    rr("period_season", PERIOD, &[node(SEASON)], en::b_season),
    rr("period_rel_season", PERIOD, &[node(RELW), node(SEASON)], en::b_rel_season),
    rr("period_season_year", PERIOD, &[node(SEASON), opt_word("van"), node(YEAR)], en::b_season_year),
    // Holidays.
    rr("hol", HOL, &[node(HOLIDAY)], b_holiday),
    rr("hol_earth", HOL, &[word("dag"), word("van"), word("de"), word("aarde")], |_, c| hol(c, lexicon::EARTH)),
    rr("hol_labor", HOL, &[word("dag"), word("van"), word("de"), word("arbeid")], |_, c| hol(c, lexicon::LABOR)),
    rr("hol_black_friday", HOL, &[word("black"), word("friday")], |_, c| hol(c, lexicon::BLACK_FRIDAY)),
    rr("hol_good_friday", HOL, &[word("goede"), word("vrijdag")], |_, c| hol(c, lexicon::GOOD_FRIDAY)),
    rr("hol_christmas_1", HOL, &[word("eerste"), word("kerstdag")], |_, c| hol(c, lexicon::CHRISTMAS)),
    rr("hol_christmas_2", HOL, &[word("tweede"), word("kerstdag")], |_, c| hol(c, lexicon::BOXING_DAY)),
    rr("hol_easter_1", HOL, &[word("eerste"), word("paasdag")], |_, c| hol(c, lexicon::EASTER)),
    rr("hol_easter_2", HOL, &[word("tweede"), word("paasdag")], |_, c| hol(c, lexicon::EASTER_MONDAY)),
    rr("hol_pentecost_1", HOL, &[word("eerste"), word("pinksterdag")], |_, c| hol(c, lexicon::PENTECOST)),
    rr("hol_pentecost_2", HOL, &[word("tweede"), word("pinksterdag")], |_, c| hol(c, lexicon::WHIT_MONDAY)),
    rr("hol_patrick", HOL, &[word("sint"), word("patrick")], |_, c| hol(c, lexicon::PATRICK)),
    rr("hol_year", HOL, &[node(HOL), opt_word(","), node(YEAR)], en::b_holiday_year),
    rr("hol_of_year", HOL, &[node(HOL), word("van"), node(PERIOD)], en::b_holiday_of_year),
    rr("hol_near", HOL, &[node(RELW), node(HOL)], en::b_holiday_near),
];
