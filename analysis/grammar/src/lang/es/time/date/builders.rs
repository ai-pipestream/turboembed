use super::*;

/// A relative word made up by a phrase: `que viene` is `next`.
pub(super) const NEXT: Value = Value::Term { value: 1, flags: rel::NEXT };
/// `este pasado`: the most recent.
pub(super) const PAST: Value = Value::Term { value: -1, flags: rel::PAST };
/// `este próximo`: the coming one.
pub(super) const UPCOMING: Value = Value::Term { value: 1, flags: rel::UPCOMING };

/// An abbreviation with its period: `sept.`, `sáb.`.
pub(super) fn b_abbr_dot(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    (term(&v[0]).1 & ABBR != 0).then_some(v[0])
}

/// A day first and a month second in digits, `27/11`; the other way
/// round only when the second cannot be a month, `11/27`.
pub(super) fn b_dm(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[0]).0 == 0 || term(&v[1]).0 == 0 {
        return None;
    }
    en::b_md_slash(&[v[1], v[0]], c)
}

/// `9 de 8 de 1971`, `17 del 11`: a month in digits after `de`.
pub(super) fn b_day_of_month_digits(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let m = en::small(v.last()?)?;
    ((1..=12).contains(&m)).then(|| en::md(c, m, term(&v[0]).0))?
}

/// `4-jul-95`: a day, a month name and a year of two or four digits.
pub(super) fn b_d_mon_y(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, en::cal(en::numeric_year(&v[4])?, term(&v[2]).0, en::small(&v[0])?, 0)?)
}

/// `6-jul`, `30-sept`.
pub(super) fn b_d_mon(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::md(c, term(&v[2]).0, en::small(&v[0])?)
}

/// `abril de 2017`, `abril 2017`; two digits only after a word, as in
/// `abril del 87`.
pub(super) fn month_year(c: &Ctx<'_>, month: &Value, year: &Value, said: bool) -> Option<Value> {
    if term(year).1 & SHORT != 0 && !said {
        return None;
    }
    emit_date(c, en::cal(year_ref(year), term(month).0, 0, 0)?)
}

/// `el lunes pasado`: the relative word after the weekday.
pub(super) fn b_wd_after(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_wd_rel(&[v[1], v[0]], c)
}

pub(super) fn b_wd_next(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_wd_rel(&[NEXT, v[0]], c)
}

pub(super) fn b_wd_past(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_wd_this_rel(&[PAST, *v.last()?], c)
}

pub(super) fn b_wd_upcoming(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_wd_this_rel(&[UPCOMING, *v.last()?], c)
}

/// `este pasado miércoles`, `este próximo viernes`: the most recent or
/// the coming one.
pub(super) fn b_wd_this_rel(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    match term(&v[0]).1 {
        rel::LAST => b_wd_past(&v[1..], c),
        rel::NEXT => b_wd_upcoming(&v[1..], c),
        _ => None,
    }
}

/// `lunes de la próxima semana`, `lunes de la semana siguiente`: the
/// weekday first and the relative word last among the children.
pub(super) fn b_wd_of_week(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let rel = v.last()?;
    if !matches!(term(rel).1, rel::THIS | rel::NEXT | rel::LAST) {
        return None;
    }
    en::b_wd_of_week(&[v[0], *rel], c)
}

pub(super) fn b_wd_of_week_next(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_wd_of_week(&[v[0], NEXT], c)
}

/// `la semana anterior - lunes`.
pub(super) fn b_week_wd(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_week_wd(&[v[0], *v.last()?], c)
}

/// `la semana pasada`, `el mes que viene`, `el trimestre actual`.
pub(super) fn b_period_after(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_period_rel(&[v[1], v[0]], c)
}

pub(super) fn b_period_next(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_period_rel(&[NEXT, v[0]], c)
}

/// `el año`, `la semana`: the reference's own, a year, month or week.
pub(super) fn b_period_the(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let unit = v.last()?;
    if !matches!(en::unit_grain(unit)?, TimeGrain::Year | TimeGrain::Month | TimeGrain::Week) {
        return None;
    }
    en::b_period_the(&[*unit], c)
}

/// `el día anterior`, `el día siguiente`.
pub(super) fn b_day_rel(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (n, kind) = term(v.last()?);
    if term(&v[0]).0 != words::unit(words::DAY, 1) || !matches!(kind, rel::NEXT | rel::LAST) {
        return None;
    }
    en::day_rel(c, n as i16)
}

/// `semana 27 del año pasado`.
pub(super) fn b_week_n_of(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_week_n(&[v[0], *v.last()?], c)
}

/// `la primera semana de 2015`.
pub(super) fn b_nth_week(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_nth_week(&[v[0], *v.last()?], c)
}

/// `el primer trimestre de 2017`, `el segundo semestre de 2019`.
pub(super) fn b_nth_part_of(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    nth_part(c, &v[0], &v[1], en::year_of(c, v.last()?)?)
}

/// `1º trimestre 2020`, `1º trimestre-2019`.
pub(super) fn b_nth_part_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    nth_part(c, &v[0], &v[1], en::full_year(v.last()?)?)
}

/// `el primer trimestre del año`, `2º semestre`: of no year said.
pub(super) fn b_nth_part_the_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let year = if term(&v[1]).0 == words::unit(words::MONTH_U, 6) { YearRef::Rel(0) } else { YearRef::Any };
    nth_part(c, &v[0], &v[1], year)
}

/// A quarter or a half of a year, by its ordinal.
pub(super) fn nth_part(c: &Ctx<'_>, ord: &Value, unit: &Value, year: YearRef) -> Option<Value> {
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
pub(super) fn b_decade_of(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (dig_len(v.last()?) == 4).then(|| en::b_decade(&[*v.last()?], c))?
}

/// `los años noventa`, `los años 90`.
pub(super) fn b_years_tens(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let n = term(v.last()?).0;
    (n % 10 == 0 && (10..100).contains(&n)).then(|| emit_date(c, DateSpec::Decade { year: YearRef::Short(n as u8) }))?
}

/// The kind of year a word names: `fiscal`, `escolar`; `calendario` is
/// the plain year.
pub(super) fn year_kind(c: &Ctx<'_>) -> Option<Option<YearKind>> {
    (c.start..c.end).find_map(|i| match c.word(i) {
        "fiscal" => Some(Some(YearKind::Fiscal)),
        "escolar" => Some(Some(YearKind::School)),
        "calendario" => Some(None),
        _ => None,
    })
}

pub(super) fn other_year(c: &Ctx<'_>, year: YearRef) -> Option<Value> {
    match year_kind(c)? {
        Some(kind) => emit_date(c, DateSpec::OtherYear { kind, year }),
        None => emit_date(c, en::cal(year, 0, 0, 0)?),
    }
}

pub(super) fn b_other_year(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    other_year(c, YearRef::Any).filter(|_| year_kind(c) != Some(None))
}

/// `año fiscal 2008`, `año calendario 18`.
pub(super) fn b_other_year_n(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    other_year(c, year_ref(v.last()?))
}

/// `este año escolar`, `el último año fiscal`.
pub(super) fn b_rel_other_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (n, kind) = term(v.iter().find(|x| matches!(x, Value::Term { flags: 1..=6, .. }))?);
    if !matches!(kind, rel::THIS | rel::NEXT | rel::LAST) {
        return None;
    }
    match year_kind(c)? {
        Some(k) => emit_date(c, DateSpec::OtherYear { kind: k, year: YearRef::Rel(n as i16) }),
        None => emit_date(c, DateSpec::Rel { grain: TimeGrain::Year, n: n as i16 }),
    }
}

pub(super) fn b_holiday_of(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_holiday_of_year(&[v[0], *v.last()?], c)
}

/// `la semana del 4 de febrero`.
pub(super) fn b_week_of(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    match date_of(c, v.last()?)? {
        DateSpec::Cal { year: YearRef::Any, month, day, weekday: 0 } if day > 0 => {
            emit_date(c, DateSpec::WeekOf { month, day })
        }
        _ => None,
    }
}

/// `el día 21 de este mes`.
pub(super) fn b_day_this_month(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_rel_month_day(&[v[0], *v.last()?], c)
}

pub(super) fn b_season_the(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_season(&[*v.last()?], c)
}

pub(super) fn b_season_after(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    en::b_rel_season(&[v[1], v[0]], c)
}

/// `el 4.º`: an ordinal day in digits; `el primero` is more often the
/// first of something else.
pub(super) fn b_dom_ordinal(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (c.lexeme(c.start).kind == Kind::Numeral).then(|| en::b_dom_ordinal(v, c))?
}

pub(super) fn hol(c: &Ctx<'_>, i: i64) -> Option<Value> {
    en::hol(c, i)
}
