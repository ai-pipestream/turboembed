use super::*;

/// The grain a unit word names as a period: `week`, `quarter`, `decade`.
pub(in crate::lang) fn unit_grain(v: &Value) -> Option<TimeGrain> {
    let u = term(v).0;
    Some(match (u & 255, u >> 8) {
        (0, 1) => TimeGrain::Year,
        (0, 10) => TimeGrain::Decade,
        (1, 1) => TimeGrain::Month,
        (1, 3) => TimeGrain::Quarter,
        (2, 1) => TimeGrain::Week,
        (2, 2) => TimeGrain::Week,
        (3, 1) => TimeGrain::Weekend,
        (6, 1) => TimeGrain::Day,
        _ => return None,
    })
}

pub(super) fn is_fortnight(v: &Value) -> bool {
    term(v).0 == lexicon::unit(lexicon::WEEK, 2)
}

pub(super) fn period(c: &Ctx<'_>, unit: &Value, n: i16) -> Option<Value> {
    if term(unit).1 & PLURAL != 0 {
        return None;
    }
    if is_fortnight(unit) {
        return emit_date(c, DateSpec::Fortnight { n: n as i8 });
    }
    let grain = unit_grain(unit)?;
    emit_date(c, DateSpec::Rel { grain, n })
}

pub(in crate::lang) fn b_period_rel(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let n = term(&v[0]).0 as i16;
    if unit_grain(&v[1]) == Some(TimeGrain::Day) {
        return None;
    }
    period(c, &v[1], n)
}

pub(in crate::lang) fn b_period_the(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    period(c, &v[0], 0)
}

pub(in crate::lang) fn b_weekend(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (unit_grain(&v[0]) == Some(TimeGrain::Weekend)).then(|| period(c, &v[0], 0))?
}

pub(in crate::lang) fn b_week_after_next(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::Rel { grain: TimeGrain::Week, n: 2 })
}

/// `the same week`, `that weekend`: a period the text does not place.
pub(in crate::lang) fn b_undefined(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[0]).1 & PLURAL != 0 {
        return None;
    }
    let grain = unit_grain(&v[0])?;
    if !matches!(grain, TimeGrain::Week | TimeGrain::Weekend | TimeGrain::Month | TimeGrain::Year)
        || is_fortnight(&v[0])
    {
        return None;
    }
    let point = Point::date(DateSpec::Undefined { grain });
    emit(c, TimeValue::Point { point, modifier: Modifier::portion(Portion::Undefined), clip: false })
}

pub(in crate::lang) fn b_week_n(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let w = small(&v[0])?;
    let year = match v.get(1) {
        Some(y) => year_of(c, y)?,
        None => YearRef::Rel(0),
    };
    ((1..=53).contains(&w)).then(|| emit_date(c, DateSpec::Week { year, week: w as u8 }))?
}

pub(super) fn week_of(c: &Ctx<'_>, n: u8, of: &Value) -> Option<Value> {
    if let Some(year) = year_of(c, of) {
        return emit_date(c, DateSpec::Week { year, week: n });
    }
    let (year, month) = month_of(c, of)?;
    emit_date(c, DateSpec::MonthWeek { year, month, week: n })
}

pub(in crate::lang) fn b_nth_week(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let n = nth(&v[0])?;
    (1..=5).contains(&n).then(|| week_of(c, n, &v[1]))?
}

pub(in crate::lang) fn b_last_week(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    week_of(c, 0, &v[0])
}

// Quarters, halves, decades, centuries.

pub(in crate::lang) fn b_qtr(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let q = small(&v[0])?;
    ((1..=4).contains(&q)).then(|| term_value(q, 0))?
}

pub(in crate::lang) fn quarter(c: &Ctx<'_>, year: YearRef, q: i64) -> Option<Value> {
    emit_date(c, DateSpec::Quarter { year, quarter: q as u8 })
}

pub(in crate::lang) fn b_quarter(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    quarter(c, YearRef::Any, term(&v[0]).0)
}

pub(in crate::lang) fn b_year_quarter(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    quarter(c, full_year(&v[0])?, term(v.last()?).0)
}

pub(in crate::lang) fn b_quarter_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    quarter(c, full_year(v.last()?)?, term(&v[0]).0)
}

pub(super) fn nth_quarter(v: &Value) -> Option<i64> {
    let q = i64::from(nth(v)?);
    (1..=4).contains(&q).then_some(q)
}

pub(in crate::lang) fn b_nth_quarter_of(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[1]).0 != lexicon::unit(lexicon::MONTH_U, 3) {
        return None;
    }
    quarter(c, year_of(c, &v[2])?, nth_quarter(&v[0])?)
}

pub(in crate::lang) fn b_nth_quarter_the_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[1]).0 != lexicon::unit(lexicon::MONTH_U, 3) {
        return None;
    }
    quarter(c, YearRef::Any, nth_quarter(&v[0])?)
}

/// `next year's first quarter`.
pub(in crate::lang) fn b_years_quarter(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[2]).0 != lexicon::unit(lexicon::MONTH_U, 3) {
        return None;
    }
    quarter(c, year_of(c, &v[0])?, nth_quarter(&v[1])?)
}

pub(in crate::lang) fn b_year_half(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let h = small(v.last()?)?;
    ((1..=2).contains(&h)).then(|| emit_date(c, DateSpec::Half { year: full_year(&v[0])?, half: h as u8 }))?
}

pub(in crate::lang) fn b_decade(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let y = term(&v[0]).0;
    if y % 10 != 0 {
        return None;
    }
    let year = match dig_len(&v[0]) {
        4 if !dig_zero(&v[0]) => YearRef::Abs(y as i16),
        2 => YearRef::Short(y as u8),
        _ => return None,
    };
    emit_date(c, DateSpec::Decade { year })
}

pub(in crate::lang) fn b_decade_word(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::Decade { year: YearRef::Short(term(&v[0]).0 as u8) })
}

/// `nineteen eighties`: a spelled century and a decade word.
pub(in crate::lang) fn b_century_decade(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let century = spelled(&v[0])?;
    let year = century * 100 + term(&v[1]).0;
    ((10..30).contains(&century)).then(|| emit_date(c, DateSpec::Decade { year: YearRef::Abs(year as i16) }))?
}

pub(in crate::lang) fn b_century(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let n = nth(&v[0])?;
    (1..=30).contains(&n).then(|| emit_date(c, DateSpec::Century { n }))?
}

pub(in crate::lang) fn b_workweek(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let n = match term(&v[0]).1 {
        rel::THIS | rel::UPCOMING => 0,
        rel::NEXT | rel::LAST => term(&v[0]).0 as i8,
        _ => return None,
    };
    emit_date(c, DateSpec::WorkWeek { n })
}

pub(in crate::lang) fn b_year_slice(v: &[Value], c: &Ctx<'_>, last: bool) -> Option<Value> {
    let year = year_of(c, &v[2])?;
    let m = term(&v[0]).0;
    if m <= 0 || m % 1000 != 0 {
        return None;
    }
    let count = u8::try_from(m / 1000).ok()?;
    let (u, f) = term(&v[1]);
    if f & PLURAL == 0 {
        return None;
    }
    let unit = match (u & 255, u >> 8) {
        (1, 1) => crate::time::duration::Unit::Month,
        (2, 1) => crate::time::duration::Unit::Week,
        (6, 1) => crate::time::duration::Unit::Day,
        _ => return None,
    };
    emit_date(c, DateSpec::YearSlice { year, unit, count, last })
}
