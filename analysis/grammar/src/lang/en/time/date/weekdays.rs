use super::*;

// Weekdays.

pub(in crate::lang) fn weekday(v: &Value) -> Option<u8> {
    let (w, f) = term(v);
    (f & PLURAL == 0).then_some(w as u8)
}

pub(in crate::lang) fn b_wd(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::Cal { year: YearRef::Any, month: 0, day: 0, weekday: weekday(&v[0])? })
}

pub(super) fn rel_weekday(c: &Ctx<'_>, rel_v: &Value, wd: u8) -> Option<Value> {
    let (n, kind) = term(rel_v);
    emit_date(
        c,
        match kind {
            rel::PAST => DateSpec::WeekdayPast { weekday: wd },
            rel::UPCOMING | rel::THIS => DateSpec::WeekdayRel { weekday: wd, weeks: 0 },
            _ => DateSpec::WeekdayRel { weekday: wd, weeks: n as i8 },
        },
    )
}

pub(in crate::lang) fn b_wd_rel(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    rel_weekday(c, &v[0], weekday(&v[1])?)
}

pub(in crate::lang) fn b_wd_this_rel(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if !matches!(term(&v[0]).1, rel::PAST | rel::UPCOMING) {
        return None;
    }
    rel_weekday(c, &v[0], weekday(&v[1])?)
}

pub(in crate::lang) fn b_wd_day(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::WeekdayDay { weekday: weekday(&v[0])?, day: term(&v[1]).0 as u8, this_month: false })
}

pub(in crate::lang) fn b_wd_the_day(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::WeekdayDay { weekday: weekday(&v[0])?, day: term(&v[1]).0 as u8, this_month: true })
}

pub(super) fn weeks(v: &Value) -> i8 {
    match term(v).1 {
        rel::THIS | rel::UPCOMING => 0,
        _ => term(v).0 as i8,
    }
}

/// `next week Wednesday`.
pub(in crate::lang) fn b_week_wd(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::WeekdayRel { weekday: weekday(&v[1])?, weeks: weeks(&v[0]) })
}

/// `Tuesday of next week`.
pub(in crate::lang) fn b_wd_of_week(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::WeekdayRel { weekday: weekday(&v[0])?, weeks: weeks(&v[1]) })
}

pub(in crate::lang) fn b_dom(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(YearRef::Any, 0, term(&v[0]).0, 0)?)
}

pub(in crate::lang) fn b_dom_ordinal(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (term(&v[0]).1 & ORDINAL != 0).then(|| b_dom(v, c))?
}

// Days named from the reference.

pub(in crate::lang) fn b_dayword(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (n, f) = term(&v[0]);
    (f & TONIGHT == 0).then(|| emit_date(c, DateSpec::Rel { grain: TimeGrain::Day, n: n as i16 }))?
}

pub(in crate::lang) fn day_rel(c: &Ctx<'_>, n: i16) -> Option<Value> {
    emit_date(c, DateSpec::Rel { grain: TimeGrain::Day, n })
}

pub(in crate::lang) fn b_rel_month_day(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let months = match v.get(1) {
        Some(r) => weeks(r),
        None => 0,
    };
    emit_date(c, DateSpec::RelMonthDay { months, day: term(&v[0]).0 as u8 })
}

/// The month a child names, and its year: `July`, `July 2016`, `next
/// month`.
pub(in crate::lang) fn month_of(c: &Ctx<'_>, v: &Value) -> Option<(YearRef, MonthRef)> {
    match date_of(c, v)? {
        DateSpec::Cal { year, month, day: 0, weekday: 0 } if month > 0 => Some((year, MonthRef::Abs(month))),
        DateSpec::Rel { grain: TimeGrain::Month, n } => Some((YearRef::Rel(0), MonthRef::Rel(n as i8))),
        _ => None,
    }
}

/// The year a child names: `2016`, `next year`.
pub(in crate::lang) fn year_of(c: &Ctx<'_>, v: &Value) -> Option<YearRef> {
    match date_of(c, v)? {
        DateSpec::Cal { year, month: 0, day: 0, weekday: 0 } => Some(year),
        DateSpec::Rel { grain: TimeGrain::Year, n } => Some(YearRef::Rel(n)),
        _ => None,
    }
}

pub(in crate::lang) fn nth(v: &Value) -> Option<u8> {
    match v {
        Value::Ordinal(o) => u8::try_from(o.value).ok(),
        _ => None,
    }
}

pub(in crate::lang) fn b_nth_weekday(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (year, month) = month_of(c, &v[2])?;
    emit_date(c, DateSpec::NthWeekday { year, month, weekday: weekday(&v[1])?, n: nth(&v[0])? })
}

pub(in crate::lang) fn b_last_weekday(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (year, month) = month_of(c, &v[1])?;
    emit_date(c, DateSpec::NthWeekday { year, month, weekday: weekday(&v[0])?, n: 0 })
}

/// `1st week of Oct. on Friday`.
pub(in crate::lang) fn b_nth_week_weekday(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let month = MonthRef::Abs(term(&v[1]).0 as u8);
    emit_date(c, DateSpec::NthWeekday { year: YearRef::Any, month, weekday: weekday(&v[2])?, n: nth(&v[0])? })
}

// Months.

pub(in crate::lang) fn b_month(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(YearRef::Any, term(&v[0]).0, 0, 0)?)
}

pub(in crate::lang) fn b_rel_month(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (n, kind) = term(&v[0]);
    if !matches!(kind, rel::THIS | rel::NEXT | rel::LAST) {
        return None;
    }
    emit_date(c, cal(YearRef::Rel(n as i16), term(&v[1]).0, 0, 0)?)
}

/// `April 2017`, `June of 1992`, `April of 87`: two digits only after
/// `of`.
pub(in crate::lang) fn b_month_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let y = v.last()?;
    if term(y).1 & SHORT != 0 && c.word(c.start + 1) != "of" {
        return None;
    }
    emit_date(c, cal(year_ref(y), term(&v[0]).0, 0, 0)?)
}

pub(in crate::lang) fn b_month_sep_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(full_year(&v[2])?, term(&v[0]).0, 0, 0)?)
}

pub(in crate::lang) fn b_year_month(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(full_year(&v[0])?, term(&v[1]).0, 0, 0)?)
}

/// `2015-3`, `2015-12`.
pub(in crate::lang) fn b_ym_digits(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let m = small(&v[2])?;
    ((1..=12).contains(&m)).then(|| emit_date(c, cal(year_digits(&v[0])?, m, 0, 0)?))?
}

/// `12-2015`.
pub(in crate::lang) fn b_my_digits(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let m = small(&v[0])?;
    ((1..=12).contains(&m)).then(|| emit_date(c, cal(year_digits(&v[1])?, m, 0, 0)?))?
}

/// `the third month of 2021`.
pub(in crate::lang) fn b_nth_month(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let m = i64::from(nth(&v[0])?);
    emit_date(c, cal(year_of(c, &v[1])?, m, 0, 0)?)
}

// Periods.

pub(in crate::lang) fn b_period_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[0]).1 & (SHORT | SPELLED) != 0 {
        return None;
    }
    emit_date(c, cal(year_ref(&v[0]), 0, 0, 0)?)
}

pub(in crate::lang) fn b_period_year_spelled(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[0]).1 & SPELLED == 0 {
        return None;
    }
    emit_date(c, cal(year_ref(&v[0]), 0, 0, 0)?)
}

pub(in crate::lang) fn b_year_word(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(full_year(&v[0])?, 0, 0, 0)?)
}
