use super::*;

// Seasons and other years.

pub(super) const SEASONS: [Season; 4] = [Season::Spring, Season::Summer, Season::Fall, Season::Winter];

pub(super) fn season(v: &Value) -> Option<Season> {
    SEASONS.get(usize::try_from(term(v).0).ok()?).copied()
}

pub(in crate::lang) fn b_season(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[0]).1 & crate::lang::en::time::AMBIG != 0 {
        return None;
    }
    emit_date(c, DateSpec::Season { year: YearRef::Any, season: season(&v[0])? })
}

pub(in crate::lang) fn b_rel_season(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::Season { year: YearRef::Rel(term(&v[0]).0 as i16), season: season(&v[1])? })
}

pub(in crate::lang) fn b_season_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::Season { year: full_year(v.last()?)?, season: season(&v[0])? })
}

pub(super) fn other(c: &Ctx<'_>, kind: YearKind, year: YearRef) -> Option<Value> {
    emit_date(c, DateSpec::OtherYear { kind, year })
}

pub(super) fn year_kind(c: &Ctx<'_>, i: usize) -> Option<YearKind> {
    match c.word(i) {
        "fiscal" | "fy" => Some(YearKind::Fiscal),
        "school" | "sy" => Some(YearKind::School),
        _ => None,
    }
}

pub(in crate::lang) fn b_other_year(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    other(c, year_kind(c, c.start)?, YearRef::Any)
}

pub(in crate::lang) fn b_other_year_n(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    other(c, year_kind(c, c.start)?, full_year(&v[0])?)
}

pub(in crate::lang) fn b_rel_other_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    other(c, year_kind(c, c.start + 1)?, YearRef::Rel(term(&v[0]).0 as i16))
}

/// `sy18`, `cy18`: two digits glued to the letters.
pub(super) fn short_year(v: &Value) -> Option<YearRef> {
    (dig_len(v) == 2).then(|| year_ref(&Value::Term { value: term(v).0, flags: SHORT }))
}

pub(in crate::lang) fn b_other_year_glued(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    other(c, year_kind(c, c.start)?, short_year(&v[0])?)
}

pub(in crate::lang) fn b_calendar_year_glued(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(short_year(&v[0])?, 0, 0, 0)?)
}

pub(in crate::lang) fn b_rel_calendar_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::Rel { grain: TimeGrain::Year, n: term(&v[0]).0 as i16 })
}

// Holidays.

pub(super) fn holiday(i: i64) -> Option<Holiday> {
    lexicon::HOLIDAYS.get(usize::try_from(i).ok()?).copied()
}

pub(in crate::lang) fn hol(c: &Ctx<'_>, i: i64) -> Option<Value> {
    emit_date(c, DateSpec::Holiday { holiday: holiday(i)?, year: YearRef::Any })
}

pub(in crate::lang) fn b_holiday(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    hol(c, term(&v[0]).0)
}

pub(super) fn holiday_of(c: &Ctx<'_>, v: &Value) -> Option<Holiday> {
    match date_of(c, v)? {
        DateSpec::Holiday { holiday, year: YearRef::Any } => Some(holiday),
        _ => None,
    }
}

pub(in crate::lang) fn b_holiday_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::Holiday { holiday: holiday_of(c, &v[0])?, year: full_year(v.last()?)? })
}

pub(in crate::lang) fn b_holiday_of_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, DateSpec::Holiday { holiday: holiday_of(c, &v[0])?, year: year_of(c, &v[1])? })
}

pub(in crate::lang) fn b_holiday_near(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (n, kind) = term(&v[0]);
    if !matches!(kind, rel::NEXT | rel::LAST) {
        return None;
    }
    emit_date(c, DateSpec::HolidayNear { holiday: holiday_of(c, &v[1])?, future: n > 0 })
}

pub(in crate::lang) fn b_holiday_weekend(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (holiday, year) = match date_of(c, v.last()?)? {
        DateSpec::Holiday { holiday, year } => (holiday, year),
        _ => return None,
    };
    emit_date(c, DateSpec::HolidayWeekend { holiday, year })
}

pub(in crate::lang) fn b_holiday_weekend_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let holiday = holiday_of(c, &v[0])?;
    let year = match v.get(1) {
        Some(y) => full_year(y)?,
        None => YearRef::Any,
    };
    emit_date(c, DateSpec::HolidayWeekend { holiday, year })
}

/// `the week of April 10th`, `w/c Feb 4`.
pub(in crate::lang) fn b_week_of(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    match date_of(c, v.last()?)? {
        DateSpec::Cal { year: YearRef::Any, month, day, weekday: 0 } if day > 0 => {
            emit_date(c, DateSpec::WeekOf { month, day })
        }
        _ => None,
    }
}

pub(in crate::lang) fn b_sep(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Empty)
}
